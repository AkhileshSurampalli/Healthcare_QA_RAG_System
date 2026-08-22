import os

import pandas as pd
from langchain_core.tools import tool

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DOSING_TABLE_PATH = os.path.join(BASE_DIR, "data", "dosing_table.csv")


def build_tools(vector_store, k: int = 3, dosing_table_path: str = DEFAULT_DOSING_TABLE_PATH):
    """
    Build the tool set available to the healthcare ReAct agent, covering three
    kinds of sources:
    - Unstructured text (PDF guideline + web reference pages), searched by
      semantic similarity via `search_clinical_guidelines`.
    - A structured dosing table (CSV), looked up by exact drug/indication match
      via `lookup_dosing_table` rather than semantic search - dosing numbers
      should come from a structured record, not a fuzzy text match.
    - Arithmetic on whatever dosing numbers either of the above returns, via
      `calculate_dose`.

    Tools are built as closures over `vector_store` (instead of module-level
    globals) so the agent can be rebuilt against a different index/k without
    import-time side effects.
    """

    def _search_guidelines(query: str) -> str:
        docs = vector_store.similarity_search(query, k=k)
        if not docs:
            return "No relevant passages found in the ingested sources."

        formatted = []
        for i, doc in enumerate(docs, start=1):
            source = doc.metadata.get("source", "unknown source")
            page = doc.metadata.get("page")
            location = f"{source}, page {page + 1}" if isinstance(page, int) else source
            formatted.append(f"[Passage {i} - {location}]\n{doc.page_content.strip()}")
        return "\n\n".join(formatted)

    @tool
    def search_clinical_guidelines(query: str) -> str:
        """Search the ingested unstructured sources (the clinical guideline PDF plus
        reference web pages on malaria, pneumonia, and diabetes) for passages
        relevant to a medical question: symptoms, diagnosis criteria, general
        treatment approach, background/epidemiology. Each result is tagged with
        which source it came from. For multi-part or comparative questions, call
        this more than once with different, specific queries rather than trying
        to answer from one search. For exact drug dosing, prefer
        `lookup_dosing_table` instead - it's a structured source, not a text match."""
        return _search_guidelines(query)

    def _cell(value) -> str:
        return "not specified" if pd.isna(value) else str(value)

    def _lookup_dosing(drug: str, indication: str = "") -> str:
        try:
            table = pd.read_csv(dosing_table_path)
        except FileNotFoundError:
            return f"Dosing table not found at {dosing_table_path}."

        matches = table[table["drug"].str.contains(drug, case=False, na=False)]
        if indication:
            matches = matches[matches["indication"].str.contains(indication, case=False, na=False)]

        if matches.empty:
            available = ", ".join(sorted(table["drug"].unique()))
            return f"No dosing entry found for '{drug}'. Available drugs in the table: {available}."

        formatted = []
        for _, row in matches.iterrows():
            max_dose = _cell(row["max_single_dose"])
            max_dose_unit = "" if max_dose == "not specified" else f" {_cell(row['max_dose_unit'])}"
            formatted.append(
                f"[Structured source: dosing_table.csv - sample data, verify against "
                f"an authoritative guideline before real-world use]\n"
                f"Drug: {row['drug']}\n"
                f"Indication: {row['indication']}\n"
                f"Population: {row['population']}\n"
                f"Dose: {_cell(row['dose_per_kg'])} {row['dose_unit']} per dose, "
                f"{row['frequency_per_day']}x/day\n"
                f"Max single dose: {max_dose}{max_dose_unit}\n"
                f"Notes: {row['notes']}"
            )
        return "\n\n".join(formatted)

    @tool
    def lookup_dosing_table(drug: str, indication: str = "") -> str:
        """Look up a medication's dosing regimen in the structured dosing table by
        drug name (and optionally indication, e.g. "pneumonia"). Use this instead
        of `search_clinical_guidelines` whenever you need exact dosing numbers -
        it's a structured record, not a similarity match, so it won't return the
        wrong drug's numbers. If the returned dose is a simple per-kg amount
        (dose_per_kg is numeric, not a weight-band or fixed-dose note), follow up
        with `calculate_dose` for the patient's actual weight. Not every drug in
        this project's sources is in the table; if it isn't, fall back to
        `search_clinical_guidelines`."""
        return _lookup_dosing(drug, indication)

    @tool
    def calculate_dose(weight_kg: float, dose_per_kg_mg: float, doses_per_day: int = 1) -> str:
        """Calculate a weight-based medication dose. Use this after getting a
        numeric mg/kg dosing regimen from `lookup_dosing_table` (or, failing that,
        `search_clinical_guidelines`) so the arithmetic is exact rather than
        estimated by the model. Returns the per-dose amount and the total daily
        amount in mg."""
        if weight_kg <= 0 or dose_per_kg_mg <= 0 or doses_per_day <= 0:
            return "Invalid input: weight, dose per kg, and doses per day must all be positive numbers."

        per_dose = weight_kg * dose_per_kg_mg
        daily_total = per_dose * doses_per_day
        return (
            f"Per-dose amount: {per_dose:.1f} mg, given {doses_per_day}x/day. "
            f"Total daily dose: {daily_total:.1f} mg."
        )

    return [search_clinical_guidelines, lookup_dosing_table, calculate_dose]
