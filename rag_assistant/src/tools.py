from langchain_core.tools import tool


def build_tools(vector_store, k: int = 3):
    """
    Build the tool set available to the healthcare ReAct agent.

    Tools are built as closures over `vector_store` (instead of module-level
    globals) so the agent can be rebuilt against a different index/k without
    import-time side effects.
    """

    def _search_guidelines(query: str) -> str:
        docs = vector_store.similarity_search(query, k=k)
        if not docs:
            return "No relevant passages found in the clinical guideline document."

        formatted = []
        for i, doc in enumerate(docs, start=1):
            page = doc.metadata.get("page")
            page_info = f" (page {page + 1})" if isinstance(page, int) else ""
            formatted.append(f"[Passage {i}{page_info}]\n{doc.page_content.strip()}")
        return "\n\n".join(formatted)

    @tool
    def search_clinical_guidelines(query: str) -> str:
        """Search the ingested clinical guideline document for passages relevant to
        a medical question (symptoms, diagnosis criteria, treatment regimens,
        dosing, etc). Call this whenever you need a grounded medical fact. For
        multi-part or comparative questions, call it more than once with
        different, specific queries rather than trying to answer from one search."""
        return _search_guidelines(query)

    @tool
    def calculate_dose(weight_kg: float, dose_per_kg_mg: float, doses_per_day: int = 1) -> str:
        """Calculate a weight-based medication dose. Use this after retrieving a
        mg/kg dosing regimen from the guideline so the arithmetic is exact rather
        than estimated by the model. Returns the per-dose amount and the total
        daily amount in mg."""
        if weight_kg <= 0 or dose_per_kg_mg <= 0 or doses_per_day <= 0:
            return "Invalid input: weight, dose per kg, and doses per day must all be positive numbers."

        per_dose = weight_kg * dose_per_kg_mg
        daily_total = per_dose * doses_per_day
        return (
            f"Per-dose amount: {per_dose:.1f} mg, given {doses_per_day}x/day. "
            f"Total daily dose: {daily_total:.1f} mg."
        )

    return [search_clinical_guidelines, calculate_dose]
