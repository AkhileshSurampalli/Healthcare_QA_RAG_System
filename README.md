# RAG Project

A Retrieval-Augmented Generation (RAG) assistant built with LangChain, OpenAI, FAISS, and Ragas.

The project loads a PDF document, splits it into chunks, embeds the chunks with OpenAI embeddings, stores them in a local FAISS vector index, retrieves relevant context for a user question, and generates grounded answers using an OpenAI chat model.

## Features

- PDF document ingestion with `PyPDFLoader`
- Recursive text chunking for retrieval-friendly document splits
- OpenAI embeddings using `text-embedding-3-small`
- Local FAISS vector store for similarity search
- RAG answer generation with LangChain LCEL
- Source chunk display for retrieved context
- Ragas-based evaluation for faithfulness, answer relevancy, context precision, and context recall
- A ReAct healthcare reasoning agent (`langchain.agents.create_agent`) that decides
  when to search the guideline, issues multiple targeted searches for multi-part
  questions, and runs an exact weight-based dose calculation instead of a single
  static retrieve-then-generate pass

## Project Structure

```text
RAG_project/
+-- LICENSE
+-- README.md
+-- rag_assistant/
    +-- data/
    |   +-- reference.pdf
    +-- faiss_index/
    |   +-- index.faiss
    |   +-- index.pkl
    +-- src/
        +-- __init__.py
        +-- ingest.py
        +-- retriever.py
        +-- chain.py
        +-- tools.py
        +-- agent.py
        +-- evaluate.py
```

## How It Works

1. `ingest.py` loads `data/reference.pdf` and splits the document into text chunks.
2. `retriever.py` converts chunks into embeddings and saves them in a FAISS vector index.
3. `chain.py` builds a **static** RAG pipeline: retrieve top-k chunks once, stuff them into a prompt, generate one answer. Good for simple, single-fact lookups.
4. `tools.py` + `agent.py` build a **ReAct agent** on top of the same FAISS index: instead of one fixed retrieve → generate pass, the agent reasons about the question, decides whether and what to search for, can issue several searches for multi-part or comparative questions, and can call an exact dose-calculation tool rather than doing arithmetic itself. See [Reason About Healthcare Questions with the ReAct Agent](#4-reason-about-healthcare-questions-with-the-react-agent) below.
5. `evaluate.py` runs a small Ragas evaluation suite against sample medical questions (against the static chain).

## Requirements

- Python 3.10+
- OpenAI API key
- LangChain
- FAISS
- Ragas
- Hugging Face `datasets`
- Python dotenv

Install dependencies:

```bash
pip install langchain langchain-community langchain-openai langchain-text-splitters langgraph faiss-cpu ragas datasets python-dotenv pypdf openai
```

`langgraph` is required by `langchain.agents.create_agent` (used by the ReAct agent in `agent.py`); `langchain>=1.0` is required for that API.

## Environment Setup

Create a `.env` file inside the `rag_assistant/` directory:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

Do not commit real API keys to GitHub.

## Usage

Run all commands from inside the `rag_assistant/` directory:

```bash
cd rag_assistant
```

### 1. Load and Chunk the PDF

```bash
python src/ingest.py
```

This loads `data/reference.pdf`, splits it into chunks, and prints a sample chunk.

### 2. Build the FAISS Vector Store

```bash
python src/retriever.py
```

This creates embeddings for the document chunks and saves the FAISS index to:

```text
faiss_index/
```

The current implementation uses:

- Embedding model: `text-embedding-3-small`
- Vector database: FAISS
- Retrieval method: similarity search
- Retrieved chunks: top 3

### 3. Ask Questions with the RAG Chain

```bash
python src/chain.py
```

The script loads the existing FAISS index, builds the RAG chain, and runs example questions such as:

```text
What are the symptoms of malaria?
How should diabetes be managed?
What is the recommended treatment for pneumonia?
What is the capital of France?
```

The prompt is designed to answer only from retrieved context. If the answer is not in the document, the assistant should respond:

```text
I don't have enough information in the document to answer this.
```

### 4. Reason About Healthcare Questions with the ReAct Agent

```bash
python src/agent.py
```

or ask a custom question directly:

```bash
python src/agent.py "A child with pneumonia weighs 12 kg. What is the amoxicillin dose?"
```

This loads the same FAISS index and builds a **ReAct agent** (`langchain.agents.create_agent`) instead of a fixed retrieve-then-generate chain. On every question the agent:

1. **Thinks** about what it needs to know.
2. **Acts** by calling a tool — `search_clinical_guidelines` (semantic search over the guideline, callable multiple times with different queries) or `calculate_dose` (exact weight-based dosing arithmetic, so the LLM never has to "guess" numbers).
3. **Observes** the tool result and decides whether it has enough grounded information yet, or needs to search again / calculate / ask a different sub-question.
4. Only then produces a final answer, citing which retrieved passages it relied on.

`ask()` in `agent.py` prints the full `Thought → Action → Observation → Answer` trace, not just the final text, so you can see the reasoning steps rather than a single opaque output.

#### Why this is different from `chain.py`

| | `chain.py` (static RAG) | `agent.py` (ReAct agent) |
|---|---|---|
| Retrieval | Always exactly one fixed top-k search | Agent decides if/when/how many times to search, and what to search for |
| Multi-part questions | One search covers the whole question (context can be diluted or incomplete) | Can decompose into multiple targeted searches, one per sub-question |
| Numeric reasoning (e.g. dosing) | LLM computes arithmetic itself in the generated text (error-prone) | Delegates arithmetic to a deterministic `calculate_dose` tool |
| Visibility | Only the final answer + raw source chunks | Full step-by-step reasoning trace (Thought/Action/Observation) |

Example questions that exercise multi-step reasoning (see `if __name__ == "__main__"` in `agent.py`):

```text
What are the symptoms of malaria and how is it treated?
Compare how malaria and pneumonia are managed according to the guideline.
A child with pneumonia weighs 12 kg. Using the amoxicillin dosing regimen in the guideline, what is their per-dose and daily dose in mg?
What is the capital of France?
```

The last question is intentionally out of scope — like the static chain, the agent should refuse rather than answer from outside the document.

### 5. Evaluate the RAG Pipeline

```bash
python src/evaluate.py
```

The evaluation script uses Ragas metrics:

- Faithfulness
- Answer relevancy
- Context precision
- Context recall

These metrics help measure whether the generated answer is grounded in the retrieved context and whether the retrieved chunks are useful.

## Example Workflow

```bash
cd rag_assistant
python src/ingest.py
python src/retriever.py
python src/chain.py
python src/agent.py
python src/evaluate.py
```

## Configuration

Important settings are currently defined directly in the source files:

- `src/ingest.py`
  - `chunk_size=500`
  - `chunk_overlap=500`
- `src/retriever.py`
  - embedding model: `text-embedding-3-small`
  - FAISS index path: `faiss_index`
- `src/chain.py`
  - chat model: `gpt-4o-mini`
  - retrieval `k=3`
  - temperature: `0`
- `src/tools.py` / `src/agent.py`
  - retrieval `k=3` (via `build_tools(vector_store, k=3)`)
  - chat model: `gpt-4o-mini`, temperature: `0`
  - system prompt controlling ReAct behavior and refusal wording

For experiments, tune chunk size, chunk overlap, retrieval `k`, prompt wording, and model choice.

## What Could Break

- The pipeline requires a valid `OPENAI_API_KEY`.
- If `reference.pdf` changes, the FAISS index should be rebuilt.
- The code assumes commands are run from the `rag_assistant/` directory.
- `FAISS.load_local(..., allow_dangerous_deserialization=True)` should only be used with indexes you trust.
- Checked-in virtual environments and `.env` files can make the repository large and may expose secrets if real keys are committed.
- `src/agent.py` requires `langchain>=1.0` (for `langchain.agents.create_agent`) and `langgraph` installed; an older `langchain` will raise `ImportError`.
- The agent makes at least one extra LLM call per tool invocation compared to the static chain, so it is slower and uses more tokens per question in exchange for more thorough, inspectable reasoning.

## How To Validate

After setup, validate the project in stages:

1. Run `python src/ingest.py` and confirm the PDF loads and chunks are created.
2. Run `python src/retriever.py` and confirm FAISS stores vectors successfully.
3. Run `python src/chain.py` and check that in-document questions receive grounded answers.
4. Ask an unrelated question and verify the assistant refuses to answer from outside knowledge.
5. Run `python src/agent.py` and confirm the printed trace shows `[Action]` / `[Observation]` steps before the `[Answer]`, that the dosing question triggers both `search_clinical_guidelines` and `calculate_dose`, and that the out-of-scope question is refused.
6. Run `python src/evaluate.py` and inspect the Ragas metric averages.

## Future Improvements

- Add a `requirements.txt` file with pinned dependency versions.
- Move model names, chunking values, and index paths into a config file.
- Support multiple PDFs or a folder of documents.
- Add automated tests for ingestion, retrieval, and refusal behavior.
- Add Ragas-style evaluation for the ReAct agent (not just the static chain), e.g. scoring tool-call correctness and final-answer groundedness.
- Remove local `venv/`, `__pycache__/`, and secret files from version control.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
