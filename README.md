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
        +-- evaluate.py
```

## How It Works

1. `ingest.py` loads `data/reference.pdf` and splits the document into text chunks.
2. `retriever.py` converts chunks into embeddings and saves them in a FAISS vector index.
3. `chain.py` builds a RAG pipeline that retrieves the top matching chunks and sends them to the LLM with the user question.
4. `evaluate.py` runs a small Ragas evaluation suite against sample medical questions.

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
pip install langchain langchain-community langchain-openai langchain-text-splitters faiss-cpu ragas datasets python-dotenv pypdf openai
```

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

### 4. Evaluate the RAG Pipeline

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

For experiments, tune chunk size, chunk overlap, retrieval `k`, prompt wording, and model choice.

## What Could Break

- The pipeline requires a valid `OPENAI_API_KEY`.
- If `reference.pdf` changes, the FAISS index should be rebuilt.
- The code assumes commands are run from the `rag_assistant/` directory.
- `FAISS.load_local(..., allow_dangerous_deserialization=True)` should only be used with indexes you trust.
- A chunk overlap equal to the chunk size may create highly redundant chunks; consider using a smaller overlap such as `50` to `100`.
- Checked-in virtual environments and `.env` files can make the repository large and may expose secrets if real keys are committed.

## How To Validate

After setup, validate the project in stages:

1. Run `python src/ingest.py` and confirm the PDF loads and chunks are created.
2. Run `python src/retriever.py` and confirm FAISS stores vectors successfully.
3. Run `python src/chain.py` and check that in-document questions receive grounded answers.
4. Ask an unrelated question and verify the assistant refuses to answer from outside knowledge.
5. Run `python src/evaluate.py` and inspect the Ragas metric averages.

## Future Improvements

- Add a `requirements.txt` file with pinned dependency versions.
- Move model names, chunking values, and index paths into a config file.
- Add a command-line interface for custom questions.
- Support multiple PDFs or a folder of documents.
- Store source metadata such as page number in final answers.
- Add automated tests for ingestion, retrieval, and refusal behavior.
- Remove local `venv/`, `__pycache__/`, and secret files from version control.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
