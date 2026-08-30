# Healthcare Q&A RAG System

A Retrieval-Augmented Generation (RAG) assistant built with LangChain, OpenAI, FAISS, and Ragas.

The project ingests **three kinds of sources** — a PDF clinical guideline, reference web pages, and a structured dosing table — embeds/indexes the unstructured ones in a local FAISS vector index, and answers questions either with a single static retrieve-then-generate pass or with a ReAct agent that reasons across all three sources.

## Features

- Multi-source ingestion: PDF (`PyPDFLoader`), web pages (`WebBaseLoader`), and a structured CSV dosing table (`pandas`) — see [Multi-Source Architecture](#multi-source-architecture)
- Recursive text chunking for retrieval-friendly document splits
- OpenAI embeddings using `text-embedding-3-small`
- Local FAISS vector store for similarity search across both unstructured sources
- RAG answer generation with LangChain LCEL
- Source chunk display for retrieved context, tagged by which document or web page it came from
- Ragas-based evaluation for faithfulness, answer relevancy, context precision, and context recall
- A ReAct healthcare reasoning agent (`langchain.agents.create_agent`) that decides
  which source to consult (unstructured search vs. the structured dosing table),
  issues multiple targeted searches for multi-part questions, and runs an exact
  weight-based dose calculation instead of a single static retrieve-then-generate pass
- A FastAPI service exposing both the static chain and the ReAct agent as HTTP endpoints
- A minimal browser frontend (vanilla HTML/JS, no build step) served by that same FastAPI app
- One pinned `requirements.txt` and a single project venv (`rag_assistant/.venv`)
- A `Dockerfile` that bakes in the pre-built FAISS index, so the built image is the deployable unit — see [Deployment](#deployment)

## Project Structure

```text
RAG_project/
+-- LICENSE
+-- README.md
+-- .gitignore
+-- rag_assistant/
    +-- .env                  (not committed)
    +-- .gitignore
    +-- .dockerignore
    +-- Dockerfile
    +-- requirements.txt
    +-- data/
    |   +-- reference.pdf
    |   +-- dosing_table.csv
    +-- faiss_index/
    |   +-- index.faiss
    |   +-- index.pkl
    +-- frontend/
    |   +-- index.html
    +-- src/
        +-- __init__.py
        +-- ingest.py
        +-- retriever.py
        +-- chain.py
        +-- tools.py
        +-- agent.py
        +-- api.py
        +-- evaluate.py
        +-- evaluate_agent.py
```

## How It Works

1. `ingest.py` loads `data/reference.pdf` **and** fetches/parses reference web pages, splitting both into text chunks tagged with where each one came from.
2. `retriever.py` combines the PDF chunks and web chunks, embeds all of them, and saves one FAISS vector index covering both unstructured sources.
3. `chain.py` builds a **static** RAG pipeline: retrieve top-k chunks once (from across both unstructured sources), stuff them into a prompt, generate one answer. Good for simple, single-fact lookups. It doesn't know about the structured dosing table.
4. `tools.py` + `agent.py` build a **ReAct agent** with three tools: `search_clinical_guidelines` (semantic search over the combined PDF + web index), `lookup_dosing_table` (exact lookup against the structured CSV), and `calculate_dose` (arithmetic). The agent decides which tool(s) a question needs, can issue several searches for multi-part or comparative questions, and prefers the structured lookup over a text search whenever exact dosing numbers matter. See [Reason About Healthcare Questions with the ReAct Agent](#4-reason-about-healthcare-questions-with-the-react-agent) and [Multi-Source Architecture](#multi-source-architecture) below.
5. `evaluate.py` runs a small Ragas evaluation suite against sample medical questions (against the static chain).
6. `evaluate_agent.py` scores the agent two ways: the same Ragas metrics against its aggregated retrieved passages, plus behavior checks that assert it actually called the right tools (multi-search, the structured lookup, the dose calculator, refusal).
7. `api.py` wraps the static chain and the agent in a FastAPI service (`/ask` and `/agent/ask`) so either can be called over HTTP instead of run as a script.

## Multi-Source Architecture

The project pulls from three sources, each handled the way its data actually behaves instead of forcing everything through one pipeline:

| Source | Loader | Where it lives | How it's queried |
|---|---|---|---|
| `data/reference.pdf` (WHO clinical guideline) | `PyPDFLoader` | FAISS index | Semantic search (`search_clinical_guidelines`) |
| Reference web pages (Wikipedia: malaria, pneumonia, diabetes) | `WebBaseLoader` | Same FAISS index | Semantic search (`search_clinical_guidelines`) |
| `data/dosing_table.csv` (drug dosing records) | `pandas.read_csv` | Not indexed — read directly | Exact filtered lookup (`lookup_dosing_table`) |

The two unstructured sources (PDF + web) are chunked the same way and land in **one combined FAISS index** — `search_clinical_guidelines` searches across both and cites which one each result came from (`doc.metadata["source"]` is the PDF path or the URL; `page` is only present for PDF chunks). They're combined rather than split into separate indices/tools because there's no benefit to forcing the agent to pick a text source ahead of time — semantic similarity already finds the right passage regardless of which document it's in.

The dosing table is deliberately **not** put through the vector store. Dosing numbers need an exact match, not a similarity match — embedding "amoxicillin 25mg/kg" and hoping it's the nearest neighbor to a question risks the agent citing the wrong drug's numbers. `lookup_dosing_table` does a direct filtered read of the CSV instead, and `agent.py`'s system prompt tells the agent to prefer it over text search whenever a question needs a specific dose.

**Important:** `data/dosing_table.csv` contains illustrative sample data for this exercise, not verified clinical guidance — every row says so in its `notes` field and the agent is instructed to flag it as such in any answer that uses it. Don't treat it as a real dosing reference.

**Adding a new source:**
- Another PDF or a folder of PDFs → extend `load_pdf_chunks` (or add a new `load_and_chunk`-style function) and add its output to the `chunks` list in `retriever.py`'s `__main__`.
- Another web page → add its URL to `DEFAULT_WEB_SOURCES` in `ingest.py`. Check first that the page is plain server-rendered HTML (view source, confirm the real text is there) — see the "What Could Break" note on JS-hydrated pages.
- Another structured dataset → follow the `lookup_dosing_table` pattern: a dedicated tool with exact/filtered lookup, not a vector search.

## Requirements

- Python 3.11+
- OpenAI API key

All dependencies are pinned in `rag_assistant/requirements.txt` — one file, one venv, for every script and the API. Set it up once:

```bash
cd rag_assistant
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

Notable pins:

- `langchain-community==0.4.1`, **not** the latest 0.4.2. `ragas==0.4.3` unconditionally imports `langchain_community.chat_models.vertexai.ChatVertexAI`, which was removed from `langchain-community` in 0.4.2. Bumping `langchain-community` without checking that import first will break `evaluate.py` and `evaluate_agent.py` with `ModuleNotFoundError`.
- `langchain==1.3.4` / `langgraph==1.2.4` — required for `langchain.agents.create_agent`, used by the ReAct agent in `agent.py`.
- `faiss-cpu==1.14.2` — the vector store backend.
- `fastapi`/`uvicorn` — only needed to run `api.py`.

If you previously had separate venvs per script (e.g. one missing `faiss-cpu`, another missing `ragas`), delete them and use the single `rag_assistant/.venv` above instead — running different scripts from different environments is what caused that split in the first place.

## Environment Setup

Create a `.env` file inside the `rag_assistant/` directory:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

`rag_assistant/.env` is listed in `.gitignore` and should never be committed. It was accidentally tracked in earlier commits (with only the placeholder key value, no real secret) and has since been untracked — if you're on an older clone, `git rm --cached rag_assistant/.env` before committing anything else.

## Usage

Run all commands from inside the `rag_assistant/` directory:

```bash
cd rag_assistant
```

### 1. Load and Chunk the Sources

```bash
python src/ingest.py
```

This loads `data/reference.pdf` **and** fetches the default web sources (Wikipedia: malaria, pneumonia, diabetes), splits each into chunks, and prints a sample chunk from a PDF source and a sample chunk from a web source.

### 2. Build the FAISS Vector Store

```bash
python src/retriever.py
```

This combines the PDF chunks and web chunks, creates embeddings for all of them, and saves one FAISS index (covering both unstructured sources) to:

```text
faiss_index/
```

Rebuilding this after `ingest.py` changes (a new PDF, a new web source, different chunk size) means re-embedding everything, which calls the OpenAI API once per chunk — for the default sources that's ~5,000 chunks, so expect it to take a few minutes and cost a small amount of API credit. The structured dosing table is **not** part of this index — see [Multi-Source Architecture](#multi-source-architecture).

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

1. **Thinks** about what it needs to know and which source is right for it.
2. **Acts** by calling a tool — `search_clinical_guidelines` (semantic search over the combined PDF + web index, callable multiple times with different queries), `lookup_dosing_table` (exact structured lookup against `dosing_table.csv`, preferred over search whenever a specific dose is needed), or `calculate_dose` (exact weight-based dosing arithmetic, so the LLM never has to "guess" numbers).
3. **Observes** the tool result and decides whether it has enough grounded information yet, or needs to search again / look up / calculate / ask a different sub-question.
4. Only then produces a final answer, citing which source (document, web page, or structured table) it relied on.

`ask()` in `agent.py` prints the full `Thought → Action → Observation → Answer` trace, not just the final text, so you can see the reasoning steps rather than a single opaque output.

#### Why this is different from `chain.py`

| | `chain.py` (static RAG) | `agent.py` (ReAct agent) |
|---|---|---|
| Sources | Unstructured index only (PDF + web) | Unstructured index **and** the structured dosing table |
| Retrieval | Always exactly one fixed top-k search | Agent decides if/when/how many times to search, and what to search for |
| Multi-part questions | One search covers the whole question (context can be diluted or incomplete) | Can decompose into multiple targeted searches, one per sub-question |
| Numeric reasoning (e.g. dosing) | LLM computes arithmetic itself in the generated text, from a text passage (error-prone twice over) | Looks up exact numbers via `lookup_dosing_table`, then delegates arithmetic to `calculate_dose` |
| Visibility | Only the final answer + raw source chunks | Full step-by-step reasoning trace (Thought/Action/Observation) |

Example questions that exercise multi-step reasoning (see `if __name__ == "__main__"` in `agent.py`):

```text
What are the symptoms of malaria and how is it treated?
Compare how malaria and pneumonia are managed according to the sources.
A child with pneumonia weighs 12 kg. What is the amoxicillin per-dose and daily dose in mg?
What is the dosing for artemether-lumefantrine and can I calculate a per-kg dose for it?
What is the capital of France?
```

The artemether-lumefantrine question is a deliberate edge case: that drug's entry in `dosing_table.csv` is weight-band dosed, not a simple mg/kg number, so the agent should look it up but *not* blindly feed it to `calculate_dose`. The France question is intentionally out of scope — like the static chain, the agent should refuse rather than answer from outside the ingested sources.

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

### 6. Evaluate the ReAct Agent

```bash
python src/evaluate_agent.py
```

This scores the agent two ways, since Ragas alone can't tell whether the agent actually reasoned:

- **Ragas metrics** — the same 10-question set from `evaluate.py`, run through the agent instead of the chain, with its retrieved passages aggregated across however many `search_clinical_guidelines` calls it made per question.
- **Behavior checks** — 4 cases the Ragas set can't exercise: the pediatric dosing question (must call both `lookup_dosing_table` and `calculate_dose`), the malaria-vs-pneumonia comparison (must call `search_clinical_guidelines` at least twice), the artemether-lumefantrine edge case (must call `lookup_dosing_table` but must **not** call `calculate_dose`, since that drug isn't simple mg/kg), and an out-of-scope question (must produce the exact refusal text). Each prints PASS/FAIL with the actual tool-call counts.

### 7. Run the API

```bash
python src/api.py
# or: uvicorn src.api:app --reload
```

Starts a FastAPI service on `http://localhost:8000` that loads the vector store and builds the chain/agent once at startup, then serves both over HTTP:

| Endpoint | Behavior |
|---|---|
| `GET /` | The browser frontend (`frontend/index.html`) |
| `GET /health` | Liveness check |
| `POST /ask` | Static chain — `{"question": "..."}` → `{"answer": "...", "sources": [{"content": "...", "source": "data/reference.pdf", "page": 3}, ...]}` (`source` is a file path or URL; `page` is only present for PDF chunks) |
| `POST /agent/ask` | ReAct agent — `{"question": "..."}` → `{"answer": "...", "trace": [{"type": "action", "tool": "...", "args": {...}}, {"type": "observation", "tool": "...", "content": "..."}, {"type": "answer", "content": "..."}]}` |

Interactive docs are auto-generated at `http://localhost:8000/docs`.

```bash
curl -X POST http://localhost:8000/agent/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "A child with pneumonia weighs 12 kg. What is the amoxicillin dose?"}'
```

The `trace` array in the agent response is the same Thought/Action/Observation data `agent.py`'s `ask()` prints to the console — `api.py` returns it as structured JSON instead of printing it, via the shared `run_with_trace()` function in `agent.py`.

#### The frontend

Open `http://localhost:8000/` in a browser: a text box, a "ReAct Agent" / "Static Chain" toggle, and a results panel. It's a single static HTML file (`frontend/index.html`) with no framework and no build step — vanilla JS `fetch()` calls to `/ask` and `/agent/ask` — mounted onto the same FastAPI app via `StaticFiles`:

```python
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
```

This mount is registered **after** the `/health`, `/ask`, and `/agent/ask` routes in `api.py`. Starlette matches routes in registration order, so those exact paths are still handled by their own functions; the mount only catches everything else (`/`, and would 404 anything not in `frontend/`). One consequence worth knowing: because the frontend is served from the same origin as the API, there's no CORS configuration anywhere — `fetch("/ask")` is a same-origin request. If you ever split the frontend out to its own domain (a separate static host, S3+CloudFront, etc.), you'd need to add `CORSMiddleware` to `api.py` and enable it for that origin.

For the agent tab, the trace renders as color-coded Action / Observation / Answer blocks — the same three step types `run_with_trace()` returns.

## Deployment

The whole app — API, agent, and frontend — is one FastAPI process, which makes it one deployable unit: a container. `Dockerfile` bakes in the pre-built `faiss_index/`, so the image doesn't need `OPENAI_API_KEY` (or network access) at *build* time, only at *run* time for the actual chat/embedding calls.

### 1. Build and run the image locally

```bash
cd rag_assistant
docker build -t healthcare-rag-assistant .
docker run -d --name rag-app -p 8000:8000 -e OPENAI_API_KEY=your_real_key healthcare-rag-assistant
```

Then open `http://localhost:8000/`. Check logs with `docker logs rag-app`; stop it with `docker rm -f rag-app`.

Two things baked into the `Dockerfile` on purpose:
- `data/`, `faiss_index/`, `frontend/`, and `src/` are copied in; `.env` is **not** (see `.dockerignore`) — the key is always passed in as a runtime environment variable, never as an image layer, so it can't leak if the image is ever pushed somewhere public.
- The index is copied in as-is rather than rebuilt during `docker build`. If you change `ingest.py` or re-run `retriever.py` locally, you must rebuild the image afterward for the new index to ship — the container has no logic to detect or rebuild a stale index itself.

### 2. Push the image to Amazon ECR

Everything below assumes the AWS CLI is installed and `aws configure` has already been run with credentials that can create ECR/App Runner resources.

```bash
aws ecr create-repository --repository-name healthcare-rag-assistant --region <your-region>

aws ecr get-login-password --region <your-region> \
  | docker login --username AWS --password-stdin <account-id>.dkr.ecr.<your-region>.amazonaws.com

docker tag healthcare-rag-assistant:latest \
  <account-id>.dkr.ecr.<your-region>.amazonaws.com/healthcare-rag-assistant:latest

docker push <account-id>.dkr.ecr.<your-region>.amazonaws.com/healthcare-rag-assistant:latest
```

(`<account-id>` is your 12-digit AWS account ID, visible in the console top-right or via `aws sts get-caller-identity`.)

### 3. Run it on AWS App Runner (recommended)

App Runner is the least AWS infrastructure to manage for a single containerized web service: point it at an image, it handles the load balancer, HTTPS, and scaling. No VPC/ALB/ECS task definitions to write by hand.

1. AWS Console → **App Runner** → **Create service**.
2. Source: **Container registry** → **Amazon ECR** → pick the image you just pushed.
3. Deployment trigger: **Manual** (simplest to start — automatic redeploys on every image push is a later refinement).
4. Service settings:
   - Port: `8000`
   - CPU/memory: at least **1 vCPU / 2 GB** — `langchain` + `ragas` + `faiss` + `pandas` have real import overhead, and the FAISS index (~5,000 vectors) sits in memory alongside them.
   - Environment variables: add `OPENAI_API_KEY` with your real key. (For anything beyond a personal demo, use **AWS Secrets Manager** and reference the secret instead of pasting the key in plaintext — App Runner supports this directly in the same environment variables section.)
5. Health check: path `/health`, which is exactly why that endpoint exists as a separate, dependency-free route.
6. Create the service. App Runner builds it, assigns a `https://<random-id>.<region>.awsapprunner.com` URL, and that's your public endpoint — the same frontend you tested locally now works over the internet, same code, same origin, no CORS setup needed.

**Cost note:** App Runner keeps at least one instance running even at zero traffic — there's no scale-to-zero — so it has a non-trivial idle cost. Fine for a demo you want reachable on-demand; if idle cost matters more than simplicity, look at ECS Fargate with a scheduled scale-down, or accept EC2's manual tradeoffs below.

### Alternative: plain EC2

More manual, but cheaper to reason about and useful if you want SSH access to debug directly:

1. Launch an EC2 instance (Ubuntu 22.04, **t3.small** or larger — the free-tier `t2.micro`'s 1 GB RAM is too tight for this dependency stack) with a security group allowing inbound `80` (and `22` for SSH).
2. SSH in, install Docker (`sudo apt-get update && sudo apt-get install -y docker.io`), and either `git clone` this repo or `scp` the `rag_assistant/` directory over — `faiss_index/` must come with it.
3. Create `rag_assistant/.env` on the instance directly (never commit it, never bake it into the image) with the real `OPENAI_API_KEY`.
4. Build and run, mapping to port 80 so no reverse proxy is needed for plain HTTP:
   ```bash
   cd rag_assistant
   sudo docker build -t healthcare-rag-assistant .
   sudo docker run -d --name rag-app -p 80:8000 --env-file .env --restart unless-stopped healthcare-rag-assistant
   ```
5. Visit `http://<instance-public-ip>/`. For HTTPS on a real domain, put an Application Load Balancer with an ACM certificate in front, or run Caddy/nginx + Certbot on the instance itself — plain EC2 gives you no HTTPS for free the way App Runner does.

### Updating a deployment

Whether on App Runner or EC2, the update flow is the same: rebuild the image locally (picking up any code, data, or FAISS index changes), push it (`docker push` to ECR, or rebuild in place on the EC2 host), and redeploy (App Runner: trigger a deployment from the new ECR image; EC2: `docker rm -f rag-app` then re-run the `docker run` command). There's no hot-reload path for a running container — a new index or a code change always means a new image.

## Example Workflow

```bash
cd rag_assistant
python src/ingest.py
python src/retriever.py
python src/chain.py
python src/agent.py
python src/evaluate.py
python src/evaluate_agent.py
python src/api.py
# or, containerized:
docker build -t healthcare-rag-assistant .
docker run -d -p 8000:8000 -e OPENAI_API_KEY=your_real_key healthcare-rag-assistant
```

## Configuration

Important settings are currently defined directly in the source files:

- `src/ingest.py`
  - `chunk_size=500`, `chunk_overlap=200`
  - `DEFAULT_WEB_SOURCES` — the list of URLs ingested alongside the PDF
- `src/retriever.py`
  - embedding model: `text-embedding-3-small`
  - FAISS index path: `faiss_index`
- `src/chain.py`
  - chat model: `gpt-4o-mini`
  - retrieval `k=3`
  - temperature: `0.7`
- `src/tools.py` / `src/agent.py`
  - retrieval `k=3` (via `build_tools(vector_store, k=3)`)
  - `dosing_table_path` — defaults to `data/dosing_table.csv` (via `build_tools(..., dosing_table_path=...)`)
  - chat model: `gpt-4o-mini`, temperature: `0.7`
  - system prompt controlling ReAct behavior, source preference (structured lookup over text search for dosing), and refusal wording
- `src/api.py`
  - host `0.0.0.0`, port `8000` (when run via `python src/api.py`; override with `uvicorn src.api:app --host ... --port ...`)
  - chain and agent are built once at startup (in the `lifespan` handler), not per-request
  - `FRONTEND_DIR` — defaults to `frontend/`, mounted at `/`
- `Dockerfile`
  - base image, port (`8000`), and `CMD` are fixed; `OPENAI_API_KEY` is supplied at `docker run` time, never baked in

For experiments, tune chunk size, chunk overlap, retrieval `k`, prompt wording, and model choice.

## What Could Break

- The pipeline requires a valid `OPENAI_API_KEY`.
- If `reference.pdf` changes, the FAISS index should be rebuilt.
- The code assumes commands are run from the `rag_assistant/` directory.
- `FAISS.load_local(..., allow_dangerous_deserialization=True)` should only be used with indexes you trust.
- A chunk overlap equal to the chunk size may create highly redundant chunks; consider using a smaller overlap such as `50` to `100`.
- `langchain-community` must stay pinned to `0.4.1` — see the note in [Requirements](#requirements). Running `pip install -U langchain-community` (or any tool that bumps it independently of `requirements.txt`) will silently break `evaluate.py`/`evaluate_agent.py`.
- Checked-in virtual environments and `.env` files can make the repository large and may expose secrets if real keys are committed.
- `src/agent.py` requires `langchain>=1.0` (for `langchain.agents.create_agent`) and `langgraph` installed; an older `langchain` will raise `ImportError`.
- The agent makes at least one extra LLM call per tool invocation compared to the static chain, so it is slower and uses more tokens per question in exchange for more thorough, inspectable reasoning.
- `api.py` builds the chain and agent once at startup; if `faiss_index/` or `.env` isn't in place before the server starts, startup will fail rather than an individual request.
- Older clones of this repo may still have `rag_assistant/venv/` and `rag_assistant/.env` tracked in git history (from before `.gitignore` was fixed) even though they're untracked going forward — history wasn't rewritten, only future commits are affected.
- `load_web_chunks()` only works for pages whose real content is in plain server-rendered HTML. Sites that hydrate their content client-side from a JS-embedded JSON blob (WHO's own fact-sheet pages are like this) will return mostly nav/footer boilerplate instead of the article — check the page source before adding a new URL to `DEFAULT_WEB_SOURCES`.
- `load_web_chunks()` targets Wikipedia's `#mw-content-text` container specifically (via `bs_kwargs`); a non-Wikipedia URL needs its own CSS/id selector for the actual content area, or it'll pull in the whole page.
- `dosing_table.csv` is illustrative sample data, not a verified clinical source — see [Multi-Source Architecture](#multi-source-architecture). Don't extend it with real dosing numbers without a citation to an actual authoritative guideline.
- `lookup_dosing_table` matches `drug` and `indication` as case-insensitive substrings, so an overly short query (e.g. a single letter) could match more rows than intended.
- The Docker image bakes in whatever `faiss_index/` exists on disk at build time — rebuilding the index (e.g. after adding a source) does nothing to a container already running from the old image. You must `docker build` again and redeploy.
- The frontend calls `/ask` and `/agent/ask` as same-origin relative paths with no CORS handling anywhere. It only works served from the same FastAPI app (`GET /`) — opening `frontend/index.html` directly as a `file://` URL, or hosting it on a different origin from the API, will fail without adding `CORSMiddleware` to `api.py`.
- App Runner (see [Deployment](#deployment)) has no built-in scale-to-zero, so it incurs cost even when idle.

## How To Validate

After setup, validate the project in stages:

1. Run `python src/ingest.py` and confirm both the PDF and the three web sources load and produce sample chunks.
2. Run `python src/retriever.py` and confirm FAISS stores vectors from all sources successfully (the printed vector count should be much larger than the PDF alone — roughly 5,000 chunks with the default sources).
3. Run `python src/chain.py` and check that in-document questions receive grounded answers.
4. Ask an unrelated question and verify the assistant refuses to answer from outside knowledge.
5. Run `python src/agent.py` and confirm: the printed trace shows `[Action]` / `[Observation]` steps before the `[Answer]`; the pneumonia dosing question triggers `lookup_dosing_table` then `calculate_dose`; the malaria/pneumonia comparison triggers `search_clinical_guidelines` at least twice with citations from more than one source; the artemether-lumefantrine question calls `lookup_dosing_table` but not `calculate_dose`; and the out-of-scope question is refused.
6. Run `python src/evaluate.py` and inspect the Ragas metric averages.
7. Run `python src/evaluate_agent.py` and confirm all agent behavior checks print `PASS`.
8. Run `python src/api.py`, then `curl http://localhost:8000/health` and confirm `{"status": "ok"}`, and `POST /ask` + `POST /agent/ask` with a question to confirm both return grounded answers (and `POST /agent/ask` includes a non-empty `trace`).
9. Open `http://localhost:8000/` in a browser, ask a question on both the "ReAct Agent" and "Static Chain" tabs, and confirm the trace/sources render correctly.
10. Run `docker build -t healthcare-rag-assistant .` then `docker run -d -p 8000:8000 -e OPENAI_API_KEY=your_real_key healthcare-rag-assistant` and repeat steps 8–9 against the container instead of the local venv — confirms the image is actually self-contained before you deploy it anywhere.

## Future Improvements

- Move model names, chunking values, and source lists into a config file.
- Support multiple PDFs or a folder of documents, not just one.
- Replace the fragile `#mw-content-text` selector with a more general content-extraction approach (e.g. `trafilatura`) so arbitrary URLs — not just Wikipedia — can be added as web sources without hand-picking a CSS selector per site.
- Add automated tests for ingestion, retrieval, refusal behavior, and the structured-lookup tool (and for the API endpoints).
- Add auth / rate limiting to the API before exposing it beyond localhost.
- Add request logging / tracing for the agent's tool-call trace (e.g. LangSmith) so agent cost and behavior can be monitored in production, not just in the CLI trace.
- Replace `dosing_table.csv`'s illustrative sample data with real, cited dosing figures if this is ever meant to inform real decisions.
- Automate the image build/push/redeploy flow (e.g. GitHub Actions → ECR → App Runner) instead of the manual `docker build && docker push` + manual redeploy trigger described in [Deployment](#deployment).
- Move the FAISS index out of the image (e.g. fetched from S3 at container startup) so a new index doesn't require a full image rebuild.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
