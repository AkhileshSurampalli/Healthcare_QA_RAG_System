import os
import sys
sys.path.append(".")

from contextlib import asynccontextmanager

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from openai import OpenAIError
from pydantic import BaseModel

MAX_QUESTION_LEN = 2000

from src.retriever import load_vector_store
from src.chain import build_rag_chain
from src.agent import build_agent, run_with_trace

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load the FAISS index and build the chain/agent once at startup instead of
    per-request, since embedding the query and loading the index are the
    expensive parts - everything after that is just an LLM call.
    """
    print("Loading vector store...")
    vector_store = load_vector_store()

    print("Building static RAG chain...")
    chain, retriever = build_rag_chain(vector_store)

    print("Building ReAct agent...")
    agent = build_agent(vector_store)

    state["chain"] = chain
    state["retriever"] = retriever
    state["agent"] = agent

    yield
    state.clear()


app = FastAPI(
    title="Healthcare RAG Assistant API",
    description="Static RAG chain and ReAct agent over a clinical guideline document.",
    lifespan=lifespan,
)


class QuestionRequest(BaseModel):
    question: str


class Source(BaseModel):
    content: str
    source: str | None = None
    page: int | None = None


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]


class TraceStep(BaseModel):
    type: str
    tool: str | None = None
    args: dict | None = None
    content: str | None = None


class AgentAskResponse(BaseModel):
    answer: str
    trace: list[TraceStep]


def _validated_question(request: QuestionRequest) -> str:
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question must not be empty")
    if len(question) > MAX_QUESTION_LEN:
        raise HTTPException(
            status_code=400,
            detail=f"question must be {MAX_QUESTION_LEN} characters or fewer (got {len(question)})",
        )
    return question


def _run_or_translate_errors(fn):
    """
    Run a chain/agent call and turn the failure modes we actually expect into
    clean HTTP responses instead of a bare, bodyless 500 (which is what every
    unhandled exception - including OpenAI auth/rate-limit/timeout errors -
    produced before this, and is genuinely uninformative to a caller).
    """
    try:
        return fn()
    except OpenAIError as e:
        # The model provider is unreachable, rejected the key, rate-limited us,
        # or timed out - not something wrong with the request itself.
        raise HTTPException(status_code=502, detail=f"Upstream model provider error: {e}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {type(e).__name__}")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask_static(request: QuestionRequest):
    """Static RAG: one fixed retrieve -> augment -> generate pass. See chain.py."""
    question = _validated_question(request)

    def _call():
        answer = state["chain"].invoke(question)
        docs = state["retriever"].invoke(question)
        sources = [
            Source(content=doc.page_content, source=doc.metadata.get("source"), page=doc.metadata.get("page"))
            for doc in docs
        ]
        return AskResponse(answer=answer, sources=sources)

    return _run_or_translate_errors(_call)


@app.post("/agent/ask", response_model=AgentAskResponse)
def ask_agent(request: QuestionRequest):
    """ReAct agent: reasons over search/calculate tool calls. See agent.py."""
    question = _validated_question(request)

    def _call():
        result = run_with_trace(state["agent"], question)
        return AgentAskResponse(answer=result["answer"], trace=[TraceStep(**step) for step in result["trace"]])

    return _run_or_translate_errors(_call)


# Mounted last and at "/" so it only catches requests that don't match an API
# route above (Starlette checks routes in registration order) - this lets one
# service serve both the JSON API and the browser UI, same-origin, no CORS setup.
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)
