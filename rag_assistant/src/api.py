import sys
sys.path.append(".")

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

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
    return question


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask_static(request: QuestionRequest):
    """Static RAG: one fixed retrieve -> augment -> generate pass. See chain.py."""
    question = _validated_question(request)

    answer = state["chain"].invoke(question)
    docs = state["retriever"].invoke(question)
    sources = [
        Source(content=doc.page_content, source=doc.metadata.get("source"), page=doc.metadata.get("page"))
        for doc in docs
    ]

    return AskResponse(answer=answer, sources=sources)


@app.post("/agent/ask", response_model=AgentAskResponse)
def ask_agent(request: QuestionRequest):
    """ReAct agent: reasons over search/calculate tool calls. See agent.py."""
    question = _validated_question(request)

    result = run_with_trace(state["agent"], question)
    return AgentAskResponse(answer=result["answer"], trace=[TraceStep(**step) for step in result["trace"]])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)
