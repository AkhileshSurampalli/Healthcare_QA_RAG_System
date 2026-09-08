import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAISS_PATH = os.path.join(BASE_DIR, "faiss_index")

def build_vector_store(chunks):
    """
    Convert each chunk into a vector and store in FAISS
    This calls the OpenAI API once per chunk
    We save to disk so we never need to re-embed unless document changes.
    """

    print("Building embeddings - calling OpenAI API...")

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key = os.getenv("OPENAI_API_KEY")
    )

    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local("faiss_index")
    print("Saved to faiss_index/")

    return vector_store

def load_vector_store():
    """
    Load existing FAISS index from disk
    Use this on every run after the first - no API calls needed
    """
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key = os.getenv("OPENAI_API_KEY")
    )
    vector_store = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )
    print(f"Loaded vector store with {vector_store.index.ntotal} vectors")
    return vector_store

def build_hybrid_retriever(vector_store, k: int = 3, dense_weight: float = 0.6):
    """
    Combine BM25 (keyword/exact-term) retrieval with FAISS's dense MMR search
    via reciprocal-rank fusion, instead of relying on embedding similarity
    alone.

    Pure dense retrieval has a real blind spot: two clinically distinct terms
    (e.g. two different antibiotics used for the same condition) can embed
    close together, so a purely semantic search can surface the wrong drug's
    passage with high confidence. BM25 catches exact-term matches dense search
    can miss; MMR still contributes result diversity on the dense side. This
    is shared by both chain.py and tools.py so hybrid retrieval only has to be
    built in one place.

    BM25Retriever needs the raw documents, not embeddings - rather than
    re-running ingestion (network calls to re-fetch the PDF/web sources), this
    pulls them straight back out of the already-loaded FAISS index's docstore,
    which holds exactly the same documents the dense index was built from.
    """
    docs = list(vector_store.docstore._dict.values())

    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = k

    dense_retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": max(k * 4, 20)}
    )

    return EnsembleRetriever(
        retrievers=[bm25_retriever, dense_retriever],
        weights=[1 - dense_weight, dense_weight],
    )


def test_retrieval(vector_store, query: str):
    """
    Test retrieval (show top 3 most relevant chunks for a query)
    Use this to verify that relevant content is being found.
    """

    print(f"\nQuery: '{query}'")
    print("Top 3 retrieved chunks: \n")

    results = vector_store.similarity_search(query, k=3)
    for i, doc in enumerate(results):
        print(f"Chunk {i+1}:")
        print(f"{doc.page_content[:300]}")
        print("-" * 50)

if __name__ == "__main__":
    import sys
    sys.path.append(".")
    from src.ingest import load_pdf_chunks, load_web_chunks

    # Multi-source: PDF chunks + web chunks go into the same index. Each chunk's
    # `source` metadata (file path vs. URL) is what lets tools.py cite where an
    # answer actually came from.
    pdf_chunks = load_pdf_chunks("data/reference.pdf")
    web_chunks = load_web_chunks()
    chunks = pdf_chunks + web_chunks
    print(f"\nTotal chunks from all sources: {len(chunks)}")

    vector_store = build_vector_store(chunks)

    # Test with healthcare questions relevant
    test_retrieval(vector_store, "What are the symptoms of malaria?")
    test_retrieval(vector_store, "How should diabetes be managed?")
    test_retrieval(vector_store, "What is the recommended treatment for pneumonia?")

