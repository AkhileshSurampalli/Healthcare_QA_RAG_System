import os
import sys
sys.path.append(".")

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from src.retriever import build_hybrid_retriever

load_dotenv()

def build_rag_chain(vector_store):
    """
    Flow:
    Question → retriever finds top 3 chunks →
    chunks + question sent to LLM → LLM generates answer
    """
    # Hybrid (BM25 + dense MMR) instead of dense-only: pure embedding similarity
    # can miss exact-term distinctions (two different drug names can embed
    # close together), and MMR alone only fixes near-duplicate results, not
    # that blind spot. See build_hybrid_retriever() in retriever.py.
    retriever = build_hybrid_retriever(vector_store, k=3)

    prompt_template = """
You are a helpful assistant. Use ONLY the context below to answer the question.
If the answer is not in the context, say exactly:
"I don't have enough information in the document to answer this."
Do not make up information. Do not use your training knowledge.

Context:
{context}

Question: {question}

Answer:"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        # 0, not 0.7: the exact refusal string this prompt depends on ("I don't
        # have enough information...") gets less reliable at higher temperature.
        temperature=0,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        max_retries=2,
        timeout=30,
    )

    def format_docs(docs):
        """Join retrieved chunks into a single context string."""
        return "\n\n".join(doc.page_content for doc in docs)

    # LCEL chain 
    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever


def ask(chain, retriever, question: str):
    """Ask a question and print answer with source chunks."""
    
    # Get answer
    answer = chain.invoke(question)
    
    # Get source chunks separately
    source_docs = retriever.invoke(question)
    
    print(f"\n{'='*60}")
    print(f"Question: {question}")
    print(f"{'='*60}")
    print(f"Answer: {answer}")
    print(f"\nBased on these chunks:")
    for i, doc in enumerate(source_docs):
        print(f"\n  Source {i+1}: {doc.page_content[:200]}...")
    
    return answer, source_docs


if __name__ == "__main__":
    from src.retriever import load_vector_store

    print("Loading vector store...")
    vector_store = load_vector_store()

    print("Building RAG chain...")
    chain, retriever = build_rag_chain(vector_store)

    print("\nRAG system ready. Testing questions...\n")

    ask(chain, retriever, "What are the symptoms of malaria?")
    ask(chain, retriever, "How should diabetes be managed?")
    ask(chain, retriever, "What is the recommended treatment for pneumonia?")
    ask(chain, retriever, "What is the capital of France?")