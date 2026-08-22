import os

os.environ.setdefault("USER_AGENT", "Mozilla/5.0 (rag-project ingestion)")

import bs4
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Wikipedia over WHO fact sheets: WHO's site hydrates its real article body from a
# JS-embedded JSON blob, so a plain requests+bs4 fetch (what WebBaseLoader does)
# only picks up nav/footer boilerplate. Wikipedia's article body is plain
# server-rendered HTML, so it actually works with this loader.
DEFAULT_WEB_SOURCES = [
    "https://en.wikipedia.org/wiki/Malaria",
    "https://en.wikipedia.org/wiki/Pneumonia",
    "https://en.wikipedia.org/wiki/Diabetes",
]


def load_pdf_chunks(pdf_path: str, chunk_size: int = 500, chunk_overlap: int = 200):
    """
    Load a PDF and split into overlapping chunks.

    Chunk size small enough to cover one specific topic, large enough that it
    contains meaningful context. Each chunk's metadata (from PyPDFLoader) already
    carries `source` (the file path) and `page`, which tools.py uses to cite it.
    """
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"Loaded {len(documents)} pages from {pdf_path}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")
    return chunks


def load_web_chunks(urls=None, chunk_size: int = 500, chunk_overlap: int = 200):
    """
    Fetch each URL and split its article body into overlapping chunks.

    Only pulls the main content element (`#mw-content-text` on Wikipedia) instead
    of the whole page, so navigation/sidebar text doesn't pollute the index. Each
    chunk's metadata carries `source` (the URL) - no `page`, which is how tools.py
    tells a web chunk apart from a PDF chunk when citing it.
    """
    urls = urls or DEFAULT_WEB_SOURCES
    all_chunks = []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    for url in urls:
        loader = WebBaseLoader(
            url,
            header_template={"User-Agent": "Mozilla/5.0"},
            bs_kwargs={"parse_only": bs4.SoupStrainer(id="mw-content-text")},
        )
        documents = loader.load()
        chunks = splitter.split_documents(documents)
        print(f"Loaded and split {url} into {len(chunks)} chunks")
        all_chunks.extend(chunks)

    return all_chunks


if __name__ == "__main__":
    pdf_chunks = load_pdf_chunks("data/reference.pdf")
    print(f"\nSample PDF chunk:\n{pdf_chunks[0].page_content[:300]}")

    web_chunks = load_web_chunks()
    print(f"\nSample web chunk:\n{web_chunks[0].page_content[:300]}")
