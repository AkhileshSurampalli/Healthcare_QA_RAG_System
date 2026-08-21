from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_and_chunk(pdf_path: str, chunk_size: int=500, chunk_overlap:int=50):
    """
    Load a PDF and split into overlapping chunks

    Chunk size small enough to cover on especific topic
    Large enough that it contains meaningful context

    """

    # Load PDF page by page
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"Loaded {len(documents)} pages from {pdf_path}")

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap,
        separators = ["\n\n", "\n", ".", " ", ""]
    )

    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")

    # Showing a sample check
    print(f"\nSample chunk:\n{chunks[0].page_content[:300]}")
    return chunks

if __name__ == "__main__":
    chunks = load_and_chunk("data/reference.pdf")

