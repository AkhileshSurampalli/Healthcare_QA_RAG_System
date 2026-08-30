FROM python:3.11-slim

WORKDIR /app

# System deps for faiss-cpu / pypdf wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Everything the app needs at runtime: source, the pre-built FAISS index, the
# PDF/CSV data files, and the static frontend. The index is baked into the
# image rather than built at container startup - see README "What Could Break"
# for why (rebuilding needs a real OPENAI_API_KEY and takes minutes).
COPY src/ src/
COPY frontend/ frontend/
COPY data/ data/
COPY faiss_index/ faiss_index/

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
