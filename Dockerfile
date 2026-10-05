FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TRANSFORMERS_VERBOSITY=error \
    HF_HUB_DISABLE_PROGRESS_BARS=1

# CPU-only PyTorch keeps the image ~2 GB smaller than the default GPU build.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install -r requirements.txt

# Download the embedding model at build time so the container starts offline.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

COPY src/ src/
COPY eval/questions.json eval/questions.json
COPY data/pdfs/ data/pdfs/

EXPOSE 8000

# Build the vector store on first start (skipped if it already exists), then serve.
CMD ["sh", "-c", "python -m src.ingest --if-empty && uvicorn src.api:app --host 0.0.0.0 --port 8000"]
