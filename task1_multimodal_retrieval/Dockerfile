# Task 1 -- Multimodal Retrieval Streamlit app, containerized.
#
# Runs in OFFLINE MOCK mode by default (no GPU, no API key, no external
# network calls at runtime) -- appropriate for a public demo deployment.
# Set DASHSCOPE_API_KEY or QWEN_VL_BACKEND=local (with the -local-qwenvl
# extra requirements) at deploy time if you want the real backend instead
# -- see task1_multimodal_retrieval/README.md for that trade-off.
#
# Build:
#   docker build -t task1-multimodal-retrieval .
# Run:
#   docker run -p 8501:8501 task1-multimodal-retrieval
# (Hugging Face Spaces expects port 7860 by default -- see PORT env below,
# which this image respects for that platform.)

FROM python:3.11-slim

WORKDIR /app

# No system packages beyond build-essential are required: ingestion.py
# uses pdfplumber's built-in PIL-based page rendering (not poppler), and
# matplotlib runs headless via the Agg backend already set in
# data/generate_corpus.py -- confirmed working with just this base image.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Corpus is already generated and committed, but regenerate at build time
# too so the image is self-contained even if data/ was excluded from the
# build context by mistake.
RUN python data/generate_corpus.py

# Streamlit's default port; Hugging Face Spaces injects PORT=7860 and
# expects the app to honor it, so we read it with a sane local fallback.
ENV PORT=8501
EXPOSE 8501
EXPOSE 7860

# Never runs any real model download/inference in the container -- offline
# mock mode only, unless the deployer explicitly sets QWEN_VL_BACKEND or
# DASHSCOPE_API_KEY as container environment variables.
CMD streamlit run app.py \
    --server.port=${PORT} \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false
