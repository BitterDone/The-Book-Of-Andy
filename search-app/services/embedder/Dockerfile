FROM docker.io/python:3.11-bullseye

# Upgrade pip & install deps
RUN pip install --no-cache-dir --upgrade pip \
    sentence-transformers \
    meilisearch

WORKDIR /app
COPY embed_new.py .
CMD ["python", "embed_new.py"]
