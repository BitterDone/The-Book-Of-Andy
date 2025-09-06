FROM docker.io/python:3.11-bullseye

# Upgrade pip & install deps
RUN pip install -q --no-cache-dir --upgrade pip \
    sentence-transformers \
    meilisearch

WORKDIR /app
COPY embed_new.py .
COPY preload_meili.py .
# This CMD is overruled when the command: section of the podman-compose.yaml is enabled
CMD ["python", "embed_new.py"]
