FROM docker.io/python:3.11-slim

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system deps (gcc, etc. for sentence-transformers)
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip & install deps
RUN pip install --no-cache-dir --upgrade pip \
    sentence-transformers \
    meilisearch


WORKDIR /app
COPY embed_new.py .
CMD ["python", "embed_new.py"]
