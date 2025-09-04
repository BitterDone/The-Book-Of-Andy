FROM python:3.11-slim
WORKDIR /app
COPY embed_new.py .
RUN pip install --no-cache-dir sentence-transformers meilisearch
CMD ["python", "embed_new.py"]
