import os
import json
import glob
import re
from sentence_transformers import SentenceTransformer
from meilisearch import Client
from meilisearch.errors import MeilisearchApiError

MEILI_URL = os.environ["MEILI_URL"]
MASTER_KEY = os.environ["MASTER_KEY"]
TRANSCRIPTS_DIR = os.environ["TRANSCRIPTS_DIR"]
PRECOMPUTED_FILE = os.environ["PRECOMPUTED_FILE"]
VECTOR_SIZE = 384  # embedding size from all-MiniLM-L6-v2
CHUNK_SIZE = 200   # words per chunk
OVERLAP_SIZE = 20  # words to overlap between chunks

# --- Verify transcripts directory ---
if not os.path.exists(TRANSCRIPTS_DIR):
    print(f"[✗] ERROR: Transcripts directory not found at {TRANSCRIPTS_DIR}")
    exit(1)

txt_files = glob.glob(os.path.join(TRANSCRIPTS_DIR, "*.txt"))
if not txt_files:
    print(f"[✗] ERROR: No .txt files found in {TRANSCRIPTS_DIR}")
    exit(1)

print(f"[✓] Found {len(txt_files)} transcript files in {TRANSCRIPTS_DIR}")

client = Client(MEILI_URL, MASTER_KEY)

# Create the index if it doesn't exist
try:
    client.get_index("transcripts")
except MeilisearchApiError:
    client.create_index(
        uid="transcripts",
        options={"primaryKey": "id"}
    )

index = client.index("transcripts")

# --- Register the user-provided embedder ---
index.update_settings({
    "embedders": {
        "all-MiniLM-L6-v2": {
            "source": "userProvided",
            "dimensions": VECTOR_SIZE
        }
    }
})

model = SentenceTransformer("all-MiniLM-L6-v2")

# Fetch existing documents
existing_docs = index.get_documents({"limit": 10000})
existing_ids = {doc["id"] for doc in existing_docs.results}

precomputed = []
if os.path.exists(PRECOMPUTED_FILE) and os.path.getsize(PRECOMPUTED_FILE) > 0:
    with open(PRECOMPUTED_FILE, "r", encoding="utf-8") as f:
        try:
            precomputed = json.load(f)
        except json.JSONDecodeError:
            print(f"[!] Warning: {PRECOMPUTED_FILE} was not valid JSON, starting fresh.")
else:
    print(f"[!] {PRECOMPUTED_FILE} not found or empty, starting fresh.")

# --- Process transcript files into overlapping chunks ---
for file in os.listdir(TRANSCRIPTS_DIR):
    if not file.endswith(".txt"):
        continue

    base_id = os.path.splitext(file)[0]
    safe_base_id = re.sub(r'[^a-zA-Z0-9_-]', '_', base_id)

    text = open(os.path.join(TRANSCRIPTS_DIR, file), encoding="utf-8").read()
    words = text.split()
    start = 0
    chunk_index = 0

    while start < len(words):
        end = min(start + CHUNK_SIZE, len(words))
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words)
        chunk_id = f"{safe_base_id}_chunk{chunk_index}"

        if chunk_id not in existing_ids:
            embedding = model.encode(chunk_text).tolist()
            precomputed.append({
                "id": chunk_id,
                "text": chunk_text,
                "_vectors": {"all-MiniLM-L6-v2": embedding}
            })
            print(f"[✓] Prepared {file} -> chunk {chunk_index}")

        chunk_index += 1
        start += CHUNK_SIZE - OVERLAP_SIZE  # move start for next chunk with overlap

# --- Save results ---
with open(PRECOMPUTED_FILE, "w", encoding="utf-8") as f:
    json.dump(precomputed, f)

print(f"[✓] Wrote {len(precomputed)} documents to {PRECOMPUTED_FILE}")