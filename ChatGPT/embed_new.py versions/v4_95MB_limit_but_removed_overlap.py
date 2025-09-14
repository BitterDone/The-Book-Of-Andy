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
VECTOR_SIZE = 384  # embedding size for all-MiniLM-L6-v2
CHUNK_SIZE = 300  # number of words per chunk
BATCH_SIZE = 50   # number of documents per Meilisearch request

# --- Verify transcripts directory ---
if not os.path.exists(TRANSCRIPTS_DIR):
    print(f"[✗] ERROR: Transcripts directory not found at {TRANSCRIPTS_DIR}")
    exit(1)

txt_files = glob.glob(os.path.join(TRANSCRIPTS_DIR, "*.txt"))
if not txt_files:
    print(f"[✗] ERROR: No .txt files found in {TRANSCRIPTS_DIR}")
    exit(1)

print(f"[✓] Found {len(txt_files)} transcript files in {TRANSCRIPTS_DIR}")

# --- Connect to Meilisearch ---
client = Client(MEILI_URL, MASTER_KEY)

# --- Create the index if it doesn't exist ---
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

# --- Load existing documents ---
existing_docs = index.get_documents({"limit": 10000})
existing_ids = {doc["id"] for doc in existing_docs.results}

# --- Load or initialize precomputed data ---
precomputed = []
if os.path.exists(PRECOMPUTED_FILE) and os.path.getsize(PRECOMPUTED_FILE) > 0:
    with open(PRECOMPUTED_FILE, "r", encoding="utf-8") as f:
        try:
            precomputed = json.load(f)
        except json.JSONDecodeError:
            print(f"[!] Warning: {PRECOMPUTED_FILE} was not valid JSON, starting fresh.")
else:
    print(f"[!] {PRECOMPUTED_FILE} not found or empty, starting fresh.")

# --- Load embedding model ---
model = SentenceTransformer("all-MiniLM-L6-v2")

# --- Process transcript files into chunks ---
for file in os.listdir(TRANSCRIPTS_DIR):
    if not file.endswith(".txt"):
        continue

    base_id = os.path.splitext(file)[0]
    safe_base_id = re.sub(r'[^a-zA-Z0-9_-]', '_', base_id)

    text = open(os.path.join(TRANSCRIPTS_DIR, file), encoding="utf-8").read()
    words = text.split()
    
    # Split transcript into chunks
    for i in range(0, len(words), CHUNK_SIZE):
        chunk_words = words[i:i + CHUNK_SIZE]
        chunk_text = " ".join(chunk_words)
        chunk_id = f"{safe_base_id}_{i//CHUNK_SIZE}"

        if chunk_id in existing_ids:
            continue

        embedding = model.encode(chunk_text).tolist()
        precomputed.append({
            "id": chunk_id,
            "text": chunk_text,
            "_vectors": {"all-MiniLM-L6-v2": embedding}
        })

        print(f"[✓] Prepared chunk {chunk_id}")

# --- Save precomputed chunks to file ---
with open(PRECOMPUTED_FILE, "w", encoding="utf-8") as f:
    json.dump(precomputed, f)

print(f"[✓] Wrote {len(precomputed)} chunk documents to {PRECOMPUTED_FILE}")

# --- Upload to Meilisearch in batches ---
for i in range(0, len(precomputed), BATCH_SIZE):
    batch = precomputed[i:i + BATCH_SIZE]
    try:
        index.add_documents(batch)
        print(f"[✓] Uploaded batch {i//BATCH_SIZE + 1} ({len(batch)} documents)")
    except Exception as e:
        print(f"[✗] Failed to upload batch {i//BATCH_SIZE + 1}: {e}")

print(f"[✓] Finished uploading all documents to Meilisearch")a