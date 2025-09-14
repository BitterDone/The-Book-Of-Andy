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
VECTOR_SIZE = 384  # size of embedding from all-MiniLM-L6-v2
CHUNK_SIZE = 200    # words per chunk
OVERLAP = 50        # words overlapping between chunks

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

# Create the index if it doesn't exist
try:
    client.get_index("transcripts")
except MeilisearchApiError:
    client.create_index(uid="transcripts", options={"primaryKey": "id"})

index = client.index("transcripts")

# --- Register user-provided embedder ---
index.update_settings({
    "embedders": {
        "all-MiniLM-L6-v2": {
            "source": "userProvided",
            "dimensions": VECTOR_SIZE
        }
    }
})

# --- Load embedding model ---
model = SentenceTransformer("all-MiniLM-L6-v2")

# --- Fetch existing documents ---
existing_docs = index.get_documents({"limit": 10000})
existing_ids = {doc["id"] for doc in existing_docs.results}

# --- Load or initialize precomputed JSON ---
precomputed = []
if os.path.exists(PRECOMPUTED_FILE) and os.path.getsize(PRECOMPUTED_FILE) > 0:
    with open(PRECOMPUTED_FILE, "r", encoding="utf-8") as f:
        try:
            precomputed = json.load(f)
        except json.JSONDecodeError:
            print(f"[!] Warning: {PRECOMPUTED_FILE} was not valid JSON, starting fresh.")
else:
    print(f"[!] {PRECOMPUTED_FILE} not found or empty, starting fresh.")

# --- Helper: split text into overlapping chunks ---
def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk_words = words[i:i + chunk_size]
        if chunk_words:
            chunks.append((" ".join(chunk_words), i))  # include start word index
    return chunks

# --- Process transcript files ---
for file in os.listdir(TRANSCRIPTS_DIR):
    if not file.endswith(".txt"):
        continue

    base_id = os.path.splitext(file)[0]
    safe_base_id = re.sub(r'[^a-zA-Z0-9_-]', '_', base_id)

    with open(os.path.join(TRANSCRIPTS_DIR, file), encoding="utf-8") as f:
        text = f.read()

    chunks = chunk_text(text)
    for idx, (chunk_text_content, start_word) in enumerate(chunks):
        chunk_id = f"{safe_base_id}_chunk{idx+1}"
        if chunk_id in existing_ids:
            continue

        embedding = model.encode(chunk_text_content).tolist()

        precomputed.append({
            "id": chunk_id,
            "file": safe_base_id,
            "chunk_index": idx+1,
            "start_word": start_word,
            "text": chunk_text_content,
            "_vectors": {"all-MiniLM-L6-v2": embedding}
        })

        print(f"[✓] Prepared {file} -> chunk {idx+1} -> id={chunk_id}")

# --- Save precomputed chunks ---
with open(PRECOMPUTED_FILE, "w", encoding="utf-8") as f:
    json.dump(precomputed, f)

print(f"[✓] Wrote {len(precomputed)} chunk documents to {PRECOMPUTED_FILE}")