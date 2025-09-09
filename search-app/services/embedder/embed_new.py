import os
import json
import glob
import re
from sentence_transformers import SentenceTransformer
from meilisearch import Client
from meilisearch.errors import MeilisearchApiError

# -------- Config via env --------
MEILI_URL        = os.environ["MEILI_URL"]
MASTER_KEY       = os.environ["MASTER_KEY"]
TRANSCRIPTS_DIR  = os.environ["TRANSCRIPTS_DIR"]
PRECOMPUTED_FILE = os.environ["PRECOMPUTED_FILE"]

# Vector + chunking params
EMBEDDER_NAME = "all-MiniLM-L6-v2"
VECTOR_SIZE   = 384          # dimensions for all-MiniLM-L6-v2
CHUNK_SIZE    = 200          # words per chunk
OVERLAP_SIZE  = 20           # words of overlap between chunks

# Batching params to stay under Meili's 95MB request limit
MAX_BATCH_DOCS = 500                   # hard cap on documents per batch
MAX_BATCH_BYTES = 80 * 1024 * 1024     # ~80MB safety cap for serialized JSON

# -------- Sanity checks --------
if not os.path.exists(TRANSCRIPTS_DIR):
    print(f"[✗] ERROR: Transcripts directory not found at {TRANSCRIPTS_DIR}")
    raise SystemExit(1)

txt_files = sorted(glob.glob(os.path.join(TRANSCRIPTS_DIR, "*.txt")))
if not txt_files:
    print(f"[✗] ERROR: No .txt files found in {TRANSCRIPTS_DIR}")
    raise SystemExit(1)

print(f"[✓] Found {len(txt_files)} transcript files in {TRANSCRIPTS_DIR}")

# Ensure PRECOMPUTED_FILE parent dir exists (useful if mapped to /search-app/data/)
os.makedirs(os.path.dirname(PRECOMPUTED_FILE), exist_ok=True)

# -------- Meilisearch client & index --------
client = Client(MEILI_URL, MASTER_KEY)

# Create index if missing
try:
    client.get_index("transcripts")
except MeilisearchApiError:
    client.create_index(uid="transcripts", options={"primaryKey": "id"})

index = client.index("transcripts")

# Register a user-provided embedder (required for pushing your own vectors)
index.update_settings({
    "embedders": {
        EMBEDDER_NAME: {
            "source": "userProvided",
            "dimensions": VECTOR_SIZE
        }
    }
})

# -------- Model --------
model = SentenceTransformer(EMBEDDER_NAME)

# -------- Utilities --------
def sanitize_id(s: str) -> str:
    """Keep only a-z A-Z 0-9 _ - in IDs."""
    return re.sub(r"[^a-zA-Z0-9_-]", "_", s)

def chunk_words(words, chunk_size, overlap):
    """Yield (start_idx, end_idx, text) for overlapping word chunks."""
    start = 0
    n = len(words)
    step = max(1, chunk_size - overlap)  # avoid infinite loop if overlap >= size
    while start < n:
        end = min(start + chunk_size, n)
        yield start, end, " ".join(words[start:end])
        if end == n:
            break
        start += step

def approx_json_size(obj) -> int:
    """Approximate serialized JSON size in bytes for batching decisions."""
    return len(json.dumps(obj, ensure_ascii=False))

def flush_batch(batch, index, which):
    if not batch:
        return
    index.add_documents(batch)
    print(f"[✓] Added batch #{which} with {len(batch)} docs")

# -------- Avoid re-adding existing docs --------
# (If you expect >10k docs, you can paginate here.)
existing_ids = set()
try:
    docs = index.get_documents({"limit": 10000})
    existing_ids = {doc["id"] for doc in docs.results}
except MeilisearchApiError:
    # If index is fresh or empty, that's fine.
    pass

# -------- Main loop: build chunks, embed, batch-upload --------
all_docs_for_debug = []
batch = []
batch_bytes = 2  # for "[]"
batch_no = 1

for path in txt_files:
    filename = os.path.basename(path)
    base_id = os.path.splitext(filename)[0]          # drop .txt
    safe_base_id = sanitize_id(base_id)

    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    words = text.split()
    chunk_idx = 0

    for start_idx, end_idx, chunk_text in chunk_words(words, CHUNK_SIZE, OVERLAP_SIZE):
        doc_id = f"{safe_base_id}_chunk{chunk_idx}"

        # Skip if already present
        if doc_id in existing_ids:
            chunk_idx += 1
            continue

        # Compute embedding for the chunk
        embedding = model.encode(chunk_text).tolist()

        # Meili v1.15 userProvided vectors go under _vectors.<embedderName>
        doc = {
            "id": doc_id,
            "file": filename,
            "chunk_index": chunk_idx,
            "word_start": start_idx,
            "word_end": end_idx,
            "text": chunk_text,
            "_vectors": {EMBEDDER_NAME: embedding}
        }

        # Try to add doc to current batch, respecting both byte and doc caps
        doc_bytes = approx_json_size(doc)
        # 1 (comma) margin per doc to be conservative
        will_exceed_bytes = (batch_bytes + doc_bytes + 1) > MAX_BATCH_BYTES
        will_exceed_count = (len(batch) + 1) > MAX_BATCH_DOCS

        if will_exceed_bytes or will_exceed_count:
            flush_batch(batch, index, batch_no)
            batch_no += 1
            batch = []
            batch_bytes = 2  # "[]"

        batch.append(doc)
        batch_bytes += doc_bytes + 1
        all_docs_for_debug.append(doc)

        print(f"[✓] Prepared {filename} -> {doc_id} "
              f"(words {start_idx}-{end_idx}, batch_bytes≈{batch_bytes})")

        chunk_idx += 1

# Flush any remaining docs
flush_batch(batch, index, batch_no)

# -------- Persist a copy for debugging/auditing --------
with open(PRECOMPUTED_FILE, "w", encoding="utf-8") as f:
    json.dump(all_docs_for_debug, f)

print(f"[✓] Wrote {len(all_docs_for_debug)} chunk-docs to {PRECOMPUTED_FILE}")
print("[✓] Done.")
