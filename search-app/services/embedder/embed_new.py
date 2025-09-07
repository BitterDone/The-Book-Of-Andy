import os
import json
import glob
from sentence_transformers import SentenceTransformer
from meilisearch import Client
from meilisearch.errors import MeilisearchApiError

MEILI_URL = os.environ["MEILI_URL"]
MASTER_KEY = os.environ["MASTER_KEY"]
TRANSCRIPTS_DIR = os.environ["TRANSCRIPTS_DIR"]
PRECOMPUTED_FILE = os.environ["PRECOMPUTED_FILE"]

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
# if "transcripts" not in [idx["uid"] for idx in client.get_indexes()]:
# if "transcripts" not in client.get_indexes():
    client.create_index(
        uid="transcripts",
        options={
            "primaryKey": "id",   # must match your document field
        }
    )
    
index = client.index("transcripts")

model = SentenceTransformer("all-MiniLM-L6-v2")

# Check which transcripts already exist in Meilisearch
# existing_ids = {doc["id"] for doc in index.get_documents({"limit": 10000})["results"]}

# Fetch existing documents
existing_docs = index.get_documents({"limit": 10000})  # returns DocumentsResults
existing_ids = {doc["id"] for doc in existing_docs.results}    # iterate directly


precomputed = []
if os.path.exists(PRECOMPUTED_FILE) and os.path.getsize(PRECOMPUTED_FILE) > 0:
    with open(PRECOMPUTED_FILE, "r", encoding="utf-8") as f:
        try:
            precomputed = json.load(f)
        except json.JSONDecodeError:
            print(f"[!] Warning: {PRECOMPUTED_FILE} was not valid JSON, starting fresh.")
else:
    print(f"[!] {PRECOMPUTED_FILE} not found or empty, starting fresh.")

for file in os.listdir(TRANSCRIPTS_DIR):
    if not file.endswith(".txt") or file in existing_ids:
        continue
    text = open(os.path.join(TRANSCRIPTS_DIR, file)).read()
    embedding = model.encode(text).tolist()
    precomputed.append({"id": file, "text": text, "embedding": embedding})
    print(f"[✓] Prepared {file}")

with open(PRECOMPUTED_FILE, "w") as f:
    json.dump(precomputed, f)
