import json
import os
from meilisearch import Client

# --- CONFIG ---
MEILI_URL = os.environ.get("MEILI_URL", "http://localhost:7700")
MASTER_KEY = os.environ.get("MASTER_KEY", "MASTER_KEY")
PRECOMPUTED_FILE = os.environ.get("PRECOMPUTED_FILE", "precomputed_transcripts.json")
INDEX_NAME = "transcripts"
VECTOR_SIZE = 384  # size of embedding from all-MiniLM-L6-v2

# --- Connect to Meilisearch ---
client = Client(MEILI_URL, MASTER_KEY)

# --- Create index if it doesn't exist ---
if INDEX_NAME not in [i["uid"] for i in client.get_indexes()]:
    client.create_index(INDEX_NAME)

index = client.index(INDEX_NAME)

# --- Enable vector search ---
index.update_settings({
    "vector": {
        "size": VECTOR_SIZE,
        "distance": "Cosine"
    }
})

# --- Load precomputed data ---
with open(PRECOMPUTED_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

# --- Bulk add documents ---
index.add_documents(data)
print(f"[✓] Preloaded {len(data)} transcript documents into Meilisearch.")
