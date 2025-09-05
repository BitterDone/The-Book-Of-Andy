import json
import os
from meilisearch import Client

# --- CONFIG ---
MEILI_URL = os.environ["MEILI_URL"]
MASTER_KEY = os.environ["MASTER_KEY"]
TRANSCRIPTS_DIR = os.environ["TRANSCRIPTS_DIR"]
PRECOMPUTED_FILE = os.environ["PRECOMPUTED_FILE"]

INDEX_NAME = "transcripts"
VECTOR_SIZE = 384  # size of embedding from all-MiniLM-L6-v2

# --- Connect to Meilisearch ---
client = Client(MEILI_URL, MASTER_KEY)

# --- Create index if it doesn't exist ---
# if INDEX_NAME not in [i["uid"] for i in client.get_indexes()]:
if "transcripts" not in client.get_indexes():
    client.create_index(
        uid="transcripts",
        options={
            "primaryKey": "id",   # must match your document field
        }
    )

index = client.index("transcripts")

# --- Enable vector search ---
index.update_settings({
     "embedders": {
        "all-MiniLM-L6-v2": {  # name of your embedding model
            "source": "userProvided",
            "dimensions": VECTOR_SIZE
        }
    }
})

# --- Load precomputed data ---
with open(PRECOMPUTED_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

# --- Bulk add documents ---
index.add_documents(data)
print(f"[✓] Preloaded {len(data)} transcript documents into Meilisearch.")
