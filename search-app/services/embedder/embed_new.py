import os
import json
from sentence_transformers import SentenceTransformer
from meilisearch import Client

MEILI_URL = os.environ["MEILI_URL"]
MASTER_KEY = os.environ["MASTER_KEY"]
TRANSCRIPTS_DIR = os.environ["TRANSCRIPTS_DIR"]
PRECOMPUTED_FILE = os.environ["PRECOMPUTED_FILE"]

client = Client(MEILI_URL, MASTER_KEY)

# Create the index if it doesn't exist
if "transcripts" not in [idx.uid for idx in client.get_indexes()]:
    client.create_index("transcripts", options={"primaryKey": "id"})
    
index = client.index("transcripts")

model = SentenceTransformer("all-MiniLM-L6-v2")

# Check which transcripts already exist in Meilisearch
existing_ids = {doc["id"] for doc in index.get_documents({"limit": 10000})["results"]}

precomputed = []
if os.path.exists(PRECOMPUTED_FILE):
    precomputed = json.load(open(PRECOMPUTED_FILE))

for file in os.listdir(TRANSCRIPTS_DIR):
    if not file.endswith(".txt") or file in existing_ids:
        continue
    text = open(os.path.join(TRANSCRIPTS_DIR, file)).read()
    embedding = model.encode(text).tolist()
    precomputed.append({"id": file, "text": text, "embedding": embedding})
    print(f"[✓] Prepared {file}")

with open(PRECOMPUTED_FILE, "w") as f:
    json.dump(precomputed, f)
