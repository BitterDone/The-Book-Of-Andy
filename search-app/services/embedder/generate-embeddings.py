from sentence_transformers import SentenceTransformer
import os
import json

model = SentenceTransformer("all-MiniLM-L6-v2")
all_data = []

for file in os.listdir("original_transcripts"):
    if not file.endswith(".txt"):
        continue
    text = open(os.path.join("original_transcripts", file)).read()
    embedding = model.encode(text).tolist()
    all_data.append({
        "id": file,
        "text": text,
        "embedding": embedding
    })

with open("precomputed_transcripts.json", "w") as f:
    json.dump(all_data, f)
