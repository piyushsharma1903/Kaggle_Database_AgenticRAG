import json
import time
from pymongo import MongoClient

# --- MongoDB setup ---
MONGO_URI = "mongodb+srv://rag_user:rag123@health-rag-cluster.qnzikcs.mongodb.net/?appName=health-rag-cluster"
DB_NAME = "health_rag"
COLLECTION_NAME = "patient_vectors"

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

# --- Load your local JSON ---
with open("data/embedded_summaries.json", "r", encoding="utf-8") as f:
    data = json.load(f)

batch_size = 200      # 🔥 SAFE FOR ATLAS
total = len(data)

print(f"Inserting {total} documents in batches of {batch_size}...")

collection.drop()  # remove if starting fresh

for start in range(0, total, batch_size):
    end = min(start + batch_size, total)
    batch = data[start:end]

    docs = [
        {
            "_id": i + start,
            "summary": item["Summary"],
            "embedding": item["embedding"]
        }
        for i, item in enumerate(batch)
    ]

    try:
        collection.insert_many(docs, ordered=False)
        print(f"✔️ Inserted batch {start}–{end}")
    except Exception as e:
        print(f"❌ Batch {start}–{end} failed: {e}")

    time.sleep(0.2)  # tiny cooldown helps

print("🎯 Finished inserting all batches!")