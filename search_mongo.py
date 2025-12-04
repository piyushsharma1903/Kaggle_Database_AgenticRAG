from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
import numpy as np
import pprint

# --- Setup ---
MONGO_URI = "mongodb+srv://rag_user:rag123@health-rag-cluster.qnzikcs.mongodb.net/?appName=health-rag-cluster"
DB_NAME = "health_rag"
COLLECTION_NAME = "patient_vectors"

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

# --- Load same model for query embedding ---
model = SentenceTransformer("all-MiniLM-L6-v2")

# --- User query ---
query = input("Who had abnormal test results for diabetes?")

# Generate embedding for query
q_emb = model.encode(query).tolist()

# --- MongoDB vector search query ---
pipeline = [
    {
        "$vectorSearch": {
            "index": "vector_index",
            "path": "embedding",
            "queryVector": q_emb,
            "numCandidates": 200,
            "limit": 5
        }
    },
    {
        "$project": {
            "_id": 0,
            "summary": 1,
            "score": {"$meta": "vectorSearchScore"}
        }
    }
]

results = list(collection.aggregate(pipeline))

print("\nTop matches:\n")
for i, r in enumerate(results):
    print(f"{i+1}. (score: {r['score']:.4f})\n{r['summary']}\n")
