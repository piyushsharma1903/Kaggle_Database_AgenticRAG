import os
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer

# -------------------------
# MongoDB Setup
# -------------------------
MONGO_URI = "mongodb+srv://rag_user:rag123@health-rag-cluster.qnzikcs.mongodb.net/?appName=health-rag-cluster"
client = MongoClient(MONGO_URI)
db = client["health_rag"]       # your DB name

collection_summaries = db["summaries"]   # vector collection
collection_patients = db["patients"]     # structured collection

# -------------------------
# Embedding Model
# -------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")

# -------------------------
# Embedding Helpers
# -------------------------
def embed_query(text: str):
    return model.encode(text).tolist()

def retrieve_topk(query_vec, topk=5, num_candidates=200):
    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index",
                "path": "embedding",
                "queryVector": query_vec,
                "numCandidates": num_candidates,
                "limit": topk
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
    return list(collection_summaries.aggregate(pipeline))

def semantic_retrieve(query: str, topk=5):
    qv = embed_query(query)
    return retrieve_topk(qv, topk)

# -------------------------
# Structured DB Functions
# -------------------------
def get_avg_stay(condition: str):
    pipeline = [
        {"$match": {"Medical Condition": condition}},
        {"$group": {"_id": None, "avg": {"$avg": "$stay_length"}}}
    ]
    result = list(collection_patients.aggregate(pipeline))
    if result:
        return result[0]["avg"]
    return None

def count_patients(filter_obj: dict):
    return collection_patients.count_documents(filter_obj)

def compare_stay(cond1: str, cond2: str):
    avg1 = get_avg_stay(cond1)
    avg2 = get_avg_stay(cond2)
    if avg1 is None or avg2 is None:
        return None
    return {
        "cond1": cond1,
        "avg1": avg1,
        "cond2": cond2,
        "avg2": avg2,
        "difference": avg1 - avg2
    }
