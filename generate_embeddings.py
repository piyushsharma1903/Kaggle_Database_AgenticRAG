import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import json
import os

# --- 1. Load your summaries ---
df = pd.read_csv("patient_summaries.csv")

# --- 2. Initialize the model ---
# This downloads the model the first time you run it (~90MB)
print("Loading model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded.")

# --- 3. Generate embeddings ---
# tqdm adds progress bar, just to keep things visible
embeddings = []
for text in tqdm(df['Summary'], desc="Embedding summaries"):
    vec = model.encode(text)
    embeddings.append(vec.tolist())

df['embedding'] = embeddings

# --- 4. Save embeddings locally ---
os.makedirs("data", exist_ok=True)
df.to_json("data/embedded_summaries.json", orient="records", indent=2)
print(f"✅ Saved {len(df)} embedded summaries to data/embedded_summaries.json")
