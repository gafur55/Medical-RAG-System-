import json
from sentence_transformers import SentenceTransformer
from config import CHUNKS_PATH, EMBEDDING_MODEL, PROCESSED_DIR
import numpy as np


with open(CHUNKS_PATH) as f:
    chunks = json.load(f)

print("Loaded", len(chunks), "chunks")

model = SentenceTransformer(EMBEDDING_MODEL)

texts = [c["text"] for c in chunks]

vectors = model.encode(texts, batch_size=32, show_progress_bar=True)

print("Shape:", vectors.shape)

np.save(PROCESSED_DIR / "vectors.npy", vectors)
print("Saved to vectors.npy")