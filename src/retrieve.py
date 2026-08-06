import json
import numpy as np
from sentence_transformers import SentenceTransformer
from config import CHUNKS_PATH, EMBEDDING_MODEL, PROCESSED_DIR


with open(CHUNKS_PATH) as f:
    chunks = json.load(f)

vectors = np.load(PROCESSED_DIR / "vectors.npy")
model = SentenceTransformer(EMBEDDING_MODEL)

question = "Karapandzic flap"

q_vector = model.encode(question)

scores = model.similarity(q_vector, vectors)[0].numpy()

top = np.argsort(scores)[::-1][:5] #first ascending, then [::-1] to change to descending and [:5] top 5

print("QUESTION:", question)
print()

for i in top:
    print("-" * 60)
    print("Page", chunks[i]["page"], "| score", round(float(scores[i]), 3))
    print(chunks[i]["text"][:300])