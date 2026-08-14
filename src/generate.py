import json
import numpy as np
import ollama
from sentence_transformers import SentenceTransformer


from config import CHUNKS_PATH, EMBEDDING_MODEL, PROCESSED_DIR

with open(CHUNKS_PATH) as f:
    chunks = json.load(f)

vectors = np.load(PROCESSED_DIR / "vectors.npy")
model = SentenceTransformer(EMBEDDING_MODEL)

question = "Reconstruction options for a 1.5 cm defect of the nasal ala"

q_vector = model.encode(question)
scores = model.similarity(q_vector, vectors)[0].numpy()
top = np.argsort(-scores)[:5]

context = ""
for i in top:
    context += f"[Page {chunks[i]['page']}]\n{chunks[i]['text']}\n\n"


prompt = f"""You are assisting a Mohs surgeon with reconstruction planning.

Answer the question using ONLY the textbook excerpts below.
Cite the page number for every claim, like this: (p. 384).
If the excerpts do not contain the answer, say so plainly. Do not use outside knowledge.

EXCERPTS:
{context}

QUESTION: {question}

ANSWER:"""

response = ollama.chat(
    model="qwen2.5:7b",
    messages=[{"role": "user", "content": prompt}],
)

print(response["message"]["content"])