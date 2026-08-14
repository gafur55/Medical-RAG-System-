import json

import numpy as np
from sentence_transformers import SentenceTransformer

from config import CHUNKS_PATH, EMBEDDING_MODEL, PROCESSED_DIR, ROOT

with open(CHUNKS_PATH) as f:
    chunks = json.load(f)

with open(ROOT / "data" / "eval" / "questions.json") as f:
    questions = json.load(f)

vectors = np.load(PROCESSED_DIR / "vectors.npy")
model = SentenceTransformer(EMBEDDING_MODEL)

lines = []

for q in questions:
    q_vector = model.encode(q["question"])
    scores = model.similarity(q_vector, vectors)[0].numpy()
    top = np.argsort(-scores)[:10]

    lines.append(f"## Q{q['id']} ({q['type']})")
    lines.append(f"**{q['question']}**\n")
    lines.append("Mark each: R = relevant, N = not relevant\n")

    for rank, i in enumerate(top, start=1):
        page = chunks[i]["page"]
        text = chunks[i]["text"]
        lines.append(f"[ ]  {rank}. **Page {page}** (score {scores[i]:.3f})")
        lines.append(f"> {text}\n")

    lines.append("**Anything important missing?**\n")
    lines.append("---\n")

out = ROOT / "data" / "eval" / "review_sheet.md"
out.write_text("\n".join(lines))

print("Wrote", out)