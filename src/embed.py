"""chunks.json -> vectors.npy

Passages are embedded PLAIN. Only queries get the BGE prefix - that
asymmetry is how the model was trained.

Re-run whenever ingest.py changes. retrieve.py asserts the counts match.

Run:  python3 -m src.embed
"""

import json
import numpy as np
from src import config


def main():
    chunks = json.loads(config.CHUNKS_PATH.read_text())
    print(f"loaded {len(chunks)} chunks")

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(config.EMBEDDING_MODEL)

    # normalize_embeddings=True scales every vector to length 1, so a dot
    # product equals cosine similarity. retrieve.py depends on this.
    vectors = model.encode(
        [c["text"] for c in chunks],
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    np.save(config.VECTORS_PATH, vectors)
    print(f"saved {vectors.shape} -> {config.VECTORS_PATH}")


if __name__ == "__main__":
    main()
