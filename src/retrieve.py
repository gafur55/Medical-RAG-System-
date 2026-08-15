"""Search the index.

This is the ONE place retrieval happens. eval/score.py, generate.py and any
future app all call search() - so an improvement here shows up everywhere,
and what you measure cannot drift from what you ship.

Run:  python3 -m src.retrieve "Karapandzic flap"
"""

import json
import sys
import numpy as np
from src import config


_model = None
_chunks = None
_vectors = None


def get_model():
    """Load the embedding model once per process."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(config.EMBEDDING_MODEL)
    return _model


def get_index():
    """Load chunks + vectors once, and check they still match.
    If you re-run ingest.py and forget embed.py, row 847 of the vector array
    no longer describes chunk 847 and every result is quietly wrong.
    """

    global _chunks, _vectors
    if _chunks is None:
        _chunks = json.loads(config.CHUNKS_PATH.read_text())
        _vectors = np.load(config.VECTORS_PATH)
        assert len(_chunks) == len(_vectors), (
            f"{len(_chunks)} chunks but {len(_vectors)} vectors - "
            "re-run: python3 -m src.embed"
        )
    return _chunks, _vectors


def embed_query(question, use_prefix=None):
    """Encode a question. BGE wants a prefix on queries but not on passages."""
    if use_prefix is None:
        use_prefix = config.USE_QUERY_PREFIX
    text = (config.QUERY_PREFIX + question) if use_prefix else question
    return get_model().encode(text, normalize_embeddings=True)


def search(question, k=None, kinds=None, use_prefix=None):
    """Return the top k chunks for a question, best first."""
    k = k or config.TOP_K
    kinds = kinds if kinds is not None else config.RETRIEVE_KINDS

    chunks, vectors = get_index()
    q = embed_query(question, use_prefix=use_prefix)

    # Both sides are normalized, so the dot product IS cosine similarity.
    # 1491 x 384 is tiny - brute force is exact and instant.
    scores = vectors @ q # matrix multiplication

    # Excluded kinds get -inf so they can never win a slot.
    allowed = np.array([c["kind"] in kinds for c in chunks])
    scores = np.where(allowed, scores, -np.inf)

    top = np.argsort(-scores)[:k]
    return [
        {"rank": rank, "score": float(scores[i]), "chunk": chunks[i]}
        for rank, i in enumerate(top, start=1)
        if np.isfinite(scores[i])
    ]




if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "Karapandzic flap"
    print("QUESTION:", question, "\n")
    for hit in search(question, k=5):
        c = hit["chunk"]
        figs = f" [{len(c['figures'])} figure]" if c["figures"] else ""
        print(f"--- {hit['rank']}. p{c['page']} | {c['kind']}{figs} | {hit['score']:.3f}")
        print(c["text"][:300], "\n")
