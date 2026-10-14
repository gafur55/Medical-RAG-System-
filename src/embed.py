"""chunks.json -> vectors.npy + vector_keys.json, re-embedding only what changed.

Passages are embedded without the query prefix. Only queries get the BGE
prefix - that asymmetry is how the model was trained. A chunk with a section
path is embedded as "Section > Subsection\\n<text>" (embed_text), so it keeps
the meaning its heading gave it; the textbook has no section path, so its
embedded text is its bare text, exactly as before.

Why incremental: with several sources indexed, enabling one more document
should not re-embed the other thousands of chunks. Every vector row gets a
key, sha256(model | embedded text). On the next run a chunk whose key is
already on disk reuses that row, and only the rest go through the model.
When nothing is new the model is never even loaded.

vector_keys.json also records which chunk each row belongs to, in
chunks.json order, and a digest of the vectors.npy it was written with.
retrieve.py checks the ids, the keys and the digest against the files on
disk, so forgetting to re-run this after ingest.py - or a run that died
between its two writes - fails loudly instead of quietly pairing row 847
with the wrong chunk.

Legacy index: before vector_keys.json existed, vectors.npy rows simply
followed chunks.json order. adopt_legacy() records that pairing as keys, so
the first multi-source build reuses the 1491 textbook vectors instead of
recomputing them. ingest.py calls it before it overwrites chunks.json.

Run:  python3 -m src.embed           # embed new or changed chunks only
      python3 -m src.embed --full    # re-embed everything
"""

import argparse
import hashlib
import json
import os

import numpy as np

from src import config


# ------------------------------------------------------------ keys
def embed_text(chunk):
    """The exact string that gets embedded for a chunk.

    Section headings are prepended (outermost first) when the chunk has a
    section path and config.EMBED_SECTION_HEADINGS is on. The textbook's
    section is [] - and old records have none - so its text is unchanged
    and its vectors stay valid.
    """
    section = chunk.get("section") or []
    if config.EMBED_SECTION_HEADINGS and section:
        return " > ".join(section) + "\n" + chunk["text"]
    return chunk["text"]


def embed_key(model, text):
    """Content key of one vector: same model and same text -> same key.

    The model is part of the key, so switching EMBEDDING_MODEL can never
    reuse a vector from the old model's space.
    """
    return hashlib.sha256(f"{model}|{text}".encode("utf-8")).hexdigest()[:16]


def chunk_keys(chunks, model):
    return [embed_key(model, embed_text(c)) for c in chunks]


def vectors_digest(vectors):
    """sha256 of a vector matrix: its shape plus its float32 bytes.

    vector_keys.json carries it to say which vectors.npy it describes. The
    ids and keys alone cannot: save() writes the two files one after the
    other, so a run that dies in between - or restoring one file without
    the other - leaves a key file whose ids look right for rows that are
    not theirs. Same rows, same digest; anything else, a different one.
    """
    v = np.ascontiguousarray(vectors, dtype=np.float32)
    return hashlib.sha256(str(v.shape).encode("utf-8") + v.tobytes()).hexdigest()


# ------------------------------------------------------------ planning
def plan(new_keys, old_keys):
    """Decide, per new row, which old row to reuse. Pure: keys in, plan out.

    Returns (reuse, todo): reuse maps new row -> old row, todo lists the new
    rows that need the model.

    Two chunks can share a key (the same text on two pages). The n-th
    occurrence of a key reuses the n-th old row with that key, falling back
    to the last one, so an unchanged index maps every row onto itself.
    """
    old_rows = {}
    for row, key in enumerate(old_keys):
        old_rows.setdefault(key, []).append(row)

    reuse, todo, seen = {}, [], {}
    for row, key in enumerate(new_keys):
        rows = old_rows.get(key)
        if rows:
            n = seen.get(key, 0)
            reuse[row] = rows[min(n, len(rows) - 1)]
            seen[key] = n + 1
        else:
            todo.append(row)
    return reuse, todo


def assemble(texts, new_keys, old_keys, old_vectors, encode):
    """Build the full vector matrix: reused rows plus freshly encoded ones.

    encode(list_of_texts) -> 2-D array is only called when something is new,
    and each distinct text is encoded once. Returns (vectors, n_reused,
    n_embedded) with vectors as float32, rows in new_keys order.
    """
    reuse, todo = plan(new_keys, old_keys)

    fresh = {}
    if todo:
        unique = list(dict.fromkeys(texts[i] for i in todo))
        encoded = np.asarray(encode(unique), dtype=np.float32)
        fresh = dict(zip(unique, encoded))

    if old_vectors is not None and len(old_vectors):
        dim = old_vectors.shape[1]
    elif fresh:
        dim = len(next(iter(fresh.values())))
    else:
        dim = 0
    vectors = np.zeros((len(new_keys), dim), dtype=np.float32)
    for row, old_row in reuse.items():
        vectors[row] = old_vectors[old_row]
    for row in todo:
        vectors[row] = fresh[texts[row]]
    return vectors, len(reuse), len(todo)


# ------------------------------------------------------------ files
def load_cache(model):
    """{"ids", "keys", "vectors"} from the last run, or None when unusable.

    Unusable: no vector_keys.json, vectors.npy missing, a different model,
    a digest that is missing or does not match vectors.npy, or a row count
    that does not match - in each case everything is re-embedded, which is
    slow but never wrong.

    The digest is what makes reuse safe: plan() maps new rows onto old row
    positions, and those positions come from the key file. A key file left
    over from a run that died before writing it (or restored without
    vectors.npy) would label the new rows with the old keys - with equal
    chunk counts nothing else notices. A key file from before digests
    existed gets one from add_missing_digest(), which embed.py runs first,
    when it still describes chunks.json exactly.
    """
    try:
        meta = json.loads(config.VECTOR_KEYS_PATH.read_text())
        vectors = np.load(config.VECTORS_PATH)
    except (OSError, ValueError):
        return None
    if meta.get("embedding_model") != model:
        print(f"embedding model changed ({meta.get('embedding_model')} -> {model}): "
              "re-embedding everything")
        return None
    if meta.get("vectors_sha256") != vectors_digest(vectors):
        print("vector_keys.json was not written for this vectors.npy (no digest, or a "
              "different one): re-embedding everything")
        return None
    keys = meta.get("keys") or []
    if len(keys) != len(vectors):
        print(f"vector_keys.json has {len(keys)} keys but vectors.npy has "
              f"{len(vectors)} rows: re-embedding everything")
        return None
    return {"ids": meta.get("ids") or [], "keys": keys, "vectors": vectors}


def keys_record(chunks, model):
    return {"embedding_model": model,
            "ids": [c["id"] for c in chunks],
            "keys": chunk_keys(chunks, model)}


def with_digest(record, vectors):
    """record plus "vectors_sha256", the digest of the vectors it describes."""
    return {**record, "vectors_sha256": vectors_digest(vectors)}


def _read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _replace(path, write):
    """Write to path.tmp, then rename over path, so a crash mid-write never
    leaves a half-written file behind."""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        write(f)
    os.replace(tmp, path)


def write_keys(record):
    _replace(config.VECTOR_KEYS_PATH,
             lambda f: f.write(json.dumps(record, indent=1).encode("utf-8")))


def save(vectors, record):
    """vectors.npy first, vector_keys.json last, the keys carrying the
    digest of exactly the vectors written.

    If the run dies in between, vectors.npy is new and the key file old,
    so its digest no longer matches: retrieve.py refuses the pair, and the
    next run's load_cache() discards it and re-embeds everything. The ids
    alone would not tell - with the same chunk count, the next run would
    have read the old keys as labels for the new rows.
    """
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    record = with_digest(record, vectors)
    _replace(config.VECTORS_PATH, lambda f: np.save(f, vectors))
    write_keys(record)


def adopt_legacy(chunks, vectors, model=None):
    """Write vector_keys.json for an index built before keys existed.

    Before the multi-source upgrade, vectors.npy had no key file: row i was
    the embedding of chunks.json record i (retrieve.py only checked that the
    counts matched). This records that pairing as keys, computed with the
    same embed_text() - plain text for the textbook - so the next embed run
    reuses those vectors instead of recomputing them. The digest of vectors
    is recorded too, as save() does.

    Only when the counts match; otherwise the pairing is already broken and
    nothing is written. Returns True when the key file was written.
    """
    model = model or config.EMBEDDING_MODEL
    if len(chunks) != len(vectors):
        return False
    write_keys(with_digest(keys_record(chunks, model), vectors))
    return True


def add_missing_digest():
    """Give a vector_keys.json written before digests existed its digest.

    Such a file cannot prove which vectors.npy it describes, so load_cache()
    would not reuse it. It is trusted - and gains the digest of today's
    vectors.npy, which is left untouched - only when it describes exactly
    the chunks.json on disk: as many rows, the same ids in the same order,
    and the keys those chunks give with its model. That is the pairing
    retrieve.py has been serving all along (it checks exactly that), so
    nothing new is trusted. Anything else is left alone, and the next embed
    run re-embeds everything - slower, never wrong.

    Returns True when the key file was rewritten.
    """
    meta = _read_json(config.VECTOR_KEYS_PATH)
    if not meta or "vectors_sha256" in meta or not config.VECTORS_PATH.exists():
        return False
    chunks = _read_json(config.CHUNKS_PATH)
    if chunks is None:
        return False
    vectors = np.load(config.VECTORS_PATH, mmap_mode="r")
    ids = meta.get("ids")
    if not (len(vectors) == len(chunks) == len(ids or [])
            and ids == [c["id"] for c in chunks]
            and meta.get("keys") == chunk_keys(chunks, meta.get("embedding_model"))):
        print("vector_keys.json has no digest and does not describe chunks.json: "
              "the next embed run re-embeds everything")
        return False
    write_keys(with_digest(meta, vectors))
    print(f"recorded the digest of vectors.npy in {config.VECTOR_KEYS_PATH}")
    return True


def adopt_existing_index():
    """Adopt the chunks.json + vectors.npy on disk, if they predate key
    files (or digests: then add_missing_digest() does the work).

    Conditions: vectors.npy exists, vector_keys.json does not, chunks.json
    has as many records as vectors.npy has rows (the test retrieve.py used
    to accept the pair), and chunks.json was written by the old ingest - no
    record has a "source" field.

    ingest.py calls this right before it overwrites chunks.json, the last
    moment the old pairing is known; embed.py calls it too, for when it is
    run without ingest first.

    Why the old-format condition: a new-style chunks.json always gets its
    key file from embed.py. One without it (deleted by hand, or a crash
    between the two writes in save()) may no longer match vectors.npy, so
    it is re-embedded from scratch - slower, never wrong.
    """
    if config.VECTOR_KEYS_PATH.exists():
        return add_missing_digest()
    if not config.VECTORS_PATH.exists():
        return False
    try:
        chunks = json.loads(config.CHUNKS_PATH.read_text())
    except (OSError, ValueError):
        return False
    if any("source" in c for c in chunks):
        return False
    vectors = np.load(config.VECTORS_PATH, mmap_mode="r")
    adopted = adopt_legacy(chunks, vectors)
    if adopted:
        print(f"adopted existing vectors: wrote keys for {len(chunks)} rows "
              f"-> {config.VECTOR_KEYS_PATH}")
    else:
        print(f"not adopting vectors.npy: {len(vectors)} rows for {len(chunks)} chunks "
              "- the next embed run re-embeds everything")
    return adopted


# ------------------------------------------------------------ model
def model_encoder(model_name):
    """encode(texts) -> vectors, loading SentenceTransformer on first call.

    Lazy so that a run with nothing new never pays for loading the model.
    normalize_embeddings=True scales every vector to length 1, so a dot
    product equals cosine similarity. retrieve.py depends on this.
    """
    def encode(texts):
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
        return model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
    return encode


def main(argv=None, encode=None):
    """Embed chunks.json. encode is injectable (tests pass a fake); the
    default loads the real model, and only if some chunk is new.

    Returns {"reused", "embedded", "written"} for callers and tests.
    """
    parser = argparse.ArgumentParser(description="Embed chunks.json incrementally.")
    parser.add_argument("--full", action="store_true",
                        help="ignore saved vectors and re-embed every chunk")
    args = parser.parse_args(argv)

    model = config.EMBEDDING_MODEL
    chunks = json.loads(config.CHUNKS_PATH.read_text())
    print(f"loaded {len(chunks)} chunks")

    cache = None
    if not args.full:
        adopt_existing_index()
        cache = load_cache(model)
    old = cache or {"ids": [], "keys": [], "vectors": None}

    texts = [embed_text(c) for c in chunks]
    record = {"embedding_model": model,
              "ids": [c["id"] for c in chunks],
              "keys": [embed_key(model, t) for t in texts]}
    vectors, n_reused, n_embedded = assemble(
        texts, record["keys"], old["keys"], old["vectors"],
        encode or model_encoder(model))
    print(f"reused {n_reused}, embedded {n_embedded}")
    summary = {"reused": n_reused, "embedded": n_embedded, "written": False}

    if cache and record["ids"] == old["ids"] and record["keys"] == old["keys"]:
        # Same chunks in the same order: every row was reused onto itself.
        print(f"already up to date: {config.VECTORS_PATH}")
        return summary
    save(vectors, record)
    print(f"saved {vectors.shape} -> {config.VECTORS_PATH}")
    summary["written"] = True
    return summary


if __name__ == "__main__":
    main()
