# Retrieval — `src/retrieve.py`

Takes a question, returns the top-k chunks. This is the **only** place
retrieval happens — `eval/score.py`, `generate.py` and any future app all
call `search()`.

```
        question
            |
            v
      embed_query()                    get_index()
   prefix + encode to 384              chunks.json + vectors.npy
            |                          asserts counts match
            |                                 |
            +-----------------+---------------+
                              |
                              v
                       vectors @ query
                    dot product = cosine
                              |
                              v
                      filter by kind
                  excluded kinds set to -inf
                              |
                              v
                        argsort, take k
                              |
                              v
              [{rank, score, chunk}, ...]
```

## Run

```bash
python3 -m src.retrieve "Reconstruction options for a nasal alar defect"
```

Requires `chunks.json` and `vectors.npy`. If you re-ran `ingest.py` without
re-running `embed.py`, `get_index()` raises with the fix in the message.

## Functions

### `get_model()` / `get_index()`
Module-level caches. The model is ~130MB and the eval harness calls
`search()` once per question — without caching you would reload it every time.

`get_index()` asserts `len(chunks) == len(vectors)`. If those drift, row 847
of the vector array no longer describes chunk 847 and every result is quietly
wrong. Loud failure beats silent nonsense.

### `embed_query(question, use_prefix=None)`
Encodes the question. See "The query prefix" below.

### `search(question, k=None, kinds=None, use_prefix=None)`
Returns `[{"rank": 1, "score": 0.81, "chunk": {...}}, ...]`, best first.

Scoring is `vectors @ q` — a plain dot product. That works only because both
sides were normalized to length 1, which makes the dot product identical to
cosine similarity. `embed.py` must keep `normalize_embeddings=True` or these
scores become meaningless (longer chunks would score higher just for being
longer).

Excluded kinds are set to `-inf` rather than removed, so the array stays
aligned with `chunks` and indices remain valid.

Brute force over 1491 × 384 is exact and instant. No vector database needed
at this scale — a second textbook would be the point to reconsider.

## The query prefix

BGE is trained **asymmetrically**. Passages are embedded plain; short queries
are meant to carry an instruction prefix:

```
Represent this sentence for searching relevant passages: {question}
```

The reasoning: a question and the passage answering it do not look alike as
text — one is short and interrogative, the other long and declarative. The
prefix tells the model "put this where its answer lives" rather than "put
this near other short questions."

`v1.5` reduced how much this matters, and its effect on a Mohs surgery
textbook is unknown. Hence a config flag rather than a hardcoded choice.

`embed.py` applies **no** prefix. Only queries get one.

## Why `use_prefix=None` and not `use_prefix=config.USE_QUERY_PREFIX`

Python evaluates default arguments **once**, at function definition. Writing
`use_prefix=config.USE_QUERY_PREFIX` would freeze whatever config said at
import time, and changing it later in the same process would do nothing.

`None` is a sentinel meaning "not specified" — distinct from `False`, which
means "specified, and off." Three states are needed because `False` is a
legitimate value to pass. The function reads config at call time, so it stays
live.

This is what lets the eval harness compare both settings in one process,
without reloading the model:

```python
with_prefix    = score_all(use_prefix=True)
without_prefix = score_all(use_prefix=False)
```

`k` and `kinds` use the same pattern. Note `kinds` uses `is not None` rather
than `or`, because an empty tuple is falsy and `or` would silently swap it
for the config default.

## Experiment knobs

Every one of these is a one-line change in `config.py`, then re-run
`score.py`. Results append to `data/eval/results.jsonl` with the config that
produced them.

| Knob | Question it answers |
|---|---|
| `USE_QUERY_PREFIX` | Does the BGE query instruction help on this corpus? |
| `RETRIEVE_KINDS` | Do figure captions help or crowd out prose? |
| `TOP_K` | How many chunks should the LLM see? |
| `EMBEDDING_MODEL` | Is `bge-base` worth the extra latency over `bge-small`? |

Workflow:

```bash
python3 -m src.eval.score --note "baseline, prefix on"
# edit config.py: USE_QUERY_PREFIX = False
python3 -m src.eval.score --note "prefix off"
```

Then compare the two lines in `results.jsonl`. **One knob at a time** — change
two and you cannot attribute the difference.
