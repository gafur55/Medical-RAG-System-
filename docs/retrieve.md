# Retrieval — `src/retrieve.py`

Takes a question, returns the top-k chunks. This is the **only** place
retrieval happens — `score.py`, `sweep.py`, `prep.py` and `generate.py` (and
through it the app) all call `search()`.

```
        question
            |
            v
      embed_query()                    get_index()
   prefix + encode to 384              chunks.json + vectors.npy
            |                          checked against vector_keys.json
            |                                 |
            +-----------------+---------------+
                              |
                              v
                       vectors @ query
                    dot product = cosine
                              |
                              v
               filter by kind, domain, front matter
                  excluded chunks set to -inf
                              |
                              v
          [hybrid_weight > 0]  fuse with BM25 keyword scores
                              |
                              v
           argsort, take k  [max_per_page: at most n per page]
                              |
                              v
              [{rank, score, chunk}, ...]
```

Everything in square brackets is an experimental knob, **off by default**.
With the defaults, `search()` returns exactly what it returned before the
knobs existed — same chunks, same order, same scores (checked for all 15
questions at k=25, see `docs/experiments.md`, Experiment C).

## Run

```bash
python3 -m src.retrieve "Reconstruction options for a nasal alar defect"
```

Requires `chunks.json` and `vectors.npy`. If you re-ran `ingest.py` without
re-running `embed.py`, `get_index()` raises with the fix in the message.

## Several sources in one index

The index can hold the textbook plus guidelines, review articles and clinic
protocols (see `sources.yaml` and `docs/sources.md`). Every chunk records
where it came from:

| field | meaning | missing on an old record → |
|---|---|---|
| `source` | id from `sources.yaml` | `chunk["book"]`, else `config.DEFAULT_SOURCE` |
| `domain` | `clinical` or `operations` | `clinical` |
| `page` | page number; `None` for html/nxml/docx/markdown | — |
| `locator` | `"p. 12"`, or the section name for unpaginated sources | `"p. {page}"` |
| `section` | heading path, outermost first; `[]` for the textbook | `[]` |

`source_of()`, `domain_of()`, `section_of()`, `locator_of()` and `place_of()`
apply those defaults, so nothing downstream special-cases old records.
`score.is_relevant()` uses `source_of()`: `gold_pages` are pages of one
source (`question["gold_source"]`, the textbook unless it says otherwise),
since p. 134 of a guideline is not p. 134 of the book. Anchors are phrases
and match in any source.

## Functions

### `get_model()` / `get_index()`
Module-level caches. The model is ~130MB and the eval harness calls
`search()` once per question — without caching you would reload it every time.

`get_index()` checks the vectors still describe the chunks
(`check_alignment()`). If those drift, row 847 of the vector array no longer
describes chunk 847 and every result is quietly wrong. Loud failure beats
silent nonsense.

- **With `vector_keys.json`** (written by `embed.py`): its `ids` must equal the
  chunk ids, in order, its row count must equal `vectors.npy`'s, and its
  `embedding_model` must be `config.EMBEDDING_MODEL` (vectors from another
  model are not comparable with this model's query vectors). Any mismatch
  raises `IndexMismatch` naming the first differing row and the fix:
  `python3 -m src.embed`.
- **Without it** (an index built before keys existed): the old check,
  `len(chunks) == len(vectors)`. That misses a re-chunk that happens to keep
  the same count, which is why the keys file exists.

### `embed_query(question, use_prefix=None)`
Encodes the question. See "The query prefix" below.

### `search(question, k=None, kinds=None, use_prefix=None, domains=None, exclude_front_matter=None, max_per_page=None, hybrid_weight=None)`
Returns `[{"rank": 1, "score": 0.81, "chunk": {...}}, ...]`, best first.

Scoring is `vectors @ q` — a plain dot product. That works only because both
sides were normalized to length 1, which makes the dot product identical to
cosine similarity. `embed.py` must keep `normalize_embeddings=True` or these
scores become meaningless (longer chunks would score higher just for being
longer).

Excluded chunks (wrong kind, wrong domain, front matter) are set to `-inf`
rather than removed, so the array stays aligned with `chunks` and indices
remain valid.

| argument | `None` reads | meaning |
|---|---|---|
| `k` | `TOP_K` | how many chunks to return |
| `kinds` | `RETRIEVE_KINDS` | chunk kinds that may be returned |
| `use_prefix` | `USE_QUERY_PREFIX` | BGE query instruction on/off |
| `domains` | `RETRIEVE_DOMAINS` | `sources.yaml` domains to search |
| `exclude_front_matter` | `EXCLUDE_FRONT_MATTER` | drop `is_front_matter()` chunks |
| `max_per_page` | `MAX_PER_PAGE` | cap per page; `0` (or `None` in config) = no cap |
| `hybrid_weight` | `HYBRID_WEIGHT` | `0` = embeddings only; `(0, 1]` mixes in BM25 |

Brute force over a few thousand × 384 is exact and instant. No vector
database needed at this scale — tens of thousands of chunks would be the
point to reconsider.

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

This is what lets the eval harness compare settings in one process, without
reloading the model — which is exactly what `src/sweep.py` does:

```python
with_prefix, _    = score_all(questions, use_prefix=True)
without_prefix, _ = score_all(questions, use_prefix=False)
```

Every argument uses the same pattern. Note `kinds` and `domains` use
`is not None` rather than `or`, because an empty tuple is falsy and `or`
would silently swap it for the config default. Likewise
`exclude_front_matter=False`, `max_per_page=0` and `hybrid_weight=0.0` are
explicit "off" values that beat whatever config says.

## Front matter — `is_front_matter(chunk, next_chunk=None)`

Experiment B showed the noise in the top 5 is not figure captions but the
packaging around the content: chapter abstracts and keyword lists (Q3 spent
three of its five slots on p. 286), the table of contents, the index, the
copyright page. They match a question's vocabulary *because* they summarise
or list everything, and they answer nothing.

`is_front_matter()` is a pure function of the chunk's text (plus its section
headings, and optionally the next chunk). It flags:

| rule | example |
|---|---|
| text starts with `Abstract` / `Keywords:` | chapter openings |
| the chapter or book describing itself | "This chapter focuses on…", "Herein, the authors…" |
| the chunk directly above a keyword list on the same page | the p. 220 scalp abstract, which has no tell-tale words |
| dense "term 123" page references | the book index, pp. 587–606 |
| names with credentials, or a list of names | contributors, title page, chapter author lines |
| publisher / copyright / trademark / e-mail address | pp. 2–6 |
| "thank you", "I acknowledge" | acknowledgements |
| `Video 8.1: …` at the start | the e-book's list of videos (p. 23) |
| section heading such as Acknowledgments, Funding, Index | new sources with headings |

What it deliberately leaves alone, because those pages are gold for some
question: chapter **key points** (the `•` lists right after an abstract — the
p. 102 ones are gold for Q13), and cross-references such as "beyond the
scope of this chapter" or "previously discussed in this chapter" (pp. 283
and 530 are gold). Figure credits with `© Thieme` are not copyright pages.

On the textbook it flags 97 of 1491 chunks: 42 of the 64 on pp. 2–36 (title,
copyright, foreword, contributors), all 32 index chunks, and 23 chapter
openings — and **none** of the chunks the gold labels count as relevant.
The rules were tuned on this one book; check the flagged list again
(`front_matter_mask(chunks)`) whenever a new source is enabled.

`search()` computes the mask once per loaded index and caches it.

## Per-page cap — `max_per_page`

At most `n` hits from one place: `(source, page)`, or `(source, locator)`
for sources without pages. Selection is greedy down the cosine ranking, so
the first k picked do not depend on how many you ask for — one deep search
still gives consistent recall@1/5/10.

The trade-off: some pages hold several right passages (p. 378, the
Karapandzic flap, has three), and a cap of 1 throws two of them away.

## Hybrid keyword search — `hybrid_weight`

Embeddings know that "lid" and "eyelid" are close; they are weaker on rare
exact terms like *Karapandzic*. BM25 is the classic keyword score: a word
counts more the rarer it is in the corpus (IDF), with diminishing returns
for repeats (k1 = 1.2) and a length correction (b = 0.75). It is implemented
in numpy (postings lists in a dict, no new dependency) over lowercase
alphanumeric tokens with a small stopword list, built once per index.

Fusion is a **convex combination of min-max-normalised scores**, both
normalised per query over the chunks that are allowed to win:

```
score = (1 - w) * minmax(cosine) + w * minmax(bm25)
```

Why both sides, not raw cosine + normalised BM25: bge cosines sit in a
narrow band (the top 10 for a question usually span ~0.05), so on raw
cosine even w = 0.2 lets BM25 take over the ranking and w stops meaning
anything. Why not reciprocal rank fusion: Bruch et al. (2023, *An analysis
of fusion functions for hybrid retrieval*, ACM TOIS) found convex
combination beats RRF and needs only a few labelled queries to tune w;
RRF also discards how far ahead the leader is. On our 13 questions
(Experiment C) the fusions differ little once their weights are matched —
raw cosine + 0.1 × BM25 behaves like the normalised version at w = 0.2, and
collapses beyond that — while RRF bought slightly higher precision@5 with
lower MRR at the small weights that matter.

With hybrid on, the returned `score` is the fused score in [0, 1], not a
cosine — keep that in mind when reading `review_sheet.md` or the app.

## Experiment knobs

Every one of these is a one-line change in `config.py`, then re-run
`score.py` — or, better, add a row to `GRID` in `src/sweep.py` and compare
them all in one process. Results append to `data/eval/results.jsonl` with
the config that produced them (`score.py` always; `sweep.py` only with
`--log`).

| Knob | Question it answers |
|---|---|
| `USE_QUERY_PREFIX` | Does the BGE query instruction help on this corpus? |
| `RETRIEVE_KINDS` | Do figure captions help or crowd out prose? |
| `RETRIEVE_DOMAINS` | Should clinical answers see operations documents (and vice versa)? |
| `EXCLUDE_FRONT_MATTER` | Do abstracts, keyword lists and index pages crowd out content? |
| `MAX_PER_PAGE` | Does one page hog the context? |
| `HYBRID_WEIGHT` | Do exact keywords rescue what embeddings miss? |
| `TOP_K` | How many chunks should the LLM see? |
| `EMBEDDING_MODEL` | Is `bge-base` worth the extra latency over `bge-small`? |

Workflow:

```bash
python3 -m src.sweep                       # compare every GRID row, write nothing
python3 -m src.sweep --log                 # same, and log each run ("sweep: <name>")
python3 -m src.score --note "baseline"     # one run with whatever config.py says
```

`TOP_K` does not change the ranking, only where it is cut, so its sweep rows
have the baseline's recall@1/5 and precision@5; read the `r@k`/`p@k` columns
(recall and precision at the k the model is shown) instead.

**One knob at a time** — change two and you cannot attribute the
difference. The combination rows in the sweep come after the single knobs
for that reason.
