# Ingestion — `src/ingest.py` and `src/embed.py`

`ingest.py` turns every enabled source in `sources.yaml` into text chunks
with metadata (`chunks.json`), a record of what went in (`manifest.json`),
and — for the textbook — a folder of extracted figure images. `embed.py`
then turns the chunks into vectors, re-using every vector whose text has
not changed. How to add, enable and license a source: `docs/sources.md`.

```
                          sources.yaml
                  load_registry(), enabled_sources()
                     enabled entries, file order
                              |
          +-------------------+--------------------+
          |                                        |
 format: pdf_textbook                    pdf / html / nxml / docx / markdown
          |                                        |
  build_chunks(id, path)                  loaders.load(source)
  layout blocks, classify(),              -> Units (paragraph, section,
  figures + caption links                    locator, page, kind)
          |                                        |
          |                               chunking.pack()
          |                               -> section-aware chunks
          |                                  with overlap
          |                                        |
          |                               chunk_records(source, pieces)
          |                                        |
          +-------------------+--------------------+
                              |
                  duplicate_ids()  (refuse on any)
                  embed.adopt_existing_index()  (first upgrade only)
                              |
                              v
               chunks.json + manifest.json
                              |
                       python3 -m src.embed
                              |
                              v
            vectors.npy + vector_keys.json
            (only new or changed chunks encoded)
```

## Run

```bash
python3 -m src.ingest            # build chunks.json + manifest.json
python3 -m src.embed             # embed new/changed chunks; prints "reused N, embedded M"
python3 -m src.embed --full      # re-embed everything
python3 -m src.ingest --list     # status of every registry entry
```

Output with only the textbook enabled (the default `sources.yaml`):

```
mohs_layers                   1491 chunks  {'prose': 1194, 'caption': 245, 'reference': 52}
  figures saved  : 282 -> data/processed/figures
  captions linked: 220 of 245
chunks written : 1491 -> data/processed/chunks.json
  by kind      : {'prose': 1194, 'caption': 245, 'reference': 52}
manifest       : index f972e1512122, 1 source(s) -> data/processed/manifest.json
next           : python3 -m src.embed
```

```
loaded 1491 chunks
reused 1491, embedded 0
already up to date: data/processed/vectors.npy
```

Re-run **both** whenever sources, a source file, a loader or the chunking
settings change. `retrieve.py` checks that `vector_keys.json` lists exactly
the chunk IDs in `chunks.json`, in order, and refuses to search otherwise.

Nothing is written until every source has loaded and every chunk ID is
unique, so a failing loader or a missing file leaves the previous index as
it was. An enabled source whose file is missing stops the build with its
download URL and notes from `sources.yaml`.

## Inputs and outputs

| | |
|---|---|
| In | `sources.yaml` — which documents, where, and on what terms |
| In | the source files, `data/raw/...` (gitignored) |
| Out | `data/processed/chunks.json` — every chunk, sources in registry order |
| Out | `data/processed/manifest.json` — index ID and per-source counts |
| Out | `data/processed/figures/*.jpeg` — textbook figures |
| Out (embed) | `data/processed/vectors.npy` — float32, one row per chunk, chunks.json order |
| Out (embed) | `data/processed/vector_keys.json` — chunk ID and content key per row |

## Chunk record

```json
{
  "id": "1364973a5dbdb729",
  "book": "mohs_layers",
  "source": "mohs_layers",
  "domain": "clinical",
  "page": 188,
  "locator": "p. 188",
  "section": [],
  "kind": "prose",
  "text": "Preservation of the subdermal plexus of vascular anastomoses...",
  "figures": []
}
```

| field | |
|---|---|
| `id` | content hash, see `chunk_id` below |
| `book` | = `source`; kept so code written before the registry keeps working |
| `source` | the entry's `id` in `sources.yaml` |
| `domain` | `clinical` or `operations` — retrieval can filter on it |
| `page` | page number for paginated sources (always for the textbook), else `null` |
| `locator` | where to look: `"p. 12"` when paginated, else the section name |
| `section` | heading path, outermost first; `[]` for the textbook |
| `kind` | `prose`, `caption`, `reference` or `table` |
| `figures`, `figure_paths` | textbook captions only: the linked figure |

Records written before the registry lack `source`, `domain`, `locator` and
`section`; readers default them to `book`, `"clinical"`, `"p. <page>"` and
`[]`.

## manifest.json

```json
{"index_id": "f972e1512122", "built_at": "2026-10-14T02:45:25+00:00", "n_chunks": 1491,
 "sources": [{"id": "mohs_layers", "label": "Mohs textbook", "title": "...", "type": "textbook",
              "domain": "clinical", "format": "pdf_textbook", "license": "...", "url": "",
              "year": null, "n_chunks": 1491}]}
```

`index_id` is the first 12 hex characters of the sha256 of all chunk IDs
joined by newlines: it changes when any chunk or the order changes, and it
is logged with every answer and score run (`config.index_id()`). The app
reads source labels from the manifest (`config.source_labels()`), so it
never needs `sources.yaml` or PyYAML. Re-running ingest on unchanged
sources keeps the old `built_at`, so the files come out byte-identical.

## The textbook path — unchanged on purpose

`format: pdf_textbook` runs the original code: the PDF is read twice, for
different things.

**Figure path.** `extract_figures()` walks every page, pulls each embedded
image, writes it to disk, and records the image's bounding box. Returns a
`{page_number: [figure, ...]}` dict.

**Text path.** `build_chunks()` walks every page again, takes layout blocks,
runs `clean()`, drops blocks under `MIN_CHUNK_CHARS`, and labels the rest
with `classify()`.

The paths join when a block is a caption: `nearest_figure_above()` compares
the caption's y-position against the figures on that page and picks one.

The only differences from the single-book version: the PDF path comes from
the registry entry (`config.BOOKS` is still the fallback for
`build_chunks()` called with no path), every record gains
`source`/`domain`/`locator`/`section`, and figure paths are made relative
to `config.ROOT` (the same directory as before). Same blocks, same text,
same IDs — so the 1491 existing vectors stay valid and every measured
result in `docs/experiments.md` still describes this index. Verified on the
real data: a rebuild reproduces every original key and value of every
record, in order.

## The path for every other source

`loaders.load(source)` reduces a PDF, HTML page, JATS `.nxml`, `.docx` or
markdown file to `Unit`s — one paragraph, caption or table each, in reading
order, with its heading path and locator. `chunking.pack()` groups them into
chunks of about `CHUNK_TARGET_CHARS` that never cross a section or page and
overlap by whole sentences (see `src/chunking.py` for the rules and why),
and `chunk_records()` adds the fields above. It is the fix for the
textbook's known limitation below, applied where it cannot disturb the
textbook's vectors.

## Functions — `ingest.py`

### `clean(text)`, `chunk_id(book_id, page, text)`
Shared with `chunking.py` (one definition each). `clean` collapses all
whitespace runs: PDF extraction inserts hard line breaks mid-sentence.

`chunk_id` is a content-derived name — same text always gives the same
16-char ID. The textbook passes its page number; other sources pass the page
when they have one, else the locator, so the same paragraph under another
heading is another chunk. Change the chunking and the text changes, so IDs
change. That is correct — it is a different chunk. Ground truth therefore
lives in `questions.json` as pages and anchor phrases, **not** as chunk IDs.

### `classify(text)`
Labels a textbook block `caption`, `reference`, or `prose`, so retrieval can
filter on `kind`. Captions are 16% of the corpus and were crowding out prose
in early results; tagging makes exclusion a measurable experiment rather
than a guess.

References use the structural pattern `[N] Surname`, not a list of journal
names. PDF extraction eats spaces (`DermatolSurg`), so literal names miss.

### `extract_figures(doc, book_id)`
PyMuPDF returns images in two steps: `get_images()` gives xrefs (the image's
ID inside the PDF), `extract_image()` gives the bytes. Images are written in
their original format, not re-encoded.

`get_image_rects()` gives the on-page position. That is the only reason
`y0`/`y1` are stored — the PDF says nothing about which caption belongs to
which figure, so it has to be inferred from geometry.

Skips images that fail to extract, images under `MIN_FIGURE_PIXELS`, and
images referenced but never placed. Existing files are not rewritten.

### `nearest_figure_above(figures_on_page, caption_top)`
Textbooks print the figure above its caption. Of the figures whose bottom
edge is at or above the caption's top edge, take the closest. y increases
downward, so "above" means a smaller y. The `+5` is slack for rounding.

Fallback when nothing sits above: whatever is vertically nearest. That
branch is a genuine guess.

### `chunk_records(source, pieces)`
`chunking.pack()` output → chunk records. Pure. A piece the loader left
without a locator gets `"p. N"` or its innermost heading.

### `duplicate_ids(chunks)`
One message per repeated ID, naming the source, the locator and the text.
A duplicate means the same text twice at the same place of one source;
ingest refuses rather than write an index where an ID names two rows.

### `build_manifest(chosen, chunks, built_at)`, `index_id(ids)`
The manifest content, pure (the time is passed in).

### `format_status(registry, manifest, present, extra_problems)`
The `--list` report, pure: file presence and the manifest are passed in.
`status_report()` gathers them from disk, plus whether `vectors.npy` still
matches `chunks.json` (`vector_problem()`).

## Functions — `embed.py`

### `embed_text(chunk)`
The exact string embedded: the chunk text, or `"A > B\n<text>"` when the
chunk has a section path and `EMBED_SECTION_HEADINGS` is on. Textbook
chunks have `section: []`, so their embedded text is unchanged.

Passages are embedded without the query prefix; only queries get the BGE
prefix — that asymmetry is how the model was trained.

### `embed_key(model, text)`
`sha256(f"{model}|{text}")[:16]`. Same model and same text → same key, so
the vector can be reused wherever the chunk moved to. The model is part of
the key: switching `EMBEDDING_MODEL` re-embeds everything.

### `plan(new_keys, old_keys)` and `assemble(...)`
`plan` decides, per new row, which old row to reuse; the rest are "todo".
Two chunks may share a key (the textbook has one block printed on two
pages); the n-th occurrence reuses the n-th old row, so an unchanged index
maps every row onto itself. `assemble` builds the matrix and calls
`encode()` only for the todo texts, each distinct text once — and not at
all when nothing is new. Both are pure and tested with fake vectors.

The model is loaded lazily inside `encode()`, with the same call as
always: `batch_size=32, normalize_embeddings=True`. Normalized vectors make
the dot product in `retrieve.py` a cosine similarity.

### `save(vectors, record)`
Writes `vectors.npy`, then `vector_keys.json`, each to a temporary file
renamed into place. If the run dies in between, the key file still names
the old chunks, `retrieve.py`'s ID check fails, and the next run re-plans.

### `adopt_legacy(chunks, vectors)` — the first upgrade

Before this version there was no `vector_keys.json`: row *i* of
`vectors.npy` was simply the embedding of record *i* of `chunks.json`
(`retrieve.py` only compared the counts). The new ingest rewrites
`chunks.json`, after which that pairing could no longer be reconstructed.

So `ingest.main()` calls `embed.adopt_existing_index()` just **before** it
overwrites `chunks.json`. If `vectors.npy` exists, `vector_keys.json` does
not, `chunks.json` was written by the old ingest (no `source` fields), and
the counts match, it writes `vector_keys.json` for the existing pairing —
keys computed with `embed_text()` of the old records, i.e. their plain text.
The rebuilt textbook chunks have the same text, hence the same keys, and
`embed` reuses all 1491 vectors:

```
adopted existing vectors: wrote keys for 1491 rows -> data/processed/vector_keys.json
...
reused 1491, embedded 0
```

`embed.main()` makes the same check, for when it is run before ingest. A
new-style `chunks.json` without a key file (deleted by hand, or a crash
between the two writes) is never adopted — it may no longer match the
vectors, so it is re-embedded from scratch: slower, never wrong.

## Known limitations

**Textbook blocks are layout units, not semantic ones.** Chunk sizes range
from 101 to 2,530 characters with no overlap, so a fact spanning a block
boundary is retrievable from neither half. The new sources use
`chunking.pack()` with overlap; the textbook keeps its blocks so its IDs and
vectors stay put. Re-chunking it is a deliberate, measured change for later
— it re-embeds the whole book and invalidates comparisons with earlier runs.

**25 captions link to no figure.** Video references (no image exists) and
figures printed on the facing page from their caption.

**Composite figures.** Multi-panel figures are a single image file — page 531
is one 928×1073 image covering panels (a) through (h). Relevant if
image-embedding models are added later; they downsample to 224×224.

**Heuristics are book-specific.** `classify()` and `nearest_figure_above()`
are tuned to this textbook and will likely need adjusting for a second one.

**Gold labels are textbook pages.** `score.py` matches `gold_pages` only
against chunks of the question's `gold_source` (the textbook by default), so
a new source that answers a question correctly does not count as a hit
unless its text contains a gold anchor.

## Tests

```bash
python3 -m tests.test_ingest            # textbook functions, no pytest needed
python3 -m pytest tests/test_index.py   # registry build, --list, incremental embed
python3 -m pytest                       # every test in tests/
```

`tests/test_ingest.py` covers `clean`, `chunk_id`, `classify` and
`nearest_figure_above`, plus integrity checks on the real `chunks.json`
(unique IDs, valid kinds, every linked figure file exists on disk).

`tests/test_index.py` builds small indexes in a temp directory with fake
loaders and a fake encoder — no PDF, no model: record fields and IDs,
duplicate detection, the manifest, the `--list` table and its problems,
registry order, legacy adoption before overwrite, and incremental embedding
(reuse, re-ordering, edits, `--full`, a model change, a corrupt cache, a
newly enabled source).

Unit tests cannot tell you whether a caption-figure link is *correct* — only
that the code ran. Open a few files in `data/processed/figures/` and compare
them to their caption text. Three were verified by hand and matched.
