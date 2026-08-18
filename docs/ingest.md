# Ingestion — `src/ingest.py`

Turns the source PDF into two things: a list of text chunks with metadata,
and a folder of extracted figure images.

```
                          Source PDF
                              |
              +---------------+---------------+
              |                               |
     extract_figures()               get_text("blocks")
     reads xrefs and rects           layout blocks per page
              |                               |
              v                               v
     figures/ on disk                clean() and filter
     282 image files                 drop under 100 chars
              |                               |
              |                               v
              |                          classify()
              |                    prose, caption, reference
              |                               |
              +---------------+---------------+
                              |
                              v
                   nearest_figure_above()
              links caption to figure, writes chunks.json
```

## Run

```bash
cd src
python3 ingest.py
```

Output on the Mohs textbook (608 pages):

```
chunks written : 1491 -> data/processed/chunks.json
  by kind      : {'prose': 1194, 'caption': 245, 'reference': 52}
figures saved  : 282 -> data/processed/figures
captions linked: 220 of 245
```

Re-run whenever you change chunking settings. **`embed.py` must be re-run
after** — `retrieve.py` asserts the chunk and vector counts match.

## Inputs and outputs

| | |
|---|---|
| In | `data/raw/<book>.pdf`, path from `config.BOOKS` |
| Out | `data/processed/chunks.json` |
| Out | `data/processed/figures/*.jpeg` |

## Chunk record

```json
{
  "id": "1364973a5dbdb729",
  "book": "mohs_layers",
  "page": 188,
  "kind": "prose",
  "text": "Preservation of the subdermal plexus of vascular anastomoses...",
  "figures": [],
  "figure_paths": []
}
```

`figures` / `figure_paths` are populated only on `kind == "caption"`.

## Two paths through the file

The PDF is read twice, for different things.

**Figure path.** `extract_figures()` walks every page, pulls each embedded
image, writes it to disk, and records the image's bounding box. Returns a
`{page_number: [figure, ...]}` dict.

**Text path.** The main loop walks every page again, takes layout blocks,
runs `clean()`, drops blocks under `MIN_CHUNK_CHARS`, and labels the rest
with `classify()`.

The paths join when a block is a caption: `nearest_figure_above()` compares
the caption's y-position against the figures on that page and picks one.

## Functions

### `clean(text)`
Collapses all whitespace runs to single spaces. PDF extraction inserts hard
line breaks mid-sentence; this removes them.

### `chunk_id(book_id, page, text)`
Content-derived name for a chunk — same text always gives the same 16-char
ID. Lets a re-run tell which chunks actually changed, so unchanged
embeddings can be reused. Also gives app feedback logs something stable to
reference.

Change the chunking and the text changes, so IDs change. That is correct —
it is a different chunk. Ground truth therefore lives in `questions.json` as
pages and anchor phrases, **not** as chunk IDs.

### `classify(text)`
Labels a block `caption`, `reference`, or `prose`, so retrieval can filter on
`kind`. Captions are 16% of the corpus and were crowding out prose in early
results; tagging makes exclusion a measurable experiment rather than a guess.

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

## Known limitations

**Blocks are layout units, not semantic ones.** Chunk sizes range from 101 to
2,530 characters with no overlap, so a fact spanning a block boundary is
retrievable from neither half. Merging blocks into target-size chunks with
overlap is a planned experiment.

**25 captions link to no figure.** Video references (no image exists) and
figures printed on the facing page from their caption.

**Composite figures.** Multi-panel figures are a single image file — page 531
is one 928×1073 image covering panels (a) through (h). Relevant if
image-embedding models are added later; they downsample to 224×224.

**Heuristics are book-specific.** `classify()` and `nearest_figure_above()`
are tuned to this textbook and will likely need adjusting for a second one.

## Tests

```bash
cd ..
python3 -m tests.test_ingest
```

30 assertions covering `clean`, `chunk_id`, `classify` and
`nearest_figure_above`, plus integrity checks on the real `chunks.json`
(unique IDs, valid kinds, every linked figure file exists on disk).

Unit tests cannot tell you whether a caption-figure link is *correct* — only
that the code ran. Open a few files in `data/processed/figures/` and compare
them to their caption text. Three were verified by hand and matched.