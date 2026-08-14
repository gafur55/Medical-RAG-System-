"""PDF -> chunks.json + extracted figures."""

import hashlib
import json
import pymupdf
import re


from config import (
    BOOKS,
    CHUNKS_PATH,
    FIGURES_DIR,
    MIN_CHUNK_CHARS,
    MIN_FIGURE_PIXELS,
    PROCESSED_DIR,
)


CAPTION_RE = re.compile(r"^(Fig\.?|Figure|Table|Video)\b", re.IGNORECASE)
REFERENCE_RE = re.compile(r"\[\d+\]\s+[A-Z][a-z]+")


def clean(text):
    return " ".join(text.split())



def chunk_id(book_id, page, text):
    """Content-derived name for a chunk: same text -> same ID, always.

    Lets a re-run of ingest.py tell which chunks actually changed, so
    embeddings for unchanged ones can be reused instead of recomputed.
    Also gives app feedback logs something stable to point at.

    Note: change the chunking (block merging, overlap, char limits) and the
    text changes, so IDs change too. That is correct - it IS a different
    chunk. Ground truth therefore lives in questions.json as pages and
    anchor phrases, not as chunk IDs.
    """
    raw = f"{book_id}|{page}|{text}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def classify(text):
    if CAPTION_RE.match(text):
        return "caption"
    if text.startswith("References") or REFERENCE_RE.search(text):
        return "reference"
    return "prose"



def extract_figures(doc, book_id):
    """Save every embedded image in the PDF to disk, keeping its position.

    PyMuPDF gives images in two steps: get_images() returns xrefs (the
    image's ID inside the PDF), extract_image() then fetches the actual
    bytes. Images are written in their original format, not re-encoded.

    get_image_rects() gives where the image was placed on the page. That is
    the whole reason y0/y1 are stored: the PDF says nothing about which
    caption belongs to which figure, so nearest_figure_above() has to infer
    it from geometry.

    Returns {page_number: [figure, ...]} rather than a flat list because
    caption matching only ever compares figures on the same page.

    Skips: images that fail to extract (broken/unsupported), images smaller
    than MIN_FIGURE_PIXELS (logos, rules), and images referenced but never
    placed on a page. Existing files are not rewritten, so re-runs are cheap.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    by_page = {}

    for page_number, page in enumerate(doc, start=1):
        for xref, *_ in page.get_images(full=True):
            try:
                data = doc.extract_image(xref)
            except Exception:
                continue

            if data["width"] < MIN_FIGURE_PIXELS or data["height"] < MIN_FIGURE_PIXELS:
                continue

            rects = page.get_image_rects(xref)
            if not rects:
                continue
            rect = rects[0]

            name = f"{book_id}_p{page_number:04d}_x{xref}.{data['ext']}"
            path = FIGURES_DIR / name
            if not path.exists():
                path.write_bytes(data["image"])

            by_page.setdefault(page_number, []).append({
                "figure_id": f"{book_id}_p{page_number}_x{xref}",
                "path": str(path.relative_to(PROCESSED_DIR.parent.parent)),
                "page": page_number,
                "width": data["width"],
                "height": data["height"],
                "y0": float(rect.y0),   # top edge on the page, in points
                "y1": float(rect.y1),   # bottom edge
            })

    return by_page



def nearest_figure_above(figures_on_page, caption_top):
    """Guess which figure a caption belongs to, from page geometry.

    The PDF stores images and text as unrelated objects - nothing in the
    file says they go together. Textbooks print the figure above its
    caption, so: of the figures whose bottom edge (y1) is at or above the
    caption's top edge, take the closest one. y increases downward here, so
    "above" means a smaller y. The +5 is slack for rounding.

    Fallback when nothing sits above (caption at the top of a column, or a
    side-by-side layout): whatever is vertically nearest. That branch is a
    real guess and the most likely source of wrong links.

    Heuristic, tuned to this book. Linked 220 of 245 captions - spot-check
    some image files against their caption text before trusting it.
    """

    if not figures_on_page:
        return None

    above = [f for f in figures_on_page if f["y1"] <= caption_top + 5]
    if above:
        return min(above, key=lambda f: caption_top - f["y1"])
    return min(figures_on_page, key=lambda f: abs(f["y0"] - caption_top))



def build_chunks(book_id="mohs_layers"):
    doc = pymupdf.open(BOOKS[book_id])
    figures_by_page = extract_figures(doc, book_id)

    chunks = []
    for page_number, page in enumerate(doc, start=1):
        figures_here = figures_by_page.get(page_number, [])

        for block in page.get_text("blocks"):
            # block = (x0, y0, x1, y1, text, block_no, block_type)
            text = clean(block[4])
            if len(text) < MIN_CHUNK_CHARS:
                continue

            kind = classify(text)
            record = {
                "id": chunk_id(book_id, page_number, text),
                "book": book_id,
                "page": page_number,
                "kind": kind,
                "text": text,
                "figures": [],
            }

            if kind == "caption":
                match = nearest_figure_above(figures_here, caption_top=float(block[1]))
                if match:
                    record["figures"] = [match["figure_id"]]
                    record["figure_paths"] = [match["path"]]

            chunks.append(record)

    doc.close()
    return chunks, figures_by_page


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    chunks, figures_by_page = build_chunks()

    ids = [c["id"] for c in chunks]
    assert len(ids) == len(set(ids)), "duplicate chunk IDs - identical text on same page"

    CHUNKS_PATH.write_text(json.dumps(chunks, indent=1))

    total_figures = sum(len(v) for v in figures_by_page.values())
    linked = sum(1 for c in chunks if c["figures"])
    by_kind = {}
    for c in chunks:
        by_kind[c["kind"]] = by_kind.get(c["kind"], 0) + 1

    print(f"chunks written : {len(chunks)} -> {CHUNKS_PATH}")
    print(f"  by kind      : {by_kind}")
    print(f"figures saved  : {total_figures} -> {FIGURES_DIR}")
    print(f"captions linked: {linked} of {by_kind.get('caption', 0)}")


if __name__ == "__main__":
    main()
