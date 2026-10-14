"""sources.yaml -> chunks.json + manifest.json (+ the textbook's figures).

Every enabled entry in the registry (src/sources.py) becomes chunks, in
registry order, in one chunks.json. Two paths:

  - format pdf_textbook (the Mohs textbook): the original block-based code
    below - layout blocks, classify(), figure extraction and caption
    linking. Unchanged on purpose: the same blocks give the same chunk IDs
    and the same text, so the existing vectors and every measured result
    stay valid. Records only gain source/domain/locator/section.
  - every other format: loaders.load() reduces the file to Units, and
    chunking.pack() groups them into overlapping, section-aware chunks.

manifest.json records what went in: the index ID (a hash of every chunk ID,
in order) and one entry per source with its label and license. The app
reads labels from it, so it never needs sources.yaml or PyYAML.

Run:  python3 -m src.ingest          # build the index from sources.yaml
      python3 -m src.ingest --list   # status of every registry entry
Then: python3 -m src.embed           # re-embeds only new or changed chunks
"""

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone

from src import chunking, config, embed, sources
# One definition of each, shared with chunking.py. They are the functions
# ingest.py always used, so the textbook's chunk IDs do not change.
from src.chunking import chunk_id, clean


CAPTION_RE = re.compile(r"^(Fig\.?|Figure|Table|Video)\b", re.IGNORECASE)
REFERENCE_RE = re.compile(r"\[\d+\]\s+[A-Z][a-z]+")

REBUILD = "python3 -m src.ingest && python3 -m src.embed"


def classify(text):
    if CAPTION_RE.match(text):
        return "caption"
    if text.startswith("References") or REFERENCE_RE.search(text):
        return "reference"
    return "prose"


# ================================================================ textbook
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

    Figure paths are stored relative to the project root (config.ROOT - the
    same directory PROCESSED_DIR.parent.parent always was), so the app finds
    them wherever the project is checked out.
    """
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    by_page = {}

    for page_number, page in enumerate(doc, start=1):
        for xref, *_ in page.get_images(full=True):
            try:
                data = doc.extract_image(xref)
            except Exception:
                continue

            if (data["width"] < config.MIN_FIGURE_PIXELS
                    or data["height"] < config.MIN_FIGURE_PIXELS):
                continue

            rects = page.get_image_rects(xref)
            if not rects:
                continue
            rect = rects[0]

            name = f"{book_id}_p{page_number:04d}_x{xref}.{data['ext']}"
            path = config.FIGURES_DIR / name
            if not path.exists():
                path.write_bytes(data["image"])

            by_page.setdefault(page_number, []).append({
                "figure_id": f"{book_id}_p{page_number}_x{xref}",
                "path": str(path.relative_to(config.ROOT)),
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


def build_chunks(book_id="mohs_layers", pdf_path=None, domain="clinical"):
    """The textbook path: one chunk per layout block of at least
    MIN_CHUNK_CHARS. pdf_path comes from the registry entry; config.BOOKS
    is the fallback so older callers keep working.

    pymupdf is imported here rather than at module level, so --list and the
    pure functions below work without it.
    """
    import pymupdf

    doc = pymupdf.open(pdf_path or config.BOOKS[book_id])
    figures_by_page = extract_figures(doc, book_id)

    chunks = []
    for page_number, page in enumerate(doc, start=1):
        figures_here = figures_by_page.get(page_number, [])

        for block in page.get_text("blocks"):
            # block = (x0, y0, x1, y1, text, block_no, block_type)
            text = clean(block[4])
            if len(text) < config.MIN_CHUNK_CHARS:
                continue

            kind = classify(text)
            record = {
                "id": chunk_id(book_id, page_number, text),
                "book": book_id,
                "source": book_id,
                "domain": domain,
                "page": page_number,
                "locator": f"p. {page_number}",
                "section": [],
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


# ============================================================ other formats
def load_units(source):
    """loaders.load(source) -> [chunking.Unit].

    Imported on first use: loaders needs BeautifulSoup / python-docx /
    pymupdf, which a textbook-only build or --list never touches. Tests
    replace this function instead of the loaders.
    """
    from src import loaders
    return loaders.load(source)


def _locator(piece):
    """The piece's locator, or one derived from its page or section if the
    loader left it empty: "p. N" for paginated sources, else the section."""
    if piece["locator"]:
        return piece["locator"]
    if piece["page"] is not None:
        return f"p. {piece['page']}"
    return piece["section"][-1] if piece["section"] else ""


def chunk_records(source, pieces):
    """chunking.pack() output -> chunk records for chunks.json. Pure.

    The ID hashes the page for paginated sources and the locator (a section
    name) for the rest, so the same paragraph under a different heading is
    a different chunk. `book` repeats the source ID for readers written
    before the multi-source index.
    """
    records = []
    for piece in pieces:
        page = piece["page"]
        locator = _locator(piece)
        records.append({
            "id": chunk_id(source.id, page if page is not None else locator, piece["text"]),
            "book": source.id,
            "source": source.id,
            "domain": source.domain,
            "page": page,
            "locator": locator,
            "section": list(piece["section"]),
            "kind": piece["kind"],
            "text": piece["text"],
            "figures": [],
        })
    return records


def build_source(source):
    """One registry entry -> (chunk records, figures_by_page).

    figures_by_page is only filled for the textbook; the loaders keep text
    only (figure images of public documents are often not reusable).
    """
    if source.format == "pdf_textbook":
        return build_chunks(source.id, source.path, source.domain)
    units = load_units(source)
    pieces = chunking.pack(units, config.CHUNK_TARGET_CHARS,
                           config.CHUNK_OVERLAP_CHARS, config.CHUNK_MIN_CHARS)
    return chunk_records(source, pieces), {}


# ================================================================== index
def duplicate_ids(chunks):
    """One message per chunk whose ID was already used. Pure.

    A duplicate means the same text twice at the same place in the same
    source (an ID hashes source, page/locator and text). Two rows with one
    ID would make every id-based lookup ambiguous, so ingest refuses.
    """
    first, problems = {}, []
    for c in chunks:
        other = first.setdefault(c["id"], c)
        if other is not c:
            problems.append(
                f"{c['id']}: {c.get('source', c['book'])} at "
                f"{c.get('locator') or c.get('page')!r} repeats the text "
                f"{c['text'][:60]!r}")
    return problems


def index_id(ids):
    """12 hex chars naming one build: changes when any chunk ID or the order
    changes. Logged with every answer and score run."""
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()[:12]


def kind_counts(chunks):
    """{kind: n}, in order of first appearance - the old by-kind printout."""
    counts = {}
    for c in chunks:
        counts[c["kind"]] = counts.get(c["kind"], 0) + 1
    return counts


def build_manifest(chosen, chunks, built_at):
    """manifest.json content. Pure: built_at is passed in."""
    per_source = {}
    for c in chunks:
        per_source[c["source"]] = per_source.get(c["source"], 0) + 1
    return {
        "index_id": index_id([c["id"] for c in chunks]),
        "built_at": built_at,
        "n_chunks": len(chunks),
        "sources": [{
            "id": s.id, "label": s.label, "title": s.title, "type": s.type,
            "domain": s.domain, "format": s.format, "license": s.license,
            "url": s.url, "year": s.year, "n_chunks": per_source.get(s.id, 0),
        } for s in chosen],
    }


def keep_built_at(new, old):
    """Re-use the old build time when nothing else in the manifest changed,
    so re-running ingest on unchanged sources rewrites identical files."""
    old = old or {}
    if old.get("built_at") and dict(old, built_at=None) == dict(new, built_at=None):
        return dict(new, built_at=old["built_at"])
    return new


def _read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _rel(path):
    try:
        return path.relative_to(config.ROOT)
    except ValueError:
        return path


def build(chosen):
    """Chunk every chosen source, then write chunks.json and manifest.json.

    Nothing is written until every source has loaded and the IDs are known
    to be unique, so a failing loader leaves the previous index intact.
    """
    if not chosen:
        raise SystemExit("no enabled sources in sources.yaml - nothing to index")

    chunks, report = [], []
    for source in chosen:
        records, figures_by_page = build_source(source)
        if not records:
            print(f"WARNING {source.id}: no chunks - check the file and its format")
        chunks.extend(records)
        report.append((source, records, figures_by_page))

    problems = duplicate_ids(chunks)
    if problems:
        raise ValueError("duplicate chunk IDs - identical text at the same place:\n  "
                         + "\n  ".join(problems))

    # The last moment the old chunks.json is on disk: if its vectors have no
    # key file yet, record their pairing now so embed.py can reuse them.
    embed.adopt_existing_index()

    built_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    manifest = keep_built_at(build_manifest(chosen, chunks, built_at),
                             _read_json(config.MANIFEST_PATH))

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    config.CHUNKS_PATH.write_text(json.dumps(chunks, indent=1))
    config.MANIFEST_PATH.write_text(json.dumps(manifest, indent=1))

    for source, records, figures_by_page in report:
        print(f"{source.id:<28} {len(records):>5} chunks  {kind_counts(records)}")
        if source.format == "pdf_textbook":
            total_figures = sum(len(v) for v in figures_by_page.values())
            captions = sum(1 for c in records if c["kind"] == "caption")
            linked = sum(1 for c in records if c["figures"])
            print(f"  figures saved  : {total_figures} -> {_rel(config.FIGURES_DIR)}")
            print(f"  captions linked: {linked} of {captions}")
    print(f"chunks written : {len(chunks)} -> {config.CHUNKS_PATH}")
    print(f"  by kind      : {kind_counts(chunks)}")
    print(f"manifest       : index {manifest['index_id']}, "
          f"{len(chosen)} source(s) -> {config.MANIFEST_PATH}")
    print("next           : python3 -m src.embed")
    return chunks, manifest


# ================================================================= --list
def abbreviate(text, width=36):
    """First sentence of a license, cut to width - enough to recognise it."""
    first = text.split(". ")[0].strip().rstrip(".")
    return first if len(first) <= width else first[:width - 3].rstrip() + "..."


def vector_problem(chunk_ids, key_ids, n_vectors):
    """Whether vectors.npy still matches chunks.json, as a message or None.

    key_ids are the ids in vector_keys.json (None if there is none);
    n_vectors is the row count of vectors.npy (None if there is none).
    """
    if chunk_ids is None:
        return None
    if n_vectors is None:
        return "no vectors.npy - run: python3 -m src.embed"
    if key_ids is None:
        if n_vectors != len(chunk_ids):
            return (f"vectors.npy has {n_vectors} rows for {len(chunk_ids)} chunks - "
                    "run: python3 -m src.embed")
        return ("no vector_keys.json yet (index built before the registry) - "
                f"run: {REBUILD}")
    if key_ids != chunk_ids:
        return "vectors.npy is out of date with chunks.json - run: python3 -m src.embed"
    return None


def format_status(registry, manifest, present, extra_problems=()):
    """The --list report: one row per registry entry, then the problems.

    Pure - file presence (present: {id: bool}) and the manifest are passed
    in, so it is tested without any files. "chunks" is the count in the
    current index according to manifest.json, "-" when not indexed.
    """
    manifest = manifest or {}
    indexed = {s["id"]: s.get("n_chunks", 0) for s in manifest.get("sources", [])}

    header = ("id", "on", "file", "format", "domain", "chunks", "license")
    rows = [(s.id, "yes" if s.enabled else "no", "yes" if present.get(s.id) else "no",
             s.format, s.domain, str(indexed[s.id]) if s.id in indexed else "-",
             abbreviate(s.license)) for s in registry]
    widths = [max(len(r[i]) for r in [header] + rows) for i in range(len(header) - 1)]

    def line(row):
        cells = [cell.rjust(w) if i == 5 else cell.ljust(w)
                 for i, (cell, w) in enumerate(zip(row, widths))]
        return "  ".join(cells + [row[-1]]).rstrip()

    out = []
    if manifest:
        out.append(f"index {manifest.get('index_id')}  built {manifest.get('built_at')}  "
                   f"{manifest.get('n_chunks')} chunks from {len(indexed)} source(s)")
    else:
        out.append("no manifest.json - the index predates the registry or was never built")
    out += ["", line(header)] + [line(r) for r in rows]

    problems = []
    known = {s.id for s in registry}
    for s in registry:
        if s.enabled and not present.get(s.id):
            problems.append(f"{s.id}: enabled but the file is missing: {_rel(s.path)}"
                            + (f" - get it from {s.url}" if s.url else ""))
        elif s.enabled and manifest and s.id not in indexed:
            problems.append(f"{s.id}: enabled but not in the index - run: {REBUILD}")
        elif not s.enabled and s.id in indexed:
            problems.append(f"{s.id}: disabled but still in the index - run: {REBUILD}")
    for sid in indexed:
        if sid not in known:
            problems.append(f"{sid}: in the index but not in sources.yaml - run: {REBUILD}")
    if not manifest:
        problems.append(f"no manifest.json - run: {REBUILD}")
    problems += [p for p in extra_problems if p]

    out.append("")
    if problems:
        out.append("problems:")
        out += [f"  - {p}" for p in problems]
    else:
        out.append("no problems")
    return "\n".join(out)


def status_report(registry):
    """Gather what format_status needs from disk."""
    present = {s.id: s.available for s in registry}
    chunks = _read_json(config.CHUNKS_PATH)
    keys = _read_json(config.VECTOR_KEYS_PATH)
    n_vectors = None
    if config.VECTORS_PATH.exists():
        import numpy as np
        n_vectors = len(np.load(config.VECTORS_PATH, mmap_mode="r"))
    vectors = vector_problem(
        [c["id"] for c in chunks] if chunks is not None else None,
        keys.get("ids") if keys else None, n_vectors)
    return format_status(registry, _read_json(config.MANIFEST_PATH), present, [vectors])


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build the index from sources.yaml.")
    parser.add_argument("--list", action="store_true",
                        help="show the status of every registry entry and exit")
    args = parser.parse_args(argv)

    try:
        registry = sources.load_registry()
        if args.list:
            print(status_report(registry))
            return None
        chosen = sources.enabled_sources(registry)
    except sources.RegistryError as err:
        raise SystemExit(str(err))
    return build(chosen)


if __name__ == "__main__":
    main()
