"""Turn a document's paragraphs into retrieval-sized chunks.

Every loader in loaders.py reduces its format to the same thing: a list of
Units - one paragraph, caption or table each, in reading order, tagged with
the section it sits under and a locator (page number, or section name for
sources without pages). pack() then groups units into chunks.

Rules, and why:
  - A chunk never crosses a section or a locator. A citation has to point
    at one place, and a fact should stay with its heading.
  - Captions and references are never merged with prose. They are separate
    kinds (see ingest.classify) and retrieval filters on kind.
  - Prose is packed greedily up to target_chars. A paragraph longer than
    that is split at sentence boundaries.
  - Consecutive prose chunks in a section overlap by up to overlap_chars of
    whole sentences, so a fact that straddles a boundary is retrievable from
    at least one side. This is the fix for the textbook's known limitation
    ("a fact spanning a block boundary is retrievable from neither half"),
    applied to new sources only - the textbook's chunks, IDs and vectors
    stay exactly as they are.
  - A final piece shorter than min_chars is merged into the chunk before
    it, or dropped if it is alone (stray headings, page furniture).

Pure functions only - no files - so all of it is unit-tested.
"""

import hashlib
import itertools
import re
from dataclasses import dataclass
from typing import Optional, Tuple


def clean(text):
    """Collapse every whitespace run to one space. Same rule as ingest.clean."""
    return " ".join(text.split())


def chunk_id(book_id, page, text):
    """Content-derived name for a chunk: same text -> same ID, always.

    Lets a re-run of ingest.py tell which chunks actually changed, so
    embeddings for unchanged ones can be reused instead of recomputed.
    Also gives app feedback logs something stable to point at.

    `page` is the page number for paginated sources and the locator string
    (e.g. a section name) for the rest.

    Note: change the chunking (block merging, overlap, char limits) and the
    text changes, so IDs change too. That is correct - it IS a different
    chunk. Ground truth therefore lives in questions.json as pages and
    anchor phrases, not as chunk IDs.
    """
    raw = f"{book_id}|{page}|{text}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


@dataclass(frozen=True)
class Unit:
    """One paragraph-like piece of a document, in reading order."""
    text: str
    section: Tuple[str, ...] = ()
    locator: str = ""
    page: Optional[int] = None
    kind: str = "prose"


# ------------------------------------------------------------ sentences
# A sentence ends at . ! or ? followed by whitespace and something that can
# start a sentence. Abbreviations common in clinical text are re-joined
# afterwards, because a look-behind cannot hold all of them.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9])")
_NO_SPLIT_AFTER = re.compile(
    r"(?:\b(?:e\.g|i\.e|et al|vs|cf|approx|Figs?|Tab|No|Dr|Drs|St|pp?|vol|"
    r"ed|eds|Ref|Refs|Inc|Ltd)\.|\b[A-Z]\.)$",
    re.IGNORECASE,
)


def split_sentences(text):
    """Split text into sentences, keeping abbreviations like 'Fig.' intact."""
    sentences = []
    for part in _SENTENCE_END.split(clean(text)):
        if sentences and _NO_SPLIT_AFTER.search(sentences[-1]):
            sentences[-1] += " " + part
        elif part:
            sentences.append(part)
    return sentences


def _hard_split(sentence, limit):
    """A single sentence longer than limit: split at the last space before
    the limit. Rare (tables flattened to text, run-on lists)."""
    pieces = []
    while len(sentence) > limit:
        cut = sentence.rfind(" ", 0, limit)
        if cut <= 0:
            cut = limit
        pieces.append(sentence[:cut].strip())
        sentence = sentence[cut:].strip()
    if sentence:
        pieces.append(sentence)
    return pieces


def split_long(paragraph, limit):
    """Paragraph -> pieces no longer than limit, cut at sentence boundaries."""
    pieces, current = [], ""
    for sentence in split_sentences(paragraph):
        for part in (_hard_split(sentence, limit) if len(sentence) > limit else [sentence]):
            if current and len(current) + 1 + len(part) > limit:
                pieces.append(current)
                current = part
            else:
                current = f"{current} {part}" if current else part
    if current:
        pieces.append(current)
    return pieces


def overlap_tail(text, overlap_chars):
    """The last whole sentences of text, at most overlap_chars long.

    Never the whole text: at least one sentence stays behind, otherwise a
    short chunk would be repeated in full at the start of the next one.
    """
    if overlap_chars <= 0:
        return ""
    sentences = split_sentences(text)
    tail = []
    for sentence in reversed(sentences[1:]):
        if len(" ".join([sentence] + tail)) > overlap_chars:
            break
        tail.insert(0, sentence)
    return " ".join(tail)


# ----------------------------------------------------------------- pack
def _pack_prose(paragraphs, target_chars, overlap_chars, min_chars):
    pieces = []
    for p in paragraphs:
        p = clean(p)
        if p:
            pieces.extend([p] if len(p) <= target_chars else split_long(p, target_chars))

    chunks, current = [], ""
    for piece in pieces:
        if not current:
            current = piece
        elif len(current) + 1 + len(piece) <= target_chars or len(current) < min_chars:
            # A too-short current piece rides along even past the target,
            # rather than becoming a chunk that min_chars would then drop.
            current = f"{current} {piece}"
        else:
            chunks.append(current)
            tail = overlap_tail(current, overlap_chars)
            fits = tail and len(tail) + 1 + len(piece) <= target_chars
            current = f"{tail} {piece}" if fits else piece
    if current:
        chunks.append(current)

    if len(chunks) > 1 and len(chunks[-1]) < min_chars:
        last = chunks.pop()
        chunks[-1] = f"{chunks[-1]} {last}"
    return [c for c in chunks if len(c) >= min_chars]


def pack(units, target_chars=1000, overlap_chars=150, min_chars=50):
    """Units -> chunk dicts {text, section, locator, page, kind}.

    Consecutive units with the same (section, locator, page) form a group;
    nothing crosses a group boundary. Within a group, runs of prose are
    packed with overlap; every other kind becomes a chunk of its own.
    """
    chunks = []
    for (section, locator, page), group in itertools.groupby(
            units, key=lambda u: (u.section, u.locator, u.page)):

        def emit(text, kind):
            chunks.append({"text": text, "section": section, "locator": locator,
                           "page": page, "kind": kind})

        prose = []
        for unit in group:
            if unit.kind == "prose":
                prose.append(unit.text)
                continue
            for text in _pack_prose(prose, target_chars, overlap_chars, min_chars):
                emit(text, "prose")
            prose = []
            text = clean(unit.text)
            if len(text) >= min_chars:
                emit(text, unit.kind)
        for text in _pack_prose(prose, target_chars, overlap_chars, min_chars):
            emit(text, "prose")
    return chunks
