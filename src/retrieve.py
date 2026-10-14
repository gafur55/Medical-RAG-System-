"""Search the index.

This is the ONE place retrieval happens. score.py, sweep.py, prep.py and
generate.py (and through it the app) all call search() - so an improvement
here shows up everywhere, and what you measure cannot drift from what you
ship.

The index can hold several sources (see sources.yaml). Every chunk says
which source and domain it came from; search() can filter on domain, and
the experimental knobs below (front-matter filter, per-page cap, BM25
fusion) all default to off, so the default ranking is plain cosine
similarity - exactly what it was before any of them existed.

Run:  python3 -m src.retrieve "Karapandzic flap"
"""

import hashlib
import json
import math
import re
import sys
from collections import Counter

import numpy as np
# embed.py imports only numpy and config (it loads the model lazily), so
# the app - and the HF Space - can check content keys and the vectors
# digest at load time without importing sentence-transformers or PyYAML.
from src import config, embed


_model = None
_chunks = None
_vectors = None
# Things computed from the chunks once and reused by every query (the
# front-matter mask, the BM25 statistics, the index_id of what was loaded).
# Keyed on the chunks list itself, so a reloaded - or, in tests, a faked -
# index never sees stale ones.
_derived = {}


class IndexMismatch(RuntimeError):
    """vectors.npy no longer describes chunks.json. The message has the fix."""


# ------------------------------------------------------------------ chunks
# Chunks written before the multi-source index lack source/domain/section/
# locator. These readers give every chunk the same shape (see
# docs/retrieve.md), so nothing downstream has to special-case old records.

def source_of(chunk):
    """Source id. Old textbook records only have "book"."""
    return chunk.get("source") or chunk.get("book") or config.DEFAULT_SOURCE


def domain_of(chunk):
    return chunk.get("domain") or "clinical"


def section_of(chunk):
    return list(chunk.get("section") or [])


def locator_of(chunk):
    """Human-readable place in the source: "p. 12", or a section name."""
    if chunk.get("locator"):
        return chunk["locator"]
    return f"p. {chunk['page']}" if chunk.get("page") else ""


def place_of(chunk):
    """(source, page) - or (source, locator) for sources without pages.

    The unit max_per_page counts in: one physical page, or one section of
    an unpaginated document.
    """
    page = chunk.get("page")
    return (source_of(chunk), page if page is not None else locator_of(chunk))


# ------------------------------------------------------------------- index
def get_model():
    """Load the embedding model once per process."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(config.EMBEDDING_MODEL, device="cpu")
    return _model


def read_vector_keys(path=None):
    """vector_keys.json as a dict, or None for a legacy index without one."""
    path = path or config.VECTOR_KEYS_PATH
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return None


def _first_difference(a, b):
    """Index of the first position where two lists differ (or the shorter
    one's length, when one is a prefix of the other)."""
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))


def alignment_problem(chunks, vectors, vector_keys=None):
    """Why row i of vectors may not describe chunks[i], as one sentence
    ending in the fix - or None when every row does. Reads no files.

    check_alignment() raises it; `ingest --list` prints it, so the status
    command cannot give an all-clear that the app would refuse.

    With vector_keys.json (written by embed.py) the check is exact, and the
    first failure is reported:
      model   the vectors come from config.EMBEDDING_MODEL
      rows    the key file lists one id per row of vectors.npy
      digest  vectors.npy is the file the key file was written with - not
              the result of an embed run that died between its two writes,
              or of restoring one file without the other
      ids     the chunk ids, in order: catches ingest re-run without embed
      keys    the content key of what embed.py would embed for each chunk
              now. A chunk id hashes the source, the page (or section) and
              the text - not EMBED_SECTION_HEADINGS, nor, for a paginated
              source, the heading path - while the embedded text includes
              both. Rename a heading, or flip the flag, and the ids still
              match vectors of the old text; the keys do not.
    Key files written before "keys" or "vectors_sha256" existed skip that
    step, as before. A legacy index has no keys file, so all we can compare
    is the count - which misses a re-chunk that happens to keep the same
    number of chunks, but is what we always had.
    """
    fix = "re-run: python3 -m src.embed"
    if vector_keys is None:
        if len(chunks) != len(vectors):
            return f"{len(chunks)} chunks but {len(vectors)} vectors - {fix}"
        return None

    model = vector_keys.get("embedding_model")
    if model and model != config.EMBEDDING_MODEL:
        return (f"vectors were embedded with {model} but config.EMBEDDING_MODEL "
                f"is {config.EMBEDDING_MODEL} - {fix}")
    ids = vector_keys.get("ids") or []
    if len(ids) != len(vectors):
        return (f"vector_keys.json lists {len(ids)} ids but vectors.npy has "
                f"{len(vectors)} rows - {fix}")
    digest = vector_keys.get("vectors_sha256")
    if digest is not None and digest != embed.vectors_digest(vectors):
        return ("vectors.npy is not the file vector_keys.json was written with (an embed "
                "run died between its two writes, or one was restored without the other) "
                f"- {fix}")
    chunk_ids = [c["id"] for c in chunks]
    if ids != chunk_ids:
        where = _first_difference(ids, chunk_ids)
        return (f"vectors.npy does not match chunks.json ({len(chunk_ids)} chunks, "
                f"{len(ids)} vectors, first difference at row {where}) - {fix}")
    keys = vector_keys.get("keys")
    if keys is not None:
        expected = embed.chunk_keys(chunks, model or config.EMBEDDING_MODEL)
        if keys != expected:
            return ("vectors.npy was embedded from different text than chunks.json now gives "
                    "(a section heading or config.EMBED_SECTION_HEADINGS changed?), first "
                    f"difference at row {_first_difference(keys, expected)} - {fix}")
    return None


def check_alignment(chunks, vectors, vector_keys=None):
    """Raise IndexMismatch unless row i of vectors describes chunks[i].

    If you re-run ingest.py and forget embed.py, row 847 of the vector array
    no longer describes chunk 847 and every result is quietly wrong. Loud
    failure beats silent nonsense. What is checked: alignment_problem().
    """
    problem = alignment_problem(chunks, vectors, vector_keys)
    if problem:
        raise IndexMismatch(problem)


def index_id(ids):
    """12 hex chars naming one build: changes when any chunk ID or the order
    changes. Logged with every answer and score run.

    ingest.py writes it into manifest.json. It lives here so the app can
    name the index it actually loaded (loaded_index_id) without importing
    ingest.py - and with it the registry and PyYAML.
    """
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()[:12]


def _index_id_of(chunks):
    return index_id([c["id"] for c in chunks])


def get_index():
    """Load chunks + vectors once, and check they still match. The index_id
    of what was loaded is recorded at the same moment (loaded_index_id)."""
    global _chunks, _vectors
    if _chunks is None:
        chunks = json.loads(config.CHUNKS_PATH.read_text())
        vectors = np.load(config.VECTORS_PATH)
        check_alignment(chunks, vectors, read_vector_keys())
        _chunks, _vectors = chunks, vectors
        derived(chunks, "index_id", _index_id_of)
    return _chunks, _vectors


def loaded_index_id():
    """index_id() of the chunks this process searches, or None before
    get_index() has loaded any.

    get_index() loads the index once and keeps it, while manifest.json
    (config.index_id) is re-read from disk on every call. Rebuild the index
    while the app runs - ingest + embed in another terminal - and the
    manifest names the new build while every answer still comes from the
    old one. generate.py stamps answers with this instead. Cached per
    chunks list, like the other derived() values, so it can never describe
    a different index than the one in memory.
    """
    if _chunks is None:
        return None
    return derived(_chunks, "index_id", _index_id_of)


def derived(chunks, name, build):
    """build(chunks), computed once per index and cached under name."""
    if _derived.get("chunks") is not chunks:
        _derived.clear()
        _derived["chunks"] = chunks
    if name not in _derived:
        _derived[name] = build(chunks)
    return _derived[name]


def embed_query(question, use_prefix=None):
    """Encode a question. BGE wants a prefix on queries but not on passages."""
    if use_prefix is None:
        use_prefix = config.USE_QUERY_PREFIX
    text = (config.QUERY_PREFIX + question) if use_prefix else question
    return get_model().encode(text, normalize_embeddings=True)


# ------------------------------------------------------------ front matter
# Experiment B found the noise in the top 5 is not captions but the pages
# around the content: chapter abstracts and keyword lists (Q3 spent three of
# five slots on p. 286), the table of contents, the index, the copyright
# page. They match a question's vocabulary because they summarise or list
# everything, and they answer nothing. Each rule below was checked against
# chunks.json: it flags none of the chunks the gold labels count as
# relevant (docs/experiments.md, Experiment C).

FRONT_MATTER_SECTIONS = re.compile(
    r"^(abstract|key ?words?|(table of )?contents|index|acknowledge?ments?|funding|"
    r"conflicts? of interest|disclosures?|author contributions|copyright|licen[cs]e|"
    r"dedication|foreword|preface|about the authors?)$", re.I)
ABSTRACT_RE = re.compile(r"^Abstract\b")
KEYWORDS_RE = re.compile(r"^(Key ?words?|Index terms)\s*:", re.I)
# The chapter talking about itself ("This chapter reviews ...", "Herein, the
# authors ..."). Cross-references inside the body - "beyond the scope of
# this chapter", "discussed earlier in this chapter" - deliberately do not
# match: those pages carry content (p. 283 and p. 530 are gold pages).
OVERVIEW_RE = re.compile(
    r"\b(this|the following) (chapter|book|textbook)s? (will |also )?"
    r"(focus|review|present|discuss|address|describe|cover|outline|introduce|use|is|has)\w*\b"
    r"|\b(the|this) (majority|goal|aim|purpose) of this (chapter|book|textbook)\b"
    r"|\b(herein|in this chapter),? (the authors?|we)\b"
    r"|\bafter reviewing this chapter\b", re.I)
# Book index lines: "bilobed flap - design 46 - execution 47". A term, then
# a bare page number; " - " and " -- " mark sub-entries. A number after a
# function word ("to 2 cups", "for 10 days") or before a unit is prose.
PAGE_REF_RE = re.compile(r"\d{1,3},?")
NOT_A_TERM = frozenset(
    "a an and are as at be by for from in into is of on or than the to with about "
    "approximately each every over under up mix take only least most".split())
UNIT_RE = re.compile(
    r"^(cm|mm|%|x|×|to|times|cups?|parts?|(year|day|week|month|hour|minute)s?)\b", re.I)
INDEX_DASH_RE = re.compile(r"(?:^| )-{1,3} ")
CREDENTIAL_RE = re.compile(r"\b(MD|PhD|MBBS|DO|FAAD|FACMS|FACS|FRCPC?|MBA|MBE|MPH|DDS)\b")
NAME_LIST_RE = re.compile(r"(?:[A-Z][a-z]+ (?:[A-Z]\. )*[A-Z][a-z]+(?: I{1,3}| Jr\.?)?, ){2,}")
BOILERPLATE_RE = re.compile(
    r"\bcopyright\b|\ball rights reserved\b|\bISBN\b|\btrademarks?\b|\bpublishers?\b|"
    r"\be-book\b|\btypesetting\b|\bcover design\b|[\w.]+@[\w-]+\.\w+", re.I)
THANKS_RE = re.compile(r"\bthank(s| you)\b|\bI (would also like to |want to )?(thank|acknowledge)\b",
                       re.I)
MEDIA_LIST_RE = re.compile(r"^Video \d+(\.\d+)*:")


def _capitalised_share(text):
    words = [w for w in text.split() if w[:1].isalpha()]
    return sum(w[0].isupper() for w in words) / len(words) if words else 0.0


def _index_entries(tokens):
    """How many "term 123" page references a token list holds."""
    n = 0
    for prev, tok, nxt in zip(tokens, tokens[1:], tokens[2:] + [""]):
        if (PAGE_REF_RE.fullmatch(tok) and prev[-1:] in "abcdefghijklmnopqrstuvwxyz)]"
                and prev.lower() not in NOT_A_TERM and not UNIT_RE.match(nxt)):
            n += 1
    return n


def _looks_like_index(text):
    """Dense page references, alone or with sub-entry dashes. (Dashes alone
    are not enough: a markdown list flattens to " - item - item".)"""
    tokens = text.split()
    if len(tokens) < 8:
        return False
    entries = _index_entries(tokens) / len(tokens)
    dashes = len(INDEX_DASH_RE.findall(text)) / len(tokens)
    return entries >= 0.12 or (entries >= 0.05 and dashes >= 0.05)


def _looks_like_people(text):
    """Contributor lists, title pages, a chapter's author line."""
    caps = _capitalised_share(text)
    if CREDENTIAL_RE.search(text) and caps >= 0.7:
        return True
    return len(text) < 400 and caps >= 0.8 and bool(NAME_LIST_RE.search(text))


def is_front_matter(chunk, next_chunk=None):
    """True if the chunk is packaging around the content, not content.

    Pure: looks only at the chunk (and, optionally, the chunk after it in
    chunks.json). Flags chapter abstracts and keyword lists, chapter
    overviews ("This chapter reviews ..."), table-of-contents / author
    lines, contributor lists, the book index, publisher and copyright
    boilerplate, acknowledgements, and the list of videos.

    next_chunk: an abstract often has no tell-tale words of its own (the
    p. 220 scalp chapter's does not), but it sits directly above the keyword
    list - so a chunk followed, on the same page, by a keyword list is one.
    """
    if any(FRONT_MATTER_SECTIONS.match(h.strip()) for h in section_of(chunk)):
        return True
    text = chunk["text"].strip()
    if (ABSTRACT_RE.match(text) or KEYWORDS_RE.match(text) or MEDIA_LIST_RE.match(text)
            or OVERVIEW_RE.search(text) or BOILERPLATE_RE.search(text)
            or THANKS_RE.search(text)):
        return True
    if _looks_like_index(text) or _looks_like_people(text):
        return True
    return (next_chunk is not None and place_of(next_chunk) == place_of(chunk)
            and bool(KEYWORDS_RE.match(next_chunk["text"].strip())))


def front_matter_mask(chunks):
    """is_front_matter for every chunk, in index order, as a bool array."""
    following = list(chunks[1:]) + [None]
    return np.array([is_front_matter(c, n) for c, n in zip(chunks, following)], dtype=bool)


# -------------------------------------------------------------------- BM25
# Keyword scoring, to fuse with the embeddings. Embeddings know that "lid"
# and "eyelid" are close; BM25 knows that "Karapandzic" is a rare word that
# the right passage must actually contain. A 1.5k-chunk corpus needs no
# library: postings lists in a dict, numpy for the arithmetic.

TOKEN_RE = re.compile(r"[a-z0-9]+")
# Lucene's English stop set plus question words. BM25's IDF already makes
# these nearly worthless; dropping them keeps "what/when/how" from adding
# noise to every question.
STOPWORDS = frozenset(
    "a an and are as at be but by for if in into is it no not of on or such that the "
    "their then there these they this to was will with what when where which who why "
    "how do does should must can".split())
BM25_K1 = 1.2    # term-frequency saturation (the usual 1.2-2.0)
BM25_B = 0.75    # length normalisation (the usual default)


def tokenize(text):
    """Lowercase alphanumeric words, stopwords removed."""
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


def bm25_text(chunk):
    """What BM25 indexes: the section path (if any) plus the text, the same
    material the embedding sees."""
    return " ".join(section_of(chunk) + [chunk["text"]])


def build_bm25(docs, k1=BM25_K1, b=BM25_B):
    """docs: list of token lists. Returns the statistics bm25_scores needs."""
    n = len(docs)
    lengths = np.array([len(d) for d in docs], dtype=float)
    avgdl = lengths.mean() if n and lengths.mean() > 0 else 1.0
    postings = {}
    for i, doc in enumerate(docs):
        for term, tf in Counter(doc).items():
            postings.setdefault(term, ([], []))
            postings[term][0].append(i)
            postings[term][1].append(tf)
    postings = {t: (np.array(ix), np.array(tf, dtype=float)) for t, (ix, tf) in postings.items()}
    # Lucene's IDF: always positive, so a term in most chunks still counts
    # a little rather than going negative as in the original formula.
    idf = {t: math.log(1 + (n - len(ix) + 0.5) / (len(ix) + 0.5)) for t, (ix, _) in postings.items()}
    return {"n": n, "k1": k1, "postings": postings, "idf": idf,
            "norm": k1 * (1 - b + b * lengths / avgdl)}


def bm25_scores(bm25, query_tokens):
    """BM25 score of every document for a query, as an array."""
    scores = np.zeros(bm25["n"])
    for term in set(query_tokens):
        if term not in bm25["postings"]:
            continue
        ix, tf = bm25["postings"][term]
        scores[ix] += bm25["idf"][term] * tf * (bm25["k1"] + 1) / (tf + bm25["norm"][ix])
    return scores


def minmax(values, allowed):
    """Rescale to [0, 1] over the allowed entries; the rest become 0.

    All-equal allowed values (e.g. no query word occurs anywhere) map to 0,
    so that side then adds nothing to the fused ranking.
    """
    out = np.zeros(len(values))
    if not allowed.any():
        return out
    lo, hi = values[allowed].min(), values[allowed].max()
    if hi > lo:
        out[allowed] = (values[allowed] - lo) / (hi - lo)
    return out


def fuse(cosine, keyword, allowed, weight):
    """Convex combination of min-max-normalised scores:

        (1 - w) * norm(cosine) + w * norm(bm25)

    Both sides are normalised, not only BM25: bge cosines sit in a narrow
    band (the top 10 for a question usually span ~0.05), so raw cosine plus
    a [0, 1] BM25 hands the ranking to BM25 from w = 0.2 on (Experiment C)
    and w stops meaning anything. Normalising both makes w the share of the
    ranking each side controls. Bruch et al. (2023, "An analysis of fusion
    functions for hybrid retrieval", ACM TOIS) found this convex combination
    beats reciprocal rank fusion and needs only a handful of labelled
    queries to tune w; RRF also discards how far ahead the leader is.
    """
    return (1 - weight) * minmax(cosine, allowed) + weight * minmax(keyword, allowed)


# ------------------------------------------------------------------ search
def allowed_mask(chunks, kinds, domains, exclude_front_matter):
    """Which chunks may be returned at all."""
    allowed = np.array([c["kind"] in kinds and domain_of(c) in domains for c in chunks],
                       dtype=bool)
    if exclude_front_matter and len(chunks):
        allowed &= ~derived(chunks, "front_matter", front_matter_mask)
    return allowed


def top_indices(scores, k, chunks, max_per_page=None):
    """Indices of the k best finite scores, best first.

    max_per_page: at most this many from one place_of() (page, or section of
    an unpaginated source), so one page cannot fill the context by itself.
    Greedy down the ranking - the first k picked do not depend on how many
    more you ask for, so recall@1/5/10 from one deep search stay consistent.
    """
    order = np.argsort(-scores)
    if not max_per_page:
        return [i for i in order[:k] if np.isfinite(scores[i])]
    picked, per_place = [], Counter()
    for i in order:
        if len(picked) == k or not np.isfinite(scores[i]):
            break
        place = place_of(chunks[i])
        if per_place[place] < max_per_page:
            per_place[place] += 1
            picked.append(i)
    return picked


def search(question, k=None, kinds=None, use_prefix=None, domains=None,
           exclude_front_matter=None, max_per_page=None, hybrid_weight=None):
    """Return the top k chunks for a question, best first.

    Every None reads config at call time (see docs/retrieve.md). Defaults
    reproduce plain cosine ranking exactly.

      domains               sources.yaml domains to search; chunks without
                            one count as "clinical"
      exclude_front_matter  drop is_front_matter() chunks
      max_per_page          cap per page (or per section, without pages);
                            0 or None in config = no cap
      hybrid_weight         0 = embeddings only; w in (0, 1] fuses BM25 in,
                            see fuse(). The returned score is then the
                            fused score, not a cosine.
    """
    k = k or config.TOP_K
    kinds = kinds if kinds is not None else config.RETRIEVE_KINDS
    domains = domains if domains is not None else config.RETRIEVE_DOMAINS
    if exclude_front_matter is None:
        exclude_front_matter = config.EXCLUDE_FRONT_MATTER
    if max_per_page is None:
        max_per_page = config.MAX_PER_PAGE
    if hybrid_weight is None:
        hybrid_weight = config.HYBRID_WEIGHT
    if not 0 <= hybrid_weight <= 1:
        raise ValueError(f"hybrid_weight must be in [0, 1], got {hybrid_weight}")

    chunks, vectors = get_index()
    q = embed_query(question, use_prefix=use_prefix)

    # Both sides are normalized, so the dot product IS cosine similarity.
    # A few thousand x 384 is tiny - brute force is exact and instant.
    scores = vectors @ q # matrix multiplication

    # Excluded chunks get -inf rather than being removed, so they can never
    # win a slot and indices stay aligned with chunks.
    allowed = allowed_mask(chunks, kinds, domains, exclude_front_matter)
    if hybrid_weight > 0:
        bm25 = derived(chunks, "bm25", lambda cs: build_bm25([tokenize(bm25_text(c)) for c in cs]))
        scores = fuse(scores, bm25_scores(bm25, tokenize(question)), allowed, hybrid_weight)
    scores = np.where(allowed, scores, -np.inf)

    top = top_indices(scores, k, chunks, max_per_page)
    return [
        {"rank": rank, "score": float(scores[i]), "chunk": chunks[i]}
        for rank, i in enumerate(top, start=1)
    ]


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "Karapandzic flap"
    print("QUESTION:", question, "\n")
    for hit in search(question, k=5):
        c = hit["chunk"]
        figs = f" [{len(c['figures'])} figure]" if c.get("figures") else ""
        print(f"--- {hit['rank']}. {source_of(c)} {locator_of(c)} | {c['kind']}{figs} | {hit['score']:.3f}")
        print(c["text"][:300], "\n")
