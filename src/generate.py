"""Retrieve, then answer with citations.

answer() is the single entry point. The app, the CLI below, and any future
grading harness all call this - one pipeline, so what the surgeon sees is
what you measure.

The LLM is a plain function, prompt in -> text out. groq_llm() is the
default; tests and dry runs pass their own, so nothing here needs an API key
or a network connection until a real answer is wanted.

Two citation styles (config.CITATION_STYLE, or style= per call):

  page      The original. Excerpts are headed [Page 145] and the model cites
            (p. 145). PROMPT and its context format are kept byte-for-byte,
            so with only the textbook indexed every answer is produced
            exactly as before and stays comparable with the ones already
            graded.
  numbered  Excerpts are headed "[1] Mohs textbook, p. 145" and the model
            cites [1]. Needed once several sources are indexed: "p. 12" no
            longer says which document, and HTML sources have no pages.

Page style over a multi-source index is ambiguous by construction. It still
works - excerpts from other sources get their label in the header, so the
model is not told a policy-manual page is a textbook page - but a citation
like (p. 12) can match pages of two documents, and an excerpt with no page
cannot be cited at all. answer() prints a one-time warning when that
happens; switch to CITATION_STYLE = "numbered".

Run:  python3 -m src.generate "Repair of a helical rim defect"
"""

import hashlib
import os
import re
import sys

from dotenv import load_dotenv

from src import config, retrieve
from src.retrieve import search

load_dotenv()


PROMPT = """You are assisting a Mohs surgeon with reconstruction planning.

Answer the question using ONLY the textbook excerpts below.
Cite the page number for every claim, like this: (p. 384).
If the excerpts do not contain the answer, say so plainly and stop.
Do not use outside knowledge. Do not guess.
Cite pages using exactly this format: (p. 384)
Do not use square brackets. Do not write the word "Page".
Every sentence containing a clinical claim must end with a citation.

EXCERPTS:
{context}

QUESTION: {question}

ANSWER:"""

# For a multi-source index. Neutral between clinical and operations
# questions, because the index now holds billing and policy documents too.
# The authority line matters: a 2026 Medicare manual and a textbook can both
# be "right" about different things, and the reader has to know which one
# said what.
NUMBERED_PROMPT = """You are assisting a Mohs surgeon and clinic staff. Questions may be clinical (diagnosis, surgery, reconstruction) or about clinic operations (coding, billing, coverage, policy).

Answer the question using ONLY the numbered excerpts below.
Each excerpt is headed by its number and the document it comes from.
The excerpts come from different documents with different dates and authority - for example a Medicare policy manual and a surgical textbook.
Cite the excerpt number for every claim, like this: [2]. Cite several excerpts like this: [1][3].
Every sentence containing a clinical or policy claim must end with a citation.
If the excerpts disagree, say so and name the document behind each position.
If the excerpts do not contain the answer, say so plainly and stop.
Do not use outside knowledge. Do not guess.
Cite excerpt numbers in square brackets, not page numbers.

EXCERPTS:
{context}

QUESTION: {question}

ANSWER:"""

PROMPTS = {"page": PROMPT, "numbered": NUMBERED_PROMPT}

# Content-derived, like chunk IDs: edit the prompt and the version changes by
# itself, so a logged answer always says which prompt produced it. A
# hand-maintained "v2" would be forgotten the first time it mattered.
PROMPT_VERSION = hashlib.sha256(PROMPT.encode("utf-8")).hexdigest()[:8]

# (p. 384) is what the prompt asks for; "page 384" catches the model slipping.
CITATION_RE = re.compile(r"(?:p\.|[Pp]age)\s*(\d+)")

# Excerpt citations: [2], [1, 3], [1; 3], [1][3] (two matches), [1-3] /
# [1–3] (a range, expanded). Also 【2】 and 【2†L4-L9】 - gpt-oss models were
# trained on that browsing-citation form and slip into it. Anything else in
# the brackets ("[Page 12]", "[figure caption]") does not match.
EXCERPT_RE = re.compile(
    r"[\[【]\s*(\d+(?:\s*[,;\-–—]\s*\d+)*)\s*(?:†[^\]】]*)?[\]】]")
_RANGE_DASH = re.compile(r"\s*[-–—]\s*")

_client = None
_warned_page_style = False


def groq_llm(prompt):
    """Default LLM: one chat completion at Groq, model from config.LLM_MODEL.

    The client is created on first use rather than at import, so importing
    this module (tests, eval scripts) works without GROQ_API_KEY. The cost:
    a missing key now fails on the first question instead of at startup.
    """
    global _client
    if _client is None:
        key = os.environ.get("GROQ_API_KEY")
        if not key:
            raise RuntimeError("GROQ_API_KEY is not set - add it to .env "
                               "(locally) or to the Space secrets")
        from groq import Groq
        _client = Groq(api_key=key)

    completion = _client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return completion.choices[0].message.content


# ---------------------------------------------------------------- styles
def resolve_style(style=None):
    """None -> config.CITATION_STYLE. An unknown style fails loudly: a typo
    would otherwise quietly run a prompt nobody meant to test."""
    style = style or config.CITATION_STYLE
    if style not in PROMPTS:
        raise ValueError(f"unknown citation style {style!r} - "
                         f"use one of {sorted(PROMPTS)}")
    return style


def prompt_version(style=None):
    """Content hash of the prompt a style uses. prompt_version("page") is
    PROMPT_VERSION, so page-style answers keep the version they always had."""
    text = PROMPTS[resolve_style(style)]
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]


# ------------------------------------------------------- chunk fields
# Chunks written before the multi-source index lack source / locator; they
# are all textbook chunks. These read every chunk the same way.
def chunk_source(chunk):
    return chunk.get("source") or chunk.get("book", config.DEFAULT_SOURCE)


def chunk_locator(chunk):
    """Where in its document a chunk sits: "p. 145", or a section name."""
    if chunk.get("locator"):
        return chunk["locator"]
    page = chunk.get("page")
    return f"p. {page}" if page else ""


def describe(label, page=None, locator=""):
    """"Mohs textbook, p. 145" for a paginated source, "StatPearls: Mohs
    AUC - Indications" for one located by section."""
    if not locator:
        return label
    sep = ", " if page is not None else " - "
    return f"{label}{sep}{locator}"


def excerpt_source(chunk, labels):
    """The citation facts for one chunk, as stored in cited_sources."""
    source = chunk_source(chunk)
    return {"source": source, "label": labels.get(source, source),
            "page": chunk.get("page"), "locator": chunk_locator(chunk),
            "chunk_id": chunk.get("id")}


def format_cited(cited_sources):
    """"[1] Mohs textbook, p. 145; [3] StatPearls: Mohs AUC - Indications"."""
    parts = []
    for c in cited_sources:
        where = describe(c["label"], c.get("page"), c.get("locator", ""))
        parts.append(f"[{c['n']}] {where}" if c.get("n") is not None else where)
    return "; ".join(parts)


# --------------------------------------------------------------- context
def _marker(chunk):
    # The textbook has no "table" chunks, so the page-style context it
    # produces is unchanged by the table marker.
    return {"caption": " [figure caption]", "table": " [table]"}.get(chunk["kind"], "")


def page_header(chunk, labels):
    """[Page 145] for the textbook - exactly the original header. Any other
    source also names its document: a bare page number would pass a policy
    manual page off as a textbook page."""
    if chunk_source(chunk) == config.DEFAULT_SOURCE:
        return f"[Page {chunk['page']}]"
    label = labels.get(chunk_source(chunk), chunk_source(chunk))
    if chunk.get("page") is not None:
        return f"[Page {chunk['page']} - {label}]"
    return f"[{describe(label, None, chunk_locator(chunk))}]"


def numbered_header(n, chunk, labels):
    """[2] StatPearls: Mohs AUC - Indications"""
    s = excerpt_source(chunk, labels)
    return f"[{n}] {describe(s['label'], s['page'], s['locator'])}"


def build_context(hits, style="page", labels=None):
    """The EXCERPTS block of the prompt.

    Numbered excerpts follow retrieval order, so excerpt n is hits[n - 1]
    and a citation maps straight back to its chunk.

    labels: {source_id: label}; None reads config.source_labels(), and only
    when a header needs a label - textbook-only page style never does.
    """
    style = resolve_style(style)
    labels_needed = style == "numbered" or any(
        chunk_source(h["chunk"]) != config.DEFAULT_SOURCE for h in hits)
    if labels is None and labels_needed:
        labels = config.source_labels()

    parts = []
    for n, h in enumerate(hits, start=1):
        c = h["chunk"]
        header = (numbered_header(n, c, labels) if style == "numbered"
                  else page_header(c, labels))
        parts.append(f"{header}{_marker(c)}\n{c['text']}")
    return "\n\n".join(parts)


# -------------------------------------------------------------- citations
def extract_citations(text):
    """Sorted, de-duplicated page numbers cited in an answer."""
    return sorted({int(p) for p in CITATION_RE.findall(text)})


def extract_excerpt_numbers(text, n_hits):
    """Sorted, de-duplicated excerpt numbers cited in a numbered-style answer.

    Numbers outside 1..n_hits are dropped: they are not excerpts (a year in
    brackets, a reference number copied from the source text). A range is
    clipped to that span, so [1-1000] cannot blow up.

    Known gap: a textbook in-text reference like "[3]" copied into the
    answer is indistinguishable from a citation of excerpt 3.
    """
    found = set()
    for group in EXCERPT_RE.findall(text):
        for part in re.split(r"\s*[,;]\s*", group):
            ends = [int(x) for x in _RANGE_DASH.split(part) if x]
            lo, hi = min(ends), max(ends)
            found.update(range(max(lo, 1), min(hi, n_hits) + 1))
    return sorted(found)


def cited_excerpts(text, hits, style):
    """[(n, chunk)] for every retrieved excerpt the answer cites.

    numbered: n is the excerpt number. page: n is None, and a cited page
    matches every retrieved chunk on that page - from any source, which is
    exactly the ambiguity page style has over several documents. A cited
    page that was never retrieved has no excerpt, so it is not here (it
    still shows in "citations").
    """
    if style == "numbered":
        return [(n, hits[n - 1]["chunk"])
                for n in extract_excerpt_numbers(text, len(hits))]
    pages = set(extract_citations(text))
    return [(None, h["chunk"]) for h in hits if h["chunk"].get("page") in pages]


def cited_pages(cites):
    """Sorted page numbers of the cited excerpts that have one."""
    return sorted({c["page"] for _, c in cites if c.get("page") is not None})


# ------------------------------------------------------------------ answer
def generation_config(style=None, domains=None):
    """run_config() plus the citation style and prompt version actually used
    - stamped on every answer, so a graded answer can always be traced to
    the model and prompt behind it. domains overrides retrieve_domains when
    the caller filtered, so the record says what was really searched.

    index_id is the index this process searched (retrieve.loaded_index_id),
    not the one manifest.json names now: the app keeps the index it loaded
    at startup, so after ingest + embed in another terminal the manifest
    would credit every later answer to a build it was not retrieved from.
    Before any search has loaded an index, the manifest's ID stands."""
    style = resolve_style(style)
    cfg = {**config.run_config(), "citation_style": style,
           "prompt_version": prompt_version(style)}
    loaded = retrieve.loaded_index_id()
    if loaded is not None:
        cfg["index_id"] = loaded
    if domains is not None:
        cfg["retrieve_domains"] = list(domains)
    return cfg


def _warn_if_ambiguous(hits, style):
    """Print once per process when page style meets a non-textbook excerpt."""
    global _warned_page_style
    if style != "page" or _warned_page_style:
        return
    others = sorted({chunk_source(h["chunk"]) for h in hits} - {config.DEFAULT_SOURCE})
    if others:
        print(f"[generate] citation style 'page' with excerpts from {', '.join(others)}: "
              "a page number no longer says which document it is from, and "
              "sources without pages cannot be cited. "
              "Set CITATION_STYLE = \"numbered\" in src/config.py.")
        _warned_page_style = True


def answer(question, k=None, llm=None, domains=None, style=None):
    """Returns the answer plus everything needed to audit it later.

    retrieved_ids is the field that makes grading useful: when the doctor
    marks an answer bad, you need to know whether a correct chunk was even
    in the context. That splits retrieval failures from generation failures.
    retrieved_sources says which document each of those chunks came from,
    since a page number alone no longer does.

    llm: function prompt -> text. None means groq_llm.
    domains: e.g. ["operations"]; None means config.RETRIEVE_DOMAINS.
    style: "page" | "numbered"; None means config.CITATION_STYLE.
    """
    style = resolve_style(style)
    hits = search(question, k=k or config.TOP_K, domains=domains)
    run = generation_config(style, domains)

    if not hits:
        # Same keys as a real answer - the app reads all of them.
        return {"question": question, "text": "Nothing retrieved.",
                "citations": [], "retrieved_ids": [], "retrieved_pages": [],
                "retrieved": [], "figures": [], "cited_sources": [],
                "retrieved_sources": [], "config": run}

    _warn_if_ambiguous(hits, style)
    labels = config.source_labels()
    llm = llm or groq_llm
    text = llm(PROMPTS[style].format(context=build_context(hits, style, labels),
                                     question=question))

    figures = []
    for h in hits:
        for path in h["chunk"].get("figure_paths", []):
            if path not in figures:
                figures.append(path)

    cites = cited_excerpts(text, hits, style)
    return {
        "question": question,
        "text": text,
        "citations": extract_citations(text) if style == "page" else cited_pages(cites),
        "retrieved_ids": [h["chunk"]["id"] for h in hits],
        "retrieved_pages": [h["chunk"].get("page") for h in hits],
        "retrieved": hits,
        "figures": figures,
        "cited_sources": [{"n": n, **excerpt_source(c, labels)} for n, c in cites],
        "retrieved_sources": [chunk_source(h["chunk"]) for h in hits],
        "config": run,
    }


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "Reconstruction options for a nasal alar defect"
    result = answer(q)
    print(result["text"])
    if result["config"]["citation_style"] == "numbered":
        print("\nsources:", format_cited(result["cited_sources"]) or "none cited")
    else:
        print("\nsources:", ", ".join(f"p{p}" for p in result["citations"]))
    if result["figures"]:
        print("figures:", *result["figures"], sep="\n  ")
