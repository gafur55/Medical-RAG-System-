"""Retrieve, then answer with citations.

answer() is the single entry point. The app, the CLI below, and any future
grading harness all call this - one pipeline, so what the surgeon sees is
what you measure.

The LLM is a plain function, prompt in -> text out. groq_llm() is the
default; tests and dry runs pass their own, so nothing here needs an API key
or a network connection until a real answer is wanted.

Run:  python3 -m src.generate "Repair of a helical rim defect"
"""

import hashlib
import os
import re
import sys

from dotenv import load_dotenv

from src import config
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

# Content-derived, like chunk IDs: edit the prompt and the version changes by
# itself, so a logged answer always says which prompt produced it. A
# hand-maintained "v2" would be forgotten the first time it mattered.
PROMPT_VERSION = hashlib.sha256(PROMPT.encode("utf-8")).hexdigest()[:8]

# (p. 384) is what the prompt asks for; "page 384" catches the model slipping.
CITATION_RE = re.compile(r"(?:p\.|[Pp]age)\s*(\d+)")

_client = None


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


def build_context(hits):
    parts = []
    for h in hits:
        c = h["chunk"]
        marker = " [figure caption]" if c["kind"] == "caption" else ""
        parts.append(f"[Page {c['page']}]{marker}\n{c['text']}")
    return "\n\n".join(parts)


def extract_citations(text):
    """Sorted, de-duplicated page numbers cited in an answer."""
    return sorted({int(p) for p in CITATION_RE.findall(text)})


def generation_config():
    """run_config() plus the prompt version - stamped on every answer, so a
    graded answer can always be traced to the model and prompt behind it."""
    return {**config.run_config(), "prompt_version": PROMPT_VERSION}


def answer(question, k=None, llm=None):
    """Returns the answer plus everything needed to audit it later.

    retrieved_ids is the field that makes grading useful: when the doctor
    marks an answer bad, you need to know whether a correct chunk was even
    in the context. That splits retrieval failures from generation failures.

    llm: function prompt -> text. None means groq_llm.
    """
    hits = search(question, k=k or config.TOP_K)

    if not hits:
        # Same keys as a real answer - the app reads all of them.
        return {"question": question, "text": "Nothing retrieved.",
                "citations": [], "retrieved_ids": [], "retrieved_pages": [],
                "retrieved": [], "figures": [], "config": generation_config()}

    llm = llm or groq_llm
    text = llm(PROMPT.format(context=build_context(hits), question=question))

    figures = []
    for h in hits:
        for path in h["chunk"].get("figure_paths", []):
            if path not in figures:
                figures.append(path)

    return {
        "question": question,
        "text": text,
        "citations": extract_citations(text),
        "retrieved_ids": [h["chunk"]["id"] for h in hits],
        "retrieved_pages": [h["chunk"]["page"] for h in hits],
        "retrieved": hits,
        "figures": figures,
        "config": generation_config(),
    }


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "Reconstruction options for a nasal alar defect"
    result = answer(q)
    print(result["text"])
    print("\nsources:", ", ".join(f"p{p}" for p in result["citations"]))
    if result["figures"]:
        print("figures:", *result["figures"], sep="\n  ")
