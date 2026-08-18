"""Retrieve, then answer with citations.

answer() is the single entry point. The app, the CLI below, and any future
grading harness all call this - one pipeline, so what the surgeon sees is
what you measure.

Run:  python3 -m src.generate "Repair of a helical rim defect"
"""

import sys
from src import config
from src.retrieve import search
import re


PROMPT = """You are assisting a Mohs surgeon with reconstruction planning.

Answer the question using ONLY the textbook excerpts below.
Cite the page number for every claim, like this: (p. 384).
If the excerpts do not contain the answer, say so plainly and stop.
Do not use outside knowledge. Do not guess.

EXCERPTS:
{context}

QUESTION: {question}

ANSWER:"""


def build_context(hits):
    parts = []
    for h in hits:
        c = h["chunk"]
        marker = " [figure caption]" if c["kind"] == "caption" else ""
        parts.append(f"[Page {c['page']}]{marker}\n{c['text']}")
    return "\n\n".join(parts)


def answer(question, k=None):
    """Returns the answer plus everything needed to audit it later.

    retrieved_ids is the field that makes grading useful: when the doctor
    marks an answer bad, you need to know whether a correct chunk was even
    in the context. That splits retrieval failures from generation failures.
    """
    hits = search(question, k=k or config.TOP_K)

    if not hits:
        return {"text": "Nothing retrieved.", "citations": [],
                "retrieved_ids": [], "retrieved": [], "figures": []}

    import ollama
    response = ollama.chat(
        model=config.LLM_MODEL,
        messages=[{"role": "user",
                   "content": PROMPT.format(context=build_context(hits),
                                            question=question)}],
    )

    text = response["message"]["content"]
    cited = sorted({int(p) for p in re.findall(r"\(p\.\s*(\d+)\)", text)})
    figures = []
    for h in hits:
        for path in h["chunk"].get("figure_paths", []):
            if path not in figures:
                figures.append(path)

    return {
        "question": question,
        "text": text,
        "citations": cited,
        "retrieved_ids": [h["chunk"]["id"] for h in hits],
        "retrieved_pages": [h["chunk"]["page"] for h in hits],
        "retrieved": hits,
        "figures": figures,
        "config": config.run_config(),
    }


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "Reconstruction options for a nasal alar defect"
    result = answer(q)
    print(result["text"])
    print("\nsources:", ", ".join(f"p{p}" for p in result["citations"]))
    if result["figures"]:
        print("figures:", *result["figures"], sep="\n  ")