"""Tests for src/chunking.py - pure functions, no files."""

import json

import pytest

from src import config
from src.chunking import (Unit, chunk_id, clean, overlap_tail, pack,
                          split_long, split_sentences)


def sentences(n, words=12, start=0):
    """n distinct, recognisable sentences."""
    return [f"Sentence {i} " + " ".join(f"w{i}x{j}" for j in range(words)) + "."
            for i in range(start, start + n)]


# ---------------------------------------------------------- basics
def test_clean_collapses_whitespace():
    assert clean("  nasal\n\nala   defect ") == "nasal ala defect"


def test_chunk_id_matches_the_existing_index():
    """ingest.py now uses this chunk_id, so it must reproduce every ID
    already in chunks.json - otherwise all vectors would be orphaned."""
    if not config.CHUNKS_PATH.exists():
        pytest.skip("no chunks.json")
    chunks = json.loads(config.CHUNKS_PATH.read_text())
    for c in chunks[:200]:
        assert chunk_id(c["book"], c["page"], c["text"]) == c["id"]


# ------------------------------------------------------- sentences
def test_split_sentences_keeps_abbreviations():
    text = ("The flap is designed (Fig. 3.2). It rotates. Dr. Smith et al. "
            "reported 12 cases. E.g. this one.")
    assert split_sentences(text) == [
        "The flap is designed (Fig. 3.2).",
        "It rotates.",
        "Dr. Smith et al. reported 12 cases.",
        "E.g. this one.",
    ]


def test_split_sentences_handles_digits_and_quotes():
    assert split_sentences('Use 17311. "Then" bill 17312. 3 stages done.') == [
        "Use 17311.", '"Then" bill 17312.', "3 stages done."]


def test_split_long_respects_limit_and_sentences():
    para = " ".join(sentences(10))
    pieces = split_long(para, 200)
    assert all(len(p) <= 200 for p in pieces)
    assert " ".join(pieces) == para
    assert all(p.endswith(".") for p in pieces)


def test_split_long_hard_splits_a_giant_sentence():
    giant = " ".join(["word"] * 400) + "."
    pieces = split_long(giant, 300)
    assert all(len(p) <= 300 for p in pieces)
    assert " ".join(pieces) == giant


def test_overlap_tail_is_bounded_and_never_everything():
    text = " ".join(sentences(4, words=5))
    tail = overlap_tail(text, 80)
    assert 0 < len(tail) <= 80
    assert text.endswith(tail)
    assert tail != text
    assert overlap_tail("One sentence only.", 500) == ""
    assert overlap_tail(text, 0) == ""


# ------------------------------------------------------------ pack
def test_pack_never_crosses_sections_or_locators():
    units = [Unit(s, section=("A",), locator="p. 1", page=1) for s in sentences(2)]
    units += [Unit(s, section=("B",), locator="p. 1", page=1) for s in sentences(2, start=2)]
    units += [Unit(s, section=("B",), locator="p. 2", page=2) for s in sentences(2, start=4)]
    chunks = pack(units, target_chars=10_000, overlap_chars=0, min_chars=10)
    assert [(c["section"], c["page"]) for c in chunks] == [(("A",), 1), (("B",), 1), (("B",), 2)]
    assert "Sentence 0" in chunks[0]["text"] and "Sentence 2" not in chunks[0]["text"]


def test_pack_respects_target_and_overlaps():
    units = [Unit(s, section=("S",)) for s in sentences(12)]
    chunks = pack(units, target_chars=400, overlap_chars=120, min_chars=10)
    assert len(chunks) > 2
    assert all(len(c["text"]) <= 400 for c in chunks)
    for prev, nxt in zip(chunks, chunks[1:]):
        last_sentence = split_sentences(prev["text"])[-1]
        assert nxt["text"].startswith(last_sentence)


def test_pack_covers_every_sentence():
    all_sentences = sentences(30, words=8)
    units = [Unit(" ".join(all_sentences[i:i + 3]), section=("S",))
             for i in range(0, 30, 3)]
    chunks = pack(units, target_chars=350, overlap_chars=100, min_chars=10)
    joined = " ".join(c["text"] for c in chunks)
    for s in all_sentences:
        assert s in joined


def test_pack_keeps_captions_separate():
    units = [Unit("Short prose paragraph about the nasal ala and its repair."),
             Unit("Fig. 3.2 Bilobed flap design for a nasal tip defect.", kind="caption"),
             Unit("More prose that follows the figure, about the same defect.")]
    chunks = pack(units, target_chars=1000, overlap_chars=0, min_chars=10)
    assert [c["kind"] for c in chunks] == ["prose", "caption", "prose"]


def test_pack_merges_a_short_tail_and_drops_lone_scraps():
    units = [Unit(s) for s in sentences(3, words=20)] + [Unit("Tiny end.")]
    chunks = pack(units, target_chars=260, overlap_chars=0, min_chars=40)
    assert chunks[-1]["text"].endswith("Tiny end.")
    alone = pack([Unit("Page 3")], min_chars=40)
    assert alone == []


def test_pack_does_not_lose_a_short_leading_paragraph():
    # 18 + 1 + 992 > 1000: without the ride-along rule the short paragraph
    # would become its own chunk and then be dropped by min_chars.
    long_para = "x " * 494 + "end."
    units = [Unit("CPT 17311 applies."), Unit(long_para)]
    chunks = pack(units, target_chars=1000, overlap_chars=0, min_chars=50)
    assert len("CPT 17311 applies.") + 1 + len(long_para) > 1000
    assert any("CPT 17311 applies." in c["text"] for c in chunks)
