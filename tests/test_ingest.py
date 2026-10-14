"""Tests for src/ingest.py

No pytest needed:  python -m tests.test_ingest

The four functions tested here are pure - same input, same output, no file
access - which is exactly why they were split out of the main loop.
"""

from src.ingest import chunk_id, classify, clean, nearest_figure_above

passed, failed = 0, 0


def check(label, got, want):
    global passed, failed
    if got == want:
        passed += 1
    else:
        failed += 1
        print(f"  FAIL {label}\n       got  {got!r}\n       want {want!r}")


# ----------------------------------------------------------------- clean
def test_clean():
    check("collapses runs of spaces", clean("a    b"), "a b")
    check("collapses newlines", clean("line one\nline two"), "line one line two")
    check("strips leading/trailing", clean("  padded  "), "padded")
    # PDF text often arrives with hard line breaks mid-sentence
    check("joins broken line", clean("nasal\nala defect"), "nasal ala defect")


# -------------------------------------------------------------- chunk_id
def test_chunk_id():
    a = chunk_id("book", 10, "some text")
    check("deterministic", chunk_id("book", 10, "some text"), a)
    check("16 chars", len(a), 16)
    check("text change -> new id", chunk_id("book", 10, "some text.") != a, True)
    check("page change -> new id", chunk_id("book", 11, "some text") != a, True)
    check("book change -> new id", chunk_id("other", 10, "some text") != a, True)
    # The separator exists to stop these two colliding
    check("separator prevents collision",
          chunk_id("b", 1, "23 abc") != chunk_id("b", 12, "3 abc"), True)


# -------------------------------------------------------------- classify
def test_classify():
    check("Fig. prefix", classify("Fig. 11.10 Combination reconstruction"), "caption")
    check("Fig without dot", classify("Fig 3.2 Bilobed flap design"), "caption")
    check("Figure spelled out", classify("Figure 4 shows the defect"), "caption")
    check("Table", classify("Table 2.1 Anesthetic options"), "caption")
    check("Video", classify("Video 8.1: Repair of a large defect"), "caption")
    check("lowercase fig", classify("fig. 9 the flap"), "caption")

    check("References header", classify("References [1] Burget GC"), "reference")
    # classify() recognises references by the structural "[N] Surname" marker,
    # not by journal names (see test_classify_known_gap for what that misses)
    check("numbered reference",
          classify("[12] Zitelli JA. J Am Acad Dermatol. 1983"), "reference")

    check("ordinary prose", classify("The nasal ala is a free margin"), "prose")
    # Must NOT fire just because the word appears mid-sentence
    check("figure mentioned in prose",
          classify("As shown in Fig. 3.2, the flap is designed"), "prose")


def test_classify_known_gap():
    """Documents a failure we measured, so it is visible rather than lurking.

    The journal-name whitelist was replaced by the structural pattern
    r'\\[\\d+\\]\\s+[A-Z][a-z]+', which fixed mangled names such as
    'DermatolSurg' when the entry starts with its [N] marker. A reference
    block that has lost its marker (a continuation line, or an unnumbered
    citation) still falls through to prose. Informational only - it prints
    instead of failing, so closing the gap is a visible event.
    """
    unmarked = [
        "prospective, randomized study. DermatolSurg. 2016; 42 (11):1248",
        "Zitelli JA. J Am Acad Dermatol. 1983",
    ]
    for text in unmarked:
        if classify(text) == "reference":
            print(f"  NOTE gap closed for: {text[:40]}...")
        else:
            print(f"  KNOWN GAP unmarked reference read as prose: {text[:40]}...")


# --------------------------------------------- nearest_figure_above
def fig(y0, y1, name):
    return {"y0": y0, "y1": y1, "figure_id": name}


def test_nearest_figure_above():
    check("no figures on page", nearest_figure_above([], 300.0), None)

    # y grows downward: a figure ending at 200 is ABOVE a caption starting at 300
    one = fig(50, 200, "A")
    check("single figure above", nearest_figure_above([one], 300.0)["figure_id"], "A")

    # Two figures above - the closer one wins
    far, near = fig(20, 100, "far"), fig(150, 280, "near")
    check("picks closest above",
          nearest_figure_above([far, near], 300.0)["figure_id"], "near")

    # Rounding slack: figure bottom 302 vs caption top 300 still counts
    check("5pt tolerance",
          nearest_figure_above([fig(100, 302, "grazing")], 300.0)["figure_id"], "grazing")

    # Nothing above -> fallback to vertically nearest
    below = fig(400, 500, "below")
    check("fallback when nothing above",
          nearest_figure_above([below], 300.0)["figure_id"], "below")

    # Fallback compares tops, so the figure starting nearest the caption wins
    check("fallback picks nearest top",
          nearest_figure_above([fig(400, 500, "close"), fig(700, 800, "distant")],
                               300.0)["figure_id"], "close")


# ------------------------------------------------ integrity of real output
def test_real_output():
    import json
    from src import config

    if not config.CHUNKS_PATH.exists():
        print("  SKIP no chunks.json - run: python -m src.ingest")
        return

    chunks = json.loads(config.CHUNKS_PATH.read_text())
    ids = [c["id"] for c in chunks]
    check("all IDs unique", len(ids), len(set(ids)))
    check("all kinds valid",
          set(c["kind"] for c in chunks) <= {"prose", "caption", "reference"}, True)
    check("only captions carry figures",
          all(c["kind"] == "caption" for c in chunks if c["figures"]), True)

    missing = [p for c in chunks for p in c.get("figure_paths", [])
               if not (config.ROOT / p).exists()]
    check("every linked figure file exists", missing, [])


if __name__ == "__main__":
    for fn in [test_clean, test_chunk_id, test_classify, test_classify_known_gap,
               test_nearest_figure_above, test_real_output]:
        print(fn.__name__)
        fn()
    print(f"\n{passed} passed, {failed} failed")