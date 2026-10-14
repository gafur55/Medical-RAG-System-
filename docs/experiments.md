# Retrieval experiments

A record of what we changed, what happened, and what we decided.

Every run appends a line to `data/eval/results.jsonl` with its settings, so
you can always check which configuration produced which numbers.

---

## What the metrics mean

We ask the system 13 questions. For each one we already wrote down which
pages of the book are correct — that list lives in `questions.json` as
`gold_pages` and `gold_anchors`. A result counts as **correct** if its page
is on that list, or if its text contains one of our marker phrases.

### recall@1

Out of 13 questions, how often was the **very first** result correct?

`0.615` means 8 of 13. The other 5 times, result number one was not useful.

### recall@5

How often was there **at least one** correct result in the top 5?

`0.923` means 12 of 13. Only one question got nothing useful in its top 5.
This asks: did the surgeon see a right answer anywhere on the screen?

### recall@10

Same, but looking at the top 10.

Ours is `1.000` — every single question has a correct result somewhere in
the top 10. That sounds great, and it does tell us something useful: the
system is not failing to *find* the answer, it is failing to put it near the
top. But the number cannot go higher than 1.0, so it is now useless for
comparing experiments. We stop watching it.

### precision@5

Of the 5 results shown, how many were correct? Averaged over all questions.

`0.477` means roughly 2.4 of every 5 results are useful and 2.6 are junk.
This is the number we care most about, because those 5 chunks are what gets
fed to the language model. Junk in that list wastes space and can push the
model toward a wrong answer.

### mrr (mean reciprocal rank)

Rewards putting the correct answer **near the top**, not just somewhere.

For each question, score = 1 divided by the position of the first correct
result. First place scores 1.0, second scores 0.5, fourth scores 0.25,
nothing found scores 0. Then average across questions.

`0.746` roughly means the first correct answer typically shows up around
position 1 or 2.

### Which to trust

- **recall@10** — ignore it, it is maxed out.
- **recall@1 and recall@5** — only 13 yes/no outcomes, so one question
  changing moves the number a lot. Treat small changes as noise.
- **precision@5** — averages 65 individual slots (13 questions × 5), so it
  is the steadiest number we have.
- **mrr** — best single summary, because it notices a good answer sliding
  from position 2 to position 4, which recall ignores completely.

**Important:** 13 questions is a small test set. A change of one or two
percentage points means nothing. Only believe a result when several metrics
move the same direction at once.

---

## Baseline

Settings: bge-small-en-v1.5, query prefix on, prose + captions searchable,
top 5.

| metric | value |
|---|---|
| recall@1 | 0.615 |
| recall@5 | 0.923 |
| recall@10 | 1.000 |
| precision@5 | 0.477 |
| mrr | 0.746 |

Read: the system almost always finds a right answer, but about half of what
it shows is not useful.

**The worst question was Q1** (nasal ala reconstruction). Its top 5 were two
chapter introductions and three case reports from the combination-repair
chapter. First useful result was at position 6.

---

## Experiment A — the query prefix

**What we changed:** `USE_QUERY_PREFIX`, on vs off.

**What that is:** the embedding model was trained expecting search questions
to start with the words *"Represent this sentence for searching relevant
passages:"*. Chunks from the book never get this; only the question does.
Version 1.5 of the model reduced how much this matters, so it was worth
checking whether it still helps on a surgery textbook.

| metric | prefix ON | prefix OFF |
|---|---|---|
| recall@1 | 0.615 | 0.615 |
| recall@5 | **0.923** | 0.846 |
| precision@5 | **0.477** | 0.415 |
| mrr | **0.746** | 0.719 |

**Decision: keep the prefix on.** It won on every metric that moved.

**But be careful how much you believe it.** The recall@5 difference is one
single question (Q3) crossing the line. If that question had landed at
position 5 by luck instead of 6, the comparison would flip. The reason we
trust the result anyway is that precision@5 moved the same direction, and
precision averages many more data points.

### A mistake worth remembering

The first attempt at this experiment produced *identical* numbers for both
settings. The edit to `config.py` had never been saved, so the same
configuration ran twice.

We only caught it because every result line records its own settings, and the
"prefix off" line still said `use_query_prefix: true`. Without that logging
we would have written down "the prefix makes no difference" as a finding, and
been wrong.

**Rule: after changing a setting, check it actually changed before believing
any number.**

```bash
grep -n "USE_QUERY_PREFIX" src/config.py
```

---

## Experiment B — removing figure captions

**What we changed:** `RETRIEVE_KINDS` from `("prose", "caption")` to
`("prose",)`.

**Why:** figure captions are 16% of the chunks and score highly, because they
are full of the same words as the question. But a caption describes a
picture. We suspected they were pushing better text out of the top 5.

| metric | captions IN | prose only |
|---|---|---|
| recall@1 | **0.615** | 0.538 |
| recall@5 | 0.923 | 0.923 |
| precision@5 | 0.477 | **0.508** |
| mrr | **0.746** | 0.685 |

**Decision: keep captions.** Precision got slightly better, but ranking got
worse, and MRR is the more reliable summary here.

**What actually happened, question by question:**

- **Q10 broke badly.** It went from a correct answer at position 1 to nothing
  in the top 5 (first hit at position 8). Its best answer is the caption on
  page 440: *"Donor sites for full-thickness skin grafting in order of
  preference: upper eyelid, preauricular, postauricular, supraclavicular."*
  That caption **is** the answer. Filtering captions deleted it.
- **Q1 improved.** Position 6 to position 4, because caption noise cleared.
- **Q2 and Q11 improved** to perfect precision.

**The real lesson:** "caption" was the wrong thing to filter. Some captions
are the best answer in the book (Q10). Some non-captions are pure noise —
Q3 spends three of its five slots on page 286, which is a chapter abstract
and keyword list.

We filtered by the wrong category. The noise is **chapter front matter and
index pages**, not captions.

---

## Results so far

| run | recall@1 | recall@5 | precision@5 | mrr |
|---|---|---|---|---|
| baseline (prefix on, captions in) | **0.615** | **0.923** | 0.477 | **0.746** |
| prefix off | 0.615 | 0.846 | 0.415 | 0.719 |
| prose only | 0.538 | 0.923 | **0.508** | 0.685 |

Baseline is still the best configuration. Neither experiment beat it.

That is a normal and useful outcome. We now know two things that do *not*
work, and we spent about ten minutes finding out instead of guessing.

---

## Experiment C — retrieval knobs sweep (2026-10-13)

**What we changed:** nothing in `config.py`. Three new search settings were
added to `retrieve.search()`, all **off by default**, and compared side by
side with `python3 -m src.sweep` — every configuration in one process, the
model loaded once. These runs were *not* logged to `results.jsonl`; run
`python3 -m src.sweep --log` to add them (each line gets the note
`sweep: <name>` and the settings it actually used).

**The three settings, in plain words:**

- **Front matter excluded** (`EXCLUDE_FRONT_MATTER`): never show chapter
  abstracts, keyword lists, chapter "this chapter covers…" overviews, the
  book index, the contributor list, the copyright pages or the list of
  videos. Experiment B pointed at exactly these.
- **Max per page** (`MAX_PER_PAGE`): at most 1 (or 2) results from the same
  page, so one page cannot fill the five slots.
- **Hybrid** (`HYBRID_WEIGHT`): mix a keyword score (BM25 — rewards passages
  that contain the question's rare words, like *Karapandzic*) into the
  embedding score. The number is how much say the keyword score gets.
- **top_k** is not a new setting: it is how many results the language model
  is shown. It does not change the ranking, only where it is cut, so for those
  rows read **r@k** and **p@k** — recall and precision at that k.

### First, a safety check

Before comparing anything: with every setting at its default, does search
still return exactly what it did before this change? Yes. For all 15
questions, the top 25 chunk IDs, their order and their scores (to 6 decimal
places) are identical to a capture taken before the code changed, and all
13 `score_one()` results are identical. Passing the "off" values explicitly
(`exclude_front_matter=False, max_per_page=0, hybrid_weight=0`) gives the
same. So the app the doctor is reviewing is unaffected until someone flips
a setting.

### Results

| run | r@1 | r@5 | p@5 | mrr | k | r@k | p@k | Δmrr |
|---|---|---|---|---|---|---|---|---|
| baseline | 0.615 | 0.923 | 0.477 | 0.746 | 5 | 0.923 | 0.477 | +0.000 |
| **front matter excluded** | **0.692** | **1.000** | **0.554** | **0.810** | 5 | 1.000 | 0.554 | **+0.064** |
| max 1 per page | 0.615 | 0.923 | 0.492 | 0.756 | 5 | 0.923 | 0.492 | +0.010 |
| max 2 per page | 0.615 | 0.923 | 0.492 | 0.750 | 5 | 0.923 | 0.492 | +0.004 |
| hybrid 0.1 | 0.615 | 0.923 | 0.492 | 0.750 | 5 | 0.923 | 0.492 | +0.004 |
| hybrid 0.2 | 0.692 | 0.923 | 0.523 | 0.763 | 5 | 0.923 | 0.523 | +0.017 |
| hybrid 0.3 | 0.615 | 0.923 | 0.523 | 0.724 | 5 | 0.923 | 0.523 | -0.022 |
| hybrid 0.5 | 0.538 | 0.846 | 0.492 | 0.664 | 5 | 0.846 | 0.492 | -0.082 |
| top_k 3 | 0.615 | 0.923 | 0.477 | 0.746 | 3 | 0.846 | 0.513 | +0.000 |
| top_k 8 | 0.615 | 0.923 | 0.477 | 0.746 | 8 | 1.000 | 0.510 | +0.000 |
| front matter + max 1 | 0.692 | 1.000 | 0.523 | 0.810 | 5 | 1.000 | 0.523 | +0.064 |
| front matter + max 2 | 0.692 | 1.000 | 0.554 | 0.810 | 5 | 1.000 | 0.554 | +0.064 |
| front matter + hybrid 0.1 | 0.692 | 1.000 | 0.554 | 0.823 | 5 | 1.000 | 0.554 | +0.077 |
| front matter + hybrid 0.2 | 0.692 | 1.000 | 0.569 | 0.791 | 5 | 1.000 | 0.569 | +0.045 |
| front matter + hybrid 0.3 | 0.615 | 0.923 | 0.569 | 0.746 | 5 | 0.923 | 0.569 | +0.000 |
| front matter + max 1 + hybrid 0.2 | 0.692 | 1.000 | 0.523 | 0.791 | 5 | 1.000 | 0.523 | +0.045 |
| front matter + top_k 3 | 0.692 | 1.000 | 0.554 | 0.810 | 3 | 0.923 | 0.564 | +0.064 |
| front matter + top_k 8 | 0.692 | 1.000 | 0.554 | 0.810 | 8 | 1.000 | 0.548 | +0.064 |

Remember the scale: one question is 0.077 of recall or MRR, and one result
slot is 0.015 of precision@5 (65 slots in total).

### What the front-matter filter removes

97 of the 1491 chunks: 42 of the 64 chunks on pages 2–36 (title page,
copyright and disclaimer, foreword, acknowledgements, contributor list),
all 32 chunks of the index (pages 587–606), and 23 chunks that introduce a
chapter rather than teach (abstracts, keyword lists, "this chapter focuses
on…" overviews — one of them the chapter-11 summary on p. 534 — and the
chapter author line on p. 101). We checked every one of the 97 against the
answer key: **none** of them is a chunk the gold labels count as correct for
any question.

Two things it keeps on purpose. Chapter **key points** (the bullet lists right
after an abstract) stay, because the key points on p. 102 are a correct
answer for Q13. And sentences that merely *mention* the chapter ("beyond the
scope of this chapter", "previously discussed in this chapter") stay,
because pages 283 and 530 say that and are correct answers.

### What moved, question by question

**Front matter excluded** — five questions improved, none got worse:

- **Q1** (nasal ala, the worst question in the baseline). Result 1 was the
  nose chapter's abstract (p. 133, "Herein, the authors begin by
  presenting a review of nasal anatomy…"). With it gone, p. 530 moves from
  position 6 to 5, so Q1 finally has a correct passage in the top 5. The
  rest of its top 5 is still combination-repair case figures (pp. 511, 518) —
  Q1 is not fixed, just less broken.
- **Q2** (lower lip). Position 3 was p. 23, the e-book's list of videos
  ("Video 8.1: Repair of a large cutaneous lateral upper lip defect…"). Gone;
  all 5 results are now correct.
- **Q3** (helical rim). The ear chapter's abstract (position 1) and keyword
  list (position 3), both p. 286, are gone. First correct result moves from
  position 5 to 3, and 3 of 5 results are correct instead of 1. The p. 286
  key points list is still at position 1 (see above for why it stays).
- **Q6** (scalp). Position 1 was the scalp chapter's abstract (p. 220). This
  one has no giveaway words; it is caught only because it sits directly above
  the chapter's keyword list. First correct result moves from 3 to 2.
- **Q12** (defect depth). Position 1 was the opening of the combination
  chapter (p. 495, "The majority of this book has focused on…") and
  position 4 the p. 220 scalp abstract. With both gone the correct p. 119
  passage is now first.

The two out-of-scope questions are not scored, but the change helps them
too: Q15 (diabetes) had a contributor's name and title as its top result,
and Q14 (melanoma chemotherapy) had the perioperative chapter's abstract and
the preface in its top 6. A model asked to refuse is better off refusing
over body text than over a list of names.

**Max per page** — Q3 improves (the three p. 286 chunks collapse to one),
but Q8 gets worse: p. 378, the Karapandzic page, holds three correct
passages and a cap of 1 throws two away. Net: one result slot, i.e. noise.
On top of the front-matter filter, a cap of 2 changes nothing and a cap of 1
costs Q2 and Q8 a correct passage each.

**Hybrid** — helps exactly where expected and hurts where it should be
expected too:

- Gains: **Q8** (all 5 results become correct — "Karapandzic" is a rare
  word only the right pages contain), **Q6** (the scalp-flap figure p. 526
  moves to position 1), **Q1**, **Q3**, **Q9**.
- Losses: **Q12** drops its only correct result out of the top 5. The
  question's words are "defect", "depth", "change", "choice",
  "reconstruction". The correct p. 119 passage says "defects",
  "reconstructing", "superficial", "deep" — the keyword score does not stem
  words, so it sees only one match, while the scalp-algorithm page (p. 231)
  literally contains "choice" and "defect" and jumps to position 1. Also
  **Q4** (first correct result 2 → 4) and **Q13** (one correct result
  pushed out).
- The keyword score *likes* front matter: abstracts and keyword lists repeat
  every topic word of their chapter. For Q3 the p. 286 abstract gets the
  maximum possible hybrid score. That is why hybrid looks better combined
  with the front-matter filter than alone.
- More weight is worse: at 0.5 recall@1, recall@5 and MRR all fall below
  the baseline.

**top_k** — the ranking is the same, only the cut moves. At k = 3 the model
would miss a correct passage for Q1 and Q3 (recall 0.846); at k = 8 every
question has one (recall 1.000), and precision barely moves (0.513,
0.477 and 0.510 at k = 3, 5 and 8). With the front-matter filter, k = 3 misses only Q1.
This is a trade-off for answer quality (more context vs. more distraction),
which retrieval metrics alone cannot settle.

### How the keyword score is combined, and why

Three ways to combine the embedding score and the keyword score were tried
(not part of the sweep; same 13 questions, no front-matter filter):

| combination | weight | r@1 | r@5 | p@5 | mrr |
|---|---|---|---|---|---|
| embeddings only (baseline) | — | 0.615 | 0.923 | 0.477 | 0.746 |
| keywords only (BM25) | — | 0.385 | 0.846 | 0.400 | 0.550 |
| both scaled to 0–1, then mixed (**shipped**) | 0.1 | 0.615 | 0.923 | 0.492 | 0.750 |
| | 0.2 | 0.692 | 0.923 | 0.523 | 0.763 |
| | 0.3 | 0.615 | 0.923 | 0.523 | 0.724 |
| raw embedding score + scaled keyword score | 0.1 | 0.692 | 0.923 | 0.523 | 0.761 |
| | 0.2 | 0.538 | 0.846 | 0.508 | 0.681 |
| | 0.3 | 0.538 | 0.846 | 0.508 | 0.656 |
| reciprocal rank fusion (k = 60) | 0.1 | 0.615 | 0.923 | 0.523 | 0.727 |
| | 0.2 | 0.538 | 1.000 | 0.523 | 0.678 |
| | 0.3 | 0.538 | 1.000 | 0.538 | 0.688 |

The raw-score version at 0.1 behaves like the shipped one at 0.2 and then
collapses: embedding scores for the top results differ by a few hundredths,
so even a small keyword weight takes over. Scaling both sides makes the
weight mean what it says. Reciprocal rank fusion bought a little precision at
the cost of MRR. Keywords alone are clearly worse than embeddings alone.
See `docs/retrieve.md` for the reasoning and the reference.

### How much to believe this

- **13 questions.** A change in one question moves recall or MRR by 0.077.
  Max per page and hybrid 0.1 are one-question changes: noise. Hybrid 0.2
  is several questions going *both* ways (five up, three down): not a
  result. Only the front-matter filter moves all four metrics the same way,
  through five different questions, with no question getting worse — and
  each removed chunk is, on reading, obviously not an answer.
- **We looked at these questions while writing the filter.** Experiment B
  told us p. 286 was the problem, and the rules were checked against this
  book. They describe general things (abstracts, keyword lists, indexes,
  copyright pages) rather than specific pages, but the test is not
  independent of the design. New questions are the real check.
- **Unreviewed passages count as wrong.** The answer key came from ticking
  the baseline's top 25 per question. A setting that brings up passages
  nobody reviewed could be under-scored. Checked: the front-matter filter
  brings no unreviewed passage into any top 5; hybrid 0.2 brings one (Q12,
  p. 428, eyelid repair options — not an answer to Q12 anyway).
- Front matter + hybrid 0.1 has the best MRR (0.823), but its difference
  from the front-matter filter alone is +0.013 — one question's rank. Not a
  reason to add a second moving part.

### Recommendation (for the doctor and the project owner to decide)

1. **Turn on `EXCLUDE_FRONT_MATTER`.** The only change in this sweep that
   improves every metric, for several questions, without hurting any. It
   removes text with no clinical content, so the risk to answers is low. If
   the doctor agrees, it is a one-line change in `config.py`; then run
   `python3 -m src.score --note "front matter excluded"` and check the logged
   config says `"exclude_front_matter": true`.
2. **Leave `MAX_PER_PAGE` off.** Good pages often hold several right
   passages.
3. **Leave `HYBRID_WEIGHT` at 0 for now.** Its gains and losses cancel on the
   textbook. Revisit when guidelines and protocols with exact terms (drug
   names, codes, policy names) are indexed, and try word stemming first —
   Q12 shows the keyword score losing to "defects" vs "defect".
4. **Keep `TOP_K` at 5** until answers, not just retrieval, are graded at 3
   and 8.

None of these defaults were changed. `config.py` still describes the
behaviour the doctor is reviewing.

---
