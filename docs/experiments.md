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

