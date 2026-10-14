# LLM judge — `src/judge.py`

Grades answers automatically, so a prompt or retrieval change gets an
answer-quality number the same day instead of waiting for the surgeon.
`score.py` says whether a right passage was *retrieved*; the judge says
whether the *answer* built on it is supported, complete, and honestly cited.

The surgeon's grades stay the ground truth. The judge is a cheap,
repeatable stand-in that has to be checked against them (see
[Calibrating against the surgeon](#calibrating-against-the-surgeon)).

```
      questions.json                     sessions.jsonl (read only)
            |                                   |
     generate.answer()                rehydrate from chunks.json
            |                          by retrieved_ids
            +----------------+------------------+
                             |
                             v
                      judge_answer()
          +------------------+-------------------+
          |                                      |
 deterministic_checks()                  judge prompt -> LLM
 no LLM: fabricated citations,           excerpts E1..Ek, question,
 relevant chunk retrieved?,              answer -> strict JSON verdict
 refusal on out-of-scope, length         (one retry if unusable)
          |                                      |
          +------------------+-------------------+
                             |
                             v
            row -> data/eval/judge_runs.jsonl (append-only)
                             |
                 summarize() / agreement()
```

## Run

```bash
python3 -m src.judge                         # answer + judge all 15 questions
python3 -m src.judge --ids 1,3 --note "numbered citations"
python3 -m src.judge --sessions              # judge the surgeon-graded answers
```

Needs `GROQ_API_KEY` in `.env`. Every run appends one row per question to
`data/eval/judge_runs.jsonl` — never overwritten, like `results.jsonl` — all
rows of one run sharing a `run_id`. `--note` says what you changed.

## What it measures

Each answer gets two kinds of output.

**`verdict`** — the judge model's JSON, on the same four-way scales the
surgeon uses in the app (codes map one-to-one to the app's labels):

| field | values |
|---|---|
| `grade` | `correct_complete`, `correct_incomplete`, `misleading`, `harmful` |
| `citation_check` | `all_supported`, `some_unsupported`, `uncited_claims`, `no_citations` |
| `refused` | the answer declines to answer |
| `answerable` | the excerpts contain enough to answer |
| `unsupported_claims` | claims no excerpt supports |
| `omissions` | relevant points in the excerpts the answer left out |
| `rationale` | at most 3 sentences |

The judge sees **exactly the excerpts the answering model saw**, rebuilt from
`result["retrieved"]` and numbered E1..Ek in retrieval order (so `[2]` in a
numbered-style answer is E2, and `(p. 145)` is the excerpt from page 145).
It is told the question and the answer — not the question type or the gold
pages, so an eval question and a surgeon's own question are judged alike.

**`checks`** — deterministic, no LLM, so they cannot be argued with:

| field | meaning |
|---|---|
| `fabricated_pages` | cited pages that were never retrieved |
| `fabricated_excerpts` | cited excerpt numbers past the last excerpt, e.g. `[7]` with 5 retrieved |
| `retrieval_hit` | a relevant chunk was in the context (`score.is_relevant`); `None` with no gold labels |
| `relevant_ranks` | which excerpts were relevant |
| `refused` | a refusal-phrase heuristic (below) |
| `refusal_correct` | out-of-scope questions only: did it refuse? |
| `refused_despite_relevant` | an in-scope question was refused although a relevant chunk was retrieved |
| `answer_words`, `answer_chars` | length |

Every row also carries the answer, its citations, `retrieved_ids` /
`retrieved_pages` / `retrieved_sources`, the answer's `config` (model,
prompt version, retrieval settings), `judge_model` and
`judge_prompt_version` — enough to trace any number back to what produced it.

## Reading the summary

```
judged 14 of 15 answers (1 errors) | judge openai/gpt-oss-120b | prompt 82f18967

  grade                         <- the judge's grades
    correct_complete     3
    correct_incomplete   8
    misleading           1
    harmful              2
  citations                     <- the judge's citation checks
    ...

  relevant chunk in context:  11/12 (0.92)    <- = recall@5 from score.py
  out-of-scope refused:       2/2 (1.00)
  fabricated citations:       0
  not fully correct:          1 search never found a relevant chunk,
                              11 had one and still fell short, 0 unlabeled
  failed (no verdict): Q2

  flagged for a human look:
    Q3: misleading
    Q4: harmful
    Q6: harmful
    Q13: refused_despite_relevant
```

(The first full run, 2026-10-14. Q2 failed on a Groq rate limit before the
answer call had backoff; it now has.)

- **Read the flagged list first.** A question is flagged for a `misleading`
  or `harmful` grade, a fabricated citation, an out-of-scope question that
  was answered instead of refused, or an in-scope question that was refused
  although a relevant chunk was in the context. Open its row and read the
  answer yourself — the flag is a pointer, not a verdict.
- **`refused_despite_relevant` catches what the judge cannot.** On the first
  run Q13 ("danger zones when undermining the lateral face") was refused and
  the judge agreed the excerpts did not answer it — but the gold labels put
  relevant chunks at ranks 1, 2, 3 and 5. Only the excerpts reach the
  judge, so it is the gold labels that catch this.
- **Fabricated citations should be 0.** They are checked mechanically; any
  non-zero count is a real bug in the answer, whatever the grade says.
- **"not fully correct" is the app's triage.** No relevant chunk retrieved ->
  fix search (`docs/retrieve.md`). A relevant chunk was there and the answer
  still fell short (any grade below `correct_complete`, or a
  `refused_despite_relevant`) -> fix the prompt or the model.
- **"failed (no verdict)"** lists questions with an error row — the answer
  call or the judge failed (rate limit after all retries, unusable judge
  JSON twice). Their deterministic checks still count where they ran;
  re-run them with `--ids`.
- **Compare runs, not absolute numbers.** Fifteen questions, two of them
  out of scope: one answer changing grade moves a count by one. Believe a
  change when it repeats across runs and the flagged answers read worse or
  better to you.
- `judge and refusal detector disagree on: ...` lists answers where the
  judge's `refused` differs from the heuristic — usually a partial answer.
  Read them; one is wrong.

## What it cannot do

- **It is not a clinical authority.** It checks the answer against the
  excerpts, nothing more. An answer that faithfully repeats a wrong or
  outdated passage passes; an answer that is clinically right but goes
  beyond the excerpts is marked unsupported. The prompt tells it to flag
  anything that could lead a surgeon to a bad decision, but it only knows
  what the excerpts say.
- **Self-preference.** `config.JUDGE_MODEL` defaults to `config.LLM_MODEL`,
  so by default the same model grades its own answers, and models tend to
  rate their own phrasing kindly. Set `JUDGE_MODEL` to a different model
  family when you can, and re-calibrate when you do.
- **It is noisy at the margin.** Temperature 0 makes it repeatable, not
  right. `correct_incomplete` vs `correct_complete` and `misleading` vs
  `harmful` are judgment calls a surgeon would also argue about.
- **The deterministic checks are heuristics too.**
  - The refusal detector looks for phrases like "the excerpts do not
    contain" in the opening 250 characters, or anywhere in an answer that
    cites nothing. Every phrase must name the material ("excerpts",
    "provided text", "the context" but not "in the context of") or answering
    itself ("I cannot answer"), so clinical wording such as "the extent cannot
    be determined clinically" or "the textbook notes a graft lacks support"
    is not a refusal. It errs that way on purpose: a false refusal on an
    out-of-scope question would hide a `failed_refusal`. A refusal worded
    some other way is missed, and shows up under "judge and refusal detector
    disagree".
  - A bracketed number past the last excerpt counts as fabricated — including
    a reference number copied out of the textbook ("[12]") or a year
    ("[2019]"). Page style forbids square brackets, so in page style any
    such number is a format slip worth seeing anyway.
  - With several paginated sources indexed, a page number alone does not
    say which document: `(p. 12)` is "not fabricated" if *any* retrieved
    chunk is on a page 12. Use `CITATION_STYLE = "numbered"` then.

## Calibrating against the surgeon

Before trusting the judge for a decision, check it agrees with the surgeon:

```bash
python3 -m src.judge --sessions
```

This reads `data/eval/sessions.jsonl` **read-only** (directly, not through
`src/storage.py`, which would pull the Hub copy over the local file), keeps
the records the surgeon graded that are not test records (the same rule as
the app: `is_test`, or listed in `data/eval/test_records.json`), rebuilds
each answer's excerpts from `chunks.json` by `retrieved_ids` — skipping
records whose chunks are gone after a re-index — judges them, and prints:

- **same grade** — exact match on the four-way grade.
- **same act / don't act** — both said "could act on it" (`correct_*`) or
  both said "could not" (`misleading` / `harmful`). The number that matters.
- **confusion** — rows are the surgeon's grade, columns the judge's.
- **judge passed, surgeon did not** — answers the surgeon called misleading
  or harmful that the judge passed. Every entry is a reason not to rely on
  the judge alone; read each one next to the surgeon's comment and, if the
  judge prompt is missing something, change it and re-run.

The rows are logged with `"mode": "sessions"`, `record_id`, and the
surgeon's `doctor_grade`, so agreement can be recomputed later with
`agreement(rows, records)`.

Right now every record in `sessions.jsonl` is listed in `test_records.json`,
so `--sessions` reports 0 graded records — expected until the surgeon grades
real answers. Rough guide once there are some: agreement measured on fewer
than ~20 answers is anecdote; you want "same act / don't act" well above
0.8, and an empty "judge passed, surgeon did not" list, before letting the
judge decide anything on its own.

Re-calibrate whenever `JUDGE_PROMPT_VERSION` or `JUDGE_MODEL` changes —
both are stamped on every row, so old and new verdicts never get mixed up.

## Cost

Two LLM calls per eval question — the answer and its verdict — and a third
when the judge's first reply is unusable. A full run is about 30 calls.
`--sessions` makes one call per record (the answers already exist).

On a 429 (rate limit) or 503, both the answer call and the judge call wait
2, 4, 8, 16 s — or the server's `retry-after` if longer — for up to 5 tries,
then the question gets an error row and the run moves on. One question
failing never stops a run. On Groq's free tier (8,000 tokens per minute for
`gpt-oss-120b`) a question costs roughly 6,000 tokens across its two calls,
so a full run takes a few minutes and some waiting is normal.

## Functions

- `judge_answer(result, question_record=None, llm=None)` — one
  `generate.answer()` result -> `{verdict, checks, error, attempts,
  raw_reply, judge_model, judge_prompt_version}`. Never raises: a failure
  comes back as `verdict: None` plus `error`, with the judge's last reply in
  `raw_reply`.
- `deterministic_checks(result, question_record=None)` — the `checks` dict.
- `parse_verdict(text)` — the first valid JSON verdict in a reply. Survives
  code fences, prose around the JSON, `<think>` blocks and wrapper objects;
  enum values are normalized ("Correct-Incomplete" -> `correct_incomplete`)
  and anything still invalid raises `VerdictError`, whose message is sent
  back to the judge on the retry.
- `groq_judge_llm(prompt)` — default judge: `config.JUDGE_MODEL`,
  temperature 0, JSON mode (dropped automatically if the API rejects it),
  rate-limit backoff. Its own Groq client, created on first use.
- `run_eval(questions, answer_fn=None, judge_llm=None, ids=None, run_id=None,
  timestamp=None, note="")` — rows for a run. `answer_fn` defaults to
  `generate.answer`; tests pass fakes for both models.
- `summarize(rows)` / `format_summary(summary)` — the summary above.
- `agreement(judged, doctor_records)` — calibration numbers; pairs rows by
  `record_id`, or aligned lists by position. Pure.

Importing `src.judge` loads no model, needs no key and touches no network:
`generate`, `score` and `groq` are imported inside the functions that use
them.
