# Nightly report — `src/report.py`

One command that checks the whole system and writes a short markdown page
saying whether it is healthy and what needs a person. It is meant to run on
its own every night, so that a problem shows up the next morning, not weeks
later when the surgeon notices bad answers.

```
python3 -m src.report
        |
        +-- 1. This run           date, code version, index ID, every setting
        +-- 2. Index and sources  what is indexed, what sources.yaml says, files line up?
        +-- 3. Search quality     the labeled questions, compared with last time
        +-- 4. Answer quality     the LLM judge on every eval question (skip with --skip-llm)
        +-- 5. Surgeon activity   new questions and grades since the last report
        +-- 6. Action items       a short to-do list built from 1-5
        |
        v
data/eval/reports/2026-10-14.md     <- read this
data/eval/reports/2026-10-14.json   <- the same numbers, for tomorrow's comparison
```

Each section is built separately. If one fails (say the Groq key expired),
that section shows its error and the rest of the report is still written.

## Run

```bash
python3 -m src.report                 # full report, including the LLM judge
python3 -m src.report --skip-llm      # free: no LLM calls, section 4 is skipped
python3 -m src.report --log           # also add tonight's search scores to results.jsonl
python3 -m src.report --out /tmp/r.md # write somewhere else (and /tmp/r.json beside it)
```

It prints the path of the report it wrote. Use `.venv/bin/python` instead
of `python3` if the virtual environment is not active.

The file is named after the **UTC** date, and a second run on the same day
gets `-2`, `-3`, ... — a report is never overwritten. In Chicago that means
a run after 7 pm (6 pm in winter) is already dated tomorrow.

"The previous report" is always the newest `.json` in `data/eval/reports/`,
even when `--out` writes tonight's report somewhere else. So a one-off run
with `--out` compares against the nightly history without becoming part of it.

## What it reads and writes

| | file | |
|---|---|---|
| reads | `data/processed/chunks.json`, `vectors.npy`, `vector_keys.json`, `manifest.json` | the index |
| reads | `sources.yaml` | the source registry |
| reads | `data/eval/questions.json` | the eval questions |
| reads | `data/eval/sessions.jsonl`, `test_records.json` | the app's records — **never written** |
| reads | `data/eval/reports/*.json` | the previous report, to compare against |
| writes | `data/eval/reports/<date>.md` and `.json` | tonight's report |
| appends | `data/eval/judge_runs.jsonl` | the judge's rows (not with `--skip-llm`) |
| appends | `data/eval/results.jsonl` | **only with `--log`**, note `"nightly report"` |

`data/eval/reports/` is gitignored: reports quote grader comments and short
answer excerpts, and the answers come from a copyrighted textbook.

## Reading the report

**Index and sources** lists what is in the index (chunks per source), the
status of every `sources.yaml` entry — enabled with its file present,
enabled but **missing** (an error: ingest will refuse to run), downloaded but
not enabled yet, or not downloaded — and whether `vectors.npy` still lines up
with `chunks.json`. If it does not, every search result is quietly wrong;
the fix is `python3 -m src.embed`.

**Search quality** scores the labeled questions exactly like
`python3 -m src.score` (see `docs/experiments.md` for what each number
means) and compares them with two baselines:

- the **previous report** — like for like, same nightly job;
- the **last logged run** in `results.jsonl` — usually your last experiment.

It flags a regression when any measure falls by 0.05 or more, or when a
question that had a right passage in its top 5 last time no longer does.
If the settings changed between the two runs, the flag says so — a drop
after you changed a setting on purpose may be expected.

**Answer quality** runs `src/judge.py`: every question in `questions.json` is
answered by the real pipeline, then a second LLM call checks the answer
against the excerpts it was given. Flagged answers (graded misleading or
harmful, citing a page that was never retrieved, or answering an
out-of-scope question) are listed with the judge's reason. The judge is not
a clinician — treat its flags as "look at this", never as a verdict.

**Surgeon activity** counts real records only. Answers made while testing
the app (the test-mode checkbox, or an id in `data/eval/test_records.json`)
are left out, the same rule the app's Grade and Summary tabs use. "New"
means the previous report did not see it: a record whose id was not in
the `sessions.jsonl` it read, or a grade it did not see on that record (a
changed grade counts as new). Each report's `.json` keeps those record ids
and `[id, grade]` pairs for the next one. Going by id rather than by time
matters because grades arrive late (next section): a grade made on Monday
and copied down on Wednesday is listed on Thursday, once. After a report
from before these ids were kept, or for a record with no id, "new" falls
back to asked (or graded) after the previous report was written. Anything
newly graded *Potentially harmful* or *Misleading* is listed with the
surgeon's comment, and new questions that are not in `questions.json` yet
are listed as candidates for the eval set.

### The surgeon's grades live on the Hub

The surgeon uses the hosted app, which saves to the private HF dataset
(`HF_DATASET`, see `src/storage.py`). The report reads the **local**
`data/eval/sessions.jsonl`, and says when that file last changed. If it is
old, download the current copy from the dataset page, or:

```bash
.venv/bin/python -c "
from huggingface_hub import hf_hub_download; import shutil, os
p = hf_hub_download(os.environ['HF_DATASET'], 'sessions.jsonl', repo_type='dataset',
                    token=os.environ['HF_TOKEN'])
shutil.copy(p, 'data/eval/sessions.jsonl')"
```

This replaces the local file with the Hub copy — the same thing the app
does when it starts — so do it only when every local record has already
been pushed.

## Run it every night (macOS)

Two things must be true at run time:

1. **The Mac is awake.** A sleeping Mac runs nothing. Keep it plugged in,
   and either leave it awake or wake it just before the run:
   `sudo pmset repeat wakeorpoweron MTWRFSU 02:55:00`
2. **The Groq key is in `.env`** at the project root (`GROQ_API_KEY=...`).
   A scheduled job does not read your shell profile; the report loads `.env`
   itself. Without the key, section 4 lists every question as an error and
   the rest of the report is unaffected.

The embedding model is read from the local HuggingFace cache, so the search
section needs no network once the model has been downloaded. (If the Mac is
often offline at night, set `HF_HUB_OFFLINE=1` for the job so the library does
not wait on the network first: an `EnvironmentVariables` dict in the plist, or
`HF_HUB_OFFLINE=1` before the command in cron.)

### Option A: launchd (recommended)

launchd is macOS's own scheduler. Unlike cron, if the Mac was asleep at
3:00 it runs the job once when the Mac wakes.

Save as `~/Library/LaunchAgents/local.mohs.nightly-report.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>local.mohs.nightly-report</string>
  <key>ProgramArguments</key>
  <array>
    <string>/Users/gafurmammadov/projects/mohs_surgery/.venv/bin/python</string>
    <string>-m</string>
    <string>src.report</string>
  </array>
  <key>WorkingDirectory</key>
  <string>/Users/gafurmammadov/projects/mohs_surgery</string>
  <key>StartCalendarInterval</key>
  <dict>
    <key>Hour</key>
    <integer>3</integer>
    <key>Minute</key>
    <integer>0</integer>
  </dict>
  <key>StandardOutPath</key>
  <string>/Users/gafurmammadov/Library/Logs/mohs_report.log</string>
  <key>StandardErrorPath</key>
  <string>/Users/gafurmammadov/Library/Logs/mohs_report.log</string>
</dict>
</plist>
```

The hour is local time. Then:

```bash
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/local.mohs.nightly-report.plist
launchctl kickstart -k gui/$(id -u)/local.mohs.nightly-report   # run once now, to test
tail ~/Library/Logs/mohs_report.log                             # last line = report path
launchctl bootout gui/$(id -u)/local.mohs.nightly-report        # stop scheduling it
```

After editing the plist, `bootout` and `bootstrap` again.

### Option B: cron

`crontab -e`, then add one line:

```
0 3 * * * cd /Users/gafurmammadov/projects/mohs_surgery && .venv/bin/python -m src.report >> /Users/gafurmammadov/Library/Logs/mohs_report.log 2>&1
```

cron skips a run the Mac slept through. If the project ever moves into
Desktop, Documents or Downloads, macOS blocks cron from reading it until
`/usr/sbin/cron` is given Full Disk Access (System Settings > Privacy &
Security).

### Adding `--log`

Add `--log` to either command to append each night's search scores to
`results.jsonl`, the experiment log. It is off in the examples above: a
line every night would bury your experiment runs, and the reports' own
`.json` files already keep the nightly history. Turn it on if you want the
nightly numbers next to the experiments; they are marked
`"note": "nightly report"`.

## Cost

Without `--skip-llm`, the judge makes **two LLM calls per eval question** —
one to answer it, one to check the answer (three if the judge's first reply
is unusable). With the 15 questions in `questions.json` that is about 30
calls a night, each a prompt of a few thousand tokens (five excerpts plus
the answer), on the models set in `config.LLM_MODEL` and `config.JUDGE_MODEL`.
Check that this fits your Groq plan's daily limits. Every other section is
free: no network, no API calls.

On nights you do not need the answer check, use `--skip-llm`.
