# The chat website — `web_app.py`

A ChatGPT-style page for asking questions, over the same `answer()` pipeline
the app, `score.py` and the judge use.

```bash
python3 web_app.py            # http://127.0.0.1:7870
```

| | |
|---|---|
| `app.py` | the surgeon's workbench: Ask / Grade / Summary (the HF Space) |
| `web_app.py` | this site: saved conversations, citations, sources, feedback |
| `chat_app.py` | the earlier minimal Gradio chat (still works, port 7861) |

## What it does

- **Conversations** in a sidebar, grouped by day, searchable, renamable,
  deletable. Stored in SQLite (`data/eval/chat.db`, gitignored), so a reload
  or a restart keeps everything. Deep links: `/c/<id>`.
- **Follow-ups** ("and for 2 cm?") are rewritten into a standalone question
  before searching; the rewrite is shown above the answer.
- **Citations** appear as chips. Clicking one opens the passages panel
  and highlights the passage it points to.
- **Figures** linked to cited captions show as thumbnails and open full size.
- **Feedback**: thumbs up/down, plus an optional "what was wrong" note.
  Stored in the database and copied onto the grading record as
  `chat_feedback`.
- **Grading**: every answer is also written to `sessions.jsonl` with
  `channel: "web"`, so it reaches the Grade queue like an Ask-tab answer.

## Robustness

| Situation | What happens |
|---|---|
| Message looks like patient data (SSN, phone, DOB, MRN, email, full date) | Refused before it is stored or sent anywhere. The text goes back to the box to edit. |
| Empty or over 2,000 characters | Refused with the reason |
| Groq rate limit or network blip | Retried with backoff (3 tries) |
| Answer still fails | Saved as an error message with a **Retry** button. The question is never lost. |
| Too many questions at once | `MOHS_WEB_CONCURRENCY` (default 2) run at a time. Others wait up to 90 s, then get a retryable "busy" message. |
| Unexpected server error | JSON error with a plain message, never a stack trace |
| Index out of date | The status line in the sidebar says so (`/api/health`) |
| Model output with HTML/script | Rendered with raw HTML off; strict Content-Security-Policy |

The embedding model and index load in the background at startup. The
sidebar shows "Loading the search index…" until they are ready.

## Settings (environment or `.env`)

| Variable | Default | |
|---|---|---|
| `MOHS_WEB_HOST` / `MOHS_WEB_PORT` | `127.0.0.1` / `7870` | use `0.0.0.0` to reach it from another device on your network |
| `MOHS_WEB_CONCURRENCY` | `2` | answers in flight at once |
| `MOHS_CHAT_DB_PATH` | `data/eval/chat.db` | conversations database |
| `MOHS_SESSIONS_PATH` | `data/eval/sessions.jsonl` | grading records (a redirected file never syncs to the Hub) |
| `MOHS_TEST_MODE=1` | off | records are marked `is_test`, shown as a "Test mode" badge |

## API

`GET /api/health`, `GET /api/meta`, `GET/POST/PATCH/DELETE /api/conversations[/{id}]`,
`POST /api/conversations/{id|new}/messages`, `POST /api/messages/{id}/retry`,
`POST /api/messages/{id}/feedback`.

## Files

`web_app.py` (server), `src/chatdb.py` (database), `src/chat.py` (shared
chat logic), `web/` (page, script, styles, icon). Tests:
`tests/test_web.py`, `tests/test_chatdb.py`.
