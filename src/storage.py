"""Where session records live.

Locally: a plain file at data/eval/sessions.jsonl.

On HuggingFace Spaces: the same file, but pushed to a private Dataset after
every write. Free Spaces have no permanent disk and sleep after inactivity,
so anything written locally disappears. Syncing to a Dataset means grades
survive, and you download them from the dataset page instead of exposing a
download button to the surgeon.

Set two secrets in the Space:
    HF_TOKEN     - a write token from huggingface.co/settings/tokens
    HF_DATASET   - e.g. "gafur55/mohs-sessions"

With neither set, this falls back to local-file-only and prints a warning.

Test records: answers produced while testing the app should never reach the
surgeon's grading queue or the summary numbers. A record counts as a test if
it carries "is_test": true (the app's test-mode checkbox), or if its id is
listed in data/eval/test_records.json. Nothing is ever deleted - the records
stay exactly as written, and un-flagging one is removing its id from the list.
"""

import json
import os
import threading

from src import config

SESSIONS_PATH = config.EVAL_DIR / "sessions.jsonl"
TEST_IDS_PATH = config.EVAL_DIR / "test_records.json"

HF_TOKEN = os.environ.get("HF_TOKEN")
HF_DATASET = os.environ.get("HF_DATASET")
REMOTE_NAME = "sessions.jsonl"

# Two writes arriving at once could interleave and corrupt the file. One lock
# around every read-modify-write keeps that from happening. Enough for a
# handful of graders; a real database is the answer beyond that.
_lock = threading.Lock()

_synced_warning = False


def _remote_enabled():
    global _synced_warning
    if HF_TOKEN and HF_DATASET:
        return True
    if not _synced_warning:
        print("[storage] HF_TOKEN / HF_DATASET not set - saving locally only. "
              "On a free Space these records will be lost on restart.")
        _synced_warning = True
    return False


def _pull():
    """Fetch the remote copy on startup, so a restart does not start empty."""
    if not _remote_enabled():
        return
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            repo_id=HF_DATASET,
            filename=REMOTE_NAME,
            repo_type="dataset",
            token=HF_TOKEN,
        )
        SESSIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
        SESSIONS_PATH.write_text(open(path).read())
        print(f"[storage] pulled {len(load_records())} records from {HF_DATASET}")
    except Exception as e:
        # A missing file on first run is normal, not an error worth crashing on.
        print(f"[storage] nothing to pull ({type(e).__name__})")


def _push():
    """Upload the whole file after a write.

    Whole-file upload rather than append, because the Hub has no append. At
    tens of records that is trivial; if it ever gets slow, that is the signal
    to move to a real database.
    """
    if not _remote_enabled() or not SESSIONS_PATH.exists():
        return
    try:
        from huggingface_hub import HfApi
        HfApi(token=HF_TOKEN).upload_file(
            path_or_fileobj=str(SESSIONS_PATH),
            path_in_repo=REMOTE_NAME,
            repo_id=HF_DATASET,
            repo_type="dataset",
        )
    except Exception as e:
        # Never let a sync failure lose the answer the surgeon is looking at.
        print(f"[storage] push failed: {type(e).__name__}: {e}")


def load_records():
    if not SESSIONS_PATH.exists():
        return []
    return [json.loads(l) for l in SESSIONS_PATH.read_text().splitlines() if l.strip()]


def test_record_ids():
    """Record ids listed in data/eval/test_records.json (empty if no file)."""
    if not TEST_IDS_PATH.exists():
        return set()
    return set(json.loads(TEST_IDS_PATH.read_text()).get("ids", []))


def is_test(record, test_ids=None):
    """True for answers made while testing the app. Pass test_ids when
    checking many records, so the file is read once rather than per record."""
    if record.get("is_test"):
        return True
    ids = test_record_ids() if test_ids is None else test_ids
    return record.get("id") in ids


def real_records():
    """Every record except test ones - what Grade and Summary should see."""
    ids = test_record_ids()
    return [r for r in load_records() if not is_test(r, ids)]


def append_record(record):
    with _lock:
        SESSIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(SESSIONS_PATH, "a") as f:
            f.write(json.dumps(record) + "\n")
    _push()


def update_record(record_id, updates):
    with _lock:
        records = load_records()
        for r in records:
            if r["id"] == record_id:
                r.update(updates)
        with open(SESSIONS_PATH, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
    _push()


# Runs once when the app imports this module.
_pull()