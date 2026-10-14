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

MOHS_SESSIONS_PATH=... puts the records in another file (a scratch file
while testing chat_app.py, say). It is read here, at import, so it already
applies to the startup pull - and only data/eval/sessions.jsonl is ever
synced. A redirected file is never overwritten by the Hub copy, and never
uploaded over it, whatever credentials are set.

Two processes can write the file at once: app.py and chat_app.py run side
by side and both save here. Every write holds an exclusive flock on a
sidecar lock file (sessions.jsonl.lock), and whole-file rewrites go to a
temp file that replaces the original in one step, so a crash or a race
leaves the old file or the new one - never a truncated or interleaved one.

Test records: answers produced while testing the app should never reach the
surgeon's grading queue or the summary numbers. A record counts as a test if
it carries "is_test": true (the app's test-mode checkbox), or if its id is
listed in data/eval/test_records.json. Nothing is ever deleted - the records
stay exactly as written, and un-flagging one is removing its id from the list.
"""

import fcntl
import json
import os
import stat
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path

from src import config

# The one file that is the dataset's sessions.jsonl.
DEFAULT_SESSIONS_PATH = config.EVAL_DIR / "sessions.jsonl"


def sessions_path(environ=None):
    """MOHS_SESSIONS_PATH if set, else data/eval/sessions.jsonl.

    Resolved by this module, at import, because the import is also when the
    remote copy is pulled: a redirect applied by the caller afterwards
    would come too late - the pull would already have run against the
    default file."""
    override = (os.environ if environ is None else environ).get("MOHS_SESSIONS_PATH")
    return Path(override) if override else DEFAULT_SESSIONS_PATH


SESSIONS_PATH = sessions_path()
TEST_IDS_PATH = config.EVAL_DIR / "test_records.json"

HF_TOKEN = os.environ.get("HF_TOKEN")
HF_DATASET = os.environ.get("HF_DATASET")
REMOTE_NAME = "sessions.jsonl"

# Two writes arriving at once could interleave and corrupt the file. The
# thread lock covers Gradio's handler threads in this process; the flock in
# _locked() covers other processes. Enough for a handful of graders; a real
# database is the answer beyond that.
_lock = threading.Lock()

_synced_warning = False
_redirect_warning = False


def is_default_file(path):
    """True when `path` is data/eval/sessions.jsonl, the file the Hub
    holds. Compared resolved, so a relative or absolute spelling of the
    same file still counts."""
    return Path(path).resolve() == DEFAULT_SESSIONS_PATH.resolve()


def _remote_enabled():
    global _synced_warning, _redirect_warning
    if not is_default_file(SESSIONS_PATH):
        # A scratch or test file. Pulling would overwrite it with the real
        # records; pushing would replace the real records with it.
        if not _redirect_warning:
            print(f"[storage] saving to {SESSIONS_PATH}, not {DEFAULT_SESSIONS_PATH} - "
                  "Hub sync is off for this file.")
            _redirect_warning = True
        return False
    if HF_TOKEN and HF_DATASET:
        return True
    if not _synced_warning:
        print("[storage] HF_TOKEN / HF_DATASET not set - saving locally only. "
              "On a free Space these records will be lost on restart.")
        _synced_warning = True
    return False


def lock_path(path):
    """The sidecar lock file for a sessions file: sessions.jsonl.lock.

    A separate file, not the sessions file itself, because rewrites replace
    the sessions file with a new one - a lock held on the old one would
    guard nothing."""
    path = Path(path)
    return path.with_name(path.name + ".lock")


@contextmanager
def _locked():
    """Hold the thread lock and an exclusive flock on the current
    SESSIONS_PATH's lock file; yields that path.

    SESSIONS_PATH is read here, at call time, not at import - tests point
    it elsewhere - and everything inside works on the yielded path, so the
    lock and the file it guards can never disagree."""
    path = Path(SESSIONS_PATH)
    with _lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(lock_path(path), "a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)    # released when the file closes
            yield path


def _replace_contents(path, text):
    """Write `text` to a temp file beside `path`, fsync it, then os.replace
    it over `path`.

    Anyone reading - or a crash part-way - sees the old file or the new
    one, never a truncated half. Same directory, so the replace is a rename
    within one filesystem. The old file's permissions carry over (mkstemp
    creates 0600). Callers hold _locked(): an append that opened the old
    file just before the replace would otherwise be lost with it."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        if path.exists():
            os.chmod(tmp, stat.S_IMODE(path.stat().st_mode))
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


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
        text = Path(path).read_text()
        with _locked() as target:
            _replace_contents(target, text)
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
    if not _remote_enabled() or not Path(SESSIONS_PATH).exists():
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


def _read(path):
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def load_records():
    return _read(Path(SESSIONS_PATH))


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
    line = json.dumps(record) + "\n"
    with _locked() as path:
        with open(path, "a") as f:
            f.write(line)
    _push()


def update_record(record_id, updates):
    """Read, change, write back - all under one lock, so no other process's
    append can land between the read and the write and be dropped."""
    with _locked() as path:
        records = _read(path)
        for r in records:
            if r["id"] == record_id:
                r.update(updates)
        _replace_contents(path, "".join(json.dumps(r) + "\n" for r in records))
    _push()


# Runs once when the app imports this module.
_pull()
