"""Tests for session storage and the test-record filter in src/storage.py.

Every test points storage at a temporary folder and disables Hub sync, so
nothing touches data/eval/sessions.jsonl or the HF dataset. The tests that
start a second Python process give it a stubbed huggingface_hub, a
temporary EVAL_DIR and no HF credentials of its own, so they cannot reach
the network or the real file either.
"""

import json
import os
import subprocess
import sys
import time

import pytest

from src import config, storage


@pytest.fixture
def tmp_storage(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "SESSIONS_PATH", tmp_path / "sessions.jsonl")
    monkeypatch.setattr(storage, "TEST_IDS_PATH", tmp_path / "test_records.json")
    monkeypatch.setattr(storage, "_push", lambda: None)
    return tmp_path


def write_sessions(path, records):
    path.write_text("".join(json.dumps(r) + "\n" for r in records))


def test_no_list_file_means_no_test_ids(tmp_storage):
    assert storage.test_record_ids() == set()


def test_is_test_by_flag_and_by_list(tmp_storage):
    (tmp_storage / "test_records.json").write_text(json.dumps({"ids": ["b"]}))
    assert storage.is_test({"id": "a", "is_test": True})
    assert storage.is_test({"id": "b"})
    assert not storage.is_test({"id": "c"})
    assert not storage.is_test({"id": "c", "is_test": False})


def test_real_records_filters_without_touching_the_file(tmp_storage):
    records = [{"id": "a"}, {"id": "b"}, {"id": "c", "is_test": True}]
    write_sessions(tmp_storage / "sessions.jsonl", records)
    (tmp_storage / "test_records.json").write_text(json.dumps({"ids": ["b"]}))
    before = (tmp_storage / "sessions.jsonl").read_text()

    assert [r["id"] for r in storage.real_records()] == ["a"]
    assert len(storage.load_records()) == 3
    assert (tmp_storage / "sessions.jsonl").read_text() == before


def test_append_and_update_round_trip(tmp_storage):
    storage.append_record({"id": "x", "grade": None, "is_test": False})
    storage.update_record("x", {"grade": "ok"})
    assert storage.load_records() == [{"id": "x", "grade": "ok", "is_test": False}]


# ------------------------------------------------- where records go, sync
def test_sessions_path_comes_from_the_environment(tmp_path):
    assert storage.sessions_path({}) == storage.DEFAULT_SESSIONS_PATH
    assert storage.sessions_path({"MOHS_SESSIONS_PATH": ""}) == storage.DEFAULT_SESSIONS_PATH
    scratch = tmp_path / "chat.jsonl"
    assert storage.sessions_path({"MOHS_SESSIONS_PATH": str(scratch)}) == scratch


def test_only_the_default_file_syncs_with_the_hub(tmp_storage, monkeypatch):
    """A redirected file is never the dataset's sessions.jsonl: with
    credentials set it still neither pulls nor pushes."""
    monkeypatch.setattr(storage, "HF_TOKEN", "token")
    monkeypatch.setattr(storage, "HF_DATASET", "user/mohs-sessions")
    assert storage._remote_enabled() is False
    monkeypatch.setattr(storage, "SESSIONS_PATH", storage.DEFAULT_SESSIONS_PATH)
    assert storage._remote_enabled() is True


# Prepended to every second-process script: huggingface_hub's two entry
# points record their calls instead of reaching the Hub, and EVAL_DIR (so
# the default sessions file) is a temporary folder.
HUB_STUB = r"""
import json, os, sys
from pathlib import Path

import huggingface_hub

calls = []

def hf_hub_download(repo_id, filename, **kwargs):
    calls.append(["pull", filename])
    if not os.environ.get("FAKE_REMOTE"):
        raise FileNotFoundError(filename)
    return os.environ["FAKE_REMOTE"]

class HfApi:
    def __init__(self, token=None):
        pass

    def upload_file(self, path_or_fileobj, path_in_repo, **kwargs):
        calls.append(["push", str(path_or_fileobj), path_in_repo])

huggingface_hub.hf_hub_download = hf_hub_download
huggingface_hub.HfApi = HfApi

from src import config
config.EVAL_DIR = Path(os.environ["FAKE_EVAL_DIR"])
"""

STORAGE_SCRIPT = HUB_STUB + r"""
from src import storage
storage.append_record({"id": "x", "grade": None})
storage.update_record("x", {"grade": "ok"})
print(json.dumps({"path": str(storage.SESSIONS_PATH), "calls": calls}))
"""


def clean_env(**extra):
    """This environment without HF credentials or a sessions redirect,
    plus `extra` - so a child process only syncs what a test asks for."""
    env = {k: v for k, v in os.environ.items()
           if k not in ("HF_TOKEN", "HF_DATASET", "MOHS_SESSIONS_PATH", "FAKE_REMOTE")}
    env.update(GRADIO_ANALYTICS_ENABLED="False", HF_HUB_OFFLINE="1", **extra)
    return env


def run_script(code, env, *args):
    out = subprocess.run([sys.executable, "-c", code, *args], cwd=config.ROOT, env=env,
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr[-2000:]
    return out


def test_redirect_is_read_at_import_and_never_synced(tmp_path):
    """MOHS_SESSIONS_PATH takes effect before the import-time pull, and a
    redirected file is neither overwritten by nor uploaded as the dataset's
    sessions.jsonl - even with HF_TOKEN and HF_DATASET set."""
    scratch = tmp_path / "chat.jsonl"
    env = clean_env(HF_TOKEN="token", HF_DATASET="user/mohs-sessions",
                    MOHS_SESSIONS_PATH=str(scratch), FAKE_EVAL_DIR=str(tmp_path / "eval"))
    result = json.loads(run_script(STORAGE_SCRIPT, env).stdout.splitlines()[-1])

    assert result == {"path": str(scratch), "calls": []}
    assert [r["id"] for r in map(json.loads, scratch.read_text().splitlines())] == ["x"]
    assert not (tmp_path / "eval" / "sessions.jsonl").exists()


def test_default_file_still_pulls_and_pushes(tmp_path):
    """No redirect: exactly the old behaviour - the remote copy is pulled
    into the default file at import, and every write pushes it back."""
    remote = tmp_path / "remote.jsonl"
    remote.write_text(json.dumps({"id": "graded", "grade": "ok"}) + "\n")
    eval_dir = tmp_path / "eval"
    env = clean_env(HF_TOKEN="token", HF_DATASET="user/mohs-sessions",
                    FAKE_EVAL_DIR=str(eval_dir), FAKE_REMOTE=str(remote))
    result = json.loads(run_script(STORAGE_SCRIPT, env).stdout.splitlines()[-1])

    default = eval_dir / "sessions.jsonl"
    push = ["push", str(default), "sessions.jsonl"]
    assert result == {"path": str(default), "calls": [["pull", "sessions.jsonl"], push, push]}
    assert [json.loads(l)["id"] for l in default.read_text().splitlines()] == ["graded", "x"]


# ------------------------------------------------- two processes, one file
# One writer process: app.py saving grades ("update") or chat_app.py
# appending answers ("append"). It says it is ready, then waits for the go
# file, so both writers really run at the same time.
WRITER = r"""
import sys, time
from pathlib import Path

from src import storage

path, kind, n, go = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]), Path(sys.argv[4])
storage.SESSIONS_PATH = path
storage._push = lambda: None
Path(f"{go}.{kind}").touch()
while not go.exists():
    time.sleep(0.001)
for i in range(n):
    if kind == "append":
        storage.append_record({"id": f"chat{i}", "grade": None, "text": "c" * 3000})
    else:
        storage.update_record("seed0", {"grade": f"g{i}"})
"""


def start_writer(path, kind, n, go):
    env = clean_env(MOHS_SESSIONS_PATH=str(path))
    return subprocess.Popen([sys.executable, "-c", WRITER, str(path), kind, str(n), str(go)],
                            cwd=config.ROOT, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def wait_for(*paths, timeout=60):
    deadline = time.monotonic() + timeout
    while not all(p.exists() for p in paths):
        assert time.monotonic() < deadline, f"timed out waiting for {paths}"
        time.sleep(0.01)


def finish(proc):
    _, err = proc.communicate(timeout=120)
    assert proc.returncode == 0, err[-2000:]


def test_concurrent_appends_and_updates_lose_nothing(tmp_storage):
    """Grades saved in one process while another appends chat answers: the
    file stays valid JSONL, every append survives, and the last grade wins.
    Without a cross-process lock most of the appends vanish."""
    path = tmp_storage / "sessions.jsonl"
    write_sessions(path, [{"id": f"seed{i}", "grade": None, "text": "s" * 1000}
                          for i in range(20)])
    go = tmp_storage / "go"
    n = 150
    procs = [start_writer(path, "update", n, go), start_writer(path, "append", n, go)]
    try:
        wait_for(tmp_storage / "go.update", tmp_storage / "go.append")
        go.touch()
    finally:
        for p in procs:
            finish(p)

    records = storage.load_records()
    ids = [r["id"] for r in records]
    assert ids[:20] == [f"seed{i}" for i in range(20)]
    assert sorted(ids[20:]) == sorted(f"chat{i}" for i in range(n))
    assert records[0]["grade"] == f"g{n - 1}"


def test_a_held_lock_blocks_another_process(tmp_storage):
    path = tmp_storage / "sessions.jsonl"
    go = tmp_storage / "go"
    go.touch()
    with storage._locked():
        proc = start_writer(path, "append", 1, go)
        wait_for(tmp_storage / "go.append")
        time.sleep(0.3)
        assert proc.poll() is None                       # still waiting for the lock
        assert storage.load_records() == []
    finish(proc)
    assert [r["id"] for r in storage.load_records()] == ["chat0"]


def test_failed_update_leaves_the_file_as_it_was(tmp_storage):
    """The rewrite goes to a temp file first, so an error part-way through
    (here: an unserialisable value) cannot truncate sessions.jsonl."""
    path = tmp_storage / "sessions.jsonl"
    write_sessions(path, [{"id": "a"}, {"id": "b"}])
    before = path.read_text()
    with pytest.raises(TypeError):
        storage.update_record("b", {"bad": object()})
    assert path.read_text() == before
    assert sorted(p.name for p in tmp_storage.iterdir()) == ["sessions.jsonl",
                                                            "sessions.jsonl.lock"]


def test_update_replaces_the_file_and_keeps_its_permissions(tmp_storage):
    path = tmp_storage / "sessions.jsonl"
    storage.append_record({"id": "x", "grade": None})
    path.chmod(0o640)
    inode = path.stat().st_ino
    storage.update_record("x", {"grade": "ok"})
    assert path.stat().st_ino != inode                   # replaced, not rewritten in place
    assert path.stat().st_mode & 0o777 == 0o640
    # The lock sits beside the file SESSIONS_PATH names now; no temp file left.
    assert sorted(p.name for p in tmp_storage.iterdir()) == ["sessions.jsonl",
                                                            "sessions.jsonl.lock"]


def test_repo_test_list_is_well_formed():
    """data/eval/test_records.json, if present, must be {"ids": [unique str]}."""
    path = config.EVAL_DIR / "test_records.json"
    if not path.exists():
        pytest.skip("no test_records.json")
    ids = json.loads(path.read_text()).get("ids")
    assert isinstance(ids, list)
    assert all(isinstance(i, str) for i in ids)
    assert len(ids) == len(set(ids))
