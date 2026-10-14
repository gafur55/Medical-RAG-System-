"""Tests for session storage and the test-record filter in src/storage.py.

Every test points storage at a temporary folder and disables Hub sync, so
nothing touches data/eval/sessions.jsonl or the HF dataset.
"""

import json

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


def test_repo_test_list_is_well_formed():
    """data/eval/test_records.json, if present, must be {"ids": [unique str]}."""
    path = config.EVAL_DIR / "test_records.json"
    if not path.exists():
        pytest.skip("no test_records.json")
    ids = json.loads(path.read_text()).get("ids")
    assert isinstance(ids, list)
    assert all(isinstance(i, str) for i in ids)
    assert len(ids) == len(set(ids))
