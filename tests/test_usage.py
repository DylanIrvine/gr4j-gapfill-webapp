# tests/test_usage.py
import json

from core.usage import increment, read_only


def test_increment_creates_and_counts_up(tmp_path):
    p = tmp_path / "count.json"
    assert read_only(p) == 0
    assert increment(p) == 1
    assert increment(p) == 2
    assert increment(p) == 3
    assert read_only(p) == 3

    data = json.loads(p.read_text())
    assert data["runs"] == 3
    assert "first_iso" in data and "last_iso" in data


def test_start_offset_is_a_floor_not_an_add(tmp_path):
    p = tmp_path / "count.json"
    assert increment(p, start=1000) == 1001
    assert increment(p, start=1000) == 1002        # floor no longer binds
    assert increment(p, start=5) == 1003           # lower start ignored


def test_corrupt_file_resets_gracefully(tmp_path):
    p = tmp_path / "count.json"
    p.write_text("not json {")
    assert increment(p) == 1


def test_unwritable_path_returns_none(tmp_path):
    # a path whose parent is a file, so mkdir/replace cannot succeed
    blocker = tmp_path / "blocker"
    blocker.write_text("x")
    assert increment(blocker / "nested" / "count.json") is None
