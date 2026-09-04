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


def test_redis_backend_is_used_when_configured(tmp_path, monkeypatch):
    seen = {}

    def fake_incr(url, token, key, start=0):
        seen["args"] = (url, token, key, start)
        return 42

    monkeypatch.setattr("core.usage._redis_incr", fake_incr)
    p = tmp_path / "count.json"
    n = increment(p, redis_url="https://x.upstash.io", redis_token="tok",
                  redis_key="hydrostitch:runs")
    assert n == 42
    assert seen["args"] == ("https://x.upstash.io", "tok", "hydrostitch:runs", 0)
    assert not p.exists()          # file backend untouched when Redis succeeds


def test_redis_failure_falls_back_to_file(tmp_path, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("no network")

    monkeypatch.setattr("core.usage._redis_incr", boom)
    p = tmp_path / "count.json"
    assert increment(p, redis_url="https://x", redis_token="t") == 1
    assert p.exists()


def test_no_redis_creds_uses_file(tmp_path):
    p = tmp_path / "count.json"
    assert increment(p, redis_url=None, redis_token=None) == 1


def test_status_reports_backend(tmp_path, monkeypatch):
    from core.usage import status

    p = tmp_path / "count.json"
    assert status(p)["backend"] == "file"
    increment(p)
    assert status(p) == {"backend": "file", "count": 1, "detail": str(p)}

    monkeypatch.setattr("core.usage._redis_get", lambda *a, **k: 4321)
    s = status(p, redis_url="https://x", redis_token="t", redis_key="hydrostitch:runs")
    assert s["backend"] == "redis" and s["count"] == 4321

    def boom(*a, **k):
        raise RuntimeError("bad token")

    monkeypatch.setattr("core.usage._redis_get", boom)
    s = status(p, redis_url="https://x", redis_token="t")
    assert s["backend"] == "redis-error" and "bad token" in s["detail"]
