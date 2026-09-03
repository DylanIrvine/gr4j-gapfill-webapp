# core/usage.py
# A "times run" counter for the app: how many browser sessions have started
# HydroSTITCH.
#
# Two backends, in order of preference:
#
#   1. Upstash Redis (serverless). An atomic INCR on a single key. Survives
#      redeploys and reboots, and is race-safe across concurrent users. Enabled
#      by passing redis_url + redis_token (from st.secrets or the
#      UPSTASH_REDIS_REST_URL / UPSTASH_REDIS_REST_TOKEN environment variables).
#      One database holds any number of counters - use a distinct redis_key per
#      app (default "hydrostitch:runs").
#
#   2. A local JSON file (.run_count.json). Zero configuration, but the container
#      filesystem on Streamlit Community Cloud is wiped on every redeploy or
#      reboot, so the count resumes from the last surviving value rather than a
#      true lifetime total. Exact for a local run or a persistent volume; point
#      HYDROSTITCH_RUN_COUNT_PATH at a persistent location to keep it exact.
#
# increment() returns None only when neither backend is available (e.g. no Redis
# configured and a read-only filesystem), so the caller can fall back to an
# in-process tally or skip the display.

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

_DEFAULT_PATH = Path(__file__).resolve().parent.parent / ".run_count.json"
DEFAULT_REDIS_KEY = "hydrostitch:runs"


# --------------------------------------------------------------------------- #
# Redis backend
# --------------------------------------------------------------------------- #
def _redis_incr(url, token, key, start=0):
    """Atomic INCR on an Upstash Redis key; seeds it to `start` first if unset."""
    from upstash_redis import Redis

    client = Redis(url=url, token=token)
    if start:
        client.set(key, int(start), nx=True)
    return int(client.incr(key))


def _redis_get(url, token, key):
    from upstash_redis import Redis

    value = Redis(url=url, token=token).get(key)
    return int(value) if value is not None else 0


# --------------------------------------------------------------------------- #
# File backend
# --------------------------------------------------------------------------- #
def _resolve_path(explicit=None):
    if explicit:
        return Path(explicit)
    env = os.environ.get("HYDROSTITCH_RUN_COUNT_PATH")
    return Path(env) if env else _DEFAULT_PATH


def _read(path):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return int(data.get("runs", 0)), dict(data)
    except (FileNotFoundError, ValueError, OSError, TypeError):
        return 0, {}


def _write_atomic(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".runcount-", suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as fh:
            json.dump(data, fh)
        os.replace(tmp, path)
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
def increment(path=None, start=0, *, redis_url=None, redis_token=None,
              redis_key=DEFAULT_REDIS_KEY):
    """Increment the run count and return the new total (or None if unavailable).

    Redis is the source of truth when redis_url and redis_token are supplied; any
    Redis error falls through to the JSON file backend so the counter still moves.
    """
    if redis_url and redis_token:
        try:
            return _redis_incr(redis_url, redis_token, redis_key or DEFAULT_REDIS_KEY,
                               start)
        except Exception:
            pass  # network / auth / missing package -> file backend

    target = _resolve_path(path)
    try:
        current, data = _read(target)
        base = max(current, int(start or 0))
        new = base + 1
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        data["runs"] = new
        data.setdefault("first_iso", now)
        data["last_iso"] = now
        _write_atomic(target, data)
        return new
    except OSError:
        return None


def read_only(path=None, *, redis_url=None, redis_token=None,
              redis_key=DEFAULT_REDIS_KEY):
    """The current count without incrementing it, or 0 if unavailable."""
    if redis_url and redis_token:
        try:
            return _redis_get(redis_url, redis_token, redis_key or DEFAULT_REDIS_KEY)
        except Exception:
            pass
    current, _ = _read(_resolve_path(path))
    return current
