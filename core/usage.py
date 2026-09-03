# core/usage.py
# A best-effort "times run" counter for the app: how many browser sessions have
# started HydroSTITCH.
#
# Persistence on Streamlit Community Cloud is limited. The container filesystem is
# wiped on every redeploy or reboot, so a file-backed count resumes from whatever
# value last survived rather than a true lifetime total. It is exact for a local
# run, or for a deployment with a persistent volume. Point the environment
# variable HYDROSTITCH_RUN_COUNT_PATH (or st.secrets['run_count_path']) at a
# persistent location to keep it exact, and set HYDROSTITCH_RUN_COUNT_START (or
# st.secrets['run_count_start']) to carry over a count from elsewhere.

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

_DEFAULT_PATH = Path(__file__).resolve().parent.parent / ".run_count.json"


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


def increment(path=None, start=0):
    """Increment the persisted run count and return the new total.

    Returns None if the count cannot be read or written (for example a read-only
    filesystem), so the caller can simply skip the display or fall back to an
    in-process tally.
    """
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


def read_only(path=None):
    """The current count without incrementing it, or 0 if unavailable."""
    current, _ = _read(_resolve_path(path))
    return current
