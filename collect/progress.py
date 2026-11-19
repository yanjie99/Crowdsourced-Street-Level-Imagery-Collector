"""Progress of the running step, for the Run page: <project>/logs/progress.json.

The runner calls start() before a step and finish() after it; long loops inside a step call update(done, total, unit).
Writes are throttled (at most one a second) and atomic, so the page can poll the file at any time. They are best
effort: if the file is held open elsewhere, an update is skipped instead of failing the step. The file holds:
  {"step", "title", "state": running|done|failed, "done", "total", "unit", "message", "started_utc", "updated_utc",
   "elapsed_s", "rate_per_s", "eta_s"}
total = null means the step cannot tell how much is left (the page then shows an indeterminate bar).
"extra" holds short notes for the page, such as the current request rate of an adaptive throttle.
Platform lanes (19 Nov 2026): a process started with the environment variable COLLECT_LANE=<platform> (one of the
parallel Mapillary / KartaView lanes) writes progress_<platform>.json instead, so each lane has its own bar.
"""
from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timezone

_state: dict = {}
_lock = threading.Lock()
_last_write = [0.0]
_failures = [0]


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write(force=False):
    p = _state.get("path")
    if not p or (not force and time.time() - _last_write[0] < 1.0):
        return
    _last_write[0] = time.time()
    el = time.time() - _state["t_step"]
    el_phase = time.time() - _state["t0"]
    done, total = _state.get("done"), _state.get("total")
    rate = (done / el_phase) if done and el_phase > 0 else None
    eta = ((total - done) / rate) if rate and total and total >= done else None
    out = {k: _state.get(k) for k in ("step", "title", "state", "done", "total", "unit", "message", "started_utc", "extra")}
    out["lane"] = lane()
    out.update(updated_utc=_now(), elapsed_s=round(el, 1), rate_per_s=None if rate is None else round(rate, 3),
               eta_s=None if eta is None else round(eta))
    # Best effort: the progress file only feeds the Run page, so it must never stop a step. On Windows the replace fails
    # with "access denied" while another program (the page polling it, a viewer, antivirus) has the file open; retry
    # briefly, then skip this update (the next one writes it again).
    tmp = p + ".tmp"
    try:
        with open(tmp, "w", encoding="utf8") as f:
            json.dump(out, f)
    except OSError:
        _failures[0] += 1
        return
    for wait in (0.05, 0.1, 0.2, 0.4, None):
        try:
            os.replace(tmp, p)
            return
        except OSError:
            if wait is None:
                break
            time.sleep(wait)
    _failures[0] += 1
    if force:  # start/finish states matter most: write in place rather than lose them
        try:
            with open(p, "w", encoding="utf8") as f:
                json.dump(out, f)
        except OSError:
            pass


def lane() -> str | None:
    """The platform lane this process runs (COLLECT_LANE), or None for the main run."""
    return os.environ.get("COLLECT_LANE") or None


def file_name() -> str:
    return f"progress_{lane()}.json" if lane() else "progress.json"


def extra(**notes):
    """Short notes shown with the bar (e.g. rate="KartaView 3.5 req/s"); written with the next update."""
    with _lock:
        if _state:
            _state.setdefault("extra", {}).update(notes)
            _write()


def read_lanes(log_dir: str) -> dict:
    """The progress of each platform lane: {platform: progress}."""
    out = {}
    for pf in ("mapillary", "kartaview"):
        p = os.path.join(log_dir, f"progress_{pf}.json")
        if os.path.exists(p):
            try:
                with open(p, encoding="utf8") as f:
                    out[pf] = json.load(f)
            except (OSError, ValueError):
                pass
    return out


def skipped() -> int:
    """How many progress updates could not be written (the file was held open elsewhere)."""
    return _failures[0]


def start(log_dir: str, step: str, title: str):
    with _lock:
        _failures[0] = 0
        _state.clear()
        _state.update(path=os.path.join(log_dir, file_name()), extra={}, step=step, title=title, state="running", done=0,
                      total=None, unit="", message="starting", started_utc=_now(), t0=time.time(), t_step=time.time())
        _write(force=True)


def update(done=None, total=None, unit=None, message=None, add=None):
    """Report progress from inside a step. add=n counts n more done (safe from worker threads)."""
    with _lock:
        if not _state:
            return
        if add is not None:
            _state["done"] = (_state.get("done") or 0) + add
        if done is not None:
            _state["done"] = done
        if total is not None:
            _state["total"] = total
        if unit is not None:
            _state["unit"] = unit
        if message is not None:
            _state["message"] = message
        _write()


def phase(message, total=None, unit=""):
    """A new part of the step with its own count (e.g. 'Mapillary sequences' then 'KartaView sequences')."""
    with _lock:
        if not _state:
            return
        _state.update(message=message, total=total, unit=unit, done=0, t0=time.time())
        _write(force=True)


def finish(state="done", message=None):
    with _lock:
        if not _state:
            return
        _state["state"] = state
        if message:
            _state["message"] = message
        if state == "done" and _state.get("total"):
            _state["done"] = _state["total"]
        _write(force=True)


def read(log_dir: str):
    p = os.path.join(log_dir, "progress.json")
    if not os.path.exists(p):
        return None
    try:
        with open(p, encoding="utf8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None
