"""HTTP helpers shared by the steps: polite throttling, retries with back-off, and a gzip JSON cache.

Every raw response is cached on disk before it is used, so a step that is stopped resumes without asking
again, and a rerun from a complete cache needs no network. With offline=True a missing cache entry is an
error instead of a request (used to check that a run reproduces from what is on disk).
"""
from __future__ import annotations

import gzip
import json
import os
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone


class OfflineMiss(RuntimeError):
    pass


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Throttle:
    """At most per_s requests a second across every thread that shares this object."""

    def __init__(self, per_s: float):
        self.gap = 1.0 / per_s
        self.last = 0.0
        self.lock = threading.Lock()

    def wait(self):
        with self.lock:
            w = self.last + self.gap - time.time()
            if w > 0:
                time.sleep(w)
            self.last = time.time()


class AdaptiveThrottle(Throttle):
    """A request rate that follows the server (19 Nov 2026): it starts at `start` requests a second, rises by `step`
    after every `every` quick, error-free replies up to `cap`, and falls at once when the server shows strain:
    halved on HTTP 429, a 5xx or a timeout (down to `min_rate`, below the usual floor, because the server is
    struggling), one step down when replies take more than twice the usual time (not below `floor`), and paused for
    as long as a Retry-After header asks. With cap == start it only ever slows down.
    on_change(rate, reason) is called whenever the rate changes (for the log and the Run page)."""

    def __init__(self, start: float, cap: float, floor: float, min_rate: float = 0.5, step: float = 0.5,
                 every: int = 60, on_change=None):
        super().__init__(start)
        self.rate, self.cap, self.floor, self.min_rate = start, max(cap, start), min(floor, start), min_rate
        self.step, self.every, self.on_change = step, every, on_change
        self.clean, self.base_latency, self.changes = 0, None, 0

    def _set(self, rate, reason):
        rate = round(min(self.cap, max(self.min_rate, rate)), 2)
        if rate != self.rate:
            self.rate, self.gap, self.changes = rate, 1.0 / rate, self.changes + 1
            if self.on_change:
                try:
                    self.on_change(rate, reason)
                except Exception:  # noqa: BLE001  reporting must never break a request
                    pass

    def ok(self, seconds: float):
        """A reply that arrived without error, in `seconds`."""
        with self.lock:
            b = self.base_latency
            self.base_latency = seconds if b is None else 0.9 * b + 0.1 * min(seconds, 3 * b)
            if b is not None and seconds > 2 * b and seconds > 1.0:
                self.clean = 0
                if self.rate > self.floor:
                    self._set(max(self.floor, self.rate - self.step), f"replies slowed to {seconds:.1f} s")
                return
            self.clean += 1
            if self.clean >= self.every and self.rate < self.cap:
                self.clean = 0
                self._set(self.rate + self.step, f"{self.every} quick replies in a row")

    def strain(self, why: str, retry_after: float | None = None):
        """The server answered 429 or 5xx, or did not answer in time: slow down (and wait if it said how long)."""
        with self.lock:
            self.clean = 0
            self._set(self.rate / 2, why)
            if retry_after:
                self.last = max(self.last, time.time() + min(float(retry_after), 600.0))


def retry_after(e) -> float | None:
    """Seconds from an HTTP error's Retry-After header (a number of seconds; dates are ignored)."""
    try:
        v = e.headers.get("Retry-After")
        return float(v) if v else None
    except (AttributeError, TypeError, ValueError):
        return None


def feedback(throttle, ok: bool, seconds: float = 0.0, why: str = "", wait: float | None = None):
    """Tell an adaptive throttle how a request went (plain throttles ignore it)."""
    if isinstance(throttle, AdaptiveThrottle):
        throttle.ok(seconds) if ok else throttle.strain(why, wait)


def platform_throttle(cfg, platform: str, key: str = "per_s", default: float = 2.0, log=None):
    """The throttle for one platform's requests. KartaView's is adaptive (platforms.kartaview.adaptive): it starts at
    per_s and may rise to max_per_s only when bulk use is approved for the project; otherwise it can only slow down."""
    k = cfg.get(f"platforms.{platform}", {}) or {}
    start = float(k.get(key, default))
    a = k.get("adaptive") or {}
    if platform != "kartaview" or not a.get("enabled", True):
        return Throttle(start)
    cap = float(a.get("max_per_s", 5.0)) if k.get("bulk_access_approved") else start
    from . import progress
    label = {"kartaview": "KartaView", "mapillary": "Mapillary"}.get(platform, platform)

    def changed(rate, reason):
        progress.extra(rate=f"{label} {rate:g} req/s")
        if log:
            log.info(f"{platform} request rate now {rate:g}/s ({reason})")
    t = AdaptiveThrottle(start, cap, floor=float(a.get("floor_per_s", 2.0)), min_rate=float(a.get("min_per_s", 0.5)),
                         step=float(a.get("step", 0.5)), every=int(a.get("every", 60)), on_change=changed)
    progress.extra(rate=f"{label} {start:g} req/s")
    return t


class Client:
    """GET JSON with the throttle, retries and statistics of the Step 1 and 3 scripts."""

    def __init__(self, base: str, per_s: float, user_agent: str, headers: dict | None = None, offline: bool = False,
                 throttle: Throttle | None = None):
        self.base, self.ua, self.headers, self.offline = base, user_agent, headers or {}, offline
        self.throttle = throttle or Throttle(per_s)
        self.stats = {"requests": 0, "retries": 0, "failures": []}
        self._count = threading.Lock()   # several worker threads can share one client

    def get(self, path: str, tries: int = 6):
        if self.offline:
            raise OfflineMiss(path)
        err = None
        for attempt in range(tries):
            self.throttle.wait()
            with self._count:
                self.stats["requests"] += 1
            t0 = time.time()
            try:
                req = urllib.request.Request(self.base + path, headers={"User-Agent": self.ua, **self.headers})
                with urllib.request.urlopen(req, timeout=90) as r:
                    data = json.loads(r.read())
                feedback(self.throttle, True, time.time() - t0)
                return data
            except urllib.error.HTTPError as e:
                err = f"HTTP {e.code}"
                if e.code == 429 or e.code >= 500:
                    feedback(self.throttle, False, why=err, wait=retry_after(e))
                time.sleep(10 * (attempt + 1) if e.code == 429 or e.code >= 500 else 2)
            except Exception as e:  # noqa: BLE001
                err = type(e).__name__
                if "timeout" in err.lower() or "timed out" in str(e):
                    feedback(self.throttle, False, why="no answer in time")
                time.sleep(5 * (attempt + 1))
            with self._count:
                self.stats["retries"] += 1
        with self._count:
            self.stats["failures"].append({"path": path, "error": err, "at": now()})
        return None

    def cached(self, path_gz: str, path: str):
        """The cached response at path_gz, or GET path and cache it (gzip JSON)."""
        if os.path.exists(path_gz):
            with gzip.open(path_gz, "rt", encoding="utf8") as f:
                return json.load(f)
        data = self.get(path)
        if data is not None:
            with gzip.open(path_gz + ".part", "wt", encoding="utf8") as f:
                json.dump(data, f)
            os.replace(path_gz + ".part", path_gz)
        return data
