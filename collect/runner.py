"""Runs a step: checks its prerequisites, fingerprints its inputs and settings, records the run in the ledger, reports
progress (logs/progress.json) and, when it fails, an explanation of the failure (logs/last_error.json)."""
from __future__ import annotations

import json
import os
import traceback
from datetime import datetime, timezone

from . import progress
from .config import Config
from .diagnose import explain
from .ledger import Ledger, fingerprint
from .log import get_logger
from .steps import PLATFORM_KEYS, PLATFORM_STEPS, STEPS, needed


def _error_file(cfg):
    return os.path.join(cfg.paths.logs, "last_error.json")


def _record_error(cfg, name, text):
    d = explain(name, text)
    d.update(time_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"), detail=text[-4000:])
    try:  # the explanation is for the Run page; failing to write it must not hide the step's own error
        with open(_error_file(cfg), "w", encoding="utf8") as f:
            json.dump(d, f, indent=1)
    except OSError:
        pass
    return d


def _warnings(summary):
    """Counts of failures a step finished with (it carried on, but some items are missing)."""
    out = []

    def walk(d, path=""):
        for k, v in (d or {}).items():
            if isinstance(v, dict):
                walk(v, f"{path}{k}.")
            elif isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0 and ("fail" in k or k in ("no_url", "no_record")):
                out.append(f"{path}{k}: {v:g}")
    walk(summary)
    if isinstance(summary, dict) and summary.get("note"):
        out.append(str(summary["note"]))
    return out


# ---- platform lanes (19 Nov 2026): Mapillary and KartaView run at the same time, each at its own polite rate
LANE_STEPS = {"mapillary": ("1a", "1b"), "kartaview": ("1c", "1d", "1e", "1f")}   # Step 1: whole steps per platform
SPLIT_STEPS = ("3b", "3d")                                                         # one step, split by platform
PLATFORMS = ("mapillary", "kartaview")


def parallel_ok(cfg: Config, opts: dict) -> bool:
    """Lanes are used when both platforms are ticked, platforms.parallel is on, and this is not already a lane."""
    return (bool(cfg.get("platforms.parallel", True)) and not opts.get("platform") and not progress.lane()
            and all(cfg.get(f"platforms.{p}.enabled", False) for p in PLATFORMS))


def _child(cfg: Config, steps: list[str], lane: str, force: bool, opts: dict, platform: str | None = None):
    """Start `collect run <steps>` for one lane (its own process: own progress file, own throttles)."""
    import subprocess
    import sys
    cmd = [sys.executable, "-m", "collect", "run", *steps, "-p", cfg.path]
    if platform:
        cmd += ["--platform", platform]
    if force:
        cmd.append("--force")
    if opts.get("offline"):
        cmd.append("--offline")
    if opts.get("regions"):
        cmd += ["--regions", ",".join(opts["regions"])]
    if opts.get("sequences"):
        cmd += ["--sequences", ",".join(opts["sequences"])]
    env = {**os.environ, "COLLECT_LANE": lane, "PYTHONIOENCODING": "utf-8"}
    return subprocess.Popen(cmd, env=env, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _wait_lanes(procs: dict, what: str):
    """Wait for every lane; each one finishes its own work even if the other fails. Raise if any failed."""
    import time
    progress.phase(f"{what}: Mapillary and KartaView lanes", len(procs), "lanes")
    failed, done = [], set()
    try:
        while len(done) < len(procs):
            for lane, p in procs.items():
                if lane not in done and p.poll() is not None:
                    done.add(lane)
                    if p.returncode != 0:
                        failed.append(lane)
            running = [lane for lane in procs if lane not in done]
            progress.update(done=len(done), message="running: " + ", ".join(running) if running else "lanes finished")
            if running:
                time.sleep(2)
    except BaseException:
        for p in procs.values():   # stopped: stop the lanes too
            if p.poll() is None:
                p.terminate()
        raise
    if failed:
        raise RuntimeError(f"{what}: the {' and '.join(failed)} lane failed (see its error above or logs/<step>.<platform>.log)")


def run_lanes(cfg: Config, groups: dict, force: bool = False, **opts):
    """Step 1's platform steps as two parallel processes: {platform: [steps]}."""
    what = "Step 1 " + ", ".join(s for v in groups.values() for s in v)
    progress.start(cfg.paths.logs, "1", what + " (parallel lanes)")
    procs = {pf: _child(cfg, steps, pf, force, opts) for pf, steps in groups.items() if steps}
    try:
        _wait_lanes(procs, what)
    except BaseException:
        progress.finish("failed", "a platform lane failed")
        raise
    progress.finish("done", "both lanes finished")


def _split_fn(name):
    """The body of a split step: run it once per platform, in parallel, and merge the summaries."""
    def fn(cfg, log, force=False, **opts):
        procs = {pf: _child(cfg, [name], pf, force, opts, platform=pf) for pf in PLATFORMS}
        _wait_lanes(procs, f"step {name}")
        led, summary = Ledger(cfg.paths.ledger), {}
        for pf in PLATFORMS:
            r = led.last(f"{name}@{pf}") or {}
            try:
                summary.update(json.loads(r.get("summary") or "{}"))
            except ValueError:
                pass
        return [os.path.join(cfg.paths.step(3), "status")], summary
    return fn


def run_step(cfg: Config, name: str, force: bool = False, **opts) -> dict:
    st = STEPS[name]
    part = opts.get("platform")                       # one platform's part of a split step (3b@kartaview)
    key = f"{name}@{part}" if part else name
    log = get_logger(f"{name}.{part}" if part else name, cfg.paths.logs)
    try:
        if cfg.selection_mode not in st.modes:
            raise RuntimeError(f"step {name} belongs to Step 2 mode {'/'.join(st.modes)}; this project uses {cfg.selection_mode}")
        reuse = cfg.get("reuse", {}) or {}
        if name[0] in {str(k) for k in reuse}:
            raise RuntimeError(f"step {name}: this project reuses Step {name[0]} from "
                               f"{reuse.get(int(name[0]), reuse.get(name[0]))}; it is read, not run here")
        led = Ledger(cfg.paths.ledger)
        for dep in st.needs:
            alts = dep.split("|")          # "2h|2w": either one (the two Step 2 modes)
            if any(a[0] in {str(k) for k in reuse} for a in alts):
                continue
            alts = [a for a in alts if needed(cfg, a)]   # an unticked platform's step is not required
            if not alts:
                continue
            if not any((led.last(a) or {}).get("status") == "ok" for a in alts):
                raise RuntimeError(f"step {name} needs step {' or '.join(alts)} to have run successfully first")
    except Exception:
        _record_error(cfg, name, traceback.format_exc())
        raise
    settings = {k: cfg.get(k, None) for k in st.settings} | {"opts": opts}
    if name in PLATFORM_STEPS:
        settings |= {k: bool(cfg.get(k, False)) for k in PLATFORM_KEYS}
    inputs = st.inputs(cfg)
    fp = fingerprint(inputs, settings)
    last = led.last(key)
    if not force and last and last["status"] == "ok" and last["fingerprint"] == fp:
        log.info(f"step {key} is up to date (same inputs and settings); use --force to rerun")
        return {"status": "up to date"}
    run_id = led.start(key, fp, inputs)
    log.info(f"step {key}: {st.title}")
    progress.start(cfg.paths.logs, name, st.title + (f" ({part})" if part else ""))
    try:
        if name in SPLIT_STEPS and parallel_ok(cfg, opts):
            outputs, summary = _split_fn(name)(cfg, log, force=force, **opts)
        else:
            outputs, summary = st.fn(cfg, log, **opts)
    except BaseException as e:  # noqa: BLE001  (KeyboardInterrupt too: the run was stopped)
        text = traceback.format_exc() if not isinstance(e, KeyboardInterrupt) else "terminated by the user"
        led.fail(run_id, text)
        d = _record_error(cfg, name, text)
        progress.finish("failed", d["title"])
        log.error(f"step {name} failed: {d['title']}. {d['cause']} What to try: {d['fix']}")
        raise
    led.finish(run_id, outputs, summary)
    w = _warnings(summary)
    progress.finish("done", f"finished with warnings: {'; '.join(w)}" if w else "finished")
    if w:
        failed = [x for x in w if x != summary.get("note")]
        log.warning(f"step {name} finished with warnings: {'; '.join(w)}."
                    + (" Rerun the step to retry the failed items." if failed else ""))
    ef = _error_file(cfg)
    if os.path.exists(ef):
        try:
            if json.load(open(ef, encoding="utf8")).get("step") == name and not part:
                os.remove(ef)          # this step now succeeds
        except (OSError, ValueError):
            pass
    log.info(f"step {key} done: {summary}")
    return {"status": "ok", **summary, **({"warnings": w} if w else {})}
