"""Local browser interface for collect (Phase 3, 18 Nov 2026).

  collect ui [--workspace <folder of projects>] [--port 8765]

Serves the page and a small JSON API on 127.0.0.1 only. Every step runs as a separate process
(`python -m collect run ...`), so a long download keeps going while the page is closed; its output goes to
<project>/logs/ui_job.log. The Mapillary token is only ever stored (Windows Credential Manager), never returned.
Requests that change anything must carry the header X-Collect: 1 (the page sends it), so another web page cannot
trigger them.
"""
from __future__ import annotations

import io
import json
import mimetypes
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import yaml

from ..config import Config, ConfigError
from ..ledger import Ledger
from ..steps import STEPS, plan

HERE = os.path.dirname(os.path.abspath(__file__))
APP_ROOT = os.path.dirname(os.path.dirname(HERE))
SAFE = re.compile(r"^[A-Za-z0-9_\-]+$")
JOBS: dict[str, dict] = {}
LOCK = threading.Lock()


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class ApiError(Exception):
    def __init__(self, status, msg):
        super().__init__(msg)
        self.status, self.msg = status, msg


# ---------------------------------------------------------------- projects
class Workspace:
    """The projects the page shows: every sub-folder of the workspace with a project.yaml, plus projects created in
    another location (19 Nov 2026), listed by id and folder in <workspace>/projects_elsewhere.json."""

    def __init__(self, folder):
        self.folder = os.path.abspath(folder)
        os.makedirs(self.folder, exist_ok=True)

    @property
    def _reg_file(self):
        return os.path.join(self.folder, "projects_elsewhere.json")

    def elsewhere(self) -> dict:
        try:
            with open(self._reg_file, encoding="utf8") as f:
                return {k: v for k, v in json.load(f).items() if SAFE.match(k)}
        except (OSError, ValueError):
            return {}

    def register(self, pid, folder):
        reg = self.elsewhere()
        reg[pid] = os.path.abspath(folder)
        with open(self._reg_file, "w", encoding="utf8") as f:
            json.dump(reg, f, indent=1)

    def folder_of(self, pid):
        return self.elsewhere().get(pid) or os.path.join(self.folder, pid)

    def path(self, pid):
        if not SAFE.match(pid or ""):
            raise ApiError(400, "bad project id")
        p = os.path.join(self.folder_of(pid), "project.yaml")
        if not os.path.exists(p):
            raise ApiError(404, f"no project {pid}")
        return p

    def cfg(self, pid):
        try:
            return Config.load(self.path(pid))
        except ConfigError as e:
            raise ApiError(400, str(e)) from e

    def list(self):
        out = []
        here = [(d, os.path.join(self.folder, d)) for d in sorted(os.listdir(self.folder))]
        for d, folder in here + sorted(self.elsewhere().items()):
            p = os.path.join(folder, "project.yaml")
            if not os.path.exists(p):
                continue
            row = {"id": d, "folder": folder}
            try:
                c = Config.load(p)
                led = {r["step"]: r["status"] for r in Ledger(c.paths.ledger).all_last()} if os.path.exists(c.paths.ledger) else {}
                pl = plan(c)
                row.update(name=c.get("project.name", d), mode=c.selection_mode, country=c.get("area.country_code"),
                           steps=len(pl), done=sum(led.get(s) == "ok" for s in pl), job=job_state(d)["state"])
            except Exception as e:  # noqa: BLE001
                row.update(name=d, error=str(e))
            out.append(row)
        return out


def project_raw(path):
    with open(path, encoding="utf8") as f:
        return f.read()


def save_project_yaml(path, text):
    data = yaml.safe_load(text) or {}
    from ..config import defaults, merge, us_names
    Config(merge(us_names(defaults()), us_names(data)), path)            # validate before writing
    bak = path + ".bak"
    shutil.copyfile(path, bak)
    with open(path, "w", encoding="utf8") as f:
        f.write(text)


def set_setting(path, key, value):
    with open(path, encoding="utf8") as f:
        text = f.read()
    head = "".join(l for l in text.splitlines(True) if l.startswith("#"))
    data = yaml.safe_load(text) or {}
    node = data
    parts = key.split(".")
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    if value is None:
        node.pop(parts[-1], None)
    else:
        node[parts[-1]] = value
    save_project_yaml(path, head + yaml.safe_dump(data, sort_keys=False, allow_unicode=True))


# ---------------------------------------------------------------- jobs
WS_REF: dict = {}          # the workspace, so job state can be found again after the server restarts


def _job_file(pid):
    ws = WS_REF.get("ws")
    return os.path.join(ws.folder_of(pid), "logs", "job.json") if ws else None


def _alive(proc_id):
    """Is the process still running? (Windows: OpenProcess + GetExitCodeProcess; elsewhere: signal 0.)"""
    if not proc_id:
        return False
    if os.name == "nt":
        import ctypes
        k = ctypes.windll.kernel32
        h = k.OpenProcess(0x1000, False, int(proc_id))          # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False
        code = ctypes.c_ulong()
        ok = k.GetExitCodeProcess(h, ctypes.byref(code))
        k.CloseHandle(h)
        return bool(ok) and code.value == 259                    # STILL_ACTIVE
    try:
        os.kill(int(proc_id), 0)
        return True
    except OSError:
        return False


def job_state(pid):
    """The project's last job. Jobs run detached from the server, so they survive a server restart; a job started by
    an earlier server is found again from logs/job.json."""
    with LOCK:
        j = JOBS.get(pid)
    if j:
        rc = j["proc"].poll()
        state = "running" if rc is None else ("finished" if rc == 0 else "failed")
        return {"state": state, "returncode": rc, "steps": j["steps"], "started_utc": j["started"], "log": j["log"],
                "pid": j["proc"].pid}
    jf = _job_file(pid)
    if not jf or not os.path.exists(jf):
        return {"state": "idle"}
    try:
        info = json.load(open(jf, encoding="utf8"))
    except (OSError, ValueError):
        return {"state": "idle"}
    if _alive(info.get("pid")):
        state, rc = "running", None
    else:                                                       # ended while no server was watching
        try:
            c = Config.load(os.path.join(WS_REF["ws"].folder_of(pid), "project.yaml"))
            led = {r["step"]: r for r in Ledger(c.paths.ledger).all_last()}
            ok = all((led.get(st) or {}).get("status") == "ok" for st in info["steps"])
        except Exception:  # noqa: BLE001
            ok = False
        state, rc = ("finished", 0) if ok else ("failed", info.get("returncode", "unknown"))
    return {"state": state, "returncode": rc, "steps": info["steps"], "started_utc": info["started_utc"],
            "log": info["log"], "pid": info.get("pid")}


def start_job(ws, pid, steps, offline=False, force=False):
    cfg = ws.cfg(pid)
    if job_state(pid)["state"] == "running":
        raise ApiError(409, "a job is already running for this project")
    known = plan(cfg)
    steps = known if steps == ["all"] else steps
    bad = [s for s in steps if s not in STEPS]
    if bad:
        raise ApiError(400, f"unknown steps: {bad}")
    logp = os.path.join(cfg.paths.logs, "ui_job.log")
    cmd = [sys.executable, "-m", "collect", "run", *steps, "-p", ws.path(pid)] + (["--offline"] if offline else []) + \
          (["--force"] if force else [])
    started = now()
    with open(logp, "w", encoding="utf8") as f:
        f.write(f"{started} $ collect run {' '.join(steps)}{' --offline' if offline else ''}{' --force' if force else ''}\n")
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf8", COLUMNS="160")
    # detached: the run keeps going if the server or the page is closed or restarted
    flags = (subprocess.CREATE_NEW_PROCESS_GROUP | 0x00000008) if os.name == "nt" else 0     # 0x8 = DETACHED_PROCESS
    with open(logp, "a", encoding="utf8") as f:
        kw = dict(cwd=APP_ROOT, stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=env,
                  start_new_session=(os.name != "nt"), close_fds=True)
        try:   # also leave the server's Windows job object, so stopping the server's process tree spares the run
            proc = subprocess.Popen(cmd, creationflags=flags | (0x01000000 if os.name == "nt" else 0), **kw)
        except OSError:                                        # the job does not allow breakaway
            proc = subprocess.Popen(cmd, creationflags=flags, **kw)
    with LOCK:
        JOBS[pid] = {"proc": proc, "steps": steps, "started": started, "log": logp}
    json.dump({"pid": proc.pid, "steps": steps, "started_utc": started, "log": logp, "cmd": cmd},
              open(os.path.join(cfg.paths.logs, "job.json"), "w", encoding="utf8"), indent=1)
    return job_state(pid)


def job_view(cfg, pid, n=200):
    """The job, its log tail, the current step's progress, the steps done in this job, and any error explained."""
    from .. import progress
    from ..diagnose import explain, stall
    j = job_state(pid)
    logs = cfg.paths.logs
    out = {**j, "tail": tail(j.get("log") or os.path.join(logs, "ui_job.log"), n), "progress": progress.read(logs)}
    steps = j.get("steps") or []
    if steps:
        led = {r["step"]: r for r in Ledger(cfg.paths.ledger).all_last()}
        started = j.get("started_utc") or ""
        out["steps_done"] = [s for s in steps if (led.get(s) or {}).get("status") == "ok"
                             and (led[s].get("finished_utc") or "") >= started]
        out["steps_skipped"] = [s for s in steps if s not in out["steps_done"] and f"{s}: " in out["tail"]
                                and '"up to date"' in out["tail"].split(f"{s}: ")[-1].split(chr(10))[0]]
    ef = os.path.join(logs, "last_error.json")
    err = json.load(open(ef, encoding="utf8")) if os.path.exists(ef) else None
    hung = [r["step"] for r in Ledger(cfg.paths.ledger).all_last() if r["status"] == "running"]         if os.path.exists(cfg.paths.ledger) and j["state"] != "running" else []
    if hung and (not err or err.get("time_utc", "") < (j.get("started_utc") or "")):
        err = None                       # explained below as an interrupted step
    elif j["state"] == "failed" and (not err or err.get("time_utc", "") < (j.get("started_utc") or "")):
        p = out.get("progress") or {}
        step = p.get("step") if p.get("started_utc", "") >= (j.get("started_utc") or "") else None
        # no step recorded an error: the process itself failed (crash before or between steps); explain from the log
        err = explain(step or (steps[0] if steps else "?"), out["tail"] if "Traceback" in out["tail"] else f"returncode {j.get('returncode')}")
        if not step and "Traceback" in out["tail"]:
            err["title"] = "The run could not start"
    if j["state"] == "running" and err and err.get("time_utc", "") < (j.get("started_utc") or ""):
        err = None                       # an error from an earlier run
    if err and err.get("step") in STEPS:
        last = Ledger(cfg.paths.ledger).last(err["step"]) or {}
        if last.get("status") == "ok" and (last.get("finished_utc") or "") > err.get("time_utc", ""):
            err = None                   # that step has succeeded since
    if not err and j["state"] != "running":
        if hung:
            err = {"rule": "interrupted", "step": hung[0], "title": f"Step {hung[0]} was interrupted",
                   "cause": "Its process ended before the step finished and nothing recorded why: the run was stopped, "
                            "the computer slept or restarted, or the process was closed from outside.",
                   "fix": "Run again (Run all skips the finished steps); the step resumes from what it already saved."}
    out["error"] = err
    # parallel platform lanes of this job (each writes progress_<platform>.json)
    started = j.get("started_utc") or ""
    out["lanes"] = {pf: p for pf, p in progress.read_lanes(logs).items() if (p.get("started_utc") or "") >= started} \
        if j["state"] == "running" else {}
    live = [p for p in out["lanes"].values() if p.get("state") == "running"]
    out["stall"] = (next((s for s in (stall(p, True) for p in live) if s), None) if live
                    else stall(out["progress"], j["state"] == "running"))
    return out


def stop_job(pid):
    st = job_state(pid)
    if st["state"] != "running":
        return {"state": st["state"]}
    with LOCK:
        j = JOBS.get(pid)
    if j:
        j["proc"].terminate()
        try:
            j["proc"].wait(10)
        except subprocess.TimeoutExpired:
            j["proc"].kill()
    elif st.get("pid"):
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(st["pid"]), "/T", "/F"], capture_output=True)
        else:
            os.kill(int(st["pid"]), 15)
    return job_state(pid)


def tail(path, n=200):
    if not os.path.exists(path):
        return ""
    with open(path, encoding="utf8", errors="replace") as f:
        lines = f.readlines()
    return "".join(lines[-n:])


# ---------------------------------------------------------------- data views for the page
def overview(cfg):
    b = cfg.boundary_ll
    return {"name": cfg.get("project.name"), "folder": cfg.folder, "mode": cfg.selection_mode,
            "area": {k: cfg.get(f"area.{k}", None) for k in ("country_code", "timezone", "driving_side", "metric_crs",
                                                               "bbox_harvest", "boundary_source")},
            "utc_offsets_h": cfg.utc_offsets_h, "area_km2": round(cfg.boundary_m.area / 1e6, 1),
            "boundary": json.loads(json.dumps(__import__("shapely.geometry", fromlist=["mapping"]).mapping(b))),
            "platforms": {p: bool(cfg.get(f"platforms.{p}.enabled", False)) for p in ("mapillary", "kartaview")},
            "road_standard_generic": "PLACEHOLDER" in str(cfg.get("step5_osm.road_standard.source", "")),
            "budget_gb": cfg.get("storage.budget_gb")}


def steps_view(cfg, job_running=False):
    led = {r["step"]: r for r in Ledger(cfg.paths.ledger).all_last()} if os.path.exists(cfg.paths.ledger) else {}
    out = []
    for n in plan(cfg):
        r = led.get(n) or {}
        status = r.get("status", "not run")
        if status == "running" and not job_running:
            status = "interrupted"        # its process ended without finishing (stopped, closed, crashed)
        out.append({"step": n, "title": STEPS[n].title, "status": status,
                    "finished_utc": r.get("finished_utc"), "summary": json.loads(r["summary"]) if r.get("summary") else None,
                    "error": (r.get("error") or "").strip().splitlines()[-1][:300] if r.get("error") else None})
    return out


def frames_sample(cfg, n=4000):
    import pandas as pd
    p = os.path.join(cfg.paths.step(2), "selection", "frames.parquet")
    if not os.path.exists(p):
        return {"points": [], "total": 0}
    F = pd.read_parquet(p, columns=["platform", "lon", "lat"])
    s = F.sample(min(n, len(F)), random_state=1) if len(F) else F
    return {"total": len(F), "points": [[round(a, 6), round(b, 6), c[0]] for a, b, c in zip(s.lat, s.lon, s.platform)]}


def read_csv_rows(p):
    import pandas as pd
    if not os.path.exists(p):
        return []
    return json.loads(pd.read_csv(p).to_json(orient="records"))


# ---- selected area (Settings > What to download)
def selected_area_view(cfg):
    """The saved download area, its size, and how many captures of the Step 1 harvest fall inside it."""
    import pandas as pd
    import shapely
    from shapely.geometry import mapping
    from shapely.ops import transform
    g = cfg.selected_area_ll
    out = {"geometry": None if g is None else mapping(g), "boundary": mapping(cfg.boundary_ll),
           "km2": None if g is None else round(transform(cfg.to_m, g).area / 1e6, 2), "counts": None}
    if g is None:
        return out
    shapely.prepare(g)
    s1 = cfg.paths.step(1)
    counts = {}
    mp = os.path.join(s1, "mapillary", "captures.parquet")
    if cfg.get("platforms.mapillary.enabled", False) and os.path.exists(mp):
        m = pd.read_parquet(mp, columns=["lon", "lat"])
        counts["mapillary"] = {"inside": int(shapely.contains_xy(g, m.lon.to_numpy(), m.lat.to_numpy()).sum()), "all": len(m)}
    kp = os.path.join(s1, "kartaview", "photos.parquet")
    if cfg.get("platforms.kartaview.enabled", False) and os.path.exists(kp):
        k = pd.read_parquet(kp, columns=["lng", "lat", "in_area", "status", "visibility"])
        k = k[k.in_area & k.status.eq("active") & k.visibility.eq("public")]
        counts["kartaview"] = {"inside": int(shapely.contains_xy(g, k.lng.to_numpy(), k.lat.to_numpy()).sum()), "all": len(k)}
    out["counts"] = counts or None
    return out


def save_selected_area(cfg, path, geometry):
    from shapely.geometry import mapping, shape
    if not geometry:
        if os.path.exists(cfg.selected_area_file):
            os.replace(cfg.selected_area_file, cfg.selected_area_file + ".removed")
        return {"saved": False, "removed": True}
    g = shape(geometry)
    if g.is_empty or g.geom_type not in ("Polygon", "MultiPolygon"):
        raise ApiError(400, "the area must be a polygon")
    if not g.intersects(cfg.boundary_ll):
        raise ApiError(400, "the area lies outside the project's boundary")
    g = g.intersection(cfg.boundary_ll) if not cfg.boundary_ll.contains(g) else g
    os.makedirs(os.path.dirname(cfg.selected_area_file), exist_ok=True)
    json.dump({"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {"saved_utc": now()},
                                                          "geometry": mapping(g)}]}, open(cfg.selected_area_file, "w", encoding="utf8"))
    if not cfg.get("step2_regions.selected_area", None):
        set_setting(path, "step2_regions.selected_area", "00_setup/selected_area.geojson")
    return {"saved": True, "clipped_to_boundary": not shape(geometry).equals(g)}


# ---- review: selection (strata mode)
def review_selection(cfg):
    import pandas as pd
    d = cfg.paths.step(2)
    out = {"mode": cfg.selection_mode}
    gj = os.path.join(d, "cells.geojson")
    cs = os.path.join(d, "cells_strata.parquet")
    if not (os.path.exists(gj) and os.path.exists(cs)):
        return {**out, "ready": False}
    C = pd.read_parquet(cs)
    st = [c[2:] for c in C.columns if c.startswith("s_")]
    keep = ["cell", "label", "valid", "rules_met", "street_km", "share_ge3", "share_own_both", "seq_mly", "seq_kv"] + [f"s_{k}" for k in st]
    info = {r["cell"]: r for r in json.loads(C[keep].to_json(orient="records"))}
    g = json.load(open(gj, encoding="utf8"))
    feats = [{"type": "Feature", "geometry": f["geometry"], "properties": info.get(f["properties"]["cell"], {"cell": f["properties"]["cell"]})}
             for f in g["features"]]
    acc = os.path.join(d, "selection_accepted.csv")
    return {**out, "ready": True, "strata": st, "cells": {"type": "FeatureCollection", "features": feats},
            "draft": read_csv_rows(os.path.join(d, "draft_selection.csv")),
            "accepted": read_csv_rows(acc) if os.path.exists(acc) else None}


def save_selection(cfg, picks):
    """picks: [{cell, stratum}]. Rows are taken from the draft where the cell is in it, else built from cells_strata."""
    import pandas as pd
    d = cfg.paths.step(2)
    C = pd.read_parquet(os.path.join(d, "cells_strata.parquet")).set_index("cell")
    D = pd.read_csv(os.path.join(d, "draft_selection.csv")) if os.path.exists(os.path.join(d, "draft_selection.csv")) else pd.DataFrame()
    rows = []
    for p in picks:
        cell, stratum = str(p["cell"]), str(p["stratum"])
        if cell not in C.index:
            raise ApiError(400, f"unknown cell {cell}")
        hit = D[(D.cell == cell) & (D.stratum == stratum)] if len(D) else D
        if len(hit):
            rows.append(hit.iloc[0].to_dict())
        else:
            r = C.loc[cell]
            rows.append({"stratum": stratum, "cell": cell, "label": r.label, "lon": r.lon, "lat": r.lat, "score": None,
                         "street_km": r.street_km, "share_ge3": r.share_ge3, "share_own_both": r.share_own_both,
                         "seq_mly": r.seq_mly, "seq_kv": r.seq_kv, "contributors": r.contributors, "manual": True})
    P = pd.DataFrame(rows)
    P.to_csv(os.path.join(d, "selection_accepted.csv"), index=False)
    return {"saved": len(P)}


# ---- review: mis-stitched panoramas
def review_misstitched(cfg):
    import pandas as pd
    d = cfg.paths.step(3)
    p = os.path.join(d, "pano_screen.parquet")
    if not os.path.exists(p):
        return {"ready": False}
    R = pd.read_parquet(p)
    O = R[R.coverage_outlier]
    acc = os.path.join(d, "mis_stitched_accepted.csv")
    A = pd.read_csv(acc, dtype=str) if os.path.exists(acc) else None
    med = R[~R.broken].groupby(["platform", "sequence_id"])[["coverage_top_deg", "coverage_bottom_deg"]].median()
    rows = []
    for r in O.itertuples():
        m = med.loc[(r.platform, r.sequence_id)] if (r.platform, r.sequence_id) in med.index else None
        rows.append({"platform": r.platform, "sequence_id": r.sequence_id, "frame_id": r.frame_id,
                     "top": round(r.coverage_top_deg, 1), "bottom": round(r.coverage_bottom_deg, 1),
                     "seq_top": None if m is None else round(float(m.coverage_top_deg), 1),
                     "seq_bottom": None if m is None else round(float(m.coverage_bottom_deg), 1),
                     "accepted": True if A is None else bool(((A.frame_id == r.frame_id) & (A.sequence_id == r.sequence_id)).any())})
    return {"ready": True, "outliers": rows, "decided": A is not None, "panoramas": len(R)}


def save_misstitched(cfg, frames):
    import pandas as pd
    P = pd.DataFrame([{"platform": f["platform"], "sequence_id": f["sequence_id"], "frame_id": f["frame_id"]} for f in frames],
                     columns=["platform", "sequence_id", "frame_id"])
    P.to_csv(os.path.join(cfg.paths.step(3), "mis_stitched_accepted.csv"), index=False)
    return {"saved": len(P), "note": "rerun step 3f (and then 3g onwards) to apply"}


# ---- review: capture class
def review_classes(cfg):
    import pandas as pd
    from ..steps.s3_views import capture_class
    mp = cfg.paths.manifest
    if not os.path.exists(mp):
        return {"ready": False}
    M = pd.read_parquet(mp)
    if "pano_status" not in M:
        return {"ready": True, "sequences": []}
    P = M[M.pano_status.notna()]
    ov = cfg.get("step3_download.capture_class.overrides", {}) or {}
    rows = []
    for (pf, seq), g in P.groupby(["platform", "sequence_id"]):
        ok = g[g.pano_status.eq("ok")].sort_values("order")
        sample = ok.frame_id.iloc[[0, len(ok) // 2, -1]].tolist() if len(ok) >= 3 else ok.frame_id.tolist()
        rows.append({"platform": pf, "sequence_id": seq, "device": str(g.device.iloc[0]), "panoramas": len(g),
                     "class": capture_class(cfg, pf, seq, g.device.iloc[0]), "override": ov.get(seq),
                     "sample": [f"{pf}/{seq}/{f}.jpg" for f in sample]})
    return {"ready": True, "sequences": rows, "classes": ["vehicle", "two_wheeler", "on_foot"]}


def image_bytes(cfg, root_name, rel, width):
    from PIL import Image
    from ..steps.s3_download import root
    if root_name not in ("images", "panoramas", "working", "views"):
        raise ApiError(400, "bad root")
    parts = rel.replace("\\", "/").split("/")
    if len(parts) != 3 or not all(re.match(r"^[A-Za-z0-9_\-\.]+$", p) and p not in ("..", ".") for p in parts) or not parts[2].endswith(".jpg"):
        raise ApiError(400, "bad path")
    p = os.path.join(root(cfg, root_name), *parts)
    if not os.path.exists(p):
        raise ApiError(404, "no such image")
    with Image.open(p) as im:
        im = im.convert("RGB")
        if width and im.width > width:
            im = im.resize((width, round(im.height * width / im.width)))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=82)
    return buf.getvalue()


# ---------------------------------------------------------------- HTTP
class Handler(BaseHTTPRequestHandler):
    ws: Workspace = None
    server_version = "collect-ui"

    def log_message(self, fmt, *args):  # quiet
        pass

    def _send(self, status, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else (json.dumps(body, default=str).encode() if ctype == "application/json" else body.encode())
        self.send_response(status)
        self.send_header("Content-Type", ctype + ("; charset=utf-8" if ctype.startswith(("text", "application/json")) else ""))
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(data)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}") if n else {}

    def _host_ok(self):
        h = (self.headers.get("Host") or "").split(":")[0]
        return h in ("127.0.0.1", "localhost")

    def do_GET(self):
        self._route("GET")

    def do_POST(self):
        self._route("POST")

    def do_PUT(self):
        self._route("PUT")

    def _route(self, method):
        try:
            if not self._host_ok():
                raise ApiError(403, "local use only")
            if method != "GET" and self.headers.get("X-Collect") != "1":
                raise ApiError(403, "missing X-Collect header")
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            parts = [p for p in u.path.split("/") if p]
            if method == "GET" and (not parts or parts[0] in ("index.html",)):
                return self._send(200, _page(), "text/html")
            # files shipped with the tool (Leaflet lives in static/leaflet/, its icons in static/leaflet/images/)
            if method == "GET" and parts[0] == "static" and 2 <= len(parts) <= 4 \
                    and all(SAFE.match(x.replace(".", "_")) for x in parts[1:]):
                p = os.path.join(HERE, "static", *parts[1:])
                if not os.path.exists(p):
                    raise ApiError(404, "not found")
                return self._send(200, open(p, "rb").read(), mimetypes.guess_type(p)[0] or "application/octet-stream")
            if parts[0] != "api":
                raise ApiError(404, "not found")
            return self._send(200, self._api(method, parts[1:], q))
        except ApiError as e:
            return self._send(e.status, {"error": e.msg})
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            return self._send(500, {"error": f"{type(e).__name__}: {e}"})

    def _api(self, m, p, q):
        ws = self.ws
        if p == ["projects"] and m == "GET":
            return {"workspace": ws.folder, "projects": ws.list()}
        if p == ["projects"] and m == "POST":
            b = self._body()
            pid = b.get("id", "")
            if not SAFE.match(pid):
                raise ApiError(400, "folder name: letters, digits, - and _ only")
            # where to save it: the workspace (default) or another existing folder, e.g. a larger disk
            loc = (b.get("location") or "").strip().strip('"')
            if loc and os.path.normcase(os.path.abspath(loc)) != os.path.normcase(ws.folder):
                if not os.path.isabs(loc):
                    raise ApiError(400, "location: give a full path, e.g. E:\\StreetImagery")
                if not os.path.isdir(loc):
                    raise ApiError(400, f"location {loc} does not exist: create the folder first")
            else:
                loc = ws.folder
            if pid in ws.elsewhere() or os.path.exists(os.path.join(ws.folder, pid, "project.yaml")):
                raise ApiError(409, f"a project called {pid} is already listed: choose another folder name")
            folder = os.path.join(os.path.abspath(loc), pid)
            if os.path.exists(os.path.join(folder, "project.yaml")):
                raise ApiError(409, "a project with that folder exists")
            from ..setup import init_project
            boundary = None
            if b.get("boundary_geojson"):
                os.makedirs(folder, exist_ok=True)
                boundary = os.path.join(folder, "uploaded_boundary.geojson")
                json.dump(b["boundary_geojson"], open(boundary, "w", encoding="utf8"))
            try:
                path, c = init_project(b.get("name") or pid, folder, b.get("place") or None, boundary,
                                       tuple(b.get("platforms") or ("mapillary", "kartaview")), b.get("budget_gb") or None,
                                       b.get("country_code"))
            except ValueError as e:
                raise ApiError(400, str(e)) from e
            if os.path.normcase(os.path.dirname(folder)) != os.path.normcase(ws.folder):
                ws.register(pid, folder)          # listed on the Projects page although it lives elsewhere
            return {"id": pid, "area": c["area"], "folder": folder}
        if p == ["folders"]:
            # the page's folder browser for "Save in" (a web page cannot read real folder paths itself). Folder names
            # only, never files; only for the page (X-Collect), like every other request that touches the disk.
            if self.headers.get("X-Collect") != "1":
                raise ApiError(403, "missing X-Collect header")
            if m == "GET":
                return list_folders(q.get("path", ""))
            if m == "POST":
                b = self._body()
                return make_folder(b.get("path", ""), b.get("name", ""))
        if p == ["geocode"] and m == "GET":
            # only for the page (it sends X-Collect); the query goes to OpenStreetMap's Nominatim, one request a second
            if self.headers.get("X-Collect") != "1":
                raise ApiError(403, "missing X-Collect header")
            from ..setup import search_places
            qq = (q.get("q") or "").strip()
            if len(qq) < 2:
                raise ApiError(400, "type a place name")
            try:
                return {"results": search_places(qq, 8)}
            except Exception as e:  # noqa: BLE001
                raise ApiError(502, f"place search failed: {type(e).__name__}: {e}") from e
        if p == ["token"] and m == "GET":
            from ..secrets import token_source
            return {"source": token_source()}
        if p == ["token"] and m == "POST":
            from ..secrets import store_token
            tok = (self._body().get("token") or "").strip()
            if not tok:
                raise ApiError(400, "empty token")
            store_token(tok)
            return {"stored": True}
        if len(p) >= 2 and p[0] == "p":
            pid, rest = p[1], p[2:]
            path = ws.path(pid)
            if rest == ["overview"]:
                return overview(ws.cfg(pid))
            if rest == ["config"] and m == "GET":
                c = ws.cfg(pid)
                return {"yaml": project_raw(path), "effective": c.data}
            if rest == ["config"] and m == "PUT":
                try:
                    save_project_yaml(path, self._body().get("yaml", ""))
                except (ConfigError, yaml.YAMLError) as e:
                    raise ApiError(400, f"not saved: {e}") from e
                return {"saved": True}
            if rest == ["setting"] and m == "POST":
                b = self._body()
                try:
                    set_setting(path, b["key"], b.get("value"))
                except (ConfigError, yaml.YAMLError) as e:
                    raise ApiError(400, f"not saved: {e}") from e
                return {"saved": True}
            if rest == ["steps"]:
                js = job_state(pid)
                return {"steps": steps_view(ws.cfg(pid), js["state"] == "running"), "job": js}
            if rest == ["run"] and m == "POST":
                b = self._body()
                return start_job(ws, pid, b.get("steps") or ["all"], bool(b.get("offline")), bool(b.get("force")))
            if rest == ["job"]:
                return job_view(ws.cfg(pid), pid, int(q.get("n", 200)))
            if rest == ["preflight"]:
                from ..diagnose import preflight
                return preflight(ws.cfg(pid))
            if rest == ["doctor"]:
                # one light request to each web service; only for the page (it sends X-Collect)
                if self.headers.get("X-Collect") != "1":
                    raise ApiError(403, "missing X-Collect header")
                from ..doctor import checks
                return {"checks": checks(ws.cfg(pid))}
            if rest == ["job", "stop"] and m == "POST":
                return stop_job(pid)
            if rest == ["report"]:
                c = ws.cfg(pid)
                rp = os.path.join(c.paths.step(6), "report.md")
                return {"report": open(rp, encoding="utf8").read() if os.path.exists(rp) else None,
                        "estimate": read_csv_rows(os.path.join(c.paths.step(3), "storage_estimate.csv"))}
            if rest == ["selected_area"]:
                c = ws.cfg(pid)
                if m == "POST":
                    r = save_selected_area(c, path, self._body().get("geometry"))
                    return {**r, **selected_area_view(ws.cfg(pid))}
                return selected_area_view(c)
            if rest == ["frames"]:
                return frames_sample(ws.cfg(pid))
            if rest == ["review", "selection"]:
                c = ws.cfg(pid)
                return save_selection(c, self._body().get("picks", [])) if m == "POST" else review_selection(c)
            if rest == ["review", "mis_stitched"]:
                c = ws.cfg(pid)
                return save_misstitched(c, self._body().get("frames", [])) if m == "POST" else review_misstitched(c)
            if rest == ["review", "classes"]:
                c = ws.cfg(pid)
                if m == "POST":
                    ov = {k: v for k, v in (self._body().get("overrides") or {}).items() if v in ("vehicle", "two_wheeler", "on_foot")}
                    set_setting(path, "step3_download.capture_class.overrides", ov)
                    return {"saved": len(ov), "note": "rerun steps 3h and 3i (delete the views of changed sequences first)"}
                return review_classes(c)
        raise ApiError(404, "no such API")


def _drives() -> list[str]:
    if os.name != "nt":
        return ["/"]
    import string
    return [f"{d}:\\" for d in string.ascii_uppercase if os.path.isdir(f"{d}:\\")]


def list_folders(path: str) -> dict:
    """Sub-folders of `path` (names only), its parent, and the drives. An empty path lists the drives."""
    path = (path or "").strip().strip('"')
    if not path:
        return {"path": "", "parent": None, "folders": [], "drives": _drives()}
    path = os.path.abspath(path)
    if not os.path.isdir(path):
        raise ApiError(404, f"{path} is not a folder")
    try:
        names = sorted((e.name for e in os.scandir(path) if e.is_dir(follow_symlinks=False)
                        and not e.name.startswith(("$", ".")) and e.name not in ("System Volume Information",)),
                       key=str.lower)
    except PermissionError:
        raise ApiError(403, f"no permission to open {path}") from None
    up = os.path.dirname(path)
    parent = "" if up == path else up           # a drive root: "up" goes to the drive list
    return {"path": path, "parent": parent, "folders": names, "drives": _drives(),
            "writable": os.access(path, os.W_OK)}


def make_folder(path: str, name: str) -> dict:
    """Create the folder `name` inside `path` (for "Save in"): a plain name, no path separators."""
    name = (name or "").strip()
    if not name or not re.match(r"^[\w .\-()]+$", name) or name in (".", "..") or name.endswith((" ", ".")):
        raise ApiError(400, "folder name: letters, digits, spaces, - _ . ( ) only")
    path = os.path.abspath((path or "").strip().strip('"'))
    if not os.path.isdir(path):
        raise ApiError(404, f"{path} is not a folder")
    new = os.path.join(path, name)
    if os.path.exists(new):
        raise ApiError(409, f"{new} already exists")
    try:
        os.makedirs(new)
    except OSError as e:
        raise ApiError(400, f"could not create {new}: {e.strerror}") from None
    return list_folders(new)


def _page() -> str:
    """index.html with the background-map tile address from the settings (services.map_tiles)."""
    from ..config import service
    html = open(os.path.join(HERE, "index.html"), encoding="utf8").read()
    return html.replace("__MAP_TILES__", service(None, "map_tiles").replace('"', "%22"))


def serve(workspace, port=8765, open_browser=False):
    class H(Handler):
        pass
    H.ws = Workspace(workspace)
    WS_REF["ws"] = H.ws

    def route(self, method, _orig=Handler._route):
        try:
            if not self._host_ok():
                raise ApiError(403, "local use only")
            if method != "GET" and self.headers.get("X-Collect") != "1":
                raise ApiError(403, "missing X-Collect header")
            u = urlparse(self.path)
            parts = [x for x in u.path.split("/") if x]
            if len(parts) >= 4 and parts[0] == "api" and parts[1] == "p" and parts[3] == "img":
                q = {k: v[0] for k, v in parse_qs(u.query).items()}
                data = image_bytes(H.ws.cfg(parts[2]), q.get("root", ""), q.get("path", ""), int(q.get("w", 640)))
                return self._send(200, data, "image/jpeg")
        except ApiError as e:
            return self._send(e.status, {"error": e.msg})
        return _orig(self, method)
    H._route = route
    httpd = ThreadingHTTPServer(("127.0.0.1", port), H)
    url = f"http://127.0.0.1:{port}/"
    print(f"collect ui on {url} (workspace {H.ws.folder}); Ctrl+C to stop", flush=True)
    if open_browser:
        import webbrowser
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
