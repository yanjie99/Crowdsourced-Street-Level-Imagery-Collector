"""Step 6: storage, checks and fetch log for the whole project (new for the app, 18 Nov 2026; the plan's Step 6).

  6a checks  - completeness: every selected frame has a status; files on disk for every ok frame (image, working
               copy, panorama); views: every usable view has its JPEG and its mask in masks.zip, and no file lacks a record
             - storage: size of each data folder and free disk space, against the budget
             - fetch log: requests, failures and licenses per platform from the steps' logs
             - the run ledger: last run of every step
             -> 06_checks/checks.json, checks_problems.csv, report.md (a readable run report with the settings,
                results and the credit lines the platforms require)
Read-only: it changes no data.
"""
from __future__ import annotations

import glob
import json
import os
import shutil
import zipfile
from datetime import datetime, timezone

import pandas as pd

from ..config import Config
from ..ledger import Ledger
from .s3_download import root

CREDITS = {"mapillary": "Images © Mapillary, CC BY-SA 4.0. No unapproved data mining, unblurring or re-identification.",
           "kartaview": "Images © Grab and KartaView Contributors, CC BY-SA 4.0.",
           "osm": "Map data © OpenStreetMap contributors, ODbL (Geofabrik extract; history via ohsome)."}


def _du(path):
    n = b = 0
    for r, _, fs in os.walk(path):
        for f in fs:
            try:
                b += os.path.getsize(os.path.join(r, f))
                n += 1
            except OSError:
                pass
    return n, b


def checks(cfg: Config, log, deep=True, **_):
    out = cfg.paths.step(6)
    problems, rep = [], {"checked_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                         "project": cfg.get("project.name", ""), "mode": cfg.selection_mode}
    # ---- completeness of the download
    mp = cfg.paths.manifest
    if os.path.exists(mp):
        M = pd.read_parquet(mp)
        rep["frames"] = {"selected": len(M), "by_platform_status": M.groupby(["platform", "status"]).size().unstack(fill_value=0).to_dict("index")}
        nd = M[M.status.eq("not_downloaded")]
        if len(nd):
            problems.append({"check": "frame has no download status", "count": len(nd),
                             "example": f"{nd.platform.iloc[0]}/{nd.sequence_id.iloc[0]}/{nd.frame_id.iloc[0]}"})
        if deep:
            ok = M[M.status.eq("ok")]
            for name, rt in (("image", root(cfg, "images")), ("working copy", root(cfg, "working"))):
                miss = [f for f in ok.file if not os.path.exists(os.path.join(rt, *f.split("/")))]
                rep.setdefault("files_missing", {})[name] = len(miss)
                if miss:
                    problems.append({"check": f"{name} missing for an ok frame", "count": len(miss), "example": miss[0]})
            if "pano_status" in M:
                po = M[M.pano_status.eq("ok")]
                miss = [f for f in po.file if not os.path.exists(os.path.join(root(cfg, "panoramas"), *f.split("/")))]
                rep["files_missing"]["panorama"] = len(miss)
                if miss:
                    problems.append({"check": "panorama missing for an ok panorama", "count": len(miss), "example": miss[0]})
        for f in ("check_problems.csv", "index_gaps.csv", "no_reconstruction.csv"):
            p = os.path.join(cfg.paths.step(3), f)
            if os.path.exists(p):
                rep.setdefault("step3_lists", {})[f] = int(len(pd.read_csv(p)))
    else:
        problems.append({"check": "no manifest (run step 3g)", "count": 1, "example": mp})
    # ---- views: records, JPEGs and masks agree
    vr = root(cfg, "views")
    vp = glob.glob(os.path.join(vr, "*", "*", "views.parquet"))
    if vp:
        nv = nu = bad = 0
        for p in vp:
            d = os.path.dirname(p)
            V = pd.read_parquet(p)
            u = V[V.usable]
            nv, nu = nv + len(V), nu + len(u)
            files = {f for f in os.listdir(d) if f.endswith(".jpg")}
            want = {os.path.basename(f) for f in u.file}
            names = set(zipfile.ZipFile(os.path.join(d, "masks.zip")).namelist()) if os.path.exists(os.path.join(d, "masks.zip")) else set()
            if files != want or names != set(u.mask_member):
                bad += 1
                problems.append({"check": "views: JPEGs or masks do not match views.parquet", "count": 1,
                                 "example": os.path.relpath(d, vr)})
        rep["views"] = {"sequences": len(vp), "views": nv, "usable": nu, "sequences_mismatched": bad}
    # ---- storage
    st = {}
    for name in ("images", "panoramas", "working", "views"):
        p = root(cfg, name)
        n, b = _du(p) if deep else (None, None)
        st[name] = {"folder": p, "files": n, "gb": None if b is None else round(b / 1e9, 2)}
    total = sum(v["gb"] or 0 for v in st.values())
    free = shutil.disk_usage(cfg.folder).free / 1e9
    budget = cfg.get("storage.budget_gb")
    rep["storage"] = {"folders": st, "total_gb": round(total, 1), "budget_gb": budget, "free_gb": round(free, 1),
                      "within_budget": total <= budget}
    if total > budget:
        problems.append({"check": "images use more than the storage budget", "count": 1, "example": f"{total:.1f} GB > {budget} GB"})
    # ---- fetch logs
    fl = {}
    for pf in ("mapillary", "kartaview"):
        p = os.path.join(root(cfg, "images"), pf, "fetch_log.jsonl")
        if os.path.exists(p):
            E = [json.loads(x) for x in open(p, encoding="utf8") if x.strip()]
            fl[pf] = {"entries": len(E), "first_utc": min((e.get("started_utc") for e in E), default=None),
                      "last_utc": max((e.get("finished_utc") for e in E), default=None),
                      "licenses": sorted({e.get("license") for e in E if e.get("license")})}
    for name, p in (("step1 mapillary tiles", os.path.join(cfg.paths.step(1), "mapillary", "harvest_log.json")),
                    ("step1 kartaview sweep", os.path.join(cfg.paths.step(1), "kartaview", "sweep_log.json")),
                    ("step5 ohsome history", os.path.join(cfg.paths.step(5), "history_log.json"))):
        if os.path.exists(p):
            j = json.load(open(p, encoding="utf8"))
            fl[name] = {k: j.get(k) for k in ("requests", "retries", "license", "data_to", "api_version") if k in j}
            if j.get("failures"):
                fl[name]["failures"] = len(j["failures"])
    rep["fetch_logs"] = fl
    # ---- ledger
    led = [r for r in Ledger(cfg.paths.ledger).all_last() if r["step"] != "6a"]   # not this run itself
    rep["steps"] = {r["step"]: {"status": r["status"], "finished_utc": r["finished_utc"]} for r in led}
    failed = [r["step"] for r in led if r["status"] != "ok"]
    if failed:
        problems.append({"check": "steps whose last run did not finish ok", "count": len(failed), "example": ", ".join(failed)})
    rep["problems"] = len(problems)
    P = pd.DataFrame(problems, columns=["check", "count", "example"])
    P.to_csv(os.path.join(out, "checks_problems.csv"), index=False)
    json.dump(rep, open(os.path.join(out, "checks.json"), "w", encoding="utf8"), indent=1, default=str)
    _report(cfg, rep, P, os.path.join(out, "report.md"))
    log.info(f"checks: {len(problems)} problem type(s); images {total:.1f} GB of {budget} GB; {free:.0f} GB free")
    return [os.path.join(out, "checks.json"), os.path.join(out, "report.md")], \
        {"problems": len(problems), "total_gb": round(total, 1), "within_budget": total <= budget}


def _report(cfg, rep, P, path):
    L = [f"# Run report: {rep['project']}", "", f"Checked {rep['checked_utc']}. Step 2 mode: {rep['mode']}.", "",
         "## Area and settings", "",
         f"- Boundary: {cfg.get('area.boundary_file')} ({cfg.get('area.country_code')}), time zone {cfg.get('area.timezone')}, "
         f"metric CRS {cfg.get('area.metric_crs')}",
         f"- Platforms: " + ", ".join(p for p in ("mapillary", "kartaview") if cfg.get(f"platforms.{p}.enabled", False)),
         f"- Image sizes: {cfg.get('step3_download.image_sizes', {})}",
         f"- OSM extract: {os.path.basename(cfg.osm_extract)} ({cfg.osm_extract_date})", "",
         "## Steps", "", "| Step | Last run | Finished (UTC) |", "|---|---|---|"]
    L += [f"| {k} | {v['status']} | {v['finished_utc'] or ''} |" for k, v in rep["steps"].items()]
    if "frames" in rep:
        L += ["", "## Frames", "", "| Platform | " + " | ".join(sorted({s for d in rep["frames"]["by_platform_status"].values() for s in d})) + " |"]
        cols = sorted({s for d in rep["frames"]["by_platform_status"].values() for s in d})
        L += ["|---" * (len(cols) + 1) + "|"]
        L += [f"| {pf} | " + " | ".join(f"{d.get(c, 0):,}" for c in cols) + " |" for pf, d in rep["frames"]["by_platform_status"].items()]
    if "views" in rep:
        v = rep["views"]
        L += ["", f"Views: {v['usable']:,} usable of {v['views']:,} in {v['sequences']} sequences; "
                  f"{v['sequences_mismatched']} sequence(s) with files that do not match their records."]
    s = rep["storage"]
    L += ["", "## Storage", "", "| Folder | Files | GB |", "|---|---|---|"]
    L += [f"| {k} | {v['files'] if v['files'] is not None else ''} | {v['gb'] if v['gb'] is not None else ''} |" for k, v in s["folders"].items()]
    L += ["", f"Total {s['total_gb']} GB against a budget of {s['budget_gb']} GB; {s['free_gb']} GB free on the disk."]
    if rep.get("fetch_logs"):
        L += ["", "## Fetch logs", ""]
        for k, v in rep["fetch_logs"].items():
            L += [f"- {k}: " + "; ".join(f"{a} {b}" for a, b in v.items() if b not in (None, [], ""))]
    L += ["", "## Problems", ""]
    L += ["None."] if P.empty else [f"- {r.check}: {r.count} (e.g. {r.example})" for r in P.itertuples()]
    L += ["", "## Credits and licenses", ""]
    L += [f"- {CREDITS[k]}" for k in ("mapillary", "kartaview") if cfg.get(f"platforms.{k}.enabled", False)] + [f"- {CREDITS['osm']}"]
    open(path, "w", encoding="utf8").write("\n".join(L) + "\n")
