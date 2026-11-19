"""Explain failures and check readiness (added 18 Nov 2026).

explain(step, error_text) turns a failed step's error (message and traceback) into
  {"title", "cause", "fix", "rule"}: what went wrong, why, and what to try. The rules cover the problems met while
building and testing the tool (network, tokens, rate limits, missing inputs, disk, files in use, selection rules);
anything else gets a general explanation with the exception's own message.
preflight(cfg) lists what would stop or weaken a run before it starts: blocking problems and warnings.
stall(progress, job) says whether a running step has stopped reporting progress.
"""
from __future__ import annotations

import os
import re
import shutil
from datetime import datetime, timezone

# (rule id, regex on the error text, title, cause, fix). First match wins; {0}, {1} ... take the regex groups.
RULES = [
    ("needs_step", r"needs step ([\w| or]+?) to have run", "An earlier step has not finished",
     "This step uses the output of step {0}, which has not run successfully yet.",
     "Run step {0} first, or use Run all: it runs the steps in order and skips the finished ones."),
    ("no_token", r"No Mapillary token", "No Mapillary token",
     "The Mapillary steps need an access token and none is stored.",
     "Create a token on Mapillary's developer dashboard and store it under Settings (or `collect token set`)."),
    ("token_rejected", r"HTTP (401|403)\b.*(graph\.mapillary|tiles\.mapillary)|(graph\.mapillary|tiles\.mapillary).*HTTP (401|403)",
     "Mapillary refused the token", "Mapillary answered 401/403: the token is wrong, expired or lacks access.",
     "Store a new token under Settings, then rerun this step."),
    ("offline_miss", r"OfflineMiss|offline: not on disk|and offline", "Needed data is not on disk (offline run)",
     "The run was offline, so nothing could be fetched, and a response or file this step needs is not cached.",
     "Untick 'offline' and run again, or point `caches` at the folder that holds the earlier download."),
    ("rate_limit", r"HTTP 429|Too Many Requests", "The platform is limiting requests",
     "The server answered 429 (too many requests).",
     "Wait a while and run the step again: it resumes where it stopped. Lower the platform's rate in Settings if it repeats."),
    ("server_error", r"HTTP 5\d\d|Bad Gateway|Service Unavailable|Gateway Time", "The platform's server had a problem",
     "A server answered with an error (5xx) after several retries.",
     "Try again later; the step resumes from what it already saved."),
    ("ohsome", r"ohsome gave up", "The ohsome history service did not answer",
     "ohsome (OpenStreetMap history) failed five times for one region; it is often busy or under maintenance.",
     "Run step 5c again later; regions already fetched are kept."),
    ("network", r"URLError|getaddrinfo|Name or service not known|ConnectionError|ConnectionResetError|RemoteDisconnected|"
                r"TimeoutError|timed out|NewConnectionError|SSLError|Max retries exceeded",
     "No connection to the server", "A request could not reach the server (no internet, a proxy, a firewall, or a timeout).",
     "Check the internet connection (and any VPN or proxy), then run the step again: it resumes."),
    ("no_extract", r"no OSM extract yet", "No OpenStreetMap extract yet",
     "The OSM steps need the dated extract that step 0a downloads.", "Run step 0a, or set osm.extract to a .osm.pbf file you have."),
    ("geofabrik_region", r"no Geofabrik region contains", "No single Geofabrik extract covers the area",
     "The boundary crosses Geofabrik's region borders.",
     "Download a larger extract (for example the whole country or continent) and set osm.extract to it."),
    ("geofabrik_dated", r"no date-stamped file", "Geofabrik's dated file was not found",
     "The region page did not list a date-stamped extract (the site may have changed or be unavailable).",
     "Try again later, or download an extract yourself and set osm.extract."),
    ("md5", r"MD5 mismatch", "The OSM extract arrived damaged", "The downloaded file does not match Geofabrik's checksum.",
     "Run step 0a again."),
    ("strata_none", r"no square of the (\d+) meets all three plan rules", "No square meets the selection rules",
     "Strata mode: none of the {0} squares passes all three plan rules (the coverage in this area is too thin for them).",
     "Relax step2_regions.rules or min_passes (and record why), or switch Selection to Whole area."),
    ("split_none", r"no split of the selected squares", "The squares cannot be split into two balanced halves",
     "No split gives each half 40-60% of each platform's frames with similar capture years.",
     "Change the selection on the Review page, or widen step2_regions.split.balance."),
    ("disk_full", r"No space left|Errno 28|not enough space|DiskFull", "The disk is full",
     "Writing failed because the disk has no free space.",
     "Free space, choose smaller image sizes or a shorter date range, or map the image folders to another disk (caches)."),
    ("in_use", r"PermissionError|being used by another process|WinError 32|Errno 13", "A file is open in another program or read-only",
     "The step could not write a file: it is open elsewhere (often Excel or a viewer) or the folder is read-only.",
     "Close the program that has the file open, check the folder's permissions, and run the step again."),
    ("web_reply", r"requests\.exceptions\.JSONDecodeError|RequestsJSONDecodeError|incomplete reply|ohsome gave up",
     "A web service sent an incomplete reply",
     "The service (for OSM history: ohsome) stopped part-way through its answer, usually because the request took longer than its time limit or the service was busy. Nothing on your disk is damaged.",
     "Run the step again; areas already fetched are kept. If it repeats, the service is likely overloaded or under maintenance: try again later, or check the service under Settings > Web services."),
    ("bad_file", r"ArrowInvalid|Parquet magic bytes|Could not open Parquet|not a gzipped file|BadGzipFile|JSONDecodeError|"
                 r"BadZipFile|EOFError", "A saved file is damaged",
     "A file written by an earlier, interrupted run cannot be read.",
     "Run the earlier step again with 'force rerun'; if it repeats, delete the damaged file named in the log."),
    ("setting_missing", r"setting ([\w.]+) not in project\.yaml", "A setting is missing",
     "The step needs the setting {0}, which is neither in project.yaml nor in the defaults.",
     "Add {0} under Settings (project.yaml)."),
    ("config", r"ConfigError|project\.yaml is missing|must be \[lon_min", "The settings are not valid",
     "project.yaml has a missing or wrong value.", "Open Settings; the message names the setting to fix."),
    ("memory", r"MemoryError|Unable to allocate", "Not enough memory",
     "The step ran out of memory.", "Lower step3_download.workers, close other programs, or run a smaller area or date range."),
    ("module", r"No module named '([\w.]+)'", "A library is missing",
     "Python could not import {0}.", "Install the package again (pip install collect_street_imagery-*.whl) so its libraries are installed."),
    ("terminated", r"returncode (-?\d+|unknown)|terminated by the user", "The run was stopped",
     "The process ended before the step finished (stopped, closed, or the computer slept).",
     "Run again: finished work is kept and the step resumes."),
]


def explain(step: str, error: str) -> dict:
    text = error or ""
    for rid, rx, title, cause, fix in RULES:
        m = re.search(rx, text, re.I | re.S)
        if m:
            g = [x for x in m.groups() if x] if m.groups() else []
            d = {"rule": rid, "step": step, "title": title, "cause": cause.format(*g) if g else cause,
                 "fix": fix.format(*g) if g else fix}
            if rid in ("in_use", "bad_file"):  # name the file, so the user knows which one to close or delete
                paths = re.findall(r"'([A-Za-z]:[\\/][^']+|/[^']+)'", text)
                if paths:
                    d["cause"] += " File: " + paths[-1].replace("\\\\", "\\")
            return d
    last = [l for l in text.strip().splitlines() if l.strip()]
    msg = last[-1].strip() if last else "unknown error"
    return {"rule": "general", "step": step, "title": f"Step {step} failed",
            "cause": f"Unexpected error: {msg[:300]}",
            "fix": "Look at the end of the log for details and run the step again. If it repeats, keep the log "
                   "(logs/<step>.log) to report the problem."}


def preflight(cfg) -> dict:
    """Problems that block a run, and warnings, before anything starts."""
    block, warn = [], []
    mly, kv = cfg.get("platforms.mapillary.enabled", False), cfg.get("platforms.kartaview.enabled", False)
    if not (mly or kv):
        block.append({"what": "No platform is enabled", "fix": "Tick Mapillary and/or KartaView under Settings."})
    if mly:
        from .secrets import token_source
        src = token_source()
        if src == "none":
            block.append({"what": "No Mapillary token is stored", "fix": "Store it under Settings (or `collect token set`)."})
        elif "environment" in src:
            # not stored in the tool: an environment variable on this computer is used instead (easy to miss)
            warn.append({"what": "The Mapillary token comes from this computer's MAPILLARY_TOKEN environment variable, "
                                 "not from Settings",
                         "fix": "The run uses that token. To use another one, store it under Settings (it then takes "
                                "precedence), or remove the variable from Windows' environment variables."})
    if "PLACEHOLDER" in str(cfg.get("step5_osm.road_standard.source", "")):
        warn.append({"what": "The road standard has generic placeholder widths",
                     "fix": "Set the national lane and line widths under Settings before using the Step 5 widths."})
    if kv and not cfg.get("platforms.kartaview.bulk_access_approved", False):
        warn.append({"what": "KartaView bulk use is not approved for this project",
                     "fix": "Requests stay light (contributor listings stop at 50 pages). Tick the box only with the platform's approval."})
    if cfg.get("area.country_code") in (None, "", "??"):
        warn.append({"what": "The country code is unknown", "fix": "Set area.country_code (KartaView's country listing uses it)."})
    try:
        free = shutil.disk_usage(cfg.folder).free / 1e9
        budget = float(cfg.get("storage.budget_gb") or 0)
        if free < 5:
            block.append({"what": f"Only {free:.1f} GB free on the project's disk", "fix": "Free space or move the project."})
        elif budget and free < budget:
            warn.append({"what": f"{free:.0f} GB free on the disk, less than the {budget:g} GB budget",
                         "fix": "Free space, lower the budget, or map the image folders to another disk (caches)."})
    except OSError:
        pass
    est = os.path.join(cfg.paths.step(3), "storage_estimate.csv")
    if os.path.exists(est):
        import pandas as pd
        E = pd.read_csv(est)
        tot = float(E.groupby("platform").head(1).est_gb.sum())
        if tot > float(cfg.get("storage.budget_gb") or 0):
            warn.append({"what": f"The storage estimate ({tot:.1f} GB) is over the budget",
                         "fix": "Narrow the date range or platforms, or choose smaller image sizes, before step 3b."})
    sel_missing = cfg.selection_mode == "selected_area" and cfg.selected_area_ll is None
    if sel_missing:
        block.append({"what": "Selection is 'selected area' but no area is drawn",
                      "fix": "Draw it under Settings > What to download, then save it. Step 1 reads only that area."})
    # the area Step 1 reads: the drawn area in selected_area mode, else the whole boundary
    km2 = (cfg.boundary_m if sel_missing else cfg.harvest_m).area / 1e6
    if km2 > 200:
        what = "selected area" if cfg.selection_mode == "selected_area" else "area"
        warn.append({"what": f"Large {what} ({km2:,.0f} km2): Step 1 reads the metadata of every capture in it",
                     "fix": "A big city can hold millions of captures (Detroit: 18 million), so Step 1 can take hours and "
                            "several GB of memory and disk. A smaller area keeps the first run quick."})
    if cfg.selection_mode == "whole_area":
        wa = cfg.get("step2_regions.whole_area", {}) or {}
        if not wa.get("date_from") and not wa.get("date_to") and cfg.boundary_m.area / 1e6 > 200:
            warn.append({"what": f"Whole area of {cfg.boundary_m.area / 1e6:,.0f} km2 without a date range",
                         "fix": "This can be a very large download: check the estimate (step 3a) or set a date range."})
    return {"blocking": block, "warnings": warn, "ok": not block}


def stall(progress: dict | None, running: bool, minutes: float = 10.0) -> dict | None:
    """A running step that has not reported for `minutes`: likely waiting on a slow or unreachable server."""
    if not running or not progress or progress.get("state") != "running":
        return None
    try:
        t = datetime.fromisoformat(progress["updated_utc"])
    except (KeyError, ValueError):
        return None
    idle = (datetime.now(timezone.utc) - t).total_seconds() / 60
    if idle < minutes:
        return None
    return {"title": f"No progress for {idle:.0f} minutes in step {progress.get('step')}",
            "cause": "The step may be waiting on a slow or unreachable server (it retries with pauses), or on a very large file.",
            "fix": "Check the last log lines. If nothing changes, stop the run and start it again: it resumes. "
                   "Check the internet connection if requests keep failing."}
