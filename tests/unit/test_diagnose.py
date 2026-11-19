"""Error explanations, progress records and stall detection (no network needed)."""
import json
from datetime import datetime, timedelta, timezone

from collect import progress
from collect.diagnose import explain, stall


def test_known_errors_are_explained():
    cases = {
        "RuntimeError: step 1b needs step 1a to have run successfully first": ("needs_step", "1a"),
        "RuntimeError: No Mapillary token: run `collect token set`": ("no_token", "token"),
        "RuntimeError: every Mapillary coverage tile failed (failed: HTTP 401) from tiles.mapillary.com": ("token_rejected", "token"),
        "urllib.error.URLError: <urlopen error [Errno 11001] getaddrinfo failed>": ("network", "internet"),
        "collect.net.OfflineMiss: /sequence/1/photos": ("offline_miss", "offline"),
        "RuntimeError: no square of the 9 meets all three plan rules": ("strata_none", "Whole area"),
        "OSError: [Errno 28] No space left on device": ("disk_full", "space"),
        "PermissionError: [WinError 32] The process cannot access the file because it is being used by another process":
            ("in_use", "Close"),
        "pyarrow.lib.ArrowInvalid: Parquet magic bytes not found": ("bad_file", "force"),
        "RuntimeError: 13_32: ohsome gave up after 5 attempts": ("ohsome", "5c"),
        "KeyError: 'setting step5_osm.match not in project.yaml'": ("setting_missing", "step5_osm.match"),
    }
    for text, (rule, word) in cases.items():
        d = explain("3b", text)
        assert d["rule"] == rule, (text, d["rule"])
        assert word.lower() in (d["cause"] + d["fix"]).lower(), (rule, d)


def test_unknown_error_falls_back():
    d = explain("2b", "Traceback ...\nZeroDivisionError: division by zero")
    assert d["rule"] == "general" and "division by zero" in d["cause"]


def test_progress_file(tmp_path):
    progress.start(str(tmp_path), "3b", "Download images and records")
    progress.phase("mapillary downloads", 10, "sequences")
    progress.update(done=4)
    progress.finish("done")
    p = progress.read(str(tmp_path))
    assert p["step"] == "3b" and p["state"] == "done" and p["done"] == 10 and p["total"] == 10


def test_in_use_names_the_file():
    d = explain("1e", r"PermissionError: [WinError 5] Access is denied: 'D:\\p\\logs\\a.tmp' -> 'D:\\p\\out\\photos.parquet'")
    assert d["rule"] == "in_use" and d["cause"].endswith(r"File: D:\p\out\photos.parquet")


def test_progress_survives_a_locked_file(tmp_path, monkeypatch):
    """The Run page (or a viewer) holding progress.json open must not fail the step."""
    import os as _os
    real = _os.replace

    def locked(a, b):
        if str(b).endswith("progress.json"):
            raise PermissionError(5, "Access is denied")
        return real(a, b)
    progress.start(str(tmp_path), "1e", "KartaView photo lists")
    monkeypatch.setattr(progress.os, "replace", locked)
    progress.phase("photo lists", 100, "sequences")      # forced write: falls back to writing in place
    for i in range(5):
        progress.update(done=i)                          # throttled writes: skipped quietly
    progress.finish("done")
    assert progress.skipped() >= 2
    assert progress.read(str(tmp_path))["state"] == "done"


def test_stall_detection():
    old = (datetime.now(timezone.utc) - timedelta(minutes=15)).isoformat(timespec="seconds")
    new = datetime.now(timezone.utc).isoformat(timespec="seconds")
    assert stall({"state": "running", "updated_utc": old, "step": "3b"}, True)["title"].startswith("No progress for 15")
    assert stall({"state": "running", "updated_utc": new, "step": "3b"}, True) is None
    assert stall({"state": "running", "updated_utc": old}, False) is None
