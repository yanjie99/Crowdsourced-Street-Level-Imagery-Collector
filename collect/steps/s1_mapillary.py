"""Step 1, Mapillary (ported 18 Nov 2026 from step1_metadata/mapillary/*.py; logic unchanged).

  1a tiles      every z14 coverage tile over the harvest box, saved unmodified; captures inside the boundary
                -> mapillary/tiles/, captures.parquet, harvest_log.json
  1b sequences  one row per sequence (moving captures, times, track, shares) plus the device of its first and
                last image from the Graph API -> sequences.parquet, sequences.csv, device_counts.csv, qa_summary.txt
"""
from __future__ import annotations

import json
import math
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute  # noqa: F401
import pyarrow.parquet as pq
import shapely

from .. import progress
from ..config import Config
from ..config import graph_url
from ..net import OfflineMiss, now


def out_dir(cfg: Config) -> str:
    d = os.path.join(cfg.paths.step(1), "mapillary")
    os.makedirs(d, exist_ok=True)
    return d


def lonlat_to_tile(lon, lat, z):
    n = 2 ** z
    return int((lon + 180) / 360 * n), int((1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n)


def tile_px_to_lonlat(x, y, z, px, py, extent):
    """mapbox_vector_tile decodes with y pointing up (origin bottom-left)."""
    n = 2 ** z
    fx, fy = x + px / extent, y + (1 - py / extent)
    return fx / n * 360 - 180, math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * fy / n))))


def tiles(cfg: Config, log, offline=False, **_):
    import mapbox_vector_tile
    m = cfg.get("platforms.mapillary")
    z, ua, bbox = m.get("tile_zoom", 14), cfg.get("platforms.user_agent"), cfg.harvest_bbox
    out = out_dir(cfg)
    tdir = cfg.cache_dir(1, "mapillary/tiles")
    poly = cfg.harvest_ll
    shapely.prepare(poly)
    x0, y0 = lonlat_to_tile(bbox[0], bbox[3], z)
    x1, y1 = lonlat_to_tile(bbox[2], bbox[1], z)
    tl = [(x, y) for x in range(x0, x1 + 1) for y in range(y0, y1 + 1)]
    started = now()
    log.info(f"{len(tl)} tiles z{z} x {x0}-{x1} y {y0}-{y1}")
    token = None

    def fetch(t):
        nonlocal token
        x, y = t
        path = os.path.join(tdir, f"{z}_{x}_{y}.mvt")
        if os.path.exists(path):
            return t, "cached", os.path.getsize(path)
        if offline:
            raise OfflineMiss(path)
        if token is None:
            from ..secrets import get_token
            token = get_token()
        url = m["tiles_url"].format(z=z, x=x, y=y) + f"?access_token={token}"
        last = None
        for attempt in range(5):
            try:
                with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": ua}), timeout=120) as r:
                    data = r.read()
                with open(path + ".part", "wb") as f:
                    f.write(data)
                os.replace(path + ".part", path)
                return t, "ok", len(data)
            except urllib.error.HTTPError as e:
                last = f"HTTP {e.code}"
                if e.code in (400, 401, 403):
                    break
            except Exception as e:  # noqa: BLE001
                last = type(e).__name__
            time.sleep(3 * (attempt + 1))
        return t, f"failed: {last}", 0

    status = {}
    progress.phase("coverage tiles", len(tl), "tiles")
    with ThreadPoolExecutor(m.get("workers", 4)) as ex:
        for i, (t, st, size) in enumerate(ex.map(fetch, tl), 1):
            status[f"{t[0]}_{t[1]}"] = {"status": st, "bytes": size}
            progress.update(done=i)
            if i % 200 == 0:
                log.info(f"tiles {i}/{len(tl)}")
    failed = [k for k, v in status.items() if v["status"].startswith("failed")]
    if tl and len(failed) == len(tl):
        why = sorted({v["status"] for v in status.values()})
        raise RuntimeError(f"every Mapillary coverage tile failed ({', '.join(why)}) from tiles.mapillary.com")

    names = ("id", "sequence_id", "captured_at", "lon", "lat", "compass_angle", "creator_id", "organization_id",
             "is_pano", "foot", "quality_score", "tile")
    cols = {k: [] for k in names}
    seen, dup = set(), 0
    for (x, y) in tl:
        path = os.path.join(tdir, f"{z}_{x}_{y}.mvt")
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            continue
        img = mapbox_vector_tile.decode(open(path, "rb").read()).get("image")
        if not img:
            continue
        extent = img.get("extent", 4096)
        for f in img["features"]:
            p = f["properties"]
            iid = p.get("id")
            if iid in seen:
                dup += 1
                continue
            seen.add(iid)
            lon, lat = tile_px_to_lonlat(x, y, z, *f["geometry"]["coordinates"], extent)
            for k, v in (("id", int(iid)), ("sequence_id", p.get("sequence_id")), ("captured_at", p.get("captured_at")),
                         ("lon", lon), ("lat", lat), ("compass_angle", p.get("compass_angle")),
                         ("creator_id", p.get("creator_id")), ("organization_id", p.get("organization_id")),
                         ("is_pano", p.get("is_pano")), ("foot", p.get("foot")),
                         ("quality_score", p.get("quality_score")), ("tile", f"{x}_{y}")):
                cols[k].append(v)
    n_all = len(cols["id"])
    inside = shapely.contains_xy(poly, cols["lon"], cols["lat"])
    keep = [i for i in range(n_all) if inside[i]]
    sel = lambda k: [cols[k][i] for i in keep]  # noqa: E731
    table = pa.table({
        "id": pa.array(sel("id"), pa.int64()), "sequence_id": pa.array(sel("sequence_id"), pa.string()),
        "captured_at_ms": pa.array(sel("captured_at"), pa.int64()), "lon": pa.array(sel("lon"), pa.float64()),
        "lat": pa.array(sel("lat"), pa.float64()), "compass_angle": pa.array(sel("compass_angle"), pa.float64()),
        "creator_id": pa.array(sel("creator_id"), pa.int64()), "organization_id": pa.array(sel("organization_id"), pa.int64()),
        "is_pano": pa.array(sel("is_pano"), pa.bool_()), "foot": pa.array(sel("foot"), pa.bool_()),
        "quality_score": pa.array(sel("quality_score"), pa.float64()), "tile": pa.array(sel("tile"), pa.string())})
    table = table.append_column("captured_at_utc", pa.compute.cast(table["captured_at_ms"], pa.timestamp("ms", tz="UTC")))
    pq.write_table(table, os.path.join(out, "captures.parquet"))
    hl = {"platform": "Mapillary", "harvest_started_utc": started, "harvest_finished_utc": now(),
          "endpoint": m["tiles_url"], "api_version": "v4 coverage tiles, mly1_public/2", "layer": "image", "zoom": z,
          "bbox": bbox, "tiles_requested": len(tl), "tiles_failed": failed, "tile_status": status,
          "features_decoded": n_all + dup, "duplicates_dropped": dup, "captures_outside_polygon": n_all - len(keep),
          "captures_kept": len(keep), "license": m.get("license")}
    json.dump(hl, open(os.path.join(out, "harvest_log.json"), "w", encoding="utf8"), indent=1)
    log.info(f"{len(keep)} captures inside the boundary from {len(tl)} tiles ({len(failed)} failed)")
    return [os.path.join(out, "captures.parquet"), os.path.join(out, "harvest_log.json")], \
        {"tiles": len(tl), "tiles_failed": len(failed), "captures": len(keep)}


def _devices(cfg, out, cap, offline, log):
    """Device of the first and last image of every sequence (Graph API, batches cached in graph_devices/)."""
    m = cfg.get("platforms.mapillary")
    fields, batch = "make,model,camera_type,width,height,sequence", m.get("graph_batch", 50)
    raw = cfg.cache_dir(1, "mapillary/graph_devices")
    c = cap.sort_values(["sequence_id", "captured_at_ms", "id"])
    g = c.groupby("sequence_id")
    ends = pd.DataFrame({"first_id": g["id"].first(), "last_id": g["id"].last()})
    ids = sorted(set(ends["first_id"].astype(str)) | set(ends["last_id"].astype(str)))
    batches = [ids[i:i + batch] for i in range(0, len(ids), batch)]
    hdr = None

    def get(url):
        nonlocal hdr
        if hdr is None:
            from ..secrets import auth_header
            hdr = auth_header()
        with urllib.request.urlopen(urllib.request.Request(url, headers=hdr), timeout=120) as r:
            return json.loads(r.read())

    def fetch(a):
        i, ids_ = a
        path = os.path.join(raw, f"batch_{i:05d}.json")
        if os.path.exists(path):
            return json.load(open(path, encoding="utf8"))
        if offline:
            raise OfflineMiss(path)
        for attempt in range(5):
            try:
                data = get(f"{graph_url(cfg)}/?ids={','.join(ids_)}&fields={fields}")
                json.dump(data, open(path, "w", encoding="utf8"))
                return data
            except urllib.error.HTTPError as e:
                if e.code in (400, 404) and len(ids_) > 1:   # one unavailable id fails the batch: one id per call
                    o = {}
                    for one in ids_:
                        try:
                            o[one] = get(f"{graph_url(cfg)}/{one}?fields={fields}")
                        except Exception:  # noqa: BLE001
                            o[one] = {"error": "unavailable"}
                    json.dump(o, open(path, "w", encoding="utf8"))
                    return o
            except Exception:  # noqa: BLE001
                pass
            time.sleep(3 * (attempt + 1))
        return {one: {"error": "failed"} for one in ids_}

    res = {}
    progress.phase("device lookups", len(batches), "batches")
    with ThreadPoolExecutor(4) as ex:
        for data in ex.map(fetch, enumerate(batches)):
            res.update(data)
            progress.update(add=1)
    log.info(f"devices for {len(ends)} sequences from {len(batches)} Graph batches")
    return ends, res


def sequences(cfg: Config, log, offline=False, **_):
    out = out_dir(cfg)
    s1 = cfg.get("step1_harvest")
    move_m, min_moving, tz = s1["move_m"], s1["min_moving_captures"], cfg.get("area.timezone")
    df = pd.read_parquet(os.path.join(out, "captures.parquet")).sort_values(["sequence_id", "captured_at_ms", "id"]).reset_index(drop=True)
    lat, lon = np.radians(df["lat"].to_numpy()), np.radians(df["lon"].to_numpy())
    same = df["sequence_id"].to_numpy()[1:] == df["sequence_id"].to_numpy()[:-1]
    a = np.sin((lat[1:] - lat[:-1]) / 2) ** 2 + np.cos(lat[1:]) * np.cos(lat[:-1]) * np.sin((lon[1:] - lon[:-1]) / 2) ** 2
    step = np.where(same, 2 * 6371008.8 * np.arcsin(np.sqrt(a)), np.nan)
    df["step_m"] = np.concatenate([[np.nan], step])
    df["moving"] = df["step_m"] >= move_m
    g = df.groupby("sequence_id", sort=False)
    seq = pd.DataFrame({
        "platform": "mapillary", "creator_id": g["creator_id"].first(), "creators_in_sequence": g["creator_id"].nunique(),
        "organization_id": g["organization_id"].first(), "capture_count": g.size(), "moving_count": g["moving"].sum(),
        "first_time_utc": g["captured_at_utc"].min(), "last_time_utc": g["captured_at_utc"].max(),
        "track_length_m": g["step_m"].sum(min_count=1).fillna(0.0),
        "lon_min": g["lon"].min(), "lat_min": g["lat"].min(), "lon_max": g["lon"].max(), "lat_max": g["lat"].max(),
        "pano_share": g["is_pano"].mean(), "foot_share": g["foot"].mean(),
        "quality_score_mean": g["quality_score"].mean()}).reset_index()
    seq["duration_s"] = (seq["last_time_utc"] - seq["first_time_utc"]).dt.total_seconds()
    seq["first_date_local"] = seq["first_time_utc"].dt.tz_convert(tz).dt.date.astype(str)
    seq["short_flag"] = seq["moving_count"] < min_moving

    ends, res = _devices(cfg, out, df[["id", "sequence_id", "captured_at_ms"]], offline, log)
    get = lambda i, f: (res.get(str(i)) or {}).get(f)  # noqa: E731
    seq = seq.set_index("sequence_id")
    e = ends.reindex(seq.index)
    for f in ("make", "model", "camera_type", "width", "height"):
        seq[f] = [get(i, f) for i in e["first_id"]]
    lm = [(get(i, "make"), get(i, "model")) for i in e["last_id"]]
    seq["device_changes_in_sequence"] = [(a, b) != c and None not in (a, b) and None not in c
                                         for a, b, c in zip(seq["make"], seq["model"], lm)]
    seq["device_lookup_failed"] = [get(i, "make") is None and get(i, "model") is None for i in e["first_id"]]
    seq["device"] = (seq["make"].fillna("?").astype(str) + " " + seq["model"].fillna("?").astype(str)).str.strip()
    seq = seq.reset_index()
    seq.to_parquet(os.path.join(out, "sequences.parquet"), index=False)
    seq.to_csv(os.path.join(out, "sequences.csv"), index=False)
    cnt = seq.groupby("device").agg(sequences=("sequence_id", "size"), captures=("capture_count", "sum"))
    cnt.sort_values("captures", ascending=False).to_csv(os.path.join(out, "device_counts.csv"))

    cap = df
    L = [f"captures: {len(cap):,}",
         f"sequences: {len(seq):,}  (short, <{min_moving} moving captures: {int(seq['short_flag'].sum()):,})",
         f"creators: {cap['creator_id'].nunique():,}   organizations: {cap['organization_id'].nunique():,}",
         f"capture time: {cap['captured_at_utc'].min()} to {cap['captured_at_utc'].max()}",
         f"panorama captures: {cap['is_pano'].mean():.1%}   pedestrian (foot) captures: {cap['foot'].mean():.1%}",
         "captures by year: " + ", ".join(f"{y}: {n:,}" for y, n in cap["captured_at_utc"].dt.year.value_counts().sort_index().items()),
         "local hour: " + ", ".join(f"{h}:{n:,}" for h, n in cap["captured_at_utc"].dt.tz_convert(tz).dt.hour.value_counts().sort_index().items()),
         f"device lookup failed: {int(seq['device_lookup_failed'].sum())} sequences; distinct devices: {seq['device'].nunique()}"]
    open(os.path.join(out, "qa_summary.txt"), "w", encoding="utf8").write("\n".join(L) + "\n")
    log.info(f"{len(seq)} Mapillary sequences, {int(seq.short_flag.sum())} short")
    return [os.path.join(out, "sequences.parquet")], {"sequences": len(seq), "short": int(seq.short_flag.sum()),
                                                      "devices": int(seq.device.nunique())}
