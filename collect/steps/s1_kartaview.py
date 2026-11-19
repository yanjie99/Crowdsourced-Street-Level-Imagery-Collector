"""Step 1, KartaView (ported 18 Nov 2026 from step1_metadata/kartaview/*.py).

  1c sweep      photo radius query on a grid over the area, then each sequence's record -> sweep_points.csv,
                sweep_hits.parquet, sequences_raw.parquet, sweep_log.json
  1d discover   sequences missed by the sweep: the country listing, every listing of each contributor seen
                (no page cap), and sequences with no bounding box located from their first photo
                -> sequences.parquet (found_by), discover_log.json
                (merges discover_extra.py and discover_fix.py: the old 5,000-page cap and its later fix are gone)
  1e photos     every photo of every sequence, with in_area -> photos.parquet, harvest_photos_sequences.csv
  1f table      one row per sequence over its active, public, in-area photos -> sequence_table.parquet, qa_summary.txt
One polite client for all of them: one request at a time, at most platforms.kartaview.per_s a second.
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import pandas as pd
import shapely
from shapely.geometry import Point, box
from shapely.ops import transform

from .. import progress
from ..config import Config
from ..net import Client, now, platform_throttle

KEYS = ["id", "status", "userId", "deviceName", "platformName", "platformVersion", "appVersion",
        "countActivePhotos", "dateAdded", "dateProcessed", "nwLat", "nwLng", "seLat", "seLng",
        "distance", "hasRawData", "metadataFileUrl", "isVideo", "sequenceType", "uploadSource",
        "orgCode", "countryCode", "address", "fieldOfView", "storage", "imageProcessingStatus",
        "matched", "obdInfo"]
FIELDS = ["id", "sequenceId", "sequenceIndex", "lat", "lng", "heading", "shotDate", "dateAdded",
          "gpsAccuracy", "projection", "fieldOfView", "width", "height", "status", "visibility",
          "matchLat", "matchLng", "wayId", "matchSegmentId", "qualityStatus", "qualityLevel",
          "autoImgProcessingResult", "isWrapped", "isUnwrapped", "orgCode", "storage", "filepath",
          "fileurl", "fileurlLTh", "fileurlProc", "videoId", "videoIndex", "rawDataId"]
NUMERIC = ["sequenceIndex", "lat", "lng", "heading", "gpsAccuracy", "fieldOfView", "width", "height",
           "matchLat", "matchLng", "qualityLevel", "videoIndex"]
NUM_SEQ = ("countActivePhotos", "distance", "nwLat", "nwLng", "seLat", "seLng")


def out_dir(cfg: Config) -> str:
    d = os.path.join(cfg.paths.step(1), "kartaview")
    os.makedirs(d, exist_ok=True)
    return d


def raw(cfg: Config, sub: str, name: str) -> str:
    """Cache file of one raw response: <kartaview/raw>/<sub>/<name> (the raw root can be mapped in `caches`)."""
    d = os.path.join(cfg.cache_dir(1, "kartaview/raw"), sub)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, name)


def client(cfg: Config, offline=False, log=None) -> Client:
    k = cfg.get("platforms.kartaview")
    return Client(k["api"], k["per_s"], cfg.get("platforms.user_agent"), offline=offline,
                  throttle=platform_throttle(cfg, "kartaview", log=log))


# ---------------------------------------------------------------- 1c sweep
def sweep(cfg: Config, log, offline=False, **_):
    k = cfg.get("platforms.kartaview")
    spacing, radius, bbox = k["sweep_spacing_m"], k["sweep_radius_m"], cfg.harvest_bbox
    out, cl, started = out_dir(cfg), client(cfg, offline, log), now()
    bx0, by0 = cfg.to_m(bbox[0], bbox[1])
    bx1, by1 = cfg.to_m(bbox[2], bbox[3])
    gx, gy = np.meshgrid(np.arange(bx0, bx1 + spacing, spacing), np.arange(by0, by1 + spacing, spacing))
    gx, gy = gx.ravel(), gy.ravel()
    zone = cfg.harvest_m.buffer(radius)
    shapely.prepare(zone)
    keep = shapely.contains_xy(zone, gx, gy)
    gx, gy = gx[keep], gy[keep]
    lon, lat = cfg.to_ll(gx, gy)
    pts = pd.DataFrame({"point": np.arange(len(gx)), "x_m": gx, "y_m": gy, "lon": lon, "lat": lat})
    # Cache files are named by the point's position (19 Nov 2026). Before, they were named by its number in the grid
    # (p00003_1), so a changed area or box read another place's answer. Files with the old names are still read, but
    # only for a point at the same position in the grid that wrote them (that grid's sweep_points.csv, next to the raw
    # cache or the step folder; read before this run replaces it).
    legacy = {}
    for f in (os.path.join(os.path.dirname(cfg.cache_dir(1, "kartaview/raw")), "sweep_points.csv"),
              os.path.join(out, "sweep_points.csv")):
        if os.path.exists(f):
            old = pd.read_csv(f)
            legacy = {(f"{a:.6f}", f"{b:.6f}"): int(n) for n, a, b in zip(old.point, old.lat, old.lon)}
            break
    pts.to_csv(os.path.join(out, "sweep_points.csv"), index=False)

    def cache_name(p, page):
        old = legacy.get((f"{p.lat:.6f}", f"{p.lon:.6f}"))
        if old is not None and os.path.exists(raw(cfg, "sweep", f"p{old:05d}_{page}.json.gz")):
            return raw(cfg, "sweep", f"p{old:05d}_{page}.json.gz")
        return raw(cfg, "sweep", f"at_{p.lat:.6f}_{p.lon:.6f}_r{radius}_{page}.json.gz")
    log.info(f"{len(pts)} grid points, spacing {spacing} m, radius {radius} m")
    hits, pages_total = [], 0
    progress.phase("grid points", len(pts), "points")
    for kk, p in pts.iterrows():
        progress.update(done=kk)
        page = 1
        while True:
            q = f"/photo/?lat={p.lat:.6f}&lng={p.lon:.6f}&radius={radius}&itemsPerPage=100&page={page}"
            data = cl.cached(cache_name(p, page), q)
            pages_total += 1
            if data is None or "result" not in data:
                break
            for r in data["result"].get("data") or []:
                hits.append({"point": int(p.point), "sequence_id": str(r.get("sequenceId")), "photo_id": str(r.get("id")),
                             "lat": r.get("lat"), "lng": r.get("lng"), "shotDate": r.get("shotDate"),
                             "distance": r.get("distance")})
            if not data["result"].get("hasMoreData"):
                break
            page += 1
        if (kk + 1) % 500 == 0:
            log.info(f"sweep {kk + 1}/{len(pts)} points, {cl.stats['requests']} requests")
    H = pd.DataFrame(hits)
    H.to_parquet(os.path.join(out, "sweep_hits.parquet"), index=False)
    seq_ids = sorted(H["sequence_id"].unique(), key=int) if len(H) else []
    rows = []
    progress.phase("sequence records", len(seq_ids), "sequences")
    for sid in seq_ids:
        progress.update(add=1)
        data = cl.cached(raw(cfg, "sequence", f"{sid}.json.gz"), f"/sequence/{sid}")
        d = ((data or {}).get("result") or {}).get("data") or {}
        rows.append({**{kk: d.get(kk) for kk in KEYS}, "sequence_id": sid, "fetch_ok": bool(d)})
    S = pd.DataFrame(rows, columns=KEYS + ["sequence_id", "fetch_ok"])
    for c in NUM_SEQ:
        S[c] = pd.to_numeric(S[c], errors="coerce")
    S.to_parquet(os.path.join(out, "sequences_raw.parquet"), index=False)
    sl = {"platform": "KartaView", "harvest_started_utc": started, "harvest_finished_utc": now(),
          "grid": {"crs": cfg.get("area.metric_crs"), "spacing_m": spacing, "radius_m": radius, "points": len(pts), "bbox": bbox},
          "rate_limit_req_per_s": k["per_s"], **cl.stats, "sweep_pages": pages_total, "sequences": len(seq_ids),
          "sequence_records_ok": int(S["fetch_ok"].sum()) if len(S) else 0, "license": k.get("license")}
    json.dump(sl, open(os.path.join(out, "sweep_log.json"), "w", encoding="utf8"), indent=1)
    log.info(f"sweep: {len(seq_ids)} sequences from {pages_total} pages")
    return [os.path.join(out, "sequences_raw.parquet")], {"points": len(pts), "sequences": len(seq_ids),
                                                          "requests": cl.stats["requests"]}


# ---------------------------------------------------------------- 1d discover
def _touches(poly, r):
    try:
        la = [float(r.get("nwLat")), float(r.get("seLat"))]
        lo = [float(r.get("nwLng")), float(r.get("seLng"))]
    except (TypeError, ValueError):
        return None
    la = [v for v in la if -90 <= v <= 90]
    lo = [v for v in lo if -180 <= v <= 180]
    if not la or not lo:
        return None
    return bool(poly.intersects(box(min(lo), min(la), max(lo), max(la))))


def _zero_box(r):
    try:
        return all(abs(float(r.get(k) or 0)) < 1e-9 for k in ("nwLat", "nwLng", "seLat", "seLng"))
    except (TypeError, ValueError):
        return True


def _list_all(cfg, cl, tag, query, max_pages):
    rows, page = [], 1
    while True:
        data = cl.cached(raw(cfg, "list", f"{tag}_p{page:05d}.json.gz"),
                         f"/sequence/?{query}&itemsPerPage=100&page={page}")
        r = (data or {}).get("result") or {}
        rows += r.get("data") or []
        if not r.get("hasMoreData") or page >= max_pages:
            return rows, page, bool(r.get("hasMoreData"))
        page += 1


def discover(cfg: Config, log, offline=False, **_):
    k = cfg.get("platforms.kartaview")
    out, cl, cc = out_dir(cfg), client(cfg, offline, log), cfg.get("area.country_code")
    # without the platform's bulk approval, keep contributor listings light (a bulk account can have 5,000+ pages)
    max_pages = k.get("max_list_pages", 40000 if k.get("bulk_access_approved") else 50)
    poly = cfg.harvest_ll
    near = poly.buffer(k["listing_buffer_deg"])
    shapely.prepare(poly)
    shapely.prepare(near)
    sweep_ = pd.read_parquet(os.path.join(out, "sequences_raw.parquet"))
    found = {sid: {"sweep"} for sid in sweep_.sequence_id}
    records = dict(zip(sweep_.sequence_id, sweep_.to_dict("records")))
    dl = {"started_utc": now()}

    def add(r, how, **extra):
        sid = str(r["id"])
        found.setdefault(sid, set()).add(how)
        records.setdefault(sid, {**{kk: r.get(kk) for kk in KEYS}, "sequence_id": sid, "fetch_ok": True, **extra})

    ctry, pages, _ = _list_all(cfg, cl, f"country_{cc}", f"countryCode={cc}", max_pages)
    # The country listing covers the whole country, which can be far larger than the study area (a 2 km box in Japan
    # listed 9,388 sequences, none inside). Keep a listed sequence only if its bounding box touches the boundary; one
    # with no usable box is located from its first photo page (the same page 1e reads first, so it costs no extra request).
    c_kept = c_out = c_none = c_located = 0
    for r in ctry:
        t = _touches(poly, r)
        if t:
            c_kept += 1
            add(r, f"country_{cc}")
            continue
        if t is False and not _zero_box(r):
            c_out += 1
            continue
        c_none += 1
        if r.get("status") not in (None, "active") or int(r.get("countActivePhotos") or 0) <= 0:
            continue
        sid = str(r["id"])
        data = cl.cached(raw(cfg, "photos", f"{sid}_p00001.json.gz"), f"/sequence/{sid}/photos?itemsPerPage=100&page=1")
        ph = ((data or {}).get("result") or {}).get("data") or []
        try:
            lat, lng = float(ph[0]["lat"]), float(ph[0]["lng"])
        except (IndexError, TypeError, ValueError, KeyError):
            continue
        if near.contains(Point(lng, lat)):
            c_located += 1
            add(r, f"country_{cc}", first_lat=lat, first_lng=lng)
    dl["country"] = {"code": cc, "sequences": len(ctry), "pages": pages, "box_touches": c_kept, "box_elsewhere": c_out,
                     "no_box": c_none, "no_box_located_near": c_located}
    log.info(f"country listing {cc}: {len(ctry)} sequences, {c_kept + c_located} near the boundary")
    # contributors seen in the sweep or with a country-listed sequence near the boundary (not every contributor in the country)
    near_ids = {sid for sid, how in found.items() if f"country_{cc}" in how}
    users = sorted({str(u) for u in sweep_.userId.dropna()}
                   | {str(r.get("userId")) for r in ctry if r.get("userId") and str(r["id"]) in near_ids}, key=int)
    no_box, dl["users"] = [], {}
    progress.phase("contributor listings", len(users), "contributors")
    for i, u in enumerate(users, 1):
        progress.update(done=i - 1)
        rows, pages, more = _list_all(cfg, cl, f"user_{u}", f"userId={u}", max_pages)
        kept = 0
        for r in rows:
            t = _touches(poly, r)
            if t is None:
                if str(r.get("countryCode")) == cc:
                    t = True
                else:
                    if r.get("status") == "active" and _zero_box(r) and int(r.get("countActivePhotos") or 0) > 0:
                        no_box.append(r)
                    continue
            if t is False and _zero_box(r) and r.get("status") == "active" and int(r.get("countActivePhotos") or 0) > 0:
                no_box.append(r)
                continue
            if t:
                kept += 1
                add(r, "user_list")
        dl["users"][u] = {"listed": len(rows), "kept": kept, "pages": pages, "still_has_more": more}
        if i % 50 == 0:
            log.info(f"contributor listings {i}/{len(users)}")
    located = kept = 0
    for r in {str(r["id"]): r for r in no_box}.values():
        sid = str(r["id"])
        if sid in records:
            continue
        data = cl.cached(raw(cfg, "locate", f"{sid}.json.gz"), f"/sequence/{sid}/photos?itemsPerPage=1&page=1")
        ph = ((data or {}).get("result") or {}).get("data") or []
        try:
            lat, lng = float(ph[0]["lat"]), float(ph[0]["lng"])
        except (IndexError, TypeError, ValueError, KeyError):
            continue
        located += 1
        if near.contains(Point(lng, lat)):
            kept += 1
            add(r, "user_list_located", first_lat=lat, first_lng=lng)
    S = pd.DataFrame(records.values())
    S["found_by"] = S.sequence_id.map(lambda s: ";".join(sorted(found[s])))
    for c in NUM_SEQ:
        S[c] = pd.to_numeric(S[c], errors="coerce")
    S = S[S.status.eq("active") | S.status.isna()]
    S.to_parquet(os.path.join(out, "sequences.parquet"), index=False)
    dl.update(finished_utc=now(), **cl.stats, no_box_with_photos=len({str(r["id"]) for r in no_box}),
              no_box_located=located, no_box_kept=kept, sequences_total=len(S),
              found_by=S.found_by.value_counts().to_dict())
    json.dump(dl, open(os.path.join(out, "discover_log.json"), "w", encoding="utf8"), indent=1)
    log.info(f"{len(S)} KartaView sequences after discovery ({kept} located from their first photo)")
    return [os.path.join(out, "sequences.parquet")], {"sequences": len(S), "located_kept": kept,
                                                      "requests": cl.stats["requests"]}


# ---------------------------------------------------------------- 1e photos
def photos(cfg: Config, log, offline=False, **_):
    out, cl = out_dir(cfg), client(cfg, offline, log)
    avail = cfg.get("existing.annotated_photo_pages", None)
    avail = cfg.resolve(avail) if avail else None
    poly = cfg.harvest_ll
    shapely.prepare(poly)
    seq = pd.read_parquet(os.path.join(out, "sequences.parquet"))
    seq["sequence_id"] = seq.sequence_id.astype(str)
    seq["n"] = pd.to_numeric(seq.countActivePhotos, errors="coerce").fillna(0)
    reported = dict(zip(seq.sequence_id, seq.n))
    ids = list(seq.sort_values("n").sequence_id)
    started, per_seq, frames = now(), [], []
    progress.phase("photo lists", len(ids), "sequences")

    def one(sid):
        """Every photo page of one sequence (several sequences are fetched at once; one throttle limits them all)."""
        pages, source = [], "api"
        if avail:
            files = glob.glob(os.path.join(avail, "raw", f"photos_{sid}_p*.json"))
            files.sort(key=lambda f: int(f.rsplit("_p", 1)[1].split(".")[0]))
            pages = [json.load(open(f, encoding="utf8")) for f in files]
            source = "annotated set"
        complete = bool(pages) and not ((pages[-1].get("result") or {}).get("hasMoreData"))
        if not complete:
            pages, complete, page, source = [], False, 1, "api"
            while page <= 5000:
                data = cl.cached(raw(cfg, "photos", f"{sid}_p{page:05d}.json.gz"),
                                 f"/sequence/{sid}/photos?itemsPerPage=100&page={page}")
                if data is None:
                    break
                pages.append(data)
                if not ((data.get("result") or {}).get("hasMoreData")):
                    complete = True
                    break
                page += 1
        rows = [{kk: p.get(kk) for kk in FIELDS} for d in pages for p in ((d.get("result") or {}).get("data") or [])]
        return (pd.DataFrame(rows, columns=FIELDS) if rows else None,
                {"sequence_id": sid, "pages": len(pages), "photos": len(rows), "reported": int(reported[sid]),
                 "complete": complete, "source": source})

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(int(cfg.get("platforms.kartaview.workers", 6))) as ex:
        for i, (fr, ps) in enumerate(ex.map(one, ids), 1):     # results in the original order: same output as before
            progress.update(done=i)
            if fr is not None:
                frames.append(fr)
            per_seq.append(ps)
            if i % 500 == 0:
                log.info(f"photo lists {i}/{len(ids)}, {cl.stats['requests']} requests")
    P = pd.concat(frames, ignore_index=True).drop_duplicates("id") if frames else pd.DataFrame(columns=FIELDS)
    for c in NUMERIC:
        P[c] = pd.to_numeric(P[c], errors="coerce")
    P = P.rename(columns={"id": "photo_id", "sequenceId": "sequence_id"})
    ok = P.lat.between(-90, 90) & P.lng.between(-180, 180) & ~((P.lat == 0) & (P.lng == 0))
    P["in_area"] = False
    P.loc[ok, "in_area"] = shapely.contains_xy(poly, P.loc[ok, "lng"].to_numpy(), P.loc[ok, "lat"].to_numpy())
    P = P.sort_values(["sequence_id", "sequenceIndex"])
    P.to_parquet(os.path.join(out, "photos.parquet"), index=False)
    S = pd.DataFrame(per_seq)
    S.to_csv(os.path.join(out, "harvest_photos_sequences.csv"), index=False)
    hl = {"started_utc": started, "finished_utc": now(), "sequences": len(S), "sequences_complete": int(S.complete.sum()),
          "sequences_count_mismatch": int((S.photos != S.reported).sum()), "photos": len(P),
          "photos_in_area": int(P.in_area.sum()), **cl.stats}
    json.dump(hl, open(os.path.join(out, "harvest_photos_log.json"), "w", encoding="utf8"), indent=1)
    log.info(f"{len(P)} photos, {int(P.in_area.sum())} inside the boundary")
    res = {"photos": len(P), "in_area": int(P.in_area.sum()), "requests": cl.stats["requests"]}
    if not res["in_area"]:
        res["note"] = ("KartaView has no photos inside the boundary: the later steps carry on with Mapillary only "
                       "(untick KartaView under Settings to skip its steps)")
        log.warning(res["note"])
    return [os.path.join(out, "photos.parquet")], res


# ---------------------------------------------------------------- 1f sequence table
def table(cfg: Config, log, **_):
    out = out_dir(cfg)
    s1 = cfg.get("step1_harvest")
    move_m, min_moving, tz = s1["move_m"], s1["min_moving_captures"], cfg.get("area.timezone")
    P = pd.read_parquet(os.path.join(out, "photos.parquet"))
    all_counts = P.groupby("sequence_id").size().rename("photos_listed_all")
    P = P[P.in_area & P.status.eq("active") & P.visibility.eq("public")].copy()
    P["t"] = pd.to_datetime(P.shotDate, utc=True, errors="coerce")
    P = P.sort_values(["sequence_id", "sequenceIndex", "photo_id"]).reset_index(drop=True)
    lat, lon = np.radians(P.lat.to_numpy()), np.radians(P.lng.to_numpy())
    same = P.sequence_id.to_numpy()[1:] == P.sequence_id.to_numpy()[:-1]
    a = np.sin((lat[1:] - lat[:-1]) / 2) ** 2 + np.cos(lat[1:]) * np.cos(lat[:-1]) * np.sin((lon[1:] - lon[:-1]) / 2) ** 2
    P["step_m"] = np.concatenate([[np.nan], np.where(same, 2 * 6371008.8 * np.arcsin(np.sqrt(a)), np.nan)])
    P["moving"] = P.step_m >= move_m
    P["gps_acc"] = P.gpsAccuracy.where(P.gpsAccuracy > 0)
    P["sphere"] = P.projection.ne("PLANE")
    g = P.groupby("sequence_id", sort=False)
    seq = pd.DataFrame({
        "platform": "kartaview", "capture_count": g.size(), "moving_count": g.moving.sum(),
        "first_time_utc": g.t.min(), "last_time_utc": g.t.max(), "track_length_m": g.step_m.sum(min_count=1).fillna(0.0),
        "lon_min": g.lng.min(), "lat_min": g.lat.min(), "lon_max": g.lng.max(), "lat_max": g.lat.max(),
        "pano_share": g.sphere.mean(), "projections": g.projection.agg(lambda s: ";".join(sorted(s.dropna().unique()))),
        "gps_accuracy_missing_share": g.gps_acc.apply(lambda s: s.isna().mean()),
        "gps_accuracy_median_m": g.gps_acc.median(), "photo_width": g.width.median(), "photo_height": g.height.median(),
        "shotdate_missing": g.t.apply(lambda s: int(s.isna().sum()))}).reset_index()
    seq["duration_s"] = (seq.last_time_utc - seq.first_time_utc).dt.total_seconds()
    seq["first_date_local"] = seq.first_time_utc.dt.tz_convert(tz).dt.date.astype(str)
    seq["short_flag"] = seq.moving_count < min_moving
    S = pd.read_parquet(os.path.join(out, "sequences.parquet"))
    S["sequence_id"] = S.sequence_id.astype(str)
    keep = ["sequence_id", "userId", "deviceName", "platformName", "platformVersion", "appVersion", "countActivePhotos",
            "dateAdded", "hasRawData", "metadataFileUrl", "isVideo", "sequenceType", "uploadSource", "orgCode",
            "countryCode", "fieldOfView", "found_by"]
    S = S[[c for c in keep if c in S.columns]].rename(columns={
        "userId": "creator_id", "deviceName": "device", "countActivePhotos": "photos_reported",
        "dateAdded": "date_added", "countryCode": "country_code"})
    seq = seq.merge(S, on="sequence_id", how="left").merge(all_counts, on="sequence_id", how="left")
    seq["clipped_share"] = 1 - seq.capture_count / seq.photos_listed_all
    seq.to_parquet(os.path.join(out, "sequence_table.parquet"), index=False)
    seq.to_csv(os.path.join(out, "sequence_table.csv"), index=False)
    L = [f"captures (active, public, in area): {len(P):,}",
         f"sequences: {len(seq):,}  (short, <{min_moving} moving captures: {int(seq.short_flag.sum()):,})",
         f"creators: {seq.creator_id.nunique():,}",
         "captures by year: " + ", ".join(f"{int(y)}: {n:,}" for y, n in P.t.dt.year.value_counts().sort_index().items()),
         "local hour: " + ", ".join(f"{int(h)}:{n:,}" for h, n in P.t.dt.tz_convert(tz).dt.hour.value_counts().sort_index().items())]
    open(os.path.join(out, "qa_summary.txt"), "w", encoding="utf8").write("\n".join(L) + "\n")
    log.info(f"{len(seq)} KartaView sequences with photos in the area")
    return [os.path.join(out, "sequence_table.parquet")], {"sequences": len(seq), "captures": len(P),
                                                           "short": int(seq.short_flag.sum())}
