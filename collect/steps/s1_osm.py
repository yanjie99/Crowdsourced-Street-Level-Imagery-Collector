"""Step 1, both platforms (ported 18 Nov 2026 from step1_metadata/osm/*.py; logic unchanged).

  1g segments   OSM ways a camera can travel, split at junctions, at most segment_max_len_m -> segments.parquet
  1h passes     each capture matched to a segment; passes (one contributor on one local date) per segment
                -> <platform>_capture_segments.parquet, <platform>_segment_passes.parquet, <platform>_passes_log.json
  1i duplicates drives uploaded to both platforms (and twice to KartaView), combined passes counting each drive once
                -> duplicate_pairs.csv, kartaview_internal_duplicates.csv, combined_segment_passes.parquet
"""
from __future__ import annotations

import json
import math
import os
from collections import Counter
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import shapely
from shapely.geometry import LineString
from shapely.ops import substring

from .. import progress
from ..config import Config
from ..net import now


def out_dir(cfg: Config) -> str:
    d = os.path.join(cfg.paths.step(1), "osm")
    os.makedirs(d, exist_ok=True)
    return d


def _bearing(x0, y0, x1, y1):
    return (math.degrees(math.atan2(x1 - x0, y1 - y0)) + 360) % 360


# ---------------------------------------------------------------- 1g segments
def segments(cfg: Config, log, **_):
    import osmium
    keep_kinds, bbox = set(cfg.get("osm.camera_highways")), cfg.harvest_bbox_osm
    max_len = cfg.get("step1_harvest.segment_max_len_m")

    class H(osmium.SimpleHandler):
        def __init__(self):
            super().__init__()
            self.ways = []

        def way(self, w):
            hw = w.tags.get("highway")
            if hw not in keep_kinds or w.tags.get("area") == "yes":
                return
            try:
                pts = [(n.ref, n.location.lon, n.location.lat) for n in w.nodes]
            except osmium.InvalidLocationError:
                pts = [(n.ref, n.location.lon, n.location.lat) for n in w.nodes if n.location.valid()]
            if len(pts) < 2 or not any(bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3] for _, x, y in pts):
                return
            self.ways.append({"way_id": w.id, "highway": hw, "name": w.tags.get("name"), "oneway": w.tags.get("oneway"), "pts": pts})

    out, started = out_dir(cfg), now()
    pbf = cfg.osm_extract
    h = H()
    h.apply_file(pbf, locations=True)
    uses = Counter()
    for w in h.ways:
        uses.update({r for r, _, _ in w["pts"]})
    poly = cfg.harvest_m
    shapely.prepare(poly)
    rows = []
    for w in h.ways:
        pts = w["pts"]
        xs, ys = cfg.to_m([p[1] for p in pts], [p[2] for p in pts])
        cuts = [0] + [i for i in range(1, len(pts) - 1) if uses[pts[i][0]] > 1] + [len(pts) - 1]
        piece = 0
        for a, b in zip(cuts[:-1], cuts[1:]):
            line = LineString(list(zip(xs[a:b + 1], ys[a:b + 1])))
            if line.length < 0.5:
                continue
            k = max(1, math.ceil(line.length / max_len))
            for j in range(k):
                sub = substring(line, j * line.length / k, (j + 1) * line.length / k)
                if sub.geom_type != "LineString" or sub.length < 0.5:
                    continue
                if not poly.contains(sub.interpolate(0.5, normalized=True)):
                    continue
                (x0, y0), (x1, y1) = sub.coords[0], sub.coords[-1]
                rows.append({"segment_id": f"{w['way_id']}_{piece}", "way_id": w["way_id"], "piece": piece,
                             "highway": w["highway"], "name": w["name"], "oneway": w["oneway"],
                             "length_m": round(sub.length, 2), "bearing_deg": round(_bearing(x0, y0, x1, y1), 1), "wkt": sub.wkt})
                piece += 1
    S = pd.DataFrame(rows)
    S.to_parquet(os.path.join(out, "segments.parquet"), index=False)
    sl = {"started_utc": started, "finished_utc": now(), "extract": os.path.basename(pbf), "max_len_m": max_len,
          "highway_kinds": sorted(keep_kinds), "ways": int(S.way_id.nunique()), "segments": len(S),
          "total_km": round(S.length_m.sum() / 1000, 1), "median_length_m": float(S.length_m.median()),
          "by_highway_km": (S.groupby("highway").length_m.sum() / 1000).round(1).sort_values(ascending=False).to_dict()}
    json.dump(sl, open(os.path.join(out, "segments_log.json"), "w", encoding="utf8"), indent=1)
    log.info(f"{len(S)} segments, {sl['total_km']} km")
    return [os.path.join(out, "segments.parquet")], {"segments": len(S), "km": sl["total_km"]}


# ---------------------------------------------------------------- 1h passes
def _load(cfg, platform):
    base = cfg.paths.step(1)
    if platform == "mapillary":
        c = pd.read_parquet(os.path.join(base, "mapillary", "captures.parquet"))
        seq = pd.read_parquet(os.path.join(base, "mapillary", "sequences.parquet"), columns=["sequence_id", "device"])
        dev = dict(zip(seq.sequence_id.astype(str), seq.device.astype(str)))
        sid = c.sequence_id.astype(str)
        return pd.DataFrame({"capture_id": c["id"].astype(str), "sequence_id": sid,
                             "t": pd.to_datetime(c.captured_at_ms, unit="ms", utc=True), "lon": c.lon, "lat": c.lat,
                             "contributor": c.creator_id.astype(str), "device": sid.map(dev).fillna("")})
    c = pd.read_parquet(os.path.join(base, "kartaview", "photos.parquet"))
    c = c[c.in_area & c.status.eq("active") & c.visibility.eq("public")]
    seq = pd.read_parquet(os.path.join(base, "kartaview", "sequence_table.parquet"), columns=["sequence_id", "creator_id", "device"])
    user = dict(zip(seq.sequence_id.astype(str), seq.creator_id.astype(str)))
    dev = dict(zip(seq.sequence_id.astype(str), seq.device.fillna("").astype(str)))
    sid = c.sequence_id.astype(str)
    return pd.DataFrame({"capture_id": c.photo_id.astype(str), "sequence_id": sid,
                         "t": pd.to_datetime(c.shotDate, utc=True, errors="coerce"), "lon": c.lng, "lat": c.lat,
                         "contributor": sid.map(user).fillna("?"), "order": c.sequenceIndex, "device": sid.map(dev).fillna("")})


def _travel(C, veh):
    order = ["sequence_id", "order"] if "order" in C else ["sequence_id", "t"]
    C = C.sort_values(order + ["capture_id"]).reset_index(drop=True)
    sp, sn = C.sequence_id.eq(C.sequence_id.shift(1)), C.sequence_id.eq(C.sequence_id.shift(-1))
    x0, y0 = np.where(sp, C.x.shift(1), C.x), np.where(sp, C.y.shift(1), C.y)
    x1, y1 = np.where(sn, C.x.shift(-1), C.x), np.where(sn, C.y.shift(-1), C.y)
    dx, dy = x1 - x0, y1 - y0
    C["travel_bearing"] = np.where(np.hypot(dx, dy) >= 1.0, (np.degrees(np.arctan2(dx, dy)) + 360) % 360, np.nan)
    step = np.hypot(np.where(sn, C.x.shift(-1), np.nan) - C.x, np.where(sn, C.y.shift(-1), np.nan) - C.y)
    dt = (C.t.shift(-1) - C.t).dt.total_seconds().where(sn)
    kmh = pd.Series(np.where((dt > 0) & (step >= 1.0), step / dt * 3.6, np.nan))
    g = kmh.groupby(C.sequence_id)
    C["seq_kmh"] = g.transform("median").round(1)
    C["seq_kmh_p90"] = g.transform(lambda v: v.quantile(0.9)).round(1)
    steps = g.transform("count")
    by_device = C.device.str.lower().str.contains("|".join(veh["car_devices"]), regex=True)
    C["vehicle"] = (C.seq_kmh >= veh["median_kmh"]) | ((C.seq_kmh_p90 >= veh["p90_kmh"]) & (steps >= veh["min_steps_p90"])) | by_device
    return C


def passes(cfg: Config, log, platforms=("mapillary", "kartaview"), **_):
    s1 = cfg.get("step1_harvest")
    rule, veh, min_year, tz = s1["match"], s1["vehicle"], s1["min_year"], cfg.get("area.timezone")
    nfv = set(cfg.get("osm.not_for_vehicles"))
    out = out_dir(cfg)
    S = pd.read_parquet(os.path.join(out, "segments.parquet"))
    geoms = shapely.from_wkt(S.wkt.to_numpy())
    tree = shapely.STRtree(geoms)
    outputs, summary = [], {}
    for platform in [p for p in platforms if cfg.get(f"platforms.{p}.enabled", True)]:
        progress.phase(f"matching {platform} captures to segments")
        started = now()
        C = _load(cfg, platform)
        C["x"], C["y"] = cfg.to_m(C.lon.to_numpy(), C.lat.to_numpy())
        C = _travel(C, veh)
        pts = shapely.points(C.x.to_numpy(), C.y.to_numpy())
        ci, si = tree.query(pts, predicate="dwithin", distance=rule["max_dist_m"])
        d = shapely.distance(pts[ci], geoms[si])
        s = shapely.line_locate_point(geoms[si], pts[ci])
        L = shapely.length(geoms[si])
        p0 = shapely.line_interpolate_point(geoms[si], np.clip(s - 2, 0, L))
        p1 = shapely.line_interpolate_point(geoms[si], np.clip(s + 2, 0, L))
        sb = (np.degrees(np.arctan2(shapely.get_x(p1) - shapely.get_x(p0), shapely.get_y(p1) - shapely.get_y(p0))) + 360) % 360
        diff = np.abs((C.travel_bearing.to_numpy()[ci] - sb + 180) % 360 - 180)
        off = np.minimum(diff, 180 - diff)
        cost = d + np.where(np.isnan(off) | (off <= rule["max_angle_deg"]), 0.0, rule["turn_penalty_m"])
        cost = np.where(C.vehicle.to_numpy()[ci] & S.highway.isin(nfv).to_numpy()[si], np.inf, cost)
        K = pd.DataFrame({"ci": ci, "si": si, "cost": cost, "dist_m": d, "angle_off_deg": off})
        K = K[np.isfinite(K.cost)].sort_values(["ci", "cost"]).drop_duplicates("ci")
        C["segment_id"] = None
        C.loc[K.ci.to_numpy(), "segment_id"] = S.segment_id.to_numpy()[K.si.to_numpy()]
        C["dist_m"] = np.nan
        C.loc[K.ci.to_numpy(), "dist_m"] = K.dist_m.round(2).to_numpy()
        C["angle_off_deg"] = np.nan
        C.loc[K.ci.to_numpy(), "angle_off_deg"] = K.angle_off_deg.round(1).to_numpy()
        local = C.t.dt.tz_convert(tz)
        C["local_date"] = local.dt.date.astype(str)
        C["date_ok"] = local.dt.year.between(min_year, datetime.now(timezone.utc).year) & C.t.notna()
        if s1.get("camera_reset_dates"):
            C["date_ok"] &= ~C.local_date.isin([str(x) for x in s1["camera_reset_dates"]])
        cp = os.path.join(out, f"{platform}_capture_segments.parquet")
        C[["capture_id", "sequence_id", "contributor", "t", "local_date", "date_ok", "segment_id", "dist_m",
           "angle_off_deg", "travel_bearing", "seq_kmh", "seq_kmh_p90", "device", "vehicle"]].to_parquet(cp, index=False)
        M = C[C.segment_id.notna()]
        P = M[M.date_ok].assign(pass_key=lambda df: df.contributor + "|" + df.local_date, year=lambda df: df.local_date.str[:4])
        agg = M.groupby("segment_id").agg(captures=("capture_id", "size"), sequences=("sequence_id", "nunique"))
        pa_ = P.groupby("segment_id").agg(passes=("pass_key", "nunique"), contributors=("contributor", "nunique"),
                                          first_date=("local_date", "min"), last_date=("local_date", "max"))
        py = P.groupby(["segment_id", "year"]).pass_key.nunique().unstack(fill_value=0)
        py.columns = [f"passes_{y}" for y in py.columns]
        T = S[["segment_id", "way_id", "highway", "length_m"]].set_index("segment_id").join(agg).join(pa_).join(py)
        num = ["captures", "sequences", "passes", "contributors"] + list(py.columns)
        T[num] = T[num].fillna(0).astype(int)
        sp = os.path.join(out, f"{platform}_segment_passes.parquet")
        T.reset_index().to_parquet(sp, index=False)
        km = lambda m: round(float(T.loc[m, "length_m"].sum()) / 1000, 1)  # noqa: E731
        pl = {"started_utc": started, "finished_utc": now(), "platform": platform, "rule": rule, "vehicle": veh,
              "min_year": min_year, "captures": len(C), "matched": int(C.segment_id.notna().sum()),
              "matched_share": round(float(C.segment_id.notna().mean()), 4),
              "sequences_vehicle": int(C[C.vehicle].sequence_id.nunique()), "captures_bad_date": int((~C.date_ok).sum()),
              "km_with_passes": {f">={k}": km(T.passes >= k) for k in (1, 2, 3, 5, 10, 20)}}
        json.dump(pl, open(os.path.join(out, f"{platform}_passes_log.json"), "w", encoding="utf8"), indent=1)
        log.info(f"{platform}: {pl['matched']} of {len(C)} captures matched")
        outputs += [cp, sp]
        summary[platform] = {"captures": len(C), "matched": pl["matched"]}
    return outputs, summary


# ---------------------------------------------------------------- 1i duplicates
def _points(cfg, path, lonlat):
    C = pd.read_parquet(path, columns=["capture_id", "sequence_id", "contributor", "t", "local_date", "date_ok", "segment_id"])
    C = C.merge(lonlat, on="capture_id", how="left")
    C["x"], C["y"] = cfg.to_m(C.lon.to_numpy(), C.lat.to_numpy())
    C["ts"] = (pd.to_datetime(C.t, utc=True) - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds()
    return C


def _join(K, M, offset_s, dp, same_platform=False):
    k = K[["capture_id", "sequence_id", "ts", "x", "y"]].copy()
    m = M[["capture_id", "sequence_id", "ts", "x", "y"]].copy()
    m["ts"] = m.ts + offset_s
    k["bin"] = np.floor(k.ts / dp["bin_s"]).astype("int64")
    m["bin"] = np.floor(m.ts / dp["bin_s"]).astype("int64")
    m = m[m.bin.between(k.bin.min() - 1, k.bin.max() + 1)]
    out = []
    for d in (-1, 0, 1):
        j = k.merge(m.assign(bin=m.bin + d), on="bin", suffixes=("_k", "_m"))
        j = j[(j.ts_k - j.ts_m).abs() <= dp["max_dt_s"]]
        if same_platform:
            j = j[j.sequence_id_k < j.sequence_id_m]
        j["dist"] = np.hypot(j.x_k - j.x_m, j.y_k - j.y_m)
        out.append(j[j.dist <= dp["max_dist_m"]])
    J = pd.concat(out, ignore_index=True)
    J["dt"] = J.ts_m - J.ts_k
    return J.sort_values("dist").drop_duplicates(["capture_id_k", "sequence_id_m"])


def duplicates(cfg: Config, log, **_):
    dp = cfg.get("step1_harvest.duplicates")
    offsets = sorted({0.0} | {s * o for o in cfg.utc_offsets_h for s in (1, -1)}, key=lambda v: (abs(v), -v))
    base, out = cfg.paths.step(1), out_dir(cfg)
    ml = pd.read_parquet(os.path.join(base, "mapillary", "captures.parquet"), columns=["id", "lon", "lat"])
    ml = ml.assign(capture_id=ml["id"].astype(str))[["capture_id", "lon", "lat"]]
    kl = pd.read_parquet(os.path.join(base, "kartaview", "photos.parquet"), columns=["photo_id", "lng", "lat"])
    kl = kl.rename(columns={"photo_id": "capture_id", "lng": "lon"})
    kl["capture_id"] = kl.capture_id.astype(str)
    M = _points(cfg, os.path.join(out, "mapillary_capture_segments.parquet"), ml)
    K = _points(cfg, os.path.join(out, "kartaview_capture_segments.parquet"), kl)
    M, K = M[M.date_ok], K[K.date_ok]
    kn, mn = K.groupby("sequence_id").size(), M.groupby("sequence_id").size()
    ks = K.drop_duplicates("sequence_id").set_index("sequence_id").contributor
    ms = M.drop_duplicates("sequence_id").set_index("sequence_id").contributor
    dl = {"started_utc": now(), **dp, "offsets_h": offsets, "offsets": {}}
    pairs, J0 = [], None
    for h in offsets:
        J = _join(K, M, h * 3600, dp)
        if h == 0:
            J0 = J
        g = J.groupby(["sequence_id_k", "sequence_id_m"])
        P = pd.DataFrame({"matched_k": g.capture_id_k.nunique(), "matched_m": g.capture_id_m.nunique(),
                          "median_dist_m": g.dist.median().round(1), "median_dt_s": g.dt.median().round(2)}).reset_index()
        # counts as numbers even when no photo matched (an empty map keeps the id column's text type and the division fails)
        P["kv_captures"] = P.sequence_id_k.map(kn).astype("int64")
        P["mly_captures"] = P.sequence_id_m.map(mn).astype("int64")
        P["share_kv"] = (P.matched_k / P.kv_captures).round(3)
        P["share_mly"] = (P.matched_m / P.mly_captures).round(3)
        P["share_shorter"] = np.where(P.kv_captures <= P.mly_captures, P.share_kv, P.share_mly)
        P["offset_h"] = int(h) if float(h).is_integer() else h
        P["duplicate"] = (P.matched_k >= dp["min_matched"]) & (P.share_shorter >= dp["min_share"])
        dl["offsets"][str(h)] = {"photo_matches": len(J), "pairs": len(P), "duplicate_pairs": int(P.duplicate.sum())}
        log.info(f"offset {h:+g} h: {len(J)} matches, {int(P.duplicate.sum())} duplicate pairs")
        pairs.append(P)
    P = pd.concat(pairs, ignore_index=True)
    P["kv_contributor"], P["mly_contributor"] = P.sequence_id_k.map(ks), P.sequence_id_m.map(ms)
    P.sort_values(["duplicate", "matched_k"], ascending=False).to_csv(os.path.join(out, "duplicate_pairs.csv"), index=False)
    D = P[P.duplicate]
    J = _join(K, K, 0, dp, same_platform=True)
    g = J.groupby(["sequence_id_k", "sequence_id_m"])
    KK = pd.DataFrame({"matched_a": g.capture_id_k.nunique(), "matched_b": g.capture_id_m.nunique(),
                       "median_dist_m": g.dist.median().round(1), "median_dt_s": g.dt.median().round(2)}).reset_index()
    KK = KK.rename(columns={"sequence_id_k": "sequence_a", "sequence_id_m": "sequence_b"})
    KK["share_shorter"] = (np.maximum(KK.matched_a, KK.matched_b).astype("float64")
                           / np.minimum(KK.sequence_a.map(kn).astype("float64"),
                                        KK.sequence_b.map(kn).astype("float64"))).clip(upper=1).round(3)
    KK["duplicate"] = (KK.matched_a >= dp["min_matched"]) & (KK.share_shorter >= dp["min_share"])
    KK["contributor_a"], KK["contributor_b"] = KK.sequence_a.map(ks), KK.sequence_b.map(ks)
    KK = KK[KK.duplicate].sort_values("matched_a", ascending=False)
    KK.to_csv(os.path.join(out, "kartaview_internal_duplicates.csv"), index=False)
    parent = {}

    def find(a):
        parent.setdefault(a, a)
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in zip(D.kv_contributor, D.mly_contributor):   # plain loops: empty tables have no text type to add to
        parent[find(f"kv:{a}")] = find(f"mly:{b}")
    for a, b in zip(KK.contributor_a, KK.contributor_b):
        parent[find(f"kv:{a}")] = find(f"kv:{b}")
    seg = pd.read_parquet(os.path.join(out, "segments.parquet"), columns=["segment_id", "highway", "length_m"])
    rows = []
    for pref, C in (("mly:", M), ("kv:", K)):
        c = C[C.segment_id.notna()]
        c = c.assign(person=c.contributor.map(lambda x: find(pref + x)), platform=pref[:-1])
        rows.append(c[["segment_id", "person", "local_date", "platform"]].drop_duplicates())
    A = pd.concat(rows, ignore_index=True)
    A["pass_key"] = A.person + "|" + A.local_date
    per = A.groupby(["segment_id", "platform"]).pass_key.nunique().unstack(fill_value=0)
    per = per.reindex(columns=["kv", "mly"], fill_value=0)   # a platform with no captures in the area still gets its column
    per.columns = [f"passes_{c}" for c in per.columns]
    T = seg.set_index("segment_id").join(per).join(A.groupby("segment_id").pass_key.nunique().rename("passes_combined")) \
        .join(A.groupby("segment_id").person.nunique().rename("contributors_combined")).fillna(0)
    for c in ["passes_kv", "passes_mly", "passes_combined", "contributors_combined"]:
        T[c] = T[c].astype(int)
    T["passes_counted_twice"] = T.passes_kv + T.passes_mly - T.passes_combined
    T.reset_index().to_parquet(os.path.join(out, "combined_segment_passes.parquet"), index=False)
    km = lambda m: round(float(T.loc[m, "length_m"].sum()) / 1000, 1)  # noqa: E731
    dl.update(finished_utc=now(), duplicate_pairs=len(D), kv_internal_duplicate_pairs=len(KK),
              mly_captures_matched_to_kv=int(J0.capture_id_m.nunique()),
              passes_combined=int(T.passes_combined.sum()), passes_counted_twice=int(T.passes_counted_twice.sum()),
              km_with_combined_passes={f">={k}": km(T.passes_combined >= k) for k in (1, 2, 3, 5, 10)})
    json.dump(dl, open(os.path.join(out, "duplicates_log.json"), "w", encoding="utf8"), indent=1)
    log.info(f"{len(D)} cross-platform and {len(KK)} KartaView-internal duplicate pairs")
    return [os.path.join(out, "combined_segment_passes.parquet")], {"duplicate_pairs": len(D), "kv_internal": len(KK),
                                                                    "passes_combined": int(T.passes_combined.sum())}
