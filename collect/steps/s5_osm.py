"""Step 5: map context from OpenStreetMap (ported 18 Nov 2026 from step5_osm/*.py; logic unchanged).

  5a ways      drivable ways and junction nodes from the dated extract     -> ways.parquet, junctions.parquet
  5b widths    carriageway width of every drivable way from the road standard -> way_widths.parquet, default_widths.csv,
                                                                                 lane_markings.csv
  5c history   ohsome edit history of drivable ways around the captures   -> history/<region>.geojson.gz,
                                                                                 regions.csv, regions.geojson, history_log.json
  5d match     every capture matched to the map on its capture date and today -> capture_map_match.parquet, match_log.json
Settings: area.*, osm.*, step5_osm.* (project.yaml). Inputs: osm.extract, area.boundary_file, the Step 3 manifest.
"""
from __future__ import annotations

import glob
import gzip
import json
import math
import os
import time
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import shapely
from shapely.geometry import LineString, mapping, shape
from shapely.ops import transform

from .. import progress
from ..config import Config

PLAN_TAGS = ["oneway", "lanes", "turn:lanes", "width", "placement", "bridge", "tunnel", "layer", "source"]
MORE_TAGS = ["name", "ref", "maxspeed", "junction", "lanes:forward", "lanes:backward", "width:carriageway"]
LINKS = {"motorway_link", "trunk_link", "primary_link", "secondary_link", "tertiary_link"}


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def out_dir(cfg: Config) -> str:
    return cfg.paths.step(5)


# ---------------------------------------------------------------- 5a ways and junctions
def ways(cfg: Config, log, **_):
    import osmium

    bbox = cfg.get("area.bbox_osm")
    drivable = set(cfg.drivable)

    class H(osmium.SimpleHandler):
        def __init__(self):
            super().__init__()
            self.ways = []

        def way(self, w):
            hw = w.tags.get("highway")
            if hw not in drivable or w.tags.get("area") == "yes":
                return
            pts = [(n.ref, n.location.lon, n.location.lat) for n in w.nodes if n.location.valid()]
            if len(pts) < 2 or not any(bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3] for _, x, y in pts):
                return
            self.ways.append({"way_id": w.id, "version": w.version, "timestamp": w.timestamp.isoformat(),
                              "highway": hw, **{t: w.tags.get(t) for t in PLAN_TAGS + MORE_TAGS}, "pts": pts})

    out = out_dir(cfg)
    pbf = cfg.osm_extract
    started = _now()
    h = H()
    h.apply_file(pbf, locations=True)
    to_m, poly = cfg.to_m, cfg.boundary_m
    shapely.prepare(poly)
    rows, arms, wn, node_xy = [], defaultdict(int), defaultdict(set), {}
    for w in h.ways:
        pts = w.pop("pts")
        xs, ys = to_m([p[1] for p in pts], [p[2] for p in pts])
        line = LineString(list(zip(xs, ys)))
        if line.length < 0.5 or not poly.intersects(line):
            continue
        refs = [p[0] for p in pts]
        seg = np.hypot(np.diff(xs), np.diff(ys))
        rows.append({**w, "n_nodes": len(refs), "length_m": round(line.length, 2),
                     "node_spacing_median_m": round(float(np.median(seg)), 2),
                     "in_polygon_share": round(poly.intersection(line).length / line.length, 4),
                     "node_refs": refs, "wkt": line.wkt})
        closed = refs[0] == refs[-1]
        for k, r in enumerate(refs[:-1] if closed else refs):
            end = (k == 0 or k == len(refs) - 1) and not closed
            arms[r] += 1 if end else 2
            wn[r].add(w["way_id"])
            node_xy[r] = (xs[k], ys[k], pts[k][1], pts[k][2])
    W = pd.DataFrame(rows)
    hw = W.set_index("way_id").highway
    J = pd.DataFrame([{"node_id": r, "x": round(node_xy[r][0], 2), "y": round(node_xy[r][1], 2), "lon": node_xy[r][2],
                       "lat": node_xy[r][3], "n_ways": len(ws), "n_arms": arms[r], "is_junction": arms[r] >= 3,
                       "way_ids": sorted(ws), "highways": sorted({hw[i] for i in ws})}
                      for r, ws in wn.items() if len(ws) >= 2])
    W.to_parquet(os.path.join(out, "ways.parquet"), index=False)
    J.to_parquet(os.path.join(out, "junctions.parquet"), index=False)
    summary = {"started_utc": started, "finished_utc": _now(), "extract": os.path.basename(pbf),
               "extract_date": cfg.osm_extract_date, "drivable_highway_kinds": sorted(drivable),
               "ways": len(W), "km": round(W.length_m.sum() / 1000, 1),
               "km_inside_polygon": round((W.length_m * W.in_polygon_share).sum() / 1000, 1),
               "by_highway_km": (W.groupby("highway").length_m.sum() / 1000).round(1).sort_values(ascending=False).to_dict(),
               "tag_fill_share": {t: round(float(W[t].notna().mean()), 4) for t in PLAN_TAGS + MORE_TAGS},
               "shared_nodes": len(J), "junctions_3plus_arms": int(J.is_junction.sum())}
    json.dump(summary, open(os.path.join(out, "extract_log.json"), "w", encoding="utf8"), indent=1, default=str)
    log.info(f"{len(W)} drivable ways, {summary['km']} km; {summary['junctions_3plus_arms']} junctions")
    return [os.path.join(out, f) for f in ("ways.parquet", "junctions.parquet", "extract_log.json")], \
        {k: summary[k] for k in ("ways", "km", "shared_nodes", "junctions_3plus_arms")}


# ---------------------------------------------------------------- 5b widths
def _one_direction(rs, hw, n, two_way):
    lw = rs["lane_width_m"]
    if hw == "motorway":
        return lw["expressway"] * n, "expressway lanes"
    if hw in LINKS:
        if n <= 1:
            return lw["one_lane_oneway"], "one-lane ramp or slip road"
        if n == 2:
            return lw["slip_two_lane"], "two-lane ramp or slip road"
        return lw["expressway"] * n, "ramp lanes"
    if n <= 1:
        return (lw["undivided_lane"], "single lane of a two-way road") if two_way else \
               (lw["one_lane_oneway"], "one-way single-lane road")
    return 2 * lw["curb_lane"] + (n - 2) * lw["inner_lane"], "curb lanes and inner lanes"


def _width(rs, hw, oneway, lanes, fwd=None, bwd=None):
    lw = rs["lane_width_m"]
    if oneway:
        return _one_direction(rs, hw, lanes, False)
    if hw in ("service", "track") and lanes <= 2:
        return lw["service_road"], "service road or track"
    if lanes <= 2:
        return lw["undivided_lane"] * 2, "undivided two-way road"
    f = fwd if fwd and bwd and fwd + bwd == lanes else math.ceil(lanes / 2)
    b = lanes - f
    (wf, _), (wb, _) = _one_direction(rs, hw, f, True), _one_direction(rs, hw, b, True)
    return wf + wb, f"two-way, {f}+{b} lanes, each direction by the one-way rule"


def widths(cfg: Config, log, **_):
    out = out_dir(cfg)
    rs = cfg.get("step5_osm.road_standard")
    min_tagged = cfg.get("step5_osm.lanes_default_min_tagged", 20)
    shoulders = {k: tuple(v) for k, v in rs.get("shoulders_m", {}).items()}
    curb = rs.get("curb_clearance_m", 0.0)
    W = pd.read_parquet(os.path.join(out, "ways.parquet"),
                        columns=["way_id", "highway", "oneway", "lanes", "lanes:forward", "lanes:backward"])
    num = lambda s: pd.to_numeric(s, errors="coerce")  # noqa: E731
    W["lanes_n"] = num(W.lanes).where(lambda s: s >= 1)
    W["oneway_b"] = W.oneway.isin(["yes", "1", "-1"]) | (W.highway.eq("motorway") & W.oneway.ne("no"))
    fallback = {True: 1, False: 2}
    rows = []
    for (hw, ow), d in W.groupby(["highway", "oneway_b"]):
        t = d.lanes_n.dropna()
        ok = len(t) >= min_tagged
        lanes = float(t.median()) if ok else float(1 if hw == "track" else fallback[bool(ow)])
        lanes = max(1, round(lanes))
        w, rule = _width(rs, hw, bool(ow), lanes)
        rows.append({"highway": hw, "oneway": bool(ow), "ways": len(d), "lanes_tagged_share": round(len(t) / len(d), 3),
                     "lanes_tagged_n": len(t), "lanes_p10": t.quantile(0.1) if len(t) else None,
                     "lanes_p90": t.quantile(0.9) if len(t) else None, "default_lanes": lanes,
                     "default_source": f"median lanes tag of {len(t)} ways" if ok else f"fallback (only {len(t)} tagged)",
                     "default_width_m": round(w, 2), "width_rule": rule,
                     "shoulder_left_m": shoulders.get(hw, (0, 0))[0], "shoulder_right_m": shoulders.get(hw, (0, 0))[1]})
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(out, "default_widths.csv"), index=False)
    dflt = T.set_index(["highway", "oneway"]).default_lanes
    res = []
    for r in W.itertuples(index=False):
        tagged = not math.isnan(r.lanes_n)
        n = int(round(r.lanes_n)) if tagged else int(dflt[(r.highway, r.oneway_b)])
        fwd, bwd = num(pd.Series([r[4], r[5]])).tolist()
        fwd = int(fwd) if fwd == fwd else None
        bwd = int(bwd) if bwd == bwd else None
        w, rule = _width(rs, r.highway, r.oneway_b, n, fwd, bwd)
        sh = shoulders.get(r.highway, (0.0, 0.0))
        res.append({"way_id": r.way_id, "highway": r.highway, "oneway": bool(r.oneway_b), "lanes_used": n,
                    "lanes_source": "tag" if tagged else "class default", "width_m": round(w, 2),
                    "shoulder_left_m": sh[0], "shoulder_right_m": sh[1],
                    "curb_clearance_m": 0.0 if r.highway == "motorway" else curb, "width_rule": rule})
    R = pd.DataFrame(res)
    R.to_parquet(os.path.join(out, "way_widths.parquet"), index=False)
    lw, pat = rs.get("line_width_mm", {}), rs.get("lane_line_pattern_m", {})
    pd.DataFrame([{"line": k, "width_mm": v} for k, v in lw.items()] +
                 [{"line": f"pattern {k}", "pattern_m": "+".join(str(x) for x in v)} for k, v in pat.items()]
                 ).to_csv(os.path.join(out, "lane_markings.csv"), index=False)
    log.info(f"widths for {len(R)} ways ({(R.lanes_source == 'tag').mean():.0%} from lanes tags)")
    return [os.path.join(out, f) for f in ("default_widths.csv", "way_widths.parquet", "lane_markings.csv")], \
        {"ways": len(R), "lanes_from_tag": int((R.lanes_source == "tag").sum())}


# ---------------------------------------------------------------- 5c history
def region_of_frames(manifest: str) -> pd.DataFrame:
    """Every selected frame with its region square. In-square frames carry region_cell; a buffer frame
    (region_cell empty) takes the square of the nearest in-square frame of its own sequence, by frame order."""
    M = pd.read_parquet(manifest, columns=["platform", "sequence_id", "order", "frame_id", "region_cell", "role",
                                           "lon", "lat", "time"])
    M["region_cell"] = M.region_cell.replace("", None)
    M = M.sort_values(["platform", "sequence_id", "order"]).reset_index(drop=True)
    out = []
    for _, g in M.groupby(["platform", "sequence_id"], sort=False):
        known = g[g.region_cell.notna()]
        if len(known) == len(g):
            out.append(g.region_cell)
            continue
        pos = pd.Series(range(len(g)), index=g.index)
        kp = pos[known.index].values
        idx = abs(pos.values[:, None] - kp[None, :]).argmin(axis=1)
        out.append(pd.Series(known.region_cell.values[idx], index=g.index))
    M["region"] = pd.concat(out)
    return M


def _frames(cfg, regions):
    M = region_of_frames(cfg.paths.manifest)
    if regions:
        M = M[M.region.isin(regions)].reset_index(drop=True)
    return M


def _quarters(poly):
    """The parts of a polygon in the four quarters of its bounding box (empty parts left out)."""
    from shapely.geometry import box
    x0, y0, x1, y1 = poly.bounds
    xm, ym = (x0 + x1) / 2, (y0 + y1) / 2
    out = [poly.intersection(box(*b)) for b in ((x0, y0, xm, ym), (xm, y0, x1, ym), (x0, ym, xm, y1), (xm, ym, x1, y1))]
    return [q for q in out if not q.is_empty and q.area > 0]


def _history_area(requests, url, poly, name, params, ua, log, pause, depth=0, max_depth=3, tries=2):
    """Full history of one area from ohsome. ohsome streams its reply, so a request that runs into the server's
    time limit can end with status 200 but a cut-off body (ohsome then appends status 413, "query is too large in
    respect to the given timeout"); such a reply counts as a failed attempt, and a 413 splits the area at once. An area that
    fails `tries` times is split into quarters (down to max_depth levels; the last level gets 5 tries). Features
    of ways that cross a split appear in more than one part and are kept once (osm id + validFrom).
    Returns (features, bytes received, requests that succeeded)."""
    n = tries if depth < max_depth else 5
    err = ""
    for attempt in range(n):
        fc = {"type": "FeatureCollection", "features": [{"type": "Feature", "id": name, "properties": {},
                                                       "geometry": mapping(poly)}]}
        try:
            resp = requests.post(url, data={**params, "bpolys": json.dumps(fc)}, headers=ua, timeout=900)
            if resp.status_code == 200:
                try:
                    return resp.json().get("features", []), len(resp.content), 1
                except ValueError:
                    # a cut-off stream; ohsome usually appends its error message at the end
                    err = f"incomplete reply ({len(resp.content) / 1e6:.1f} MB), ends with: {resp.text[-300:]!r}"
            else:
                err = f"HTTP {resp.status_code}: {resp.text[:200]}"
        except requests.RequestException as e:
            err = repr(e)
        log.info(f"{name} attempt {attempt + 1} failed: {err}")
        if depth < max_depth and ("413" in err or "too large" in err):
            break                                        # ohsome says the area is too heavy: retrying the same request won't help
        if attempt + 1 < n:
            time.sleep(30 * (attempt + 1))
    if depth >= max_depth:
        raise RuntimeError(f"{name}: ohsome gave up after {n} attempts and {depth} splits; last error: {err}")
    log.info(f"{name}: splitting into quarters (smaller requests finish within ohsome's time limit)")
    feats, nbytes, parts, seen = [], 0, 0, set()
    for i, q in enumerate(_quarters(poly)):
        f, b, k = _history_area(requests, url, q, f"{name}.{i}", params, ua, log, pause, depth + 1, max_depth, tries)
        nbytes, parts = nbytes + b, parts + k
        for ft in f:
            key = (ft["properties"].get("@osmId"), ft["properties"].get("@validFrom"))
            if key not in seen:
                seen.add(key)
                feats.append(ft)
        time.sleep(pause)
    return feats, nbytes, parts


def history(cfg: Config, log, regions=None, **_):
    import requests
    out = out_dir(cfg)
    os.makedirs(os.path.join(out, "history"), exist_ok=True)
    api = cfg.get("osm.history_api")
    ua = {"User-Agent": cfg.get("platforms.user_agent", "collect")}
    margin, simp = cfg.get("step5_osm.history_margin_m"), cfg.get("step5_osm.history_simplify_m")
    pause = cfg.get("step5_osm.history_pause_s")
    meta = requests.get(f"{api}/metadata", headers=ua, timeout=60).json()
    t0, t1 = meta["extractRegion"]["temporalExtent"]["fromTimestamp"], meta["extractRegion"]["temporalExtent"]["toTimestamp"]
    M = _frames(cfg, regions)
    regs, polys = [], {}
    for cell, g in M.groupby("region"):
        x, y = cfg.to_m(g.lon.values, g.lat.values)
        area = shapely.union_all(shapely.buffer(shapely.points(np.c_[x, y]), margin, quad_segs=4)).simplify(simp)
        ll = transform(cfg.to_ll, area)
        polys[cell] = ll
        b = ll.bounds
        regs.append({"cell": cell, "captures": len(g), "min_lon": round(b[0], 6), "min_lat": round(b[1], 6),
                     "max_lon": round(b[2], 6), "max_lat": round(b[3], 6), "area_km2": round(area.area / 1e6, 2),
                     "vertices": int(shapely.get_num_coordinates(ll))})
    R = pd.DataFrame(regs)
    R.to_csv(os.path.join(out, "regions.csv"), index=False)
    json.dump({"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {"cell": c}, "geometry": mapping(p)}
                                                         for c, p in polys.items()]},
              open(os.path.join(out, "regions.geojson"), "w", encoding="utf8"))
    flt = "type:way and highway in (" + ", ".join(cfg.drivable) + ")"
    hl = {"started_utc": _now(), "api": api, "api_version": meta.get("apiVersion"), "data_from": t0, "data_to": t1,
          "filter": flt, "margin_m": margin, "license": "ODbL, © OpenStreetMap contributors (via ohsome)", "regions": []}
    progress.phase("history regions", len(R), "regions")
    for r in R.itertuples():
        progress.update(add=1)
        path = os.path.join(out, "history", f"{r.cell}.geojson.gz")
        if os.path.exists(path):
            continue
        t = time.time()
        feats, nbytes, parts = _history_area(requests, f"{api}/elementsFullHistory/geometry", polys[r.cell], r.cell,
                                             {"time": f"{t0},{t1}", "filter": flt, "properties": "tags,metadata",
                                              "clipGeometry": "false"}, ua, log, pause)
        js = {"type": "FeatureCollection", "features": feats}
        with gzip.open(path + ".tmp", "wt", encoding="utf8") as f:
            json.dump(js, f)
        os.replace(path + ".tmp", path)
        n = len(feats)
        nw = len({ft["properties"]["@osmId"] for ft in feats})
        hl["regions"].append({"cell": r.cell, "request_utc": _now(), "seconds": round(time.time() - t, 1),
                              "states": n, "ways": nw, "bytes": nbytes, "requests": parts})
        log.info(f"{r.cell}: {nw} ways, {n} states, {nbytes / 1e6:.1f} MB in {parts} request(s), {time.time() - t:.0f} s")
        time.sleep(pause)
    hl["finished_utc"] = _now()
    lp = os.path.join(out, "history_log.json")
    if os.path.exists(lp):
        hl["regions"] = json.load(open(lp, encoding="utf8"))["regions"] + hl["regions"]
    json.dump(hl, open(lp, "w", encoding="utf8"), indent=1)
    return [os.path.join(out, "history"), os.path.join(out, "regions.csv"), lp], \
        {"regions": len(R), "data_to": t1, "fetched_now": len(hl["regions"])}


# ---------------------------------------------------------------- 5d match on the capture date
def _load_states(cfg, d, regions=None):
    rows, seen = [], set()
    files = sorted(glob.glob(os.path.join(d, "history", "*.geojson.gz")))
    if regions:
        files = [f for f in files if os.path.basename(f).split(".")[0] in regions]
    for f in files:
        for ft in json.load(gzip.open(f, "rt", encoding="utf8"))["features"]:
            p = ft["properties"]
            wid = int(str(p["@osmId"]).split("/")[-1])
            key = (wid, p["@validFrom"])
            if key in seen or ft.get("geometry") is None:
                continue
            seen.add(key)
            g = transform(cfg.to_m, shape(ft["geometry"]))
            if g.geom_type in ("Polygon", "MultiPolygon"):
                g = g.boundary
            if g.is_empty or g.geom_type not in ("LineString", "MultiLineString"):
                continue
            rows.append({"way_id": wid, "version": p.get("@version"), "valid_from": p["@validFrom"],
                         "valid_to": p["@validTo"], "highway": p.get("highway"), "geom": g})
    S = pd.DataFrame(rows)
    for c in ("valid_from", "valid_to"):
        S[c] = pd.to_datetime(S[c], utc=True)
    return S


def _travel_bearing(cfg, M, min_step):
    x, y = cfg.to_m(M.lon.values, M.lat.values)
    M["x"], M["y"] = x, y
    b = np.full(len(M), np.nan)
    for _, idx in M.groupby(["platform", "sequence_id"], sort=False).indices.items():
        xs, ys = x[idx], y[idx]
        prev = np.r_[0, np.arange(len(idx) - 1)]
        nxt = np.r_[np.arange(1, len(idx)), len(idx) - 1]
        dx, dy = xs[nxt] - xs[prev], ys[nxt] - ys[prev]
        ok = np.hypot(dx, dy) >= min_step
        b[idx[ok]] = (np.degrees(np.arctan2(dx[ok], dy[ok])) + 360) % 360
    M["travel_bearing_deg"] = b


def _local_bearing(g, pt):
    s = g.project(pt)
    a, c = g.interpolate(max(s - 5, 0)), g.interpolate(min(s + 5, g.length))
    return (math.degrees(math.atan2(c.x - a.x, c.y - a.y)) + 360) % 360


def _match(M, geoms, ok_fn, rule, way_ids=None):
    pts = shapely.points(M.x.values, M.y.values)
    tree = shapely.STRtree(geoms)
    ci, gi = tree.query(pts, predicate="dwithin", distance=rule["max_dist_m"])
    keep = ok_fn(ci, gi)
    ci, gi = ci[keep], gi[keep]
    dist = shapely.distance(pts[ci], geoms[gi])
    ang = np.full(len(ci), np.nan)
    tb = M.travel_bearing_deg.values
    for k in range(len(ci)):
        if not np.isnan(tb[ci[k]]):
            dd = abs(_local_bearing(geoms[gi[k]], pts[ci[k]]) - tb[ci[k]]) % 180
            ang[k] = min(dd, 180 - dd)
    score = dist + np.where(ang > rule["max_angle_deg"], rule["penalty_m"], 0.0)
    C = pd.DataFrame({"c": ci, "g": gi, "dist": dist, "ang": ang, "score": score,
                      "wid": way_ids[gi] if way_ids is not None else gi})
    # ties (a capture equally far from two ways meeting at a node) go to the way closer to the travel direction,
    # then to the lower way id, so the result does not depend on array order (Phase 1 change, 18 Nov 2026)
    best = (C.assign(ang_s=C.ang.fillna(999.0)).sort_values(["c", "score", "ang_s", "wid"])
            .drop_duplicates("c").set_index("c"))
    return best, C


def match(cfg: Config, log, regions=None, **_):
    d = out_dir(cfg)
    rule = cfg.get("step5_osm.match")
    started = _now()
    hlog = json.load(open(os.path.join(d, "history_log.json"), encoding="utf8"))
    ts = pd.Timestamp(hlog["data_to"])
    data_to = ts.tz_convert("UTC") if ts.tzinfo else ts.tz_localize("UTC")
    M = _frames(cfg, regions)
    M["time_utc"] = pd.to_datetime(M.time, format="ISO8601", utc=True)
    _travel_bearing(cfg, M, rule["min_step_m"])
    M["after_history_end"] = M.time_utc >= data_to
    t_eff = M.time_utc.where(~M.after_history_end, data_to - pd.Timedelta(seconds=1)).values

    S = _load_states(cfg, d, regions)
    log.info(f"{len(S)} way states of {S.way_id.nunique()} ways")
    vf, vt = S.valid_from.values, S.valid_to.values
    sg = np.array(S.geom.values, dtype=object)
    best, C = _match(M, sg, lambda ci, gi: (vf[gi] <= t_eff[ci]) & (t_eff[ci] < vt[gi]), rule, S.way_id.values)
    C["way"] = S.way_id.values[C.g]
    ncand = C.groupby("c").way.nunique()

    W = pd.read_parquet(os.path.join(d, "ways.parquet"), columns=["way_id", "wkt"])
    feats = json.load(open(os.path.join(d, "regions.geojson"), encoding="utf8"))["features"]
    if regions:
        feats = [f for f in feats if f["properties"]["cell"] in regions]
    reg = shapely.union_all([shape(f["geometry"]) for f in feats])
    wg = shapely.from_wkt(W.wkt.values)
    near = shapely.intersects(wg, transform(cfg.to_m, reg).buffer(rule["max_dist_m"]))
    W, wg = W[near].reset_index(drop=True), wg[near]
    bn, Cn = _match(M, wg, lambda ci, gi: np.ones(len(ci), bool), rule, W.way_id.values)
    ncn = Cn.groupby("c").g.nunique()

    O = M[["platform", "sequence_id", "frame_id", "order", "region", "time_utc", "after_history_end", "travel_bearing_deg"]].copy()
    i = np.arange(len(M))
    O["date_way_id"] = pd.Series(S.way_id.values[best.g], index=best.index).reindex(i).astype("Int64").values
    O["date_version"] = pd.Series(S.version.values[best.g], index=best.index).reindex(i).astype("Int64").values
    O["date_valid_from"] = pd.to_datetime(pd.Series(S.valid_from.values[best.g], index=best.index).reindex(i).values, utc=True)
    O["date_valid_to"] = pd.to_datetime(pd.Series(S.valid_to.values[best.g], index=best.index).reindex(i).values, utc=True)
    O["date_dist_m"] = best.dist.reindex(i).round(2).values
    O["date_angle_deg"] = best.ang.reindex(i).round(1).values
    O["date_candidates"] = ncand.reindex(i).fillna(0).astype(int).values
    O["now_way_id"] = pd.Series(W.way_id.values[bn.g], index=bn.index).reindex(i).astype("Int64").values
    O["now_dist_m"] = bn.dist.reindex(i).round(2).values
    O["now_angle_deg"] = bn.ang.reindex(i).round(1).values
    O["now_candidates"] = ncn.reindex(i).fillna(0).astype(int).values
    O["same_way"] = (O.date_way_id == O.now_way_id).fillna(False).astype(bool)
    O["date_way_absent_now"] = O.date_way_id.notna() & ~O.date_way_id.isin(W.way_id)
    O.to_parquet(os.path.join(d, "capture_map_match.parquet"), index=False)
    m, n = O.date_way_id.notna(), O.now_way_id.notna()
    summary = {"started_utc": started, "finished_utc": _now(), "history_data_to": str(data_to), "rule": rule,
               "captures": len(O), "after_history_end": int(O.after_history_end.sum()), "way_states": len(S),
               "matched_on_date": int(m.sum()), "matched_now": int(n.sum()), "both_same_way": int(O.same_way.sum()),
               "both_different_way": int((m & n & ~O.same_way).sum()),
               "date_way_absent_now": int(O.date_way_absent_now.sum()),
               "median_dist_date_m": float(O.date_dist_m.median()), "median_dist_now_m": float(O.now_dist_m.median())}
    json.dump(summary, open(os.path.join(d, "match_log.json"), "w", encoding="utf8"), indent=1, default=str)
    log.info(f"{summary['matched_on_date']} of {len(O)} captures matched on their date")
    return [os.path.join(d, "capture_map_match.parquet"), os.path.join(d, "match_log.json")], \
        {k: summary[k] for k in ("captures", "matched_on_date", "matched_now", "both_same_way", "date_way_absent_now")}
