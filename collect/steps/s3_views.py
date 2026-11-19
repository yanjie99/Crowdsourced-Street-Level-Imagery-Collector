"""Step 3: travel yaw, leveling and perspective views from the panoramas (ported 18 Nov 2026 from
step3_download/travel_yaw.py and make_views.py; logic unchanged; Panorama Processing Scheme).

  3h travel yaw  one value per vehicle sequence: track bearing minus camera heading, circular median
                 -> travel_yaw.csv, travel_yaw_frames.parquet
  3i views       four views per usable panorama (step3_download.crop), Mapillary panoramas leveled frame by
                 frame from computed_rotation, vehicle sequences turned by their travel yaw; validity masks in one
                 zip per sequence -> <views>/<platform>/<sequence>/ (views.parquet, panos.parquet, masks.zip)
Capture class (step3_download.capture_class): rules tried in order, the first that matches wins, e.g.
  {platform: kartaview, class: vehicle}; {platform: mapillary, device_contains: GoPro, class: two_wheeler}; ...
plus overrides {sequence_id: class} set at the review gate. Only vehicle sequences get a travel yaw.
"""
from __future__ import annotations

import glob
import gzip
import json
import os
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from .. import progress
from ..config import Config
from .s3_download import d3, root, s3

COPY = ["half", "role", "region_cell", "stratum"]


def capture_class(cfg, platform, seq, device):
    cc = s3(cfg, "capture_class", {}) or {}
    ov = cc.get("overrides") or {}
    if seq in ov:
        return ov[seq]
    for r in cc.get("rules") or [{"platform": "kartaview", "class": "vehicle"},
                                 {"platform": "mapillary", "device_contains": "GoPro", "class": "two_wheeler"},
                                 {"platform": "mapillary", "class": "on_foot"}]:
        if r.get("platform", platform) != platform:
            continue
        if r.get("device_contains") and r["device_contains"] not in str(device):
            continue
        return r["class"]
    return "on_foot"


def _wrap(a):
    a = np.mod(a, 360.0)
    return np.where(a > 180.0, a - 360.0, a)


def _circ_median(a):
    a = np.asarray(a, float)
    if len(a) == 0:
        return np.nan
    c = np.degrees(np.arctan2(np.sin(np.radians(a)).mean(), np.cos(np.radians(a)).mean()))
    return float(_wrap(c + np.median(_wrap(a - c))))


def _bearings(g, lon, lat, min_span):
    g = g.sort_values("order")
    o = g.order.to_numpy()
    x = np.radians(g[lon].to_numpy()) * 6371000.0 * np.cos(np.radians(g[lat].to_numpy().mean()))
    y = np.radians(g[lat].to_numpy()) * 6371000.0
    b, span = np.full(len(g), np.nan), np.full(len(g), np.nan)
    for i in range(1, len(g) - 1):
        if o[i] - o[i - 1] == 1 and o[i + 1] - o[i] == 1:
            dx, dy = x[i + 1] - x[i - 1], y[i + 1] - y[i - 1]
            span[i] = np.hypot(dx, dy)
            if span[i] >= min_span:
                b[i] = np.degrees(np.arctan2(dx, dy)) % 360.0
    g = g.copy()
    g["bearing"], g["span_m"] = b, span
    return g


def _mly_records(cfg, seq):
    rec = {}
    for f in glob.glob(os.path.join(root(cfg, "images"), "mapillary", seq, "records", "graph_*.json.gz")):
        for k, r in json.load(gzip.open(f)).items():
            cg = (r.get("computed_geometry") or {}).get("coordinates")
            sc = r.get("sfm_cluster")
            rec[str(k)] = {"compass": r.get("compass_angle"), "computed_compass": r.get("computed_compass_angle"),
                           "c_lon": cg[0] if cg else np.nan, "c_lat": cg[1] if cg else np.nan,
                           "rotation": r.get("computed_rotation"), "cluster": sc.get("id") if isinstance(sc, dict) else None}
    return rec


def travel_yaw(cfg: Config, log, **_):
    from .. import pano as pv
    ty = s3(cfg, "travel_yaw", {}) or {}
    min_span = s3(cfg, "crop.travel_yaw_min_span_m", 3.0)
    min_frames, min_agree = ty.get("min_frames", 10), ty.get("min_agree", 0.5)
    fallback = ty.get("fallback_deg", {"kartaview": 0.0})
    M = pd.read_parquet(cfg.paths.manifest)
    M = M[M.pano_status.eq("ok")] if "pano_status" in M else M.iloc[0:0]
    frames = []
    for (pf, seq), g in M.groupby(["platform", "sequence_id"]):
        if capture_class(cfg, pf, seq, g.device.iloc[0]) != "vehicle":
            continue
        if pf == "kartaview":
            g = _bearings(g, "lon", "lat", min_span)
            g["camera_heading"] = g.heading.where(g.heading != 0)
            g["heading_source"] = "kartaview heading"
        else:
            r = pd.DataFrame.from_dict(_mly_records(cfg, seq), orient="index").rename_axis("frame_id").reset_index()
            g = g.merge(r, on="frame_id", how="left")
            use_c = g.computed_compass.notna() & g.c_lon.notna()
            g["p_lon"], g["p_lat"] = np.where(use_c, g.c_lon, g.lon), np.where(use_c, g.c_lat, g.lat)
            g = _bearings(g, "p_lon", "p_lat", min_span)
            g["camera_heading"] = np.where(g.computed_compass.notna(), g.computed_compass, g.heading)
            g["heading_source"] = np.where(g.computed_compass.notna(), "mapillary computed_compass_angle", "mapillary compass_angle")
        frames.append(g)
    if not frames:
        pd.DataFrame(columns=["platform", "sequence_id", "travel_yaw_used"]).to_csv(os.path.join(d3(cfg), "travel_yaw.csv"), index=False)
        return [], {"vehicle_sequences": 0}
    F = pd.concat(frames, ignore_index=True)
    F["yaw"] = [pv.travel_yaw(b, h) if b == b and h == h else np.nan for b, h in zip(F.bearing, F.camera_heading)]
    rows = []
    for (pf, seq), g in F.groupby(["platform", "sequence_id"]):
        y = g.yaw.dropna().to_numpy()
        med = _circ_median(y)
        dev = np.abs(_wrap(y - med)) if len(y) else np.array([])
        r = {"platform": pf, "sequence_id": seq, "device": g.device.iloc[0], "frames": len(g), "frames_used": len(y),
             "travel_yaw_deg": round(med, 1) if med == med else np.nan,
             "mad_deg": round(float(np.median(dev)), 1) if len(y) else np.nan,
             "share_within_15": round(float((dev <= 15).mean()), 3) if len(y) else np.nan,
             "heading_source": g.heading_source.mode().iloc[0]}
        ok = len(y) >= min_frames and r["share_within_15"] >= min_agree
        r["travel_yaw_used"] = r["travel_yaw_deg"] if ok else fallback.get(pf, np.nan)
        r["yaw_source"] = "computed" if ok else ("rig (fallback_deg)" if pf in fallback else "none")
        rows.append(r)
    S = pd.DataFrame(rows)
    S.to_csv(os.path.join(d3(cfg), "travel_yaw.csv"), index=False)
    F[["platform", "sequence_id", "frame_id", "order", "bearing", "span_m", "camera_heading", "heading_source", "yaw"]].to_parquet(
        os.path.join(d3(cfg), "travel_yaw_frames.parquet"), index=False)
    log.info(f"travel yaw for {len(S)} vehicle sequences ({int(S.yaw_source.eq('computed').sum())} computed)")
    return [os.path.join(d3(cfg), "travel_yaw.csv")], {"vehicle_sequences": len(S), "computed": int(S.yaw_source.eq("computed").sum())}


def views(cfg: Config, log, regions=None, sequences=None, out_root=None, **_):
    import cv2
    from .. import pano as pv
    from .s3_download import seqs_of_regions
    crop = s3(cfg, "crop", {})
    pv.configure(crop)
    jpeg_q, level_min = crop.get("jpeg_quality", 95), crop.get("level_min_deg", 0.01)
    out = out_root or root(cfg, "views")
    K, dropped = pv.crop_list(pd.read_parquet(cfg.paths.manifest), root(cfg, "panoramas"))
    seqs = sequences or seqs_of_regions(cfg, regions)
    if seqs:
        K = K[K.sequence_id.isin(set(seqs))]
    T = pd.read_csv(os.path.join(d3(cfg), "travel_yaw.csv"), dtype={"sequence_id": str}).set_index("sequence_id")
    workers = s3(cfg, "workers", 16)
    totals = {"panoramas": 0, "views": 0, "usable": 0}

    def level_for(rot, fid):
        rv, cluster = rot.get(fid, (None, None))
        if rv is None:
            return None, {"leveled": False, "level_source": "none: no computed_rotation", "sfm_cluster": cluster}
        C, inf = pv.level_rotation(rv)
        if inf["cam_tilt_deg"] < level_min:
            return None, {"leveled": False, "level_source": "none: not reconstructed", "sfm_cluster": cluster, **inf}
        return C, {"leveled": True, "level_source": "computed_rotation", "sfm_cluster": cluster, **inf}

    def one(r, ty, cls, rot):
        im = cv2.imread(r["pano_path"])
        base = {"platform": r["platform"], "sequence_id": r["sequence_id"], "frame_id": r["frame_id"], "order": r["order"]}
        if im is None:
            return {**base, "status": "unreadable"}, []
        if r["platform"] == "mapillary":
            C, lev = level_for(rot, r["frame_id"])
        else:
            C, lev = None, {"leveled": False, "level_source": "none: fixed rig (kartaview)"}
        pinfo, vs = pv.views_from_pano(im, travel_yaw_deg=ty, edge_black=(r["platform"] == "kartaview"), level=C)
        prow = {**base, "status": "broken" if pinfo["broken"] else "ok", "capture_class": cls, "travel_yaw_deg": ty, **lev,
                **{k: pinfo[k] for k in ("pano_width", "pano_height", "placeholder", "pano_valid_share",
                                         "coverage_top_deg", "coverage_bottom_deg", "coverage_class")}}
        rows = []
        d = os.path.join(out, r["platform"], r["sequence_id"])
        for name, (view, vm, info) in vs.items():
            vid = f"{r['frame_id']}_{name}"
            path = mpath = png = None
            if info["usable"]:
                path, mpath = f"{vid}.jpg", f"{vid}_mask.png"
                cv2.imwrite(os.path.join(d, path), view, [cv2.IMWRITE_JPEG_QUALITY, jpeg_q])
                ok, png = cv2.imencode(".png", (vm * 255).astype(np.uint8))
            rows.append({"view_id": vid, "capture_id": r["frame_id"], **base, **{c: r.get(c) for c in COPY},
                         "capture_class": cls, "leveled": lev["leveled"], "level_source": lev["level_source"],
                         "cam_tilt_deg": lev.get("cam_tilt_deg", np.nan),
                         "yaw_from_pano_center_deg": float(info["yaw_deg"] % 360.0),
                         "yaw_from_travel_deg": pv.YAWS[name] if cls == "vehicle" else np.nan,
                         **{k: v for k, v in info.items() if k != "yaw_deg"},
                         "file": None if path is None else f"{r['platform']}/{r['sequence_id']}/{path}",
                         "mask_file": None if mpath is None else f"{r['platform']}/{r['sequence_id']}/masks.zip",
                         "mask_member": mpath, "_png": None if png is None else png.tobytes()})
        return prow, rows

    progress.phase("views", int(K.groupby(["platform", "sequence_id"]).ngroups), "sequences")
    for (pf, seq), g in K.groupby(["platform", "sequence_id"]):
        progress.update(add=1, message=f"views: {totals['usable']:,} usable views so far")
        d = os.path.join(out, pf, seq)
        if os.path.exists(os.path.join(d, "views.parquet")):
            continue
        os.makedirs(d, exist_ok=True)
        g = g.sort_values("order")
        cls = capture_class(cfg, pf, seq, g.device.iloc[0])
        tyv = float(T.loc[seq, "travel_yaw_used"]) if cls == "vehicle" and seq in T.index else 0.0
        rot = {k: (v["rotation"], v["cluster"]) for k, v in _mly_records(cfg, seq).items()} if pf == "mapillary" else {}
        t0 = time.time()
        with ThreadPoolExecutor(workers) as ex:
            res = list(ex.map(lambda r: one(r, tyv, cls, rot), g.to_dict("records")))
        P = pd.DataFrame([p for p, _ in res])
        V = pd.DataFrame([v for _, vs in res for v in vs])
        if len(V):
            with zipfile.ZipFile(os.path.join(d, "masks.zip"), "w", zipfile.ZIP_STORED) as z:
                for m, b in zip(V.mask_member, V._png):
                    if b is not None:
                        z.writestr(m, b)
            V = V.drop(columns="_png")
        P.to_parquet(os.path.join(d, "panos.parquet"), index=False)
        V.to_parquet(os.path.join(d, "views.parquet"), index=False)
        totals["panoramas"] += len(P)
        totals["views"] += len(V)
        totals["usable"] += int(V.usable.sum()) if len(V) else 0
        log.info(f"{pf}/{seq} ({cls}, travel yaw {tyv:+.1f}): {len(P)} panoramas, {int(V.usable.sum()) if len(V) else 0} views, "
                 f"{time.time() - t0:.0f} s")
    return [out], {**totals, "dropped_by_quality_filter": dropped}
