"""Step 2: what to download (ported 18 Nov 2026 from step2_regions/*.py; logic unchanged).

Two modes (step2_regions.mode):
  whole_area  2w only: every capture inside the boundary (optionally a date range, one copy of duplicated drives)
  strata      2a-2h: the descriptor-validation design below (squares per stratum, tuning and evaluation halves)

  2a features   OSM way attributes, buildings, land use and trees from the dated extract
  2b cells      square grid over the area: coverage from Step 1 and stratum features -> cells.parquet, cell_sequences.parquet
  2c layers     optional local footprint layer (Singapore: HDB blocks) and tall-building split -> cells_hdb.parquet
  2d strata     stratum rules, plan rules, ranking -> cells_strata.parquet, strata_summary.csv, strata_top.csv
  2e draft      a first pick of squares per stratum -> draft_selection.csv          (review gate: accept or edit it)
  2f split      tuning and evaluation halves, sequence by sequence -> region_halves.csv, sequence_halves.parquet
  2g represent  selected regions against the whole area; flags cameras whose clock is off by the UTC offset
                -> representativeness.csv, representativeness_summary.csv, clock_suspects.csv
  2h lists      frame and sequence lists to download -> selection/frames.parquet, selection/sequences.csv
The accepted selection is <02_regions>/selection_accepted.csv when present (written at the review gate),
else draft_selection.csv.
"""
from __future__ import annotations

import itertools
import json
import os

import numpy as np
import pandas as pd
import shapely
from shapely.geometry import box, mapping
from shapely import wkb as swkb
from shapely.ops import transform

from ..config import Config
from ..net import now

ARTERIAL = {"trunk", "primary", "secondary", "tertiary"}
ROAD_CLASS = {"motorway": "expressway", "motorway_link": "expressway", "trunk": "major arterial",
              "trunk_link": "major arterial", "primary": "major arterial", "primary_link": "major arterial",
              "secondary": "minor arterial", "secondary_link": "minor arterial", "tertiary": "collector",
              "tertiary_link": "collector"}
HOURS = [(0, 6, "00-06 night"), (6, 8, "06-08 dawn"), (8, 12, "08-12 morning"), (12, 16, "12-16 midday"),
         (16, 19, "16-19 afternoon"), (19, 20, "19-20 dusk"), (20, 24, "20-24 night")]
PASS_BINS = [(0, 0, "0"), (1, 1, "1"), (2, 2, "2"), (3, 4, "3-4"), (5, 9, "5-9"), (10, 10**9, "10+")]
COLS = ["whole", "selected", "tuning", "evaluation"]


def d2(cfg):
    return cfg.paths.step(2)


def s2(cfg, key, default=None):
    return cfg.get(f"step2_regions.{key}", default)


def cell_xy(cfg, x, y):
    m = s2(cfg, "cell_m")
    return np.floor(np.asarray(x) / m).astype(int).astype(str) + "_" + np.floor(np.asarray(y) / m).astype(int).astype(str)


def _s1(cfg, *p):
    return os.path.join(cfg.paths.step(1), *p)


def _dup_drops(cfg):
    dup = pd.read_csv(_s1(cfg, "osm", "duplicate_pairs.csv"), dtype=str)
    drop_m = set(dup[dup.duplicate.eq("True")].sequence_id_m)
    kk = pd.read_csv(_s1(cfg, "osm", "kartaview_internal_duplicates.csv"), dtype=str)
    drop_k = set(np.where(kk.sequence_a.astype(int) > kk.sequence_b.astype(int), kk.sequence_a, kk.sequence_b))
    return drop_m, drop_k


# ---------------------------------------------------------------- 2a features
def features(cfg: Config, log, **_):
    import osmium
    bbox = cfg.get("area.bbox_osm")
    landuse = set(s2(cfg, "landuse_kinds", ["industrial", "residential", "commercial", "retail", "port", "military",
                                             "construction", "grass", "forest", "recreation_ground", "cemetery",
                                             "farmland", "brownfield", "railway"]))
    way_tags = ["highway", "tunnel", "bridge", "layer", "lanes", "maxspeed", "junction", "oneway", "parking:lane:both",
                "parking:lane:left", "parking:lane:right", "parking:both", "parking:left", "parking:right", "covered",
                "man_made", "name", "ref"]
    to_m, fab = cfg.to_m, osmium.geom.WKBFactory()
    inside = lambda lon, lat: bbox[0] <= lon <= bbox[2] and bbox[1] <= lat <= bbox[3]  # noqa: E731

    class H(osmium.SimpleHandler):
        def __init__(self):
            super().__init__()
            self.ways, self.bld, self.lu, self.trees = [], [], [], []

        def node(self, n):
            if n.tags.get("natural") == "tree" and n.location.valid() and inside(n.location.lon, n.location.lat):
                x, y = to_m(n.location.lon, n.location.lat)
                self.trees.append({"osm_id": f"n{n.id}", "kind": "tree", "wkt": f"POINT ({x:.1f} {y:.1f})"})

        def way(self, w):
            t = w.tags
            if "highway" in t or t.get("natural") == "tree_row":
                try:
                    n0 = w.nodes[0].location
                    if not inside(n0.lon, n0.lat):
                        return
                except Exception:  # noqa: BLE001
                    return
                if t.get("natural") == "tree_row":
                    try:
                        g = transform(to_m, swkb.loads(fab.create_linestring(w), hex=True))
                        self.trees.append({"osm_id": f"w{w.id}", "kind": "tree_row", "wkt": g.wkt})
                    except Exception:  # noqa: BLE001
                        pass
                if "highway" in t:
                    self.ways.append({"way_id": w.id, **{k: t.get(k) for k in way_tags}})

        def area(self, a):
            t = a.tags
            b, lu = t.get("building"), t.get("landuse")
            if not b and lu not in landuse:
                return
            try:
                g = swkb.loads(fab.create_multipolygon(a), hex=True)
            except Exception:  # noqa: BLE001
                return
            c = g.representative_point()
            if not inside(c.x, c.y):
                return
            g = transform(to_m, g)
            if b:
                c = g.centroid
                self.bld.append({"osm_id": a.orig_id(), "from_way": a.from_way(), "building": b,
                                 "levels": t.get("building:levels"), "height": t.get("height"),
                                 "x": round(c.x, 1), "y": round(c.y, 1), "area_m2": round(g.area, 1)})
            if lu in landuse:
                self.lu.append({"osm_id": a.orig_id(), "from_way": a.from_way(), "landuse": lu,
                                "area_m2": round(g.area, 1), "wkt": shapely.set_precision(g, 0.5).wkt})

    out = os.path.join(d2(cfg), "osm")
    os.makedirs(out, exist_ok=True)
    h = H()
    h.apply_file(cfg.osm_extract, locations=True)
    counts = {}
    for name, rows in (("way_attrs", h.ways), ("buildings", h.bld), ("landuse", h.lu), ("trees", h.trees)):
        pd.DataFrame(rows).to_parquet(os.path.join(out, f"{name}.parquet"), index=False)
        counts[name] = len(rows)
    log.info(f"OSM features: {counts}")
    return [out], counts


# ---------------------------------------------------------------- 2b cells
def _num(s):
    return pd.to_numeric(pd.Series(s).astype(str).str.extract(r"([\d.]+)")[0], errors="coerce")


def _wmedian(v, w):
    o = np.argsort(v)
    cw = np.cumsum(w[o])
    return float(v[o][np.searchsorted(cw, cw[-1] / 2)])


def cells(cfg: Config, log, **_):
    out, feat = d2(cfg), os.path.join(d2(cfg), "osm")
    m_ = s2(cfg, "cell_m")
    street = set(cfg.get("osm.street_highways"))
    tall = s2(cfg, "tall_building")
    bt = s2(cfg, "building_types")
    landed_set, blocks_set = set(bt["landed"]), set(bt["blocks"])
    small_m2 = s2(cfg, "block_min_footprint_m2", 600)
    rules = s2(cfg, "rules", {"min_own_share_each_platform": 0.2, "min_share_passes": 0.5, "max_top_contributor_share": 0.5})
    min_passes, min_caps = s2(cfg, "min_passes"), s2(cfg, "min_captures_per_cell")
    poly = cfg.boundary_m
    shapely.prepare(poly)
    seg = pd.read_parquet(_s1(cfg, "osm", "combined_segment_passes.parquet"))
    geo = pd.read_parquet(_s1(cfg, "osm", "segments.parquet"), columns=["segment_id", "way_id", "wkt"])
    seg = seg.merge(geo, on="segment_id")
    mid = shapely.line_interpolate_point(shapely.from_wkt(seg.wkt.to_numpy()), 0.5, normalized=True)
    seg["cell"] = cell_xy(cfg, shapely.get_x(mid), shapely.get_y(mid))
    seg["street"] = seg.highway.isin(street)
    seg["own_mly"] = (seg.passes_mly - seg.passes_counted_twice) > 0
    seg["own_kv"] = (seg.passes_kv - seg.passes_counted_twice) > 0
    seg = seg.merge(pd.read_parquet(os.path.join(feat, "way_attrs.parquet"), columns=["way_id", "tunnel", "bridge"]),
                    on="way_id", how="left")
    st = seg[seg.street].copy()
    ws = lambda df, m: (df.length_m * m).sum() / df.length_m.sum()  # noqa: E731
    rows = {}
    for c, d in st.groupby("cell"):
        L = d.length_m
        mw = d.highway.isin({"motorway", "motorway_link"})
        rows[c] = {"street_km": L.sum() / 1000, "share_ge1": ws(d, d.passes_combined >= 1),
                   "share_ge3": ws(d, d.passes_combined >= min_passes), "share_ge5": ws(d, d.passes_combined >= 5),
                   "share_ge1_mly": ws(d, d.passes_mly >= 1), "share_ge1_kv": ws(d, d.passes_kv >= 1),
                   "share_ge3_mly": ws(d, d.passes_mly >= min_passes), "share_ge3_kv": ws(d, d.passes_kv >= min_passes),
                   "share_own_mly": ws(d, d.own_mly), "share_own_kv": ws(d, d.own_kv),
                   "share_own_both": ws(d, d.own_mly & d.own_kv),
                   "median_passes": _wmedian(d.passes_combined.to_numpy(), L.to_numpy()),
                   "motorway_km": L[mw].sum() / 1000, "motorway_tunnel_km": L[mw & d.tunnel.eq("yes")].sum() / 1000,
                   "motorway_bridge_km": L[mw & d.bridge.isin(["yes", "viaduct"])].sum() / 1000,
                   "residential_share": ws(d, d.highway.isin({"residential", "living_street"})),
                   "arterial_km": L[d.highway.isin(ARTERIAL)].sum() / 1000}
    C = pd.DataFrame.from_dict(rows, orient="index")
    C.index.name = "cell"
    ij = C.index.to_series().str.split("_", expand=True).astype(int)
    C["i"], C["j"] = ij[0].to_numpy(), ij[1].to_numpy()
    C["cell_in_area_km2"] = [poly.intersection(box(i * m_, j * m_, (i + 1) * m_, (j + 1) * m_)).area / 1e6 for i, j in zip(C.i, C.j)]
    sg = shapely.from_wkt(st.wkt.to_numpy())
    ends = np.concatenate([shapely.get_coordinates(shapely.get_point(sg, 0)), shapely.get_coordinates(shapely.get_point(sg, -1))])
    deg = pd.Series([f"{x:.1f}_{y:.1f}" for x, y in ends]).value_counts()
    jn = deg[deg >= 3].index.str.split("_", expand=True).to_frame().astype(float)
    C["junctions"] = pd.Series(cell_xy(cfg, jn[0], jn[1])).value_counts().reindex(C.index).fillna(0).astype(int)
    b = pd.read_parquet(os.path.join(feat, "buildings.parquet"))
    b = b[shapely.contains_xy(poly, b.x, b.y)]
    b["cell"] = cell_xy(cfg, b.x, b.y)
    lv, ht = _num(b.levels).to_numpy(), _num(b.height).to_numpy()
    b["tall"] = (lv >= tall["min_levels"]) | (ht >= tall["min_height_m"])
    small = b.area_m2.to_numpy() < small_m2
    b["landed"] = b.building.isin(landed_set) | (b.building.isin(blocks_set) & ((lv <= 4) | (np.isnan(lv) & small)))
    b["block"] = b.building.isin(blocks_set) & ((lv >= 5) | (np.isnan(lv) & ~small))
    gb = b.groupby("cell")
    C["tall_buildings"] = gb.tall.sum().reindex(C.index).fillna(0).astype(int)
    C["landed_houses"] = gb.landed.sum().reindex(C.index).fillna(0).astype(int)
    C["apartment_blocks"] = gb.block.sum().reindex(C.index).fillna(0).astype(int)
    C["building_cover"] = (gb.area_m2.sum().reindex(C.index).fillna(0) / 1e6 / C.cell_in_area_km2.clip(lower=0.05)).clip(upper=1)
    lu = pd.read_parquet(os.path.join(feat, "landuse.parquet"))
    ind = shapely.union_all(shapely.make_valid(shapely.from_wkt(
        lu[lu.landuse.isin(s2(cfg, "industrial_landuse", ["industrial", "port"]))].wkt.to_numpy())))
    shapely.prepare(ind)
    share = []
    for i, j in zip(C.i, C.j):
        xs, ys = np.meshgrid(np.arange(i * m_ + 25, (i + 1) * m_, 50), np.arange(j * m_ + 25, (j + 1) * m_, 50))
        xs, ys = xs.ravel(), ys.ravel()
        inn = shapely.contains_xy(poly, xs, ys)
        share.append(float(shapely.contains_xy(ind, xs[inn], ys[inn]).mean()) if inn.any() else 0.0)
    C["industrial_share"] = share
    t = pd.read_parquet(os.path.join(feat, "trees.parquet"))
    tg = shapely.from_wkt(t.wkt.to_numpy())
    art = st[st.highway.isin(ARTERIAL)]
    ai, ti = shapely.STRtree(tg).query(shapely.from_wkt(art.wkt.to_numpy()), predicate="dwithin", distance=20)
    near = pd.DataFrame({"cell": art.cell.to_numpy()[ai], "t": ti}).drop_duplicates()
    w = np.where(t.kind.to_numpy()[near.t] == "tree_row", shapely.length(tg[near.t]) / 10, 1.0)
    C["arterial_trees"] = pd.Series(w).groupby(near.cell.to_numpy()).sum().reindex(C.index).fillna(0)
    C["arterial_trees_per_km"] = C.arterial_trees / C.arterial_km.replace(0, np.nan)
    drop_m, drop_k = _dup_drops(cfg)
    seg_cell = st.set_index("segment_id").cell
    parts = []
    for pf, drop in (("mapillary", drop_m), ("kartaview", drop_k)):
        cs = pd.read_parquet(_s1(cfg, "osm", f"{pf}_capture_segments.parquet"),
                             columns=["sequence_id", "contributor", "local_date", "date_ok", "segment_id", "device"])
        cs = cs[cs.segment_id.isin(seg_cell.index) & cs.date_ok & ~cs.sequence_id.isin(drop)]
        cs["cell"] = cs.segment_id.map(seg_cell)
        q = cs.groupby(["cell", "sequence_id"]).agg(n=("segment_id", "size"), contributor=("contributor", "first"),
                                                     device=("device", "first"), year=("local_date", lambda s: s.iloc[0][:4]))
        q = q[q.n >= min_caps].reset_index()
        q["platform"] = pf
        parts.append(q)
    Q = pd.concat(parts, ignore_index=True)
    Q.to_parquet(os.path.join(out, "cell_sequences.parquet"), index=False)
    gq = Q.groupby("cell")
    C["seq_mly"] = Q[Q.platform == "mapillary"].groupby("cell").size().reindex(C.index).fillna(0).astype(int)
    C["seq_kv"] = Q[Q.platform == "kartaview"].groupby("cell").size().reindex(C.index).fillna(0).astype(int)
    C["contributors"] = gq.apply(lambda d: (d.platform + ":" + d.contributor).nunique()).reindex(C.index).fillna(0).astype(int)
    C["top_contributor_share"] = gq.apply(lambda d: (d.platform + ":" + d.contributor).value_counts(normalize=True).iloc[0]).reindex(C.index)
    C["top_device_share"] = gq.apply(lambda d: d.device.str.lower().value_counts(normalize=True).iloc[0]).reindex(C.index)
    C["years"] = gq.year.apply(lambda s: ";".join(sorted(s.unique()))).reindex(C.index)
    C["n_years"] = gq.year.nunique().reindex(C.index).fillna(0).astype(int)
    r = rules["min_own_share_each_platform"]
    C["rule_both_platforms"] = (C.share_own_mly >= r) & (C.share_own_kv >= r)
    C["rule_passes"] = C.share_ge3 >= rules["min_share_passes"]
    C["rule_contributor"] = C.top_contributor_share <= rules["max_top_contributor_share"]
    C = C.reset_index()
    C["lon"], C["lat"] = cfg.to_ll(((C.i + 0.5) * m_).to_numpy(), ((C.j + 0.5) * m_).to_numpy())
    C = C.round(4)
    C.to_parquet(os.path.join(out, "cells.parquet"), index=False)
    feats = []
    for rr in C.itertuples():
        g0 = transform(cfg.to_ll, box(rr.i * m_, rr.j * m_, (rr.i + 1) * m_, (rr.j + 1) * m_))
        props = {k: (None if isinstance(v, float) and np.isnan(v) else (v.item() if hasattr(v, "item") else v))
                 for k, v in rr._asdict().items() if k != "Index"}
        feats.append({"type": "Feature", "geometry": mapping(g0), "properties": props})
    json.dump({"type": "FeatureCollection", "features": feats}, open(os.path.join(out, "cells.geojson"), "w", encoding="utf8"))
    sm = {"cells": len(C), "street_km": round(float(C.street_km.sum()), 1),
          "cells_all_three_rules": int((C.rule_both_platforms & C.rule_passes & C.rule_contributor).sum())}
    json.dump(sm, open(os.path.join(out, "cells_log.json"), "w", encoding="utf8"), indent=1)
    log.info(f"{len(C)} cells, {sm['cells_all_three_rules']} meet all three plan rules")
    return [os.path.join(out, "cells.parquet"), os.path.join(out, "cell_sequences.parquet")], sm


# ---------------------------------------------------------------- 2c local layers
def layers(cfg: Config, log, **_):
    out = d2(cfg)
    C = pd.read_parquet(os.path.join(out, "cells.parquet"), columns=["cell"])
    O = pd.read_parquet(os.path.join(out, "osm", "buildings.parquet"))
    tall, bt = s2(cfg, "tall_building"), s2(cfg, "building_types")
    blocks_set = set(bt["blocks"])
    lv = pd.to_numeric(O.levels, errors="coerce").to_numpy()
    ht = pd.to_numeric(O.height.astype(str).str.extract(r"([\d.]+)")[0], errors="coerce").to_numpy()
    O["tall"] = (lv >= tall["min_levels"]) | (ht >= tall["min_height_m"])
    O["cell"] = cell_xy(cfg, O.x, O.y)
    O["is_hdb"] = False
    lay = (s2(cfg, "local_layers", {}) or {}).get("hdb_blocks")
    C = C.set_index("cell")
    sm = {"layer": None}
    if lay and lay.get("file") and os.path.exists(cfg.resolve(lay["file"])):
        d = json.load(open(cfg.resolve(lay["file"]), encoding="utf8"))
        tr = cfg.to_m
        geoms = shapely.transform(shapely.from_geojson([json.dumps(f["geometry"]) for f in d["features"]]),
                                  lambda xy: np.column_stack(tr(xy[:, 0], xy[:, 1])))
        props = pd.DataFrame([f["properties"] for f in d["features"]])
        rp = shapely.point_on_surface(geoms)
        idf = lay.get("id_fields", {"postal": "POSTAL_COD", "blk_no": "BLK_NO", "street_code": "ST_COD"})
        B = pd.DataFrame({"postal": props[idf["postal"]].astype(str).str.zfill(6) if idf.get("postal") in props else None,
                          "blk_no": props.get(idf.get("blk_no")), "street_code": props.get(idf.get("street_code")),
                          "area_m2": shapely.area(geoms).round(1),
                          "x": shapely.get_x(rp).round(1), "y": shapely.get_y(rp).round(1)})
        B["cell"] = cell_xy(cfg, B.x, B.y)
        B["kept"] = B.area_m2 >= lay["min_block_m2"]
        B.assign(wkt=shapely.to_wkt(shapely.set_precision(geoms, 0.1))).to_parquet(os.path.join(out, "hdb_blocks.parquet"), index=False)
        hit = shapely.STRtree(geoms).query(shapely.points(O.x.to_numpy(), O.y.to_numpy()), predicate="dwithin",
                                           distance=lay["match_buffer_m"])
        O.loc[O.index[np.unique(hit[0])], "is_hdb"] = True
        K = B[B.kept]
        C["hdb_blocks"] = K.groupby("cell").size().reindex(C.index).fillna(0).astype(int)
        C["hdb_footprint_m2"] = K.groupby("cell").area_m2.sum().reindex(C.index).fillna(0).round(0)
        sm = {"layer": os.path.basename(lay["file"]), "blocks_kept": int(B.kept.sum()), "osm_matched": int(O.is_hdb.sum())}
    else:
        C["hdb_blocks"], C["hdb_footprint_m2"] = 0, 0.0
        log.info("no local footprint layer: hdb_blocks set to 0; strata that need it are switched off")
    g = O.groupby("cell")
    C["tall_hdb"] = g.apply(lambda s: int((s.tall & s.is_hdb).sum())).reindex(C.index).fillna(0).astype(int)
    C["tall_non_hdb"] = g.apply(lambda s: int((s.tall & ~s.is_hdb).sum())).reindex(C.index).fillna(0).astype(int)
    C["tall_non_res"] = g.apply(lambda s: int((s.tall & ~s.is_hdb & ~s.building.isin(blocks_set | {"house"})).sum())).reindex(C.index).fillna(0).astype(int)
    other = O[O.building.isin(blocks_set) & ~O.is_hdb & (O.area_m2 >= 600)].groupby("cell").size()
    C["other_apartment_blocks"] = other.reindex(C.index).fillna(0).astype(int)
    C["hdb_share_blocks"] = (C.hdb_blocks / (C.hdb_blocks + C.other_apartment_blocks)).fillna(0).round(3)
    C.reset_index().to_parquet(os.path.join(out, "cells_hdb.parquet"), index=False)
    return [os.path.join(out, "cells_hdb.parquet")], sm


# ---------------------------------------------------------------- 2d strata
def strata(cfg: Config, log, **_):
    out = d2(cfg)
    m_ = s2(cfg, "cell_m")
    have_layer = bool(((s2(cfg, "local_layers", {}) or {}).get("hdb_blocks") or {}).get("file"))
    C = pd.read_parquet(os.path.join(out, "cells.parquet")).merge(pd.read_parquet(os.path.join(out, "cells_hdb.parquet")),
                                                                  on="cell", how="left")
    C["motorway_tunnel_bridge"] = C.motorway_tunnel_km + C.motorway_bridge_km
    st = {k: v for k, v in s2(cfg, "strata").items() if have_layer or not v.get("needs_layer")}
    for k, v in st.items():
        C[f"s_{k}"] = C.eval(v["rule"]).astype(bool)
    C["rules_met"] = C[["rule_both_platforms", "rule_passes", "rule_contributor"]].sum(axis=1)
    C["valid"] = C.rules_met == 3
    S = pd.read_parquet(_s1(cfg, "osm", "segments.parquet"), columns=["way_id", "highway", "length_m", "wkt"])
    S = S[S.highway.isin(["motorway", "trunk", "primary", "secondary", "tertiary", "residential"])]
    S = S.merge(pd.read_parquet(os.path.join(out, "osm", "way_attrs.parquet"), columns=["way_id", "name"]).dropna(), on="way_id")
    mid = shapely.line_interpolate_point(shapely.from_wkt(S.wkt.to_numpy()), 0.5, normalized=True)
    S["cell"] = cell_xy(cfg, shapely.get_x(mid), shapely.get_y(mid))
    lab = S.groupby(["cell", "name"]).length_m.sum().reset_index().sort_values("length_m", ascending=False)
    C["label"] = C.cell.map(lab.groupby("cell").name.apply(lambda s: " / ".join(s.head(2)))).fillna("")
    C = C.drop(columns="motorway_tunnel_bridge")
    C.to_parquet(os.path.join(out, "cells_strata.parquet"), index=False)
    order = ["rules_met", "share_own_both", "share_ge3", "contributors"]
    summ, tops = [], []
    for k in st:
        m = C[C[f"s_{k}"]]
        v = m[m.valid]
        summ.append({"stratum": k, "candidate_cells": len(m), "valid_cells": len(v),
                     "fail_both_platforms": int((~m.rule_both_platforms).sum()), "fail_passes": int((~m.rule_passes).sum()),
                     "fail_contributor": int((~m.rule_contributor).sum()), "street_km_valid": round(float(v.street_km.sum()), 1),
                     "seq_mly_valid": int(v.seq_mly.sum()), "seq_kv_valid": int(v.seq_kv.sum()),
                     "median_contributors_valid": float(v.contributors.median()) if len(v) else None})
        tops.append(m.sort_values(order, ascending=False).head(8).assign(stratum=k))
    Sm = pd.DataFrame(summ)
    Sm.to_csv(os.path.join(out, "strata_summary.csv"), index=False)
    pd.concat(tops, ignore_index=True).to_csv(os.path.join(out, "strata_top.csv"), index=False)
    off = [k for k, v in s2(cfg, "strata").items() if k not in st]
    log.info(f"{int(C.valid.sum())} valid cells; strata switched off: {off or 'none'}")
    return [os.path.join(out, "cells_strata.parquet")], {"cells": len(C), "valid": int(C.valid.sum()), "strata_off": off}


# ---------------------------------------------------------------- 2e draft
def draft(cfg: Config, log, **_):
    out = d2(cfg)
    sel, st = s2(cfg, "selection"), s2(cfg, "strata")
    m_, buf = s2(cfg, "cell_m"), s2(cfg, "clip_buffer_frames")
    mb, budget = cfg.get("storage.mb_per_frame"), cfg.get("storage.budget_gb")
    C = pd.read_parquet(os.path.join(out, "cells_strata.parquet"))
    C["motorway_tunnel_bridge"] = C.motorway_tunnel_km + C.motorway_bridge_km
    used, picks = set(), []
    for k in [s for s in sel["order"] if f"s_{s}" in C]:
        m = C[C[f"s_{k}"] & C.valid & ~C.cell.isin(used)].copy()
        m["feature_pct"] = C[st[k]["feature"]].rank(pct=True).reindex(m.index)
        m["score"] = 0.5 * m.feature_pct + 0.5 * m.share_own_both
        gap, n = st[k].get("min_gap_m", sel["min_gap_m"]), st[k].get("picks", sel["picks_per_stratum"])
        got = []
        for _, r in m.sort_values("score", ascending=False).iterrows():
            if all(np.hypot((r.i - g.i) * m_, (r.j - g.j) * m_) >= gap for g in got):
                got.append(r)
            if len(got) == n:
                break
        for r in got:
            used.add(r.cell)
            picks.append({"stratum": k, **r[["cell", "label", "lon", "lat", "score", "feature_pct", "street_km", "share_ge3",
                                             "share_own_both", "seq_mly", "seq_kv", "contributors", "top_contributor_share",
                                             "years"]].to_dict()})
    P = pd.DataFrame(picks)
    if P.empty:
        R = C[["rule_both_platforms", "rule_passes", "rule_contributor"]].sum()
        raise RuntimeError(
            f"no square of the {len(C)} meets all three plan rules (both platforms: {int(R.rule_both_platforms)}, passes: "
            f"{int(R.rule_passes)}, contributor: {int(R.rule_contributor)} squares pass), so no region can be picked. "
            "Relax step2_regions.rules or min_passes (and record why), or use step2_regions.mode: whole_area.")
    S = pd.read_parquet(os.path.join(out, "cell_sequences.parquet"))
    S = S[S.cell.isin(P.cell)].copy()
    S["frames"] = S.n + 2 * buf
    S["gb"] = S.frames * S.platform.map(mb) / 1000
    F = S.pivot_table(index="cell", columns="platform", values=["frames", "gb"], aggfunc="sum", fill_value=0)
    F.columns = [f"{a}_{b[:3]}" for a, b in F.columns]
    P = P.merge(F.reset_index(), on="cell", how="left")
    P["gb"] = P.gb_kar + P.gb_map
    P.to_csv(os.path.join(out, "draft_selection.csv"), index=False)
    gb = round(float(P.gb.sum()), 1)
    log.info(f"draft: {len(P)} squares, about {gb} GB of images (budget {budget} GB)")
    return [os.path.join(out, "draft_selection.csv")], {"squares": len(P), "est_gb": gb, "fits_budget": gb <= budget}


def accepted(cfg: Config) -> str:
    p = os.path.join(d2(cfg), "selection_accepted.csv")
    return p if os.path.exists(p) else os.path.join(d2(cfg), "draft_selection.csv")


# ---------------------------------------------------------------- 2f split
def split(cfg: Config, log, **_):
    out = d2(cfg)
    sp = s2(cfg, "split")
    buf = s2(cfg, "clip_buffer_frames")
    lo, hi = sp.get("balance", [0.40, 0.60])
    ry, gapmax = sp["recent_year"], sp["max_recent_gap"]
    P = pd.read_csv(accepted(cfg))
    S = pd.read_parquet(os.path.join(out, "cell_sequences.parquet"))
    S = S[S.cell.isin(P.cell)].merge(P[["cell", "stratum"]], on="cell")
    S["frames"] = S.n + 2 * buf
    S["year"] = S.year.astype(int)
    cl = P.cell.tolist()
    ci = {c: k for k, c in enumerate(cl)}
    S["key"] = S.platform + ":" + S.sequence_id
    seqs = S.key.unique()
    si = {s: k for k, s in enumerate(seqs)}
    M = np.zeros((len(seqs), len(cl)), dtype=np.int16)
    M[S.key.map(si), S.cell.map(ci)] = 1
    recent = S.drop_duplicates("key").set_index("key").year.reindex(seqs).to_numpy() >= ry
    Fm = np.zeros((2, len(cl)))
    for (pf, c), v in S.groupby(["platform", "cell"]).frames.sum().items():
        Fm[0 if pf == "mapillary" else 1, ci[c]] = v
    opts = []
    for _, g in P.groupby("stratum", sort=False):
        idx = [ci[c] for c in g.cell]
        opts.append([tuple(idx[j] for j in comb) for comb in itertools.combinations(range(len(idx)), (len(idx) + 1) // 2)])
    best, n_ok = None, 0
    for choice in itertools.product(*opts):
        a = np.zeros(len(cl), dtype=np.int16)
        for comb in choice:
            a[list(comb)] = 1
        ev = (Fm @ a) / Fm.sum(axis=1)
        if not ((ev >= lo) & (ev <= hi)).all():
            continue
        te, tt = (M @ a) > 0, (M @ (1 - a)) > 0
        gap = abs(float(recent[te].mean() - recent[tt].mean()))
        if gap > gapmax:
            continue
        n_ok += 1
        cross, dev = int((te & tt).sum()), float(np.abs(ev - 0.5).max())
        if best is None or (cross, dev) < best[0]:
            best = ((cross, dev), a.copy(), ev, gap)
    if best is None:
        raise RuntimeError("no split of the selected squares meets the balance rules; review the selection")
    (cross, dev), a, ev, gap = best
    half = {c: ("evaluation" if a[ci[c]] else "tuning") for c in cl}
    S["half_cell"] = S.cell.map(half)
    cap = S.groupby(["key", "half_cell"]).n.sum().unstack(fill_value=0)
    for h in ("evaluation", "tuning"):
        if h not in cap:
            cap[h] = 0
    S["half"] = S.key.map(pd.Series(np.where(cap.evaluation >= cap.tuning, "evaluation", "tuning"), index=cap.index))
    S["kept"] = S.half == S.half_cell
    dropped = S[~S.kept]
    K = S[S.kept].drop(columns=["key", "half_cell", "kept"])
    K.to_parquet(os.path.join(out, "sequence_halves.parquet"), index=False)
    R = P[["stratum", "cell", "label"]].copy()
    R["half"] = R.cell.map(half)
    g = K.groupby(["cell", "platform"]).agg(seqs=("sequence_id", "size"), frames=("frames", "sum")).unstack(fill_value=0)
    g.columns = [f"{x}_{'mly' if y == 'mapillary' else 'kv'}" for x, y in g.columns]
    R = R.merge(g.reset_index(), on="cell", how="left").fillna(0)
    R["clips_dropped"] = R.cell.map(dropped.groupby("cell").size()).fillna(0).astype(int)
    R.to_csv(os.path.join(out, "region_halves.csv"), index=False)
    sm = {"splits_balanced": n_ok, "sequences_touching_both_halves": cross, "recent_share_gap": round(gap, 3),
          "evaluation_frame_share": {"mapillary": round(float(ev[0]), 3), "kartaview": round(float(ev[1]), 3)},
          "clips_dropped": int(len(dropped))}
    json.dump(sm, open(os.path.join(out, "split_log.json"), "w", encoding="utf8"), indent=1)
    log.info(f"split: {n_ok} balanced splits, {cross} sequences touch both halves, {len(dropped)} clips dropped")
    return [os.path.join(out, "region_halves.csv"), os.path.join(out, "sequence_halves.parquet")], sm


# ---------------------------------------------------------------- 2g representativeness and camera clocks
def _bins(v, table, closed_hi=False):
    out = np.empty(len(v), dtype=object)
    for lo, hi, name in table:
        out[(v >= lo) & ((v <= hi) if closed_hi else (v < hi))] = name
    return out


def represent(cfg: Config, log, **_):
    out = d2(cfg)
    street, tz = set(cfg.get("osm.street_highways")), cfg.get("area.timezone")
    ck_cfg = cfg.get("step1_harvest.clock_check")
    top_devices = 6
    seg = pd.read_parquet(_s1(cfg, "osm", "combined_segment_passes.parquet"))
    geo = pd.read_parquet(_s1(cfg, "osm", "segments.parquet"), columns=["segment_id", "wkt"])
    seg = seg[seg.highway.isin(street)].merge(geo, on="segment_id")
    mid = shapely.line_interpolate_point(shapely.from_wkt(seg.wkt.to_numpy()), 0.5, normalized=True)
    seg["cell"] = cell_xy(cfg, shapely.get_x(mid), shapely.get_y(mid))
    seg["road_class"] = seg.highway.map(ROAD_CLASS).fillna("local")
    seg = seg.drop(columns="wkt")
    seg = seg[seg.cell.isin(set(pd.read_parquet(os.path.join(out, "cells.parquet"), columns=["cell"]).cell))]
    H = pd.read_csv(os.path.join(out, "region_halves.csv"))
    half_of_cell = dict(zip(H.cell, H.half))
    K = pd.read_parquet(os.path.join(out, "sequence_halves.parquet"), columns=["cell", "sequence_id", "platform", "half"])
    drop_m, drop_k = _dup_drops(cfg)
    seg_i = seg.set_index("segment_id")
    rows, summary, rl, clocks = [], [], {}, []

    def shares(df, col, weight=None):
        res = {}
        for c in COLS:
            d = df if c == "whole" else df[df.sel] if c == "selected" else df[df.half == c]
            w = d[weight] if weight else pd.Series(1.0, index=d.index)
            s = w.groupby(d[col]).sum()
            res[c] = s / s.sum() if s.sum() else s
        return pd.DataFrame(res).fillna(0)

    def add(dim, platform, unit, T):
        for cat, r in T.iterrows():
            rows.append({"dimension": dim, "platform": platform, "unit": unit, "category": cat, **{c: round(float(r[c]), 4) for c in COLS}})
        summary.append({"dimension": dim, "platform": platform, "unit": unit,
                        **{f"tvd_{c}": round(float(0.5 * (T[c] - T.whole).abs().sum()), 3) for c in ("selected", "tuning", "evaluation")}})

    nh0, nh1 = ck_cfg["night_hours"]
    dh0, dh1 = ck_cfg["shifted_day_hours"]
    for pf, drop in (("mapillary", drop_m), ("kartaview", drop_k)):
        cs = pd.read_parquet(_s1(cfg, "osm", f"{pf}_capture_segments.parquet"),
                             columns=["sequence_id", "contributor", "t", "local_date", "date_ok", "segment_id", "device"])
        cs = cs[cs.segment_id.isin(seg_i.index) & cs.date_ok & ~cs.sequence_id.isin(drop)].copy()
        cs["cell"] = cs.segment_id.map(seg_i.cell)
        cs["road_class"] = cs.segment_id.map(seg_i.road_class)
        cs = cs.merge(K[K.platform == pf][["cell", "sequence_id", "half"]], on=["cell", "sequence_id"], how="left")
        cs["sel"] = cs.half.notna()
        cs["device"] = cs.device.replace("", np.nan).fillna("unknown")
        top = cs.device.value_counts().head(top_devices).index
        cs["device_g"] = np.where(cs.device.isin(top), cs.device, "other")
        add("device", pf, "captures", shares(cs, "device_g"))
        for c in COLS:
            d = cs if c == "whole" else cs[cs.sel] if c == "selected" else cs[cs.half == c]
            v = d.contributor.value_counts(normalize=True)
            rl.setdefault(pf, {})[f"contributors_{c}"] = int(d.contributor.nunique())
            rl[pf][f"top1_share_{c}"] = round(float(v.iloc[0]), 3) if len(v) else None
        top5 = cs.contributor.value_counts().head(5).index
        cs["contrib_g"] = np.where(cs.contributor.isin(top5), "contributor " + cs.contributor.astype(str), "other contributors")
        add("contributor", pf, "captures", shares(cs, "contrib_g"))
        cs["year"] = cs.local_date.str[:4]
        add("capture year", pf, "captures", shares(cs, "year"))
        lt = cs.t.dt.tz_convert(tz)
        h = lt.dt.hour.astype("float")
        off = (lt.dt.tz_localize(None) - cs.t.dt.tz_localize(None)).dt.total_seconds() / 3600   # the area's UTC offset
        cs["hour_g"] = _bins(h.to_numpy(), HOURS)
        cs.loc[h.isna(), "hour_g"] = "unknown"
        cs["h"], cs["hs"] = h, ((h + off) % 24)
        ck = cs.groupby(["contributor", "device"]).agg(captures=("h", "size"), sequences=("sequence_id", "nunique"),
                                                       share_00_06=("h", lambda x: ((x >= nh0) & (x < nh1)).mean()),
                                                       share_07_19_if_plus8=("hs", lambda x: x.between(dh0, dh1 - 1).mean()),
                                                       captures_selected=("sel", "sum")).reset_index()
        ck = ck[(ck.captures >= ck_cfg["min_captures"]) & (ck.share_00_06 >= ck_cfg["min_night_share"])
                & (ck.share_07_19_if_plus8 >= ck_cfg["min_shifted_day_share"])]
        clocks.append(ck.assign(platform=pf))
        bad = cs.set_index(["contributor", "device"]).index.isin(ck.set_index(["contributor", "device"]).index)
        offs = cfg.utc_offsets_h
        cs.loc[bad, "hour_g"] = f"clock suspect (+{offs[0]:g} h)" if len(offs) == 1 else "clock suspect (+UTC offset)"
        add("time of day", pf, "captures", shares(cs, "hour_g"))
        add("road class", pf, "captures", shares(cs, "road_class"))
    seg["sel"] = seg.cell.isin(half_of_cell)
    seg["half"] = seg.cell.map(half_of_cell)
    seg["km"] = seg.length_m / 1000
    add("road class", "streets", "street km", shares(seg, "road_class", "km"))
    seg["pass_g"] = _bins(seg.passes_combined.to_numpy(), PASS_BINS, closed_hi=True)
    add("passes per segment (combined)", "streets", "street km", shares(seg, "pass_g", "km"))
    pd.concat(clocks).round(3).to_csv(os.path.join(out, "clock_suspects.csv"), index=False)
    pd.DataFrame(rows).to_csv(os.path.join(out, "representativeness.csv"), index=False)
    Sm = pd.DataFrame(summary)
    Sm.to_csv(os.path.join(out, "representativeness_summary.csv"), index=False)
    json.dump(rl, open(os.path.join(out, "representativeness_log.json"), "w", encoding="utf8"), indent=1)
    n_ck = int(sum(len(c) for c in clocks))
    log.info(f"representativeness written; {n_ck} camera(s) flagged with a clock off by the UTC offset")
    return [os.path.join(out, "clock_suspects.csv"), os.path.join(out, "representativeness_summary.csv")], \
        {"clock_suspect_cameras": n_ck, "max_tvd_selected": float(Sm.tvd_selected.max())}


# ---------------------------------------------------------------- 2h frame and sequence lists
def _runs(n, inside, buf):
    keep = np.zeros(n, dtype=bool)
    idx = np.flatnonzero(inside)
    if len(idx):
        d = np.zeros(n + 1, dtype=int)
        np.add.at(d, np.clip(idx - buf, 0, n - 1), 1)
        np.add.at(d, np.clip(idx + buf, 0, n - 1) + 1, -1)
        keep = np.cumsum(d)[:n] > 0
    return keep


def lists(cfg: Config, log, **_):
    s2d = d2(cfg)
    out = os.path.join(s2d, "selection")
    os.makedirs(out, exist_ok=True)
    buf, mb = s2(cfg, "clip_buffer_frames"), cfg.get("storage.mb_per_frame")
    K = pd.read_parquet(os.path.join(s2d, "sequence_halves.parquet"))
    H = pd.read_csv(os.path.join(s2d, "region_halves.csv"))
    half_of_cell, stratum_of_cell = dict(zip(H.cell, H.half)), dict(zip(H.cell, H.stratum))
    clocks = pd.read_csv(os.path.join(s2d, "clock_suspects.csv"), dtype=str)
    clock_keys = set(zip(clocks.platform, clocks.contributor, clocks.device))
    man = cfg.get("existing.annotated_window_manifest", None)
    done = set(pd.read_csv(cfg.resolve(man), dtype=str).iloc[:, 0]) if man else set()
    frames, sl = [], {}
    for pf in ("kartaview", "mapillary"):
        kp = K[K.platform == pf]
        want = set(kp.sequence_id)
        if pf == "kartaview":
            F = pd.read_parquet(_s1(cfg, "kartaview", "photos.parquet"),
                                columns=["photo_id", "sequence_id", "sequenceIndex", "lat", "lng", "heading", "shotDate",
                                         "projection", "width", "height", "status", "visibility", "fileurlLTh", "fileurl", "fileurlProc"])
            F["sequence_id"] = F.sequence_id.astype(str)
            F = F[F.sequence_id.isin(want) & F.status.eq("active") & F.visibility.eq("public")]
            F = F.sort_values(["sequence_id", "sequenceIndex"]).rename(columns={"photo_id": "frame_id", "lng": "lon", "shotDate": "time"})
            F["frame_id"] = F.frame_id.astype(str)
            F["order"] = F.sequenceIndex
            F["is_pano"] = F.projection.eq("SPHERE")
            F["already_downloaded"] = F.frame_id.isin(done)
            F = F.drop(columns=["status", "visibility", "sequenceIndex"])
        else:
            F = pd.read_parquet(_s1(cfg, "mapillary", "captures.parquet"),
                                columns=["id", "sequence_id", "captured_at_ms", "lon", "lat", "compass_angle", "is_pano", "captured_at_utc"])
            F = F[F.sequence_id.isin(want)].sort_values(["sequence_id", "captured_at_ms", "id"])
            F = F.rename(columns={"id": "frame_id", "compass_angle": "heading", "captured_at_utc": "time"})
            F["frame_id"] = F.frame_id.astype(str)
            F["order"] = F.groupby("sequence_id").cumcount()
            F["already_downloaded"] = False
            F = F.drop(columns=["captured_at_ms"])
        x, y = cfg.to_m(F.lon.to_numpy(), F.lat.to_numpy())
        F["cell"] = cell_xy(cfg, x, y)
        clip = kp[["sequence_id", "cell", "half"]]
        F = F.merge(clip.drop_duplicates("sequence_id")[["sequence_id", "half"]], on="sequence_id")
        F = F.merge(clip.assign(kept_cell=True)[["sequence_id", "cell", "kept_cell"]], on=["sequence_id", "cell"], how="left")
        F["kept_cell"] = F.kept_cell.fillna(False).astype(bool)
        keep = np.zeros(len(F), dtype=bool)
        for _, ix in F.groupby("sequence_id").indices.items():
            keep[ix] = _runs(len(ix), F.kept_cell.to_numpy()[ix], buf)
        F = F[keep].copy()
        F["role"] = np.where(F.kept_cell, "in_cell", "buffer")
        oc = F.cell.map(half_of_cell)
        F["in_other_half_cell"] = oc.notna() & (oc != F.half)
        F["region_cell"] = np.where(F.kept_cell, F.cell, "")
        F["stratum"] = F.region_cell.map(stratum_of_cell).fillna("")
        F["platform"] = pf
        frames.append(F.drop(columns=["kept_cell"]))
        sl[pf] = {"sequences": int(F.sequence_id.nunique()), "frames": int(len(F)), "in_cell": int((F.role == "in_cell").sum()),
                  "already_downloaded": int(F.already_downloaded.sum())}
    A = pd.concat(frames, ignore_index=True)
    A["time"] = A.time.astype(str)
    cols = ["platform", "sequence_id", "order", "frame_id", "half", "role", "region_cell", "stratum", "cell",
            "in_other_half_cell", "lon", "lat", "heading", "time", "is_pano", "projection", "width", "height",
            "already_downloaded", "fileurlLTh", "fileurl", "fileurlProc"]
    A = A[cols]
    A.to_parquet(os.path.join(out, "frames.parquet"), index=False)
    m = pd.read_parquet(_s1(cfg, "mapillary", "sequences.parquet"), columns=["sequence_id", "creator_id", "device", "first_date_local"])
    m["platform"] = "mapillary"
    k = pd.read_parquet(_s1(cfg, "kartaview", "sequence_table.parquet"), columns=["sequence_id", "creator_id", "device", "first_date_local"])
    k["sequence_id"] = k.sequence_id.astype(str)
    k["platform"] = "kartaview"
    meta = pd.concat([m, k]).rename(columns={"creator_id": "contributor"})
    meta["contributor"] = meta.contributor.astype(str)
    meta["device"] = meta.device.fillna("unknown").replace("", "unknown")
    meta["year"] = meta.first_date_local.astype(str).str[:4]
    g = A.groupby(["platform", "sequence_id"])
    S = g.agg(half=("half", "first"), frames=("frame_id", "size"),
              frames_in_cell=("role", lambda r: int((r == "in_cell").sum())),
              frames_buffer=("role", lambda r: int((r == "buffer").sum())),
              frames_in_other_half_cell=("in_other_half_cell", "sum"),
              frames_already_downloaded=("already_downloaded", "sum"), pano_share=("is_pano", "mean")).reset_index()
    rc = A[A.role == "in_cell"].groupby(["platform", "sequence_id"]).region_cell.agg(lambda c: ";".join(sorted(set(c))))
    S["cells"] = S.set_index(["platform", "sequence_id"]).index.map(rc)
    S["strata"] = S.cells.map(lambda c: ";".join(sorted({stratum_of_cell[x] for x in c.split(";")})))
    S = S.merge(meta[["platform", "sequence_id", "contributor", "device", "year"]], on=["platform", "sequence_id"], how="left")
    S["clock_suspect"] = [(p, c, d) in clock_keys for p, c, d in zip(S.platform, S.contributor, S.device)]
    # numbers even when nothing is selected (an empty table's mapped column keeps the text type and the division fails)
    S["est_gb"] = ((S.frames.astype("int64") - S.frames_already_downloaded.astype("int64"))
                   * S.platform.map(mb).astype("float64") / 1000).round(3)
    dup = pd.read_csv(_s1(cfg, "osm", "duplicate_pairs.csv"), dtype=str)
    dup = dup[dup.duplicate.eq("True")]
    lk = dup.groupby("sequence_id_k").sequence_id_m.agg(lambda s: ";".join("mapillary:" + s))
    kk = pd.read_csv(_s1(cfg, "osm", "kartaview_internal_duplicates.csv"), dtype=str)
    kk = kk[kk.duplicate.eq("True")]
    lo_ = np.where(kk.sequence_a.astype(int) > kk.sequence_b.astype(int), kk.sequence_b, kk.sequence_a)
    hi_ = np.where(kk.sequence_a.astype(int) > kk.sequence_b.astype(int), kk.sequence_a, kk.sequence_b)
    lk2 = pd.Series(hi_, index=lo_).groupby(level=0).agg(lambda s: ";".join("kartaview:" + s))
    kv = S.platform.eq("kartaview")
    l1, l2 = S.sequence_id.map(lk).where(kv, None), S.sequence_id.map(lk2).where(kv, None)
    S["duplicate_copies_dropped"] = [";".join(x for x in (a, b) if isinstance(x, str)) for a, b in zip(l1, l2)]
    S = S.sort_values(["half", "platform", "cells", "sequence_id"])
    S.to_csv(os.path.join(out, "sequences.csv"), index=False)
    sl["total"] = {"sequences": len(S), "frames": int(len(A)), "est_gb": round(float(S.est_gb.sum()), 1)}
    json.dump(sl, open(os.path.join(out, "selection_log.json"), "w", encoding="utf8"), indent=1)
    log.info(f"selection: {len(S)} sequences, {len(A)} frames, about {sl['total']['est_gb']} GB")
    return [os.path.join(out, "frames.parquet"), os.path.join(out, "sequences.csv")], sl["total"]


# ---------------------------------------------------------------- 2w whole area (no strata)
def _clock_suspects(cfg, drops, pfs=("mapillary", "kartaview")):
    """Cameras whose clock is off by the UTC offset, over every dated capture of the area (whole-area mode)."""
    ck_cfg, tz = cfg.get("step1_harvest.clock_check"), cfg.get("area.timezone")
    nh0, nh1 = ck_cfg["night_hours"]
    dh0, dh1 = ck_cfg["shifted_day_hours"]
    out = []
    for pf in [p for p in ("mapillary", "kartaview") if p in pfs]:   # only the platforms being downloaded
        p = _s1(cfg, "osm", f"{pf}_capture_segments.parquet")
        if not os.path.exists(p):
            continue
        cs = pd.read_parquet(p, columns=["sequence_id", "contributor", "t", "date_ok", "device"])
        cs = cs[cs.date_ok & ~cs.sequence_id.isin(drops[pf])].copy()
        cs["device"] = cs.device.replace("", np.nan).fillna("unknown")
        lt = cs.t.dt.tz_convert(tz)
        cs["h"] = lt.dt.hour.astype("float")
        cs["hs"] = (cs.h + (lt.dt.tz_localize(None) - cs.t.dt.tz_localize(None)).dt.total_seconds() / 3600) % 24
        ck = cs.groupby(["contributor", "device"]).agg(captures=("h", "size"), sequences=("sequence_id", "nunique"),
                                                       share_00_06=("h", lambda x: ((x >= nh0) & (x < nh1)).mean()),
                                                       share_07_19_if_plus8=("hs", lambda x: x.between(dh0, dh1 - 1).mean())).reset_index()
        ck = ck[(ck.captures >= ck_cfg["min_captures"]) & (ck.share_00_06 >= ck_cfg["min_night_share"])
                & (ck.share_07_19_if_plus8 >= ck_cfg["min_shifted_day_share"])]
        out.append(ck.assign(platform=pf))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["platform", "contributor", "device"])


def whole_area(cfg: Config, log, **_):
    """Every capture inside the boundary (whole_area) or inside the drawn area (selected_area), no strata, no squares,
    no halves. Options (step2_regions.whole_area): date_from / date_to (local dates, inclusive; blank = all dates),
    drop_duplicate_drives (whole_area only: keep one copy of a drive uploaded to both platforms or twice to
    KartaView), platforms (default: the enabled ones). In selected_area mode each enabled platform is handled on its
    own: every one of its captures inside the area (within the dates) is kept."""
    wa = s2(cfg, "whole_area", {}) or {}
    sel = None
    if cfg.selection_mode == "selected_area":      # only captures inside the drawn area
        sel = cfg.selected_area_ll
        if sel is None:
            raise RuntimeError("no selected area saved: draw it under Settings > What to download "
                               f"(expected {cfg.selected_area_file})")
        shapely.prepare(sel)
    s2d = d2(cfg)
    out = os.path.join(s2d, "selection")
    os.makedirs(out, exist_ok=True)
    tz, mb, budget = cfg.get("area.timezone"), cfg.get("storage.mb_per_frame"), cfg.get("storage.budget_gb")
    pfs = wa.get("platforms") or [p for p in ("kartaview", "mapillary") if cfg.get(f"platforms.{p}.enabled", False)]
    # selected_area: each ticked platform's captures inside the area are downloaded on their own, all of them (no drive
    # is dropped because it is also on the other platform; 1i does not run). whole_area keeps the duplicate option.
    dedup = sel is None and wa.get("drop_duplicate_drives", True)
    drop_m, drop_k = _dup_drops(cfg) if dedup and os.path.exists(_s1(cfg, "osm", "duplicate_pairs.csv")) else (set(), set())
    drops = {"mapillary": drop_m, "kartaview": drop_k}
    man = cfg.get("existing.annotated_window_manifest", None)
    done = set(pd.read_csv(cfg.resolve(man), dtype=str).iloc[:, 0]) if man else set()
    frames, sl, months = [], {"mode": cfg.selection_mode}, {}
    for pf in pfs:
        if pf == "kartaview":
            F = pd.read_parquet(_s1(cfg, "kartaview", "photos.parquet"),
                                columns=["photo_id", "sequence_id", "sequenceIndex", "lat", "lng", "heading", "shotDate", "projection",
                                         "width", "height", "status", "visibility", "in_area", "fileurlLTh", "fileurl", "fileurlProc"])
            F["sequence_id"] = F.sequence_id.astype(str)
            F = F[F.in_area & F.status.eq("active") & F.visibility.eq("public")]
            F = F.sort_values(["sequence_id", "sequenceIndex"]).rename(columns={"photo_id": "frame_id", "lng": "lon", "shotDate": "time"})
            F["frame_id"] = F.frame_id.astype(str)
            F["order"] = F.sequenceIndex
            F["is_pano"] = F.projection.eq("SPHERE")
            F["already_downloaded"] = F.frame_id.isin(done)
            F["t"] = pd.to_datetime(F.time, utc=True, errors="coerce")
            F = F.drop(columns=["status", "visibility", "sequenceIndex", "in_area"])
        else:
            F = pd.read_parquet(_s1(cfg, "mapillary", "captures.parquet"),
                                columns=["id", "sequence_id", "captured_at_ms", "lon", "lat", "compass_angle", "is_pano", "captured_at_utc"])
            F = F.sort_values(["sequence_id", "captured_at_ms", "id"])
            F = F.rename(columns={"id": "frame_id", "compass_angle": "heading", "captured_at_utc": "time"})
            F["frame_id"] = F.frame_id.astype(str)
            F["order"] = F.groupby("sequence_id").cumcount()
            F["already_downloaded"] = False
            F["t"] = F.time
            for c in ("projection", "width", "height", "fileurlLTh", "fileurl", "fileurlProc"):
                F[c] = None
            F = F.drop(columns=["captured_at_ms"])
        n0 = len(F)
        if sel is not None:
            F = F[shapely.contains_xy(sel, F.lon.to_numpy(), F.lat.to_numpy())]
        n_sel = len(F)
        F = F[~F.sequence_id.isin(drops[pf])]
        ld = F.t.dt.tz_convert(tz).dt.date.astype(str)
        months[pf] = ld.str[:7].value_counts().sort_index().to_dict()   # to explain an empty date range
        if wa.get("date_from"):
            F = F[ld >= str(wa["date_from"])]
        if wa.get("date_to"):
            F = F[ld.reindex(F.index) <= str(wa["date_to"])]
        x, y = cfg.to_m(F.lon.to_numpy(), F.lat.to_numpy())
        F = F.assign(cell=cell_xy(cfg, x, y), half="", role="in_area", stratum="", in_other_half_cell=False, platform=pf)
        F["region_cell"] = F.cell
        frames.append(F.drop(columns=["t"]))
        sl[pf] = {"captures_in_area": n0, "inside_selected_area": n_sel if sel is not None else None,
                  "frames": len(F), "sequences": int(F.sequence_id.nunique()), "dropped_duplicate_or_date": n_sel - len(F), "duplicate_drives_dropped": bool(dedup)}
    A = pd.concat(frames, ignore_index=True)
    A["time"] = A.time.astype(str)
    cols = ["platform", "sequence_id", "order", "frame_id", "half", "role", "region_cell", "stratum", "cell",
            "in_other_half_cell", "lon", "lat", "heading", "time", "is_pano", "projection", "width", "height",
            "already_downloaded", "fileurlLTh", "fileurl", "fileurlProc"]
    A = A[cols]
    A.to_parquet(os.path.join(out, "frames.parquet"), index=False)
    ck = _clock_suspects(cfg, drops, pfs)
    ck.round(3).to_csv(os.path.join(s2d, "clock_suspects.csv"), index=False)
    keys = set(zip(ck.platform, ck.contributor, ck.device))
    meta = []
    if "mapillary" in pfs:
        m = pd.read_parquet(_s1(cfg, "mapillary", "sequences.parquet"), columns=["sequence_id", "creator_id", "device", "first_date_local"])
        meta.append(m.assign(platform="mapillary"))
    if "kartaview" in pfs:
        k = pd.read_parquet(_s1(cfg, "kartaview", "sequence_table.parquet"), columns=["sequence_id", "creator_id", "device", "first_date_local"])
        meta.append(k.assign(platform="kartaview", sequence_id=k.sequence_id.astype(str)))
    meta = pd.concat(meta).rename(columns={"creator_id": "contributor"})
    meta["contributor"] = meta.contributor.astype(str)
    meta["device"] = meta.device.fillna("unknown").replace("", "unknown")
    meta["year"] = meta.first_date_local.astype(str).str[:4]
    S = A.groupby(["platform", "sequence_id"]).agg(frames=("frame_id", "size"), cells=("cell", "nunique"),
                                                   frames_already_downloaded=("already_downloaded", "sum"),
                                                   pano_share=("is_pano", "mean")).reset_index()
    S = S.merge(meta[["platform", "sequence_id", "contributor", "device", "year"]], on=["platform", "sequence_id"], how="left")
    S["clock_suspect"] = [(p, c, d) in keys for p, c, d in zip(S.platform, S.contributor, S.device)]
    # numbers even when nothing is selected (an empty table's mapped column keeps the text type and the division fails)
    S["est_gb"] = ((S.frames.astype("int64") - S.frames_already_downloaded.astype("int64"))
                   * S.platform.map(mb).astype("float64") / 1000).round(3)
    S.to_csv(os.path.join(out, "sequences.csv"), index=False)
    gb = round(float(S.est_gb.sum()), 1)
    sl["total"] = {"sequences": len(S), "frames": len(A), "est_gb": gb, "budget_gb": budget, "fits_budget": gb <= budget}
    sl["months_before_date_filter"] = months
    if not len(A):
        where = "inside the selected area" if sel is not None else "in the area"
        have = {m: n for d in months.values() for m, n in d.items()}
        rng = f"{wa.get('date_from') or 'any'} to {wa.get('date_to') or 'any'}"
        if have:
            top = ", ".join(f"{m} ({sum(d.get(m, 0) for d in months.values())})" for m in sorted(have))
            note = (f"Nothing to download: no capture {where} falls in the date range {rng}. "
                    f"Months with captures {where}: {top}. Change the dates under Settings > What to download.")
        else:
            note = f"Nothing to download: no capture {where}" + (" (draw a larger selected area or switch Selection to Whole area)."
                                                                if sel is not None else ".")
        sl["total"]["note"] = note
        log.warning(note)
    json.dump(sl, open(os.path.join(out, "selection_log.json"), "w", encoding="utf8"), indent=1)
    msg = f"whole area: {len(S)} sequences, {len(A)} frames, about {gb} GB"
    log.info(msg + ("" if gb <= budget else f" -- OVER the {budget} GB budget: narrow the dates or platforms, or choose a smaller image size"))
    return [os.path.join(out, "frames.parquet"), os.path.join(out, "sequences.csv")], sl["total"]
