"""Phase 1 check: the ported Step 5 reproduces the Singapore Step 5 outputs (step5_osm, run 18 Nov 2026).

Compares, value by value (except ties: a capture equally far from two ways now goes to the way closer to its
travel direction, where the original took whichever came first in the array): ways and junctions (whole island), widths (numeric columns), the ohsome history of the
regions the check run fetched, and the date matches of those regions' captures.
Usage: python -I compare_step5.py <original step5_osm dir> <check project 05_osm dir>
"""
import glob
import gzip
import json
import os
import sys

import numpy as np
import pandas as pd


def same(name, a, b, key, cols=None):
    a = a.sort_values(key).reset_index(drop=True)
    b = b.sort_values(key).reset_index(drop=True)
    cols = cols or [c for c in a.columns if c in b.columns]
    if len(a) != len(b):
        print(f"FAIL {name}: {len(a)} rows vs {len(b)}")
        return False
    bad = []
    for c in cols:
        x, y = a[c], b[c]
        if x.dtype.kind in "fc" or y.dtype.kind in "fc":
            ok = np.allclose(pd.to_numeric(x, errors="coerce"), pd.to_numeric(y, errors="coerce"), equal_nan=True)
        else:
            ok = x.astype(str).equals(y.astype(str))
        if not ok:
            bad.append(c)
    print(("FAIL " if bad else "ok   ") + f"{name}: {len(a)} rows, {len(cols)} columns" + (f", differ: {bad}" if bad else ""))
    return not bad


def main():
    orig, new = sys.argv[1:3]
    ok = True
    ok &= same("ways", pd.read_parquet(f"{orig}/ways.parquet"), pd.read_parquet(f"{new}/ways.parquet"), "way_id")
    ok &= same("junctions", pd.read_parquet(f"{orig}/junctions.parquet"), pd.read_parquet(f"{new}/junctions.parquet"), "node_id")
    ok &= same("way widths", pd.read_parquet(f"{orig}/way_widths.parquet").rename(columns={"kerb_clearance_m": "curb_clearance_m"}),
               pd.read_parquet(f"{new}/way_widths.parquet"), "way_id",
               ["way_id", "highway", "oneway", "lanes_used", "lanes_source", "width_m", "shoulder_left_m",
                "shoulder_right_m", "curb_clearance_m"])
    ok &= same("default widths", pd.read_csv(f"{orig}/default_widths.csv"), pd.read_csv(f"{new}/default_widths.csv"),
               ["highway", "oneway"], ["highway", "oneway", "ways", "default_lanes", "default_width_m"])
    regions = [os.path.basename(f).split(".")[0] for f in glob.glob(f"{new}/history/*.geojson.gz")]
    for r in regions:
        fa = json.load(gzip.open(f"{orig}/history/{r}.geojson.gz", "rt", encoding="utf8"))["features"]
        fb = json.load(gzip.open(f"{new}/history/{r}.geojson.gz", "rt", encoding="utf8"))["features"]
        sa = sorted(json.dumps(f, sort_keys=True) for f in fa)
        sb = sorted(json.dumps(f, sort_keys=True) for f in fb)
        print(("ok   " if sa == sb else "FAIL ") + f"history {r}: {len(fa)} vs {len(fb)} states")
        ok &= sa == sb
    A = pd.read_parquet(f"{orig}/capture_map_match.parquet")
    A = A[A.region.isin(regions)]
    B = pd.read_parquet(f"{new}/capture_map_match.parquet")
    k = ["platform", "sequence_id", "frame_id"]
    m = A.merge(B, on=k, suffixes=("_a", "_b"))
    if len(m) != len(A) or len(m) != len(B):
        print(f"FAIL capture matches: {len(A)} vs {len(B)} rows")
        ok = False
    for p in ("date", "now"):
        d = m[m[f"{p}_way_id_a"].astype(str) != m[f"{p}_way_id_b"].astype(str)]
        score = lambda dist, ang: dist + np.where(ang > 45, 15, 0)  # noqa: E731
        ties = np.isclose(score(d[f"{p}_dist_m_a"], d[f"{p}_angle_deg_a"]), score(d[f"{p}_dist_m_b"], d[f"{p}_angle_deg_b"])) &             (d[f"{p}_angle_deg_b"].fillna(0) <= d[f"{p}_angle_deg_a"].fillna(0))
        good = bool(ties.all())
        print(("ok   " if good else "FAIL ") + f"capture matches ({p}): {len(m)} captures, {len(d)} differ, all of them equal-score ties"
              " now given to the way closer to the travel direction" if good else f"{len(d) - int(ties.sum())} real differences")
        ok &= good
    rest = ["after_history_end", "travel_bearing_deg", "date_candidates", "now_candidates", "date_dist_m", "now_dist_m"]
    ok &= same("capture matches (other columns)", A, B, k, k + rest)
    print("ALL SAME" if ok else "DIFFERENCES FOUND")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
