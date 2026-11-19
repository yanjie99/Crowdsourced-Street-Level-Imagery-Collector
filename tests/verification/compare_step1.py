"""Phase 1 check: the ported Step 1, run offline from the saved caches, reproduces the Singapore Step 1 tables
(step1_metadata, 12-13 Nov 2026).

Every table is compared row by row on its key and value by value on the shared columns. Known, intended
differences are reported separately instead of failing:
  - KartaView found_by: discovery now lists every contributor without the old 5,000-page cap, so a sequence on a
    late page of user 44 that the sweep also found is labeled "sweep;user_list" instead of "sweep".
  - kartaview/sequences.parquet (19 Nov 2026): the country listing is now limited to sequences near the boundary, so
    97 Singapore sequences with no bounding box and no photos are no longer listed (2,759 instead of 2,856 rows). Every
    photo and every other table is unchanged; this table's FAIL line is expected.
Usage: python -I compare_step1.py <original step1_metadata dir> <check project 01_metadata dir>
"""
import os
import sys

import numpy as np
import pandas as pd


# Column names of the Singapore originals that the package now spells in US English (18 Nov 2026)
US = {"levelled": "leveled", "yaw_from_pano_centre_deg": "yaw_from_pano_center_deg", "kerb_clearance_m": "curb_clearance_m",
      "centre_scale": "center_scale"}


def read(p):
    d = pd.read_parquet(p) if p.endswith(".parquet") else pd.read_csv(p, dtype=str, keep_default_na=False)
    return d.rename(columns=US)


def compare(name, a, b, key, skip=()):
    a = a.copy()
    b = b.copy()
    for k in key:
        a[k], b[k] = a[k].astype(str), b[k].astype(str)
    if a.duplicated(key).any() or b.duplicated(key).any():
        print(f"FAIL {name}: duplicate keys")
        return False
    ka, kb = set(map(tuple, a[key].values)), set(map(tuple, b[key].values))
    if ka != kb:
        print(f"FAIL {name}: {len(ka)} vs {len(kb)} rows, {len(ka - kb)} only in original, {len(kb - ka)} only in new")
        return False
    a, b = a.sort_values(key).reset_index(drop=True), b.sort_values(key).reset_index(drop=True)
    cols = [c for c in a.columns if c in b.columns and c not in key and c not in skip]
    bad = []
    for c in cols:
        x, y = a[c], b[c]
        if x.dtype.kind in "fiub" and y.dtype.kind in "fiub":
            ok = np.allclose(x.astype(float), y.astype(float), equal_nan=True)
        elif x.dtype.kind == "M" or y.dtype.kind == "M":
            ok = pd.to_datetime(x, utc=True).equals(pd.to_datetime(y, utc=True))
        else:
            ok = x.astype(str).equals(y.astype(str))
        if not ok:
            bad.append(c)
    missing = [c for c in a.columns if c not in b.columns and c not in skip]
    status = "FAIL" if bad else "ok  "
    print(f"{status} {name}: {len(a)} rows, {len(cols)} columns" + (f"; differ: {bad}" if bad else "")
          + (f"; columns not in new: {missing}" if missing else ""))
    return not bad


def main():
    o, n = sys.argv[1:3]
    ok = True
    T = [("mapillary/captures.parquet", ["id"], ()),
         ("mapillary/sequences.parquet", ["sequence_id"], ()),
         ("kartaview/sweep_hits.parquet", ["point", "photo_id"], ()),
         ("kartaview/sequences_raw.parquet", ["sequence_id"], ()),
         ("kartaview/sequences.parquet", ["sequence_id"], ("found_by",)),
         ("kartaview/photos.parquet", ["photo_id"], ()),
         ("kartaview/sequence_table.parquet", ["sequence_id"], ("found_by",)),
         ("osm/segments.parquet", ["segment_id"], ()),
         ("osm/mapillary_capture_segments.parquet", ["capture_id"], ()),
         ("osm/kartaview_capture_segments.parquet", ["capture_id"], ()),
         ("osm/mapillary_segment_passes.parquet", ["segment_id"], ()),
         ("osm/kartaview_segment_passes.parquet", ["segment_id"], ()),
         ("osm/duplicate_pairs.csv", ["sequence_id_k", "sequence_id_m", "offset_h"], ()),
         ("osm/kartaview_internal_duplicates.csv", ["sequence_a", "sequence_b"], ()),
         ("osm/combined_segment_passes.parquet", ["segment_id"], ())]
    for f, key, skip in T:
        pa, pb = os.path.join(o, f), os.path.join(n, f)
        if not os.path.exists(pb):
            print(f"FAIL {f}: not produced")
            ok = False
            continue
        ok &= compare(f, read(pa), read(pb), key, skip)
    a = read(os.path.join(o, "kartaview/sequences.parquet")).set_index("sequence_id").found_by
    b = read(os.path.join(n, "kartaview/sequences.parquet")).set_index("sequence_id").found_by
    d = (a != b.reindex(a.index))
    pairs = pd.Series(list(zip(a[d], b.reindex(a.index)[d]))).value_counts()
    print(f"note kartaview found_by: {int(d.sum())} labels differ: " + "; ".join(f"{x} -> {y}: {c}" for (x, y), c in pairs.items()))
    print("ALL SAME" if ok else "DIFFERENCES FOUND")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
