"""Phase 1 check: the ported Step 3, run offline on a set of sequences, reproduces the Singapore Step 3 outputs
(step3_download: data, data_pano, data_1024, data_views) for those sequences.

Offline means nothing is fetched: a frame the original run could not get (failed, no_url) has no file, so the
port reports it as failed or no_url; every frame the original has (ok, black_at_source) must match exactly.
Usage: python -I compare_step3.py <step3_download dir> <check project dir> <comma-separated sequences>
"""
import glob
import os
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from compare_step1 import US, compare  # noqa: E402

HAVE = ["ok", "black_at_source"]


def status_pair(o, n, seqs, kind):
    rows_o, rows_n = [], []
    for pf in ("kartaview", "mapillary"):
        for s in seqs:
            fo = os.path.join(o, "data" if kind == "frames" else "data_pano", pf, s, "frames_status.csv" if kind == "frames" else "pano_status.csv")
            fn = os.path.join(n, "03_download", "status", pf, f"{s}_{kind}.csv")
            if os.path.exists(fo):
                rows_o.append(pd.read_csv(fo, dtype={"frame_id": str}).assign(platform=pf, sequence_id=s))
            if os.path.exists(fn):
                rows_n.append(pd.read_csv(fn, dtype={"frame_id": str}).assign(platform=pf, sequence_id=s))
    return pd.concat(rows_o, ignore_index=True), pd.concat(rows_n, ignore_index=True)


def check_status(name, A, B, cols):
    ok = True
    k = ["platform", "sequence_id", "frame_id"]
    m = A.merge(B, on=k, how="outer", suffixes=("_o", "_n"), indicator=True)
    if (m._merge != "both").any():
        print(f"FAIL {name}: frames differ ({int((m._merge != 'both').sum())})")
        return False
    have = m.status_o.isin(HAVE + ["mis_stitched"])
    st_ok = (m.status_o[have] == m.status_n[have]).all()
    miss_ok = m.status_n[~have].isin(["failed", "no_url", "no_record"]).all()
    bad = [c for c in cols if not np.allclose(m.loc[have, f"{c}_o"].astype(float), m.loc[have, f"{c}_n"].astype(float), equal_nan=True)]
    ok = st_ok and miss_ok and not bad
    print(("ok   " if ok else "FAIL ") + f"{name}: {len(m)} frames; {int(have.sum())} on disk match status"
          + ("" if st_ok else " (status differs)") + f"; {int((~have).sum())} not on disk reported as failed/no_url"
          + ("" if miss_ok else " (wrong)") + (f"; differ: {bad}" if bad else f"; {', '.join(cols)} equal"))
    return ok


def main():
    o, n, seqs = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
    ok = True
    A, B = status_pair(o, n, seqs, "frames")
    ok &= check_status("image status", A, B, ["bytes", "width", "height"])
    am, bm = A[A.platform == "mapillary"], B[B.platform == "mapillary"]
    if len(am):
        ok &= compare("mapillary reconstruction fields", am, bm, ["sequence_id", "frame_id"],
                      tuple(c for c in am.columns if c not in ("sfm_cluster", "has_computed", "api_width", "api_height")))
    A, B = status_pair(o, n, seqs, "pano")
    ok &= check_status("panorama status (incl. mis_stitched)", A, B, ["bytes", "width", "height"])
    io_ = pd.read_parquet(os.path.join(o, "data", "image_screen.parquet"))
    io_ = io_[io_.sequence_id.isin(seqs)]
    # status is the label at screening time: the original screened before marking mis-stitched panoramas
    ok &= compare("image screen (measures)", io_, pd.read_parquet(os.path.join(n, "03_download", "image_screen.parquet")),
                  ["set", "platform", "sequence_id", "frame_id"], ("status",))
    po = pd.read_parquet(os.path.join(o, "data_pano", "pano_screen.parquet"))
    po = po[po.sequence_id.isin(seqs)]
    ok &= compare("panorama screen", po, pd.read_parquet(os.path.join(n, "03_download", "pano_screen.parquet")),
                  ["platform", "sequence_id", "frame_id"])
    mo = pd.read_parquet(os.path.join(o, "data", "manifest.parquet"))
    mo = mo[mo.sequence_id.isin(seqs) & mo.status.isin(HAVE)]
    mn = pd.read_parquet(os.path.join(n, "03_download", "manifest.parquet"))
    mn = mn[mn.sequence_id.isin(seqs) & mn.frame_id.isin(mo.frame_id)]
    ok &= compare("manifest (frames on disk)", mo, mn, ["platform", "sequence_id", "frame_id"], ("source", "note"))
    to = pd.read_csv(os.path.join(o, "data_pano", "travel_yaw.csv"), dtype={"sequence_id": str})
    tn = pd.read_csv(os.path.join(n, "03_download", "travel_yaw.csv"), dtype={"sequence_id": str})
    to = to[to.sequence_id.isin(tn.sequence_id)]
    ok &= compare("travel yaw (vehicle sequences)", to, tn, ["sequence_id"], ("yaw_source",))
    vo, vn = os.path.join(o, "data_views"), os.path.join(n, "03_download", "views_check")
    for f in ("views.parquet", "panos.parquet"):
        a = pd.concat([pd.read_parquet(p) for p in glob.glob(os.path.join(vo, "*", "*", f)) if p.replace("\\", "/").split("/")[-2] in seqs]).rename(columns=US)
        b = pd.concat([pd.read_parquet(p) for p in glob.glob(os.path.join(vn, "*", "*", f))])
        ok &= compare(f"views: {f}", a, b, ["view_id"] if f == "views.parquet" else ["platform", "sequence_id", "frame_id"])
    import cv2
    rng = np.random.default_rng(3)
    jp = sorted(glob.glob(os.path.join(vn, "*", "*", "*.jpg")))
    same = sum(open(p, "rb").read() == open(os.path.join(vo, *p.replace("\\", "/").split("/")[-3:]), "rb").read()
               for p in rng.choice(jp, min(60, len(jp)), replace=False))
    print(("ok   " if same == min(60, len(jp)) else "FAIL ") + f"view JPEGs: {same} of {min(60, len(jp))} sampled files byte-identical")
    ok &= same == min(60, len(jp))
    import zipfile
    zs = 0
    for z in glob.glob(os.path.join(vn, "*", "*", "masks.zip")):
        za = zipfile.ZipFile(os.path.join(vo, *z.replace("\\", "/").split("/")[-3:]))
        zb = zipfile.ZipFile(z)
        zs += all(za.read(m) == zb.read(m) for m in zb.namelist()) and set(za.namelist()) == set(zb.namelist())
    nz = len(glob.glob(os.path.join(vn, "*", "*", "masks.zip")))
    print(("ok   " if zs == nz else "FAIL ") + f"masks: {zs} of {nz} sequence zips identical")
    ok &= zs == nz
    from collect.steps.s3_download import _convert
    src = [p for p in glob.glob(os.path.join(o, "data", "*", "*", "*.jpg")) if p.replace("\\", "/").split("/")[-2] in seqs]
    tmp = tempfile.mkdtemp()
    pix = 0
    smp = rng.choice(src, min(40, len(src)), replace=False)
    for i, p in enumerate(smp):
        dst = os.path.join(tmp, f"{i}.jpg")
        _convert((p, dst, 1024, 90))
        orig = os.path.join(o, "data_1024", *p.replace("\\", "/").split("/")[-3:])
        pix += open(dst, "rb").read() == open(orig, "rb").read()
    print(("ok   " if pix == len(smp) else "FAIL ") + f"working copies: {pix} of {len(smp)} remade copies byte-identical")
    ok &= pix == len(smp)
    print("ALL SAME" if ok else "DIFFERENCES FOUND")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
