"""Step 3: downloads, screening, panoramas and the manifest (ported 18 Nov 2026 from step3_download/*.py).

  3a estimate   fetch a small random sample at each candidate image size and estimate the storage of the
                selection at each size, against the budget -> storage_estimate.csv            (online only)
  3b download   per frame: the first available size of the platform's fallback list (image_sizes), plus
                any extra sizes for a sample (extra_sizes); Mapillary Graph records and reconstruction
                files; KartaView photo pages, sequence record and raw sensor log
  3c working    1024 px working copies (EXIF orientation, Lanczos, JPEG 90)
  3d panoramas  the full panorama of every 360 frame (KartaView wrapped_lth/proc, Mapillary original)
  3e screen     black share, flat rows and mean of every image and panorama (platform faults served as ordinary files)
  3f pano check screen every panorama; mark mis-stitched ones (sequence coverage outliers)  (review gate)
  3g manifest   one row per selected frame with status, size, flags, pano_status; completeness checks
Folders (each root can be mapped to another folder in project.yaml `caches`):
  images      <03_download>/data/<platform>/<sequence>/<frame>.jpg, records/, (Mapillary) _sfm/
  panoramas   <03_download>/data_pano/<platform>/<sequence>/<frame>.jpg
  working     <03_download>/data_1024/<platform>/<sequence>/<frame>.jpg
Status files are always written in the project: <03_download>/status/<platform>/<sequence>_frames.csv and
_pano.csv. With offline=True nothing is requested and nothing is written outside the project: missing
files are reported as failed ("offline: not on disk").
Image sizes: Mapillary thumb_256, thumb_1024, thumb_2048, original; KartaView th, lth, proc, original
(paths /th/, /lth/, /proc/, /ori/); panoramas KartaView wrapped_lth, wrapped_proc, Mapillary original.
"""
from __future__ import annotations

import glob
import gzip
import io
import json
import os
import random
import re
import shutil
import threading
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np
import pandas as pd

from .. import progress
from ..config import Config
from ..config import graph_url
from ..net import Throttle, feedback, now, platform_throttle, retry_after

MLY_URL = {"thumb_256": "thumb_256_url", "thumb_1024": "thumb_1024_url", "thumb_2048": "thumb_2048_url",
           "original": "thumb_original_url"}
KV_PATH = {"th": "/th/", "lth": "/lth/", "proc": "/proc/", "original": "/ori/",
           "wrapped_lth": "/wrapped_lth/", "wrapped_proc": "/wrapped_proc/"}
MLY_FIELDS = ("id,sequence,captured_at,geometry,computed_geometry,compass_angle,computed_compass_angle,altitude,"
              "computed_altitude,computed_rotation,camera_type,camera_parameters,width,height,exif_orientation,is_pano,"
              "make,model,creator,sfm_cluster,mesh,merge_cc,atomic_scale,quality_score,thumb_256_url,thumb_1024_url,"
              "thumb_2048_url,thumb_original_url")
_wlock = threading.Lock()


# ---------------------------------------------------------------- helpers
class FetchError(Exception):
    pass


def fetch(url, throttle, headers=None, tries=4, timeout=120, ua="collect"):
    import time
    import urllib.error
    import urllib.request
    h = {"User-Agent": ua, **(headers or {})}
    err = None
    for a in range(tries):
        throttle.wait()
        t0 = time.time()
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=timeout) as r:
                data = r.read()
            feedback(throttle, True, time.time() - t0)
            return data
        except urllib.error.HTTPError as e:
            if e.code == 429 or e.code >= 500:
                feedback(throttle, False, why=f"HTTP {e.code}", wait=retry_after(e))
            body = ""
            try:
                body = e.read(400).decode("utf8", "ignore")
            except Exception:  # noqa: BLE001
                pass
            err = body.split("<Code>")[1].split("</Code>")[0] if "<Code>" in body else f"HTTP {e.code}"
            if e.code in (400, 401, 403, 404, 409, 410):
                break
            time.sleep((30 if e.code == 429 else 5) * (a + 1))
        except Exception as e:  # noqa: BLE001
            err = type(e).__name__
            if "timeout" in err.lower() or "timed out" in str(e):
                feedback(throttle, False, why="no answer in time")
            time.sleep(5 * (a + 1))
    raise FetchError(err)


def image_size(x):
    from PIL import Image
    try:
        with Image.open(x if isinstance(x, str) else io.BytesIO(x)) as im:
            im.load()
            return im.size
    except Exception:  # noqa: BLE001
        return None


def is_black(x, max_level=8):
    from PIL import Image
    try:
        with Image.open(x if isinstance(x, str) else io.BytesIO(x)) as im:
            im.draft("L", (max(1, im.width // 8), max(1, im.height // 8)))
            return im.convert("L").getextrema()[1] <= max_level
    except Exception:  # noqa: BLE001
        return False


def write_atomic(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".part", "wb") as f:
        f.write(data)
    os.replace(path + ".part", path)


def log_jsonl(path, obj):
    with _wlock:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf8") as f:
            f.write(json.dumps(obj, ensure_ascii=False, default=str) + "\n")


def d3(cfg):
    return cfg.paths.step(3)


def root(cfg, name):
    """images | panoramas | working | views: a data root, mappable in `caches`."""
    sub = {"images": "data", "panoramas": "data_pano", "working": "data_1024", "views": "data_views"}[name]
    mapped = (cfg.get("caches", {}) or {}).get(name)
    p = cfg.resolve(mapped) if mapped else os.path.join(d3(cfg), sub)
    os.makedirs(p, exist_ok=True)
    return p


def status_path(cfg, pf, seq, kind):
    p = os.path.join(d3(cfg), "status", pf)
    os.makedirs(p, exist_ok=True)
    return os.path.join(p, f"{seq}_{kind}.csv")


def s3(cfg, key, default=None):
    return cfg.get(f"step3_download.{key}", default)


def sizes(cfg, pf):
    """Fallback order of image sizes for a platform (first = preferred)."""
    v = (s3(cfg, "image_sizes", {}) or {}).get(pf) or [cfg.get(f"platforms.{pf}.image_size")]
    return [v] if isinstance(v, str) else list(v)


def selection(cfg, sequences=None):
    F = pd.read_parquet(os.path.join(cfg.paths.step(2), "selection", "frames.parquet"))
    if sequences:
        F = F[F.sequence_id.isin(set(sequences))]
    return F


def seqs_of_regions(cfg, regions):
    if not regions:
        return None
    F = pd.read_parquet(os.path.join(cfg.paths.step(2), "selection", "frames.parquet"), columns=["sequence_id", "region_cell"])
    return sorted(F[F.region_cell.isin(set(regions))].sequence_id.unique())


def kv_url(lth_url, size):
    if not isinstance(lth_url, str) or "/lth/" not in lth_url:
        return None
    return lth_url.replace("/lth/", KV_PATH[size])


def _result(p, data=None):
    sz = image_size(data if data is not None else p)
    if not sz:
        return None
    return {"status": "black_at_source" if is_black(data if data is not None else p) else "ok",
            "bytes": len(data) if data is not None else os.path.getsize(p), "width": sz[0], "height": sz[1]}


def save_first(urls, path, th, offline, ua, extra_note=""):
    """Try (size, url) in order; keep the first image that arrives as path. Returns a status row."""
    if os.path.exists(path):
        r = _result(path)
        if r:
            return {**r, "size": None, "source": "on_disk", "note": extra_note or "on_disk"}
    if urls and all(u is None for _, u in urls):
        return {"status": "no_url", "size": None, "source": "", "note": ""}
    if offline:
        return {"status": "failed", "size": None, "source": "", "note": "offline: not on disk"}
    notes = []
    for size, url in urls:
        if not url:
            notes.append(f"{size}: no url")
            continue
        try:
            data = fetch(url, th, ua=ua)
        except FetchError as e:
            notes.append(f"{size}: {e}")
            continue
        r = _result(path, data)
        if not r:
            notes.append(f"{size}: not an image")
            continue
        write_atomic(path, data)
        return {**r, "size": size, "source": size, "note": "; ".join(notes)}
    return {"status": "failed" if urls else "no_url", "size": None, "source": "", "note": "; ".join(notes)}


def _extras(cfg, pf, frame_ids):
    """frame ids that also get the extra sizes (step3_download.extra_sizes / extra_sample), seeded."""
    ex = (s3(cfg, "extra_sizes", {}) or {}).get(pf) or []
    if not ex:
        return [], set()
    share = float(s3(cfg, "extra_sample", 1.0))
    ids = sorted(frame_ids)
    k = len(ids) if share >= 1 else int(round(len(ids) * share))
    return list(ex), set(random.Random(7).sample(ids, k))


# ---------------------------------------------------------------- 3a storage estimate
def estimate(cfg: Config, log, offline=False, n=None, **_):
    out = d3(cfg)
    if offline:
        log.info("storage estimate needs the network; skipped (offline)")
        return [], {"skipped": "offline"}
    n = n or int(s3(cfg, "estimate_sample", 20))
    F = selection(cfg)
    ua, budget = cfg.get("platforms.user_agent"), cfg.get("storage.budget_gb")
    rows = []
    for pf in [p for p in ("mapillary", "kartaview") if cfg.get(f"platforms.{p}.enabled", False) and (F.platform == p).any()]:
        g = F[F.platform == pf]
        smp = g.sample(min(n, len(g)), random_state=7)
        cand = list(dict.fromkeys(sizes(cfg, pf) + ((s3(cfg, "extra_sizes", {}) or {}).get(pf) or [])))
        th = (platform_throttle(cfg, "kartaview", log=log) if pf == "kartaview"
              else Throttle(cfg.get("platforms.mapillary.files_per_s", 8.0)))
        urls = {}
        if pf == "mapillary":
            from ..secrets import auth_header
            raw = fetch(f"{graph_url(cfg)}/?ids={','.join(smp.frame_id)}&fields=id,{','.join(MLY_URL[s] for s in cand)}",
                        Throttle(cfg.get("platforms.mapillary.graph_per_s", 3.0)), auth_header(), ua=ua)
            meta = json.loads(raw)
            urls = {s: [(meta.get(f) or {}).get(MLY_URL[s]) for f in smp.frame_id] for s in cand}
        else:
            urls = {s: [kv_url(u, s) for u in smp.fileurlLTh] for s in cand}
        for s in cand:
            b = []
            for u in urls[s]:
                try:
                    b.append(len(fetch(u, th, ua=ua)) if u else None)
                except FetchError:
                    b.append(None)
            got = [x for x in b if x]
            mb = float(np.mean(got)) / 1e6 if got else np.nan
            rows.append({"platform": pf, "size": s, "sampled": len(b), "available": len(got),
                         "available_share": round(len(got) / len(b), 3) if b else np.nan, "mb_per_frame": round(mb, 3),
                         "frames": len(g), "est_gb": round(mb * len(g) / 1000, 1) if got else np.nan})
            log.info(f"{pf} {s}: {len(got)}/{len(b)} available, {mb:.3f} MB a frame -> {rows[-1]['est_gb']} GB")
    E = pd.DataFrame(rows, columns=["platform", "size", "sampled", "available", "available_share", "mb_per_frame", "frames", "est_gb"])
    E["budget_gb"] = budget
    E.to_csv(os.path.join(out, "storage_estimate.csv"), index=False)
    pref = E.groupby("platform").head(1)
    tot = float(pref.est_gb.sum())
    return [os.path.join(out, "storage_estimate.csv")], {"preferred_sizes_gb": round(tot, 1), "budget_gb": budget,
                                                         "fits": tot <= budget}


# ---------------------------------------------------------------- 3b download
# A failure that suggests the saved link itself is dead (the file moved, the storage host was renamed or retired),
# not a busy server: 404/403/410, S3 "NoSuchKey"/"AccessDenied", an unknown host or a refused connection.
DEAD_LINK = re.compile(r"HTTP 40[34]|HTTP 410|NoSuchKey|AccessDenied|URLError|gaierror|ConnectionRefused", re.I)


def _kv_refresh_links(cfg, seq, fr, R, rec, one, log_to=None):
    """KartaView image links come from Step 1e's saved photo lists, which can be weeks or months old. When frames
    fail with a dead-link error, fetch this sequence's photo list again from the API (not from the cache), and retry
    those frames with the links it gives now. The fresh pages are kept in records/ as photos_refreshed_p*.json.gz;
    Step 1's own cache is left as it was."""
    bad = R.status.eq("failed") & R.note.fillna("").str.contains(DEAD_LINK)
    if not bad.any():
        return R
    from .s1_kartaview import client
    cl, fresh, page = client(cfg), {}, 1
    while page <= 5000:
        data = cl.get(f"/sequence/{seq}/photos?itemsPerPage=100&page={page}")
        if data is None:
            break
        with gzip.open(os.path.join(rec, f"photos_refreshed_p{page:05d}.json.gz"), "wt", encoding="utf8") as z:
            json.dump(data, z)
        res = data.get("result") or {}
        fresh.update({str(p.get("id")): p.get("fileurlLTh") for p in res.get("data") or []})
        if not res.get("hasMoreData"):
            break
        page += 1
    if not fresh:
        return R
    rows = fr.set_index(fr.frame_id.astype(str))
    redo = [fid for fid in R.frame_id[bad].astype(str) if fresh.get(fid) and fid in rows.index]
    changed = 0
    R = R.astype(object)          # retried rows bring text and numbers into any column
    for fid in redo:
        r = rows.loc[fid].to_dict()
        r["frame_id"] = fid
        i = R.index[R.frame_id.astype(str) == fid][0]
        if r.get("fileurlLTh") == fresh[fid]:
            # the API still gives the same link: the file is missing on KartaView's own storage (seen for 2016
            # sequences on storage1.openstreetcam.org, listed as active and public); a retry cannot help
            R.at[i, "note"] = (str(R.at[i, "note"] or "") + "; same link from the API today: the image is missing "
                               "on KartaView's storage").strip("; ")
            continue
        changed += 1
        r["fileurlLTh"] = fresh[fid]
        new = one(r)
        new["note"] = ("retried with a refreshed link; " + str(new.get("note") or "")).strip("; ")
        for k, v in new.items():
            if k in R.columns:
                R.at[i, k] = v
    if log_to:
        ok = int((R[R.frame_id.astype(str).isin(redo)].status == "ok").sum())
        log_jsonl(log_to, {"sequence_id": seq, "refreshed_links_utc": now(), "dead_link_frames": int(bad.sum()),
                           "retried": len(redo), "links_changed": changed, "ok_after_retry": ok})
    return R


def _kv_sequence(cfg, seq, fr, raw_info, th, offline, ua, flog):
    img = os.path.join(root(cfg, "images"), "kartaview", seq)
    rec = os.path.join(img, "records")
    kraw = cfg.cache_dir(1, "kartaview/raw")
    win = cfg.get("existing.annotated_window_dir", None)
    win = cfg.resolve(win) if win else None
    order = sizes(cfg, "kartaview")
    ex_sizes, ex_ids = _extras(cfg, "kartaview", fr.frame_id)
    t0 = now()
    if not offline:
        os.makedirs(rec, exist_ok=True)
        for f in glob.glob(os.path.join(kraw, "photos", f"{seq}_p*.json.gz")):
            dst = os.path.join(rec, os.path.basename(f))
            if not os.path.exists(dst):
                shutil.copyfile(f, dst)
        sf = os.path.join(kraw, "sequence", f"{seq}.json.gz")
        if os.path.exists(sf) and not os.path.exists(os.path.join(rec, "sequence.json.gz")):
            shutil.copyfile(sf, os.path.join(rec, "sequence.json.gz"))

    def one(r):
        p = os.path.join(img, f"{r['frame_id']}.jpg")
        base = {"frame_id": r["frame_id"], "order": r["order"], "role": r["role"]}
        if not os.path.exists(p) and win and order[0] == "lth":
            w = os.path.join(win, r["sequence_id"], f"{r['frame_id']}.jpg")
            if os.path.exists(w) and image_size(w) and not offline:
                shutil.copyfile(w, p)
                return {**base, **_result(p), "size": "lth", "source": "annotated_windows", "note": ""}
        res = save_first([(s, kv_url(r.get("fileurlLTh"), s)) for s in order], p, th, offline, ua)
        for s in ex_sizes if r["frame_id"] in ex_ids else []:
            save_first([(s, kv_url(r.get("fileurlLTh"), s))], os.path.join(img, f"{r['frame_id']}_{s}.jpg"), th, offline, ua)
        return {**base, **res}

    with ThreadPoolExecutor(cfg.get("platforms.kartaview.workers", 3)) as ex:
        R = pd.DataFrame(list(ex.map(one, fr.to_dict("records"))))
    if not offline:
        R = _kv_refresh_links(cfg, seq, fr, R, rec, one, log_to=flog)
    rawres = None
    has, url = raw_info.get(seq, (0, None))
    if str(has) == "1" and isinstance(url, str) and url.startswith("https://storage") and "." in url.rsplit("/", 1)[-1]:
        rp = os.path.join(img, "rawdata", url.rsplit("/", 1)[-1])
        if os.path.exists(rp) and os.path.getsize(rp) > 0:
            rawres = "ok (on disk)"
        elif offline:
            rawres = "offline: not on disk"
        else:
            try:
                write_atomic(rp, fetch(url, th, timeout=300, ua=ua))
                rawres = "ok"
            except FetchError as e:
                rawres = f"failed: {e}"
    elif str(has) == "1":
        rawres = "no valid url"
    if not offline:
        log_jsonl(flog, {"sequence_id": seq, "started_utc": t0, "finished_utc": now(), "sizes": order, "frames": len(R),
                         "status": R.status.value_counts().to_dict(), "size_got": R["size"].fillna("on_disk").value_counts().to_dict(),
                         "rawdata": rawres, "license": cfg.get("platforms.kartaview.license")})
    return R


def _mly_records(img, ids):
    meta = {}
    for f in sorted(glob.glob(os.path.join(img, "records", "graph_*.json.gz"))):
        try:
            meta.update(json.load(gzip.open(f)))
        except (OSError, ValueError):
            pass
    return meta


def _mly_sequence(cfg, seq, fr, tg, tf, sfm, offline, ua, flog):
    img = os.path.join(root(cfg, "images"), "mapillary", seq)
    sfm_dir = os.path.join(root(cfg, "images"), "mapillary", "_sfm")
    order = sizes(cfg, "mapillary")
    ex_sizes, ex_ids = _extras(cfg, "mapillary", fr.frame_id)
    ids, t0 = fr.frame_id.tolist(), now()
    batch = cfg.get("platforms.mapillary.graph_batch", 50)
    meta = _mly_records(img, ids)
    missing = [f for f in ids if not os.path.exists(os.path.join(img, f"{f}.jpg"))]
    gfail = []
    if not offline and (missing or not set(ids) <= set(meta)):
        from ..secrets import auth_header
        hdr = auth_header()
        os.makedirs(os.path.join(img, "records"), exist_ok=True)
        meta = {}
        for b in range(0, len(ids), batch):
            try:
                raw = fetch(f"{graph_url(cfg)}/?ids={','.join(ids[b:b + batch])}&fields={MLY_FIELDS}", tg, hdr, timeout=180, ua=ua)
                with gzip.open(os.path.join(img, "records", f"graph_{b // batch:03d}.json.gz"), "wb") as g:
                    g.write(raw)
                meta.update(json.loads(raw))
            except (FetchError, ValueError) as e:
                gfail.append(str(e))
        if gfail and not meta:
            raise RuntimeError(f"every Graph API request for sequence {seq} failed ({', '.join(sorted(set(gfail)))}) "
                               "from graph.mapillary.com")

    def one(fid):
        m = meta.get(fid) or {}
        sc = m.get("sfm_cluster") or {}
        base = {"frame_id": fid, "sfm_cluster": sc.get("id"), "has_computed": "computed_geometry" in m,
                "is_pano": m.get("is_pano"), "api_width": m.get("width"), "api_height": m.get("height")}
        p = os.path.join(img, f"{fid}.jpg")
        if not m and not os.path.exists(p):
            return {**base, "status": "no_record", "size": None, "source": "", "note": "id not returned by Graph API"}
        res = save_first([(s, m.get(MLY_URL[s])) for s in order], p, tf, offline, ua)
        for s in ex_sizes if fid in ex_ids else []:
            save_first([(s, m.get(MLY_URL[s]))], os.path.join(img, f"{fid}_{s}.jpg"), tf, offline, ua)
        return {**base, **res}

    w = cfg.get("platforms.mapillary.workers", 6)
    with ThreadPoolExecutor(w) as ex:
        R = pd.DataFrame(list(ex.map(one, ids)))
    R.insert(1, "order", fr.order.to_numpy())
    R.insert(2, "role", fr.role.to_numpy())
    clusters = {(m.get("sfm_cluster") or {}).get("id"): (m.get("sfm_cluster") or {}).get("url") for m in meta.values()}
    clusters = {c: u for c, u in clusters.items() if c and u}

    def get_sfm(cu):
        c, u = cu
        p = os.path.join(sfm_dir, f"{c}.json.zlib")
        with _wlock:
            if c in sfm:
                return sfm[c]
            sfm[c] = "pending"
        if os.path.exists(p) and os.path.getsize(p) > 0:
            r = "ok"
        elif offline:
            r = "offline: not on disk"
        else:
            try:
                write_atomic(p, fetch(u, tf, timeout=300, ua=ua))
                r = "ok"
            except FetchError as e:
                r = f"failed: {e}"
        with _wlock:
            sfm[c] = r
        return r

    with ThreadPoolExecutor(w) as ex:
        sres = dict(zip(clusters, ex.map(get_sfm, clusters.items())))
    R["sfm_status"] = R.sfm_cluster.map(sres)
    if not offline:
        log_jsonl(flog, {"sequence_id": seq, "started_utc": t0, "finished_utc": now(), "sizes": order, "frames": len(R),
                         "graph_failures": gfail, "status": R.status.value_counts().to_dict(),
                         "size_got": R["size"].fillna("on_disk").value_counts().to_dict(), "sfm_clusters": len(clusters),
                         "license": cfg.get("platforms.mapillary.license")})
    return R


def download(cfg: Config, log, offline=False, regions=None, sequences=None, platform=None, **_):
    seqs = sequences or seqs_of_regions(cfg, regions)
    F = selection(cfg, seqs)
    ua = cfg.get("platforms.user_agent")
    summary = {}
    for pf in [p for p in ("mapillary", "kartaview") if not platform or p == platform]:   # one lane: one platform
        g = F[F.platform == pf]
        if g.empty or not cfg.get(f"platforms.{pf}.enabled", False):
            continue
        flog = os.path.join(root(cfg, "images"), pf, "fetch_log.jsonl")
        if pf == "kartaview":
            S = pd.read_parquet(os.path.join(cfg.paths.step(1), "kartaview", "sequences.parquet"),
                                columns=["sequence_id", "hasRawData", "metadataFileUrl"])
            raw_info = dict(zip(S.sequence_id.astype(str), zip(S.hasRawData, S.metadataFileUrl)))
            th = platform_throttle(cfg, "kartaview", log=log)
        else:
            tg = Throttle(cfg.get("platforms.mapillary.graph_per_s", 3.0))
            tf = Throttle(cfg.get("platforms.mapillary.files_per_s", 8.0))
            sfm = {}
        counts = {}
        sl = sorted(g.sequence_id.unique())
        progress.phase(f"{pf} downloads", len(sl), "sequences")
        for i, seq in enumerate(sl, 1):
            progress.update(done=i - 1, message=f"{pf} downloads: {counts.get('ok', 0):,} frames ok so far")
            st = status_path(cfg, pf, seq, "frames")
            fr = g[g.sequence_id == seq].sort_values("order")
            if os.path.exists(st):
                old = pd.read_csv(st, dtype={"frame_id": str})
                if len(old) == len(fr) and old.status.isin(["ok", "black_at_source"]).all():
                    for k, v in old.status.value_counts().items():
                        counts[k] = counts.get(k, 0) + int(v)
                    continue
            R = (_kv_sequence(cfg, seq, fr, raw_info, th, offline, ua, flog) if pf == "kartaview"
                 else _mly_sequence(cfg, seq, fr, tg, tf, sfm, offline, ua, flog))
            R.to_csv(st, index=False)
            for k, v in R.status.value_counts().items():
                counts[k] = counts.get(k, 0) + int(v)
            if i % 50 == 0 or i == len(sl):
                log.info(f"{pf}: {i}/{len(sl)} sequences; {counts}")
        summary[pf] = counts
    return [os.path.join(d3(cfg), "status")], summary


# ---------------------------------------------------------------- 3c working copies
def _convert(args):
    src, dst, width, quality = args
    from PIL import Image, ImageOps
    if os.path.exists(dst):
        return "skip"
    try:
        with Image.open(src) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            if im.width > width:
                im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            im.save(dst + ".part", "JPEG", quality=quality, optimize=True)
        os.replace(dst + ".part", dst)
        return "ok"
    except Exception as e:  # noqa: BLE001
        return f"failed: {type(e).__name__}"


def working(cfg: Config, log, offline=False, regions=None, sequences=None, out_root=None, **_):
    wc = s3(cfg, "working_copy")
    src_root, dst_root = root(cfg, "images"), out_root or root(cfg, "working")
    seqs = sequences or seqs_of_regions(cfg, regions)
    jobs = []
    for st in glob.glob(os.path.join(d3(cfg), "status", "*", "*_frames.csv")):
        pf, seq = os.path.basename(os.path.dirname(st)), os.path.basename(st)[:-len("_frames.csv")]
        if seqs and seq not in seqs:
            continue
        S = pd.read_csv(st, dtype={"frame_id": str})
        for f in S.frame_id[S.status.isin(["ok", "black_at_source"])]:
            jobs.append((os.path.join(src_root, pf, seq, f"{f}.jpg"), os.path.join(dst_root, pf, seq, f"{f}.jpg"),
                         wc["width"], wc["jpeg_quality"]))
    if offline and out_root is None:
        have = sum(os.path.exists(j[1]) for j in jobs)
        log.info(f"offline: {have} of {len(jobs)} working copies already exist; none made")
        return [], {"images": len(jobs), "existing": have}
    res = {}
    progress.phase("working copies", len(jobs), "images")
    with ProcessPoolExecutor(s3(cfg, "workers", 4)) as ex:
        for r in ex.map(_convert, jobs, chunksize=64):
            progress.update(add=1)
            k = r.split(":")[0]
            res[k] = res.get(k, 0) + 1
    log.info(f"working copies: {res}")
    return [dst_root], {"images": len(jobs), **res}


# ---------------------------------------------------------------- 3d screen images
def _measure(path):
    from PIL import Image
    try:
        with Image.open(path) as im:
            im.draft("L", (max(1, im.width // 8), max(1, im.height // 8)))
            a = np.asarray(im.convert("L"), dtype=np.float32)
    except Exception:  # noqa: BLE001
        return (np.nan, np.nan, np.nan)
    sd, mu = a.std(axis=1), a.mean(axis=1)
    return (float((a <= 4).mean()), float(((sd < 1) & (mu > 8)).mean()), float(a.mean()))


def screen(cfg: Config, log, regions=None, sequences=None, **_):
    seqs = sequences or seqs_of_regions(cfg, regions)
    rows = []
    for kind, base, suf in (("images", root(cfg, "images"), "_frames.csv"), ("panoramas", root(cfg, "panoramas"), "_pano.csv")):
        for st in glob.glob(os.path.join(d3(cfg), "status", "*", f"*{suf}")):
            pf, seq = os.path.basename(os.path.dirname(st)), os.path.basename(st)[:-len(suf)]
            if seqs and seq not in seqs:
                continue
            S = pd.read_csv(st, dtype={"frame_id": str})
            S = S[S.status.isin(["ok", "black_at_source", "mis_stitched"])]
            rows += [(kind, pf, seq, f, s, os.path.join(base, pf, seq, f"{f}.jpg")) for f, s in zip(S.frame_id, S.status)]
    R = pd.DataFrame(rows, columns=["set", "platform", "sequence_id", "frame_id", "status", "path"])
    progress.phase("screening images", len(R), "images")
    m = []
    with ProcessPoolExecutor(s3(cfg, "workers", 4)) as ex:
        for v in ex.map(_measure, R.path, chunksize=256):
            m.append(v)
            progress.update(add=1)
    R[["black_share", "flat_rows", "mean"]] = np.array(m) if m else np.empty((0, 3))
    p = os.path.join(d3(cfg), "image_screen.parquet")
    R.drop(columns="path").to_parquet(p, index=False)
    log.info(f"screened {len(R)} images")
    return [p], {"images": len(R)}


# ---------------------------------------------------------------- 3e panoramas
FINAL = {"ok", "black_at_source", "mis_stitched"}


def panoramas(cfg: Config, log, offline=False, regions=None, sequences=None, platform=None, **_):
    if not s3(cfg, "panoramas.download", True):
        return [], {"skipped": "panoramas.download is false"}
    seqs = sequences or seqs_of_regions(cfg, regions)
    F = selection(cfg, seqs)
    ua, base = cfg.get("platforms.user_agent"), root(cfg, "panoramas")
    psz = s3(cfg, "panoramas.sizes", {}) or {}
    summary = {}
    for pf in [p for p in ("mapillary", "kartaview") if not platform or p == platform]:   # one lane: one platform
        if not cfg.get(f"platforms.{pf}.enabled", False):
            continue
        rows = []
        for st in glob.glob(os.path.join(d3(cfg), "status", pf, "*_frames.csv")):
            seq = os.path.basename(st)[:-len("_frames.csv")]
            if seqs and seq not in seqs:
                continue
            S = pd.read_csv(st, dtype={"frame_id": str})
            S["sequence_id"] = seq
            rows.append(S)
        if not rows:
            continue
        S = pd.concat(rows, ignore_index=True)
        # frames whose image is black are tried too: the panorama is a separate file (Singapore: black as well)
        S = S[S.status.isin(["ok", "black_at_source"])].merge(F[["sequence_id", "frame_id", "projection", "is_pano", "fileurlLTh"]],
                                       on=["sequence_id", "frame_id"], how="left", suffixes=("", "_sel"))
        if pf == "kartaview":
            S = S[S.projection.eq("SPHERE")]
            order = psz.get("kartaview") or ["wrapped_lth"]
            th = platform_throttle(cfg, "kartaview", "pano_per_s" if cfg.get("platforms.kartaview.pano_per_s", None)
                                   else "per_s", log=log)
        else:
            S = S[S.is_pano.astype(str).eq("True")]
            order = psz.get("mapillary") or ["original"]
            tg = Throttle(cfg.get("platforms.mapillary.graph_per_s", 3.0))
            th = Throttle(cfg.get("platforms.mapillary.files_per_s", 8.0))
        counts = {}
        progress.phase(f"{pf} panoramas", int(S.sequence_id.nunique()), "sequences")
        for seq, g in S.sort_values("order").groupby("sequence_id"):
            progress.update(add=1)
            st = status_path(cfg, pf, seq, "pano")
            if os.path.exists(st):
                old = pd.read_csv(st, dtype={"frame_id": str, "note": object})
                if set(g.frame_id) <= set(old.frame_id[old.status.isin(FINAL)]):
                    for k, v in old.status.value_counts().items():
                        counts[k] = counts.get(k, 0) + int(v)
                    continue
            d = os.path.join(base, pf, seq)
            urls = {}
            if pf == "kartaview":
                urls = {r.frame_id: [(s, kv_url(r.fileurlLTh, s)) for s in order] for r in g.itertuples()}
            else:
                missing = [f for f in g.frame_id if not os.path.exists(os.path.join(d, f"{f}.jpg"))]
                if missing and not offline:
                    from ..secrets import auth_header
                    hdr, meta = auth_header(), {}
                    os.makedirs(os.path.join(d, "records"), exist_ok=True)
                    ids = g.frame_id.tolist()
                    for b in range(0, len(ids), 50):
                        try:
                            raw = fetch(f"{graph_url(cfg)}/?ids={','.join(ids[b:b + 50])}&fields=id,width,height,is_pano,"
                                        + ",".join(MLY_URL[s] for s in order), tg, hdr, timeout=180, ua=ua)
                            with gzip.open(os.path.join(d, "records", f"original_{b // 50:03d}.json.gz"), "wb") as z:
                                z.write(raw)
                            meta.update(json.loads(raw))
                        except (FetchError, ValueError):
                            pass
                    urls = {f: [(s, (meta.get(f) or {}).get(MLY_URL[s])) for s in order] for f in ids}

            def one(r):
                res = save_first(urls.get(r["frame_id"], []), os.path.join(d, f"{r['frame_id']}.jpg"), th, offline, ua)
                return {"frame_id": r["frame_id"], "order": r["order"], "role": r["role"], **res}
            with ThreadPoolExecutor(cfg.get(f"platforms.{pf}.workers", 4)) as ex:
                R = pd.DataFrame(list(ex.map(one, g.to_dict("records"))))
            if os.path.exists(st):                    # keep mis_stitched from the review gate
                old = pd.read_csv(st, dtype={"frame_id": str, "note": object})
                q = old[old.status.eq("mis_stitched")].set_index("frame_id")
                hit = R.frame_id.isin(q.index) & R.status.eq("ok")
                R.loc[hit, "status"] = "mis_stitched"
                R["note"] = R["note"].astype(object)
                R.loc[hit, "note"] = R.loc[hit, "frame_id"].map(q.note)
            R.to_csv(st, index=False)
            for k, v in R.status.value_counts().items():
                counts[k] = counts.get(k, 0) + int(v)
        summary[pf] = counts
        log.info(f"{pf} panoramas: {counts}")
    return [os.path.join(d3(cfg), "status")], summary


# ---------------------------------------------------------------- 3f panorama screening and mis-stitch marking
def _screen_one(job):
    import cv2
    from .. import pano as pv
    pf, seq, fid, path, crop = job
    pv.configure(crop)
    im = cv2.imread(path)
    mask, info = pv.screen_pano(im, edge_black=(pf == "kartaview"))
    row = {"platform": pf, "sequence_id": seq, "frame_id": fid, **{k: info[k] for k in (
        "pano_width", "placeholder", "broken", "pano_valid_share", "coverage_top_deg", "coverage_bottom_deg", "coverage_class")}}
    H, W = im.shape[:2]
    for name, yaw in pv.YAWS.items():
        if mask is None or info["broken"]:
            row[f"valid_{name}"] = np.nan
            continue
        m1, m2, _ = pv._grid(W, H, round(yaw % 360.0, 3), pv.VIEW_W, pv.VIEW_H, pv.HFOV_DEG, pv.PITCH_DOWN_DEG)
        row[f"valid_{name}"] = float((cv2.remap(mask.astype(np.uint8), m1, m2, cv2.INTER_NEAREST, borderMode=cv2.BORDER_WRAP) > 0).mean())
    return row


def pano_check(cfg: Config, log, regions=None, sequences=None, **_):
    from .. import pano as pv
    crop = s3(cfg, "crop", {})
    pv.configure(crop)
    seqs = sequences or seqs_of_regions(cfg, regions)
    jobs = []
    for st in glob.glob(os.path.join(d3(cfg), "status", "*", "*_pano.csv")):
        pf, seq = os.path.basename(os.path.dirname(st)), os.path.basename(st)[:-len("_pano.csv")]
        if seqs and seq not in seqs:
            continue
        S = pd.read_csv(st, dtype={"frame_id": str})
        S = S[S.status.isin(["ok", "mis_stitched"])]
        jobs += [(pf, seq, f, os.path.join(root(cfg, "panoramas"), pf, seq, f"{f}.jpg"), crop) for f in S.frame_id]
    progress.phase("screening panoramas", len(jobs), "panoramas")
    rows_ = []
    with ThreadPoolExecutor(s3(cfg, "workers", 8)) as ex:
        for r_ in ex.map(_screen_one, jobs):
            rows_.append(r_)
            progress.update(add=1)
    # no panoramas (none in the selection, or none downloaded yet): an empty table that still has the columns
    R = pd.DataFrame(rows_) if rows_ else pd.DataFrame(columns=["platform", "sequence_id", "frame_id", "broken"])
    R["coverage_outlier"] = False
    if len(R):
        for (pf, seq), g in R[~R.broken].groupby(["platform", "sequence_id"]):
            if pf == "kartaview":
                R.loc[g.index, "coverage_outlier"] = pv.coverage_outliers(g.coverage_top_deg, g.coverage_bottom_deg)
    p = os.path.join(d3(cfg), "pano_screen.parquet")
    R.to_parquet(p, index=False)
    # review gate: <03_download>/mis_stitched_accepted.csv (platform, sequence_id, frame_id) when present,
    # else every coverage outlier is marked (as decided for Singapore after the contact-sheet check)
    acc = os.path.join(d3(cfg), "mis_stitched_accepted.csv")
    O = pd.read_csv(acc, dtype=str) if os.path.exists(acc) else R[R.coverage_outlier][["platform", "sequence_id", "frame_id"]]
    note = f"band edge off its sequence median by more than {pv.COVERAGE_TOL_DEG:g} deg (screening); file kept"
    n = 0
    for (pf, seq), g in O.groupby(["platform", "sequence_id"]):
        st = status_path(cfg, pf, seq, "pano")
        S = pd.read_csv(st, dtype={"frame_id": str, "note": object})
        hit = S.frame_id.isin(set(g.frame_id)) & S.status.isin(["ok", "mis_stitched"])
        S.loc[hit, "status"] = "mis_stitched"
        S.loc[hit, "note"] = note
        S.to_csv(st, index=False)
        n += int(hit.sum())
    O.to_csv(os.path.join(d3(cfg), "mis_stitched_frames.csv"), index=False)
    log.info(f"screened {len(R)} panoramas; {n} marked mis_stitched ({'accepted list' if os.path.exists(acc) else 'all outliers'})")
    return [p], {"panoramas": len(R), "broken": int(R.broken.sum()) if len(R) else 0, "mis_stitched": n}


# ---------------------------------------------------------------- 3g manifest and completeness checks
def manifest(cfg: Config, log, **_):
    out = d3(cfg)
    F = pd.read_parquet(os.path.join(cfg.paths.step(2), "selection", "frames.parquet")).rename(
        columns={"width": "record_width", "height": "record_height"})
    Sq = pd.read_csv(os.path.join(cfg.paths.step(2), "selection", "sequences.csv"), dtype=str)

    def read_status(kind):
        parts = []
        for f in glob.glob(os.path.join(out, "status", "*", f"*_{kind}.csv")):
            s = pd.read_csv(f, dtype={"frame_id": str})
            s["platform"] = os.path.basename(os.path.dirname(f))
            s["sequence_id"] = os.path.basename(f)[:-len(f"_{kind}.csv")]
            parts.append(s)
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["platform", "sequence_id", "frame_id", "status"])
    St = read_status("frames")
    St = St.drop(columns=[c for c in ("order", "role", "is_pano") if c in St])
    M = F.merge(St, on=["platform", "sequence_id", "frame_id"], how="left")
    M["status"] = M.status.fillna("not_downloaded")
    M["file"] = M.platform + "/" + M.sequence_id + "/" + M.frame_id + ".jpg"
    M = M.merge(Sq[["platform", "sequence_id", "clock_suspect", "contributor", "device"]], on=["platform", "sequence_id"], how="left")
    Ps = read_status("pano")
    if len(Ps):
        M = M.merge(Ps[["platform", "sequence_id", "frame_id", "status"]].rename(columns={"status": "pano_status"}),
                    on=["platform", "sequence_id", "frame_id"], how="left")
        M["quality_flag"] = np.where(M.pano_status.eq("mis_stitched") & M.status.eq("ok"), "mis_stitched", None)
    else:                                            # no panoramas in this selection: keep the columns, left empty
        M["pano_status"], M["quality_flag"] = pd.Series(None, index=M.index, dtype=object), pd.Series(None, index=M.index, dtype=object)
    M["selection_reason"] = np.where(M.role == "in_cell", "region " + M.region_cell + " (" + M.stratum + ")",
                                     np.where(M.role == "in_area", "whole area", "buffer of region clip"))
    prob = []
    bad = M[M.status != "ok"]
    prob.append(bad.assign(problem="status " + bad.status + " " + bad.get("note", pd.Series("", index=bad.index)).fillna("").astype(str)))
    ok = M[M.status == "ok"].copy()
    kv = ok[(ok.platform == "kartaview") & ok["size"].isin([None, np.nan, "lth"]) if "size" in ok else (ok.platform == "kartaview")]
    rw, rh = kv.record_width.astype(float), kv.record_height.astype(float)
    r_rec = np.maximum(rw, rh) / np.minimum(rw, rh)
    r_kv = np.maximum(kv.width, kv.height) / np.minimum(kv.width, kv.height)
    prob.append(kv[(np.maximum(kv.width, kv.height) != 1280) | ((np.abs(r_rec - r_kv) > 0.02) & r_rec.notna())]
                .assign(problem="kartaview size does not match record"))
    my = ok[ok.platform == "mapillary"]
    if "api_width" in my:
        r_api = np.maximum(my.api_width.astype(float), my.api_height.astype(float)) / np.minimum(my.api_width.astype(float), my.api_height.astype(float))
        r_img = np.maximum(my.width, my.height) / np.minimum(my.width, my.height)
        lim = {"thumb_256": 256, "thumb_1024": 1024, "thumb_2048": 2048}.get(sizes(cfg, "mapillary")[0], 10**6)
        prob.append(my[(np.abs(r_api - r_img) > 0.02) | (my.width > lim)].assign(problem="mapillary size does not match record"))
    P = pd.concat(prob, ignore_index=True)[["platform", "sequence_id", "frame_id", "problem"]]
    P.to_csv(os.path.join(out, "check_problems.csv"), index=False)
    gaps = []
    kp = os.path.join(cfg.paths.step(1), "kartaview", "photos.parquet")
    have = {}
    if os.path.exists(kp):
        Kp = pd.read_parquet(kp, columns=["sequence_id", "sequenceIndex", "status", "visibility"])
        Kp["sequence_id"] = Kp.sequence_id.astype(str)
        Kp = Kp[Kp.status.eq("active") & Kp.visibility.eq("public") & Kp.sequence_id.isin(set(F.sequence_id))]
        have = {s: np.sort(g.sequenceIndex.to_numpy()) for s, g in Kp.groupby("sequence_id")}
    for (pf, seq), g in M.sort_values("order").groupby(["platform", "sequence_id"]):
        o = g.order.to_numpy()
        for i in np.flatnonzero(np.diff(o) > 1):
            a, b = int(o[i]), int(o[i + 1])
            if pf == "kartaview":
                h = have.get(seq, np.array([]))
                exist = int(np.searchsorted(h, b) - np.searchsorted(h, a + 1))
            else:
                exist = b - a - 1
            gaps.append({"platform": pf, "sequence_id": seq, "after_order": a, "next_order": b, "skipped": b - a - 1,
                         "on_platform": exist, "type": "between_clips" if exist == b - a - 1 else ("platform_gap" if exist == 0 else "mixed")})
    G = pd.DataFrame(gaps, columns=["platform", "sequence_id", "after_order", "next_order", "skipped", "on_platform", "type"])
    G.to_csv(os.path.join(out, "index_gaps.csv"), index=False)
    if "has_computed" in M:
        nr = M[(M.platform == "mapillary") & M.status.eq("ok") & (~M.has_computed.fillna(False).astype(bool) | M.sfm_cluster.isna())]
        nr[["sequence_id", "frame_id", "has_computed", "sfm_cluster"]].to_csv(os.path.join(out, "no_reconstruction.csv"), index=False)
    keep = ["platform", "sequence_id", "order", "frame_id", "file", "status", "source", "size", "bytes", "width", "height", "half", "role",
            "region_cell", "stratum", "selection_reason", "in_other_half_cell", "already_downloaded", "clock_suspect", "is_pano",
            "projection", "lon", "lat", "heading", "time", "contributor", "device", "sfm_cluster", "has_computed", "pano_status", "quality_flag"]
    M[[c for c in keep if c in M]].to_parquet(cfg.paths.manifest, index=False)
    sm = {"frames": len(M), "by_status": M.status.value_counts().to_dict(), "problems": len(P), "gaps": len(G),
          "gb": {k: round(v / 1e9, 1) for k, v in ok.groupby("platform").bytes.sum().items()}}
    json.dump(sm, open(os.path.join(out, "check_log.json"), "w", encoding="utf8"), indent=1, default=str)
    log.info(f"manifest: {len(M)} frames {sm['by_status']}")
    return [cfg.paths.manifest], sm
