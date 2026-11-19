"""Perspective views from equirectangular panoramas, for the quality assessment framework.

Scheme (15 Nov 2026, set from the Singapore panorama collection): four views at yaw
0, 90, 180, 270 deg, horizontal field of view 90 deg, pitch 5 deg down, 1024 x 576 px.
Each view then spans about 24 deg above to 34 deg below the horizon, which fits inside
the KartaView band panoramas (about 42 deg above to 40 deg below). One geometry for every
panorama size and for panoramas with reduced vertical coverage. Each view comes with a validity
mask (False where the panorama has no image data) and its exact pinhole intrinsics.

Revised 16 Nov 2026 after a test on the full collection (166,052 panoramas):
  - screen_pano() runs before any view is made: placeholder (all black), validity mask, coverage,
    and a 'broken' flag for panoramas that keep only a patch of picture or are cut off with gray.
    coverage_outliers() then flags mis-stitched frames by comparing coverage within a sequence.
  - The black-edge rule is switched off for full-sphere cameras (edge_black=False, Mapillary): it
    took dark night skies for missing data.
  - The validity mask also removes flat gray fill joined to the bottom edge (a truncated image,
    found on 2 KartaView panoramas and their front-view thumbnails), besides black joined to the
    top or bottom edge.
  - Each view reports 'usable' (valid share at least VIEW_MIN_VALID); 0.2% of fleet views reach
    a black stitching gap.
  - Resolution is measured per view pixel from the projection itself (including the equirectangular
    stretch with latitude): center scale, the share of the view that is enlarged from the source,
    and the largest enlargement. Effective focal length remains the center value.
  - Orientation: travel_yaw_deg is the direction of travel measured clockwise from the panorama
    center, i.e. track bearing minus camera heading (travel_yaw()). The views are turned by it, so
    'front' faces the direction of travel. A mount yaw defined as heading minus bearing has the
    opposite sign.
  - Sampling grids are cached per panorama size and yaw, and fixed-point maps are used, so a
    panorama costs one blur and four remaps. The blur is applied once per panorama.
  - Rows are clamped at the poles instead of wrapping vertically; coverage_class adds 'cut_top'.

Revised 17 Nov 2026: quality filter before cropping. crop_list() reads the manifest and keeps only
panoramas with pano_status == CROP_STATUS ("ok"); the 365 mis-stitched panoramas (pano_status
"mis_stitched") and the 413 black ones are dropped, but their files are not deleted. Panoramas that
pass are still screened per image by views_from_pano (placeholder, broken, views under 99% valid).

Revised 17 Nov 2026 (crop pilot): leveling. Mapillary panoramas are not leveled at source; the
cameras (handheld, helmet and bike mounts) are tilted by a median 1-18 deg per sequence, up to about
30 deg, which gave diagonal horizons and front views looking into the sky. level_rotation() turns
each frame's computed_rotation into a leveled frame (down on gravity, panorama-center heading kept)
and views_from_pano(level=...) takes the views in it. Checked by eye per SfM cluster
(review/levelling): per-frame leveling is right for nearly all frames; a per-sequence median is
not. KartaView is not leveled.
"""
import os
from functools import lru_cache

import cv2
import numpy as np

VIEW_W, VIEW_H = 1024, 576
HFOV_DEG = 90.0
PITCH_DOWN_DEG = 5.0
YAWS = {"front": 0.0, "right": 90.0, "back": 180.0, "left": 270.0}
VIEW_MIN_VALID = 0.99      # a view below this share of valid pixels is not usable
PANO_MIN_VALID = 0.20      # fleet band panoramas are about 0.46 valid (0.3 with a stitching gap), NCTech 0.8,
                           # full spheres 1.0; below 0.2 only a patch of picture is left
COVERAGE_TOL_DEG = 8.0     # a frame whose band edge is this far from its sequence median is mis-stitched
CROP_STATUS = "ok"          # quality filter: only panoramas whose manifest pano_status is "ok" are cropped.
                           # Dropped: black_at_source (all black), mis_stitched (365 KartaView frames,
                           # mark_misstitched.py), failed, no_url, and frames without a panorama. The
                           # dropped panorama files stay on disk; they are only not cropped.


def travel_yaw(track_bearing_deg, camera_heading_deg):
    """Direction of travel relative to the panorama center, clockwise, in (-180, 180]."""
    d = (track_bearing_deg - camera_heading_deg) % 360.0
    return d - 360.0 if d > 180.0 else d


def valid_mask(pano_bgr, black_max=4, min_share=0.002, edge_black=True):
    """True where the panorama holds image data. Invalid: pure-black regions joined to the top or
    bottom edge (missing vertical coverage, stitching gaps that reach the band edge), and flat gray
    fill joined to the bottom edge (an image cut off part-way).
    edge_black=False skips the black rule, for full-sphere cameras (all Mapillary panoramas), where
    a dark night sky would otherwise be taken for missing data."""
    g = cv2.cvtColor(pano_bgr, cv2.COLOR_BGR2GRAY)
    h = g.shape[0]
    bad = np.zeros(g.shape, bool)
    if edge_black:
        black = (g <= black_max).astype(np.uint8)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(black, connectivity=8)
        top, hh, area = stats[:, cv2.CC_STAT_TOP], stats[:, cv2.CC_STAT_HEIGHT], stats[:, cv2.CC_STAT_AREA]
        keep = np.zeros(n, bool)
        keep[1:] = ((top[1:] == 0) | (top[1:] + hh[1:] >= h)) & (area[1:] >= min_share * g.size)
        bad = keep[lab]
    # flat gray fill: rows of one tone (not black, not saturated) running down to the bottom edge
    rf = g.astype(np.float32)
    flat = (rf.std(axis=1) < 1.0) & (rf.mean(axis=1) > black_max * 2) & (rf.mean(axis=1) < 245)
    if flat[-1]:
        r = h - 1
        while r > 0 and flat[r - 1]:
            r -= 1
        if (h - r) >= 0.02 * h:            # longer than a pole cap, which is flat by geometry
            bad[max(0, r - 16):] = True      # the cut lies inside the last 16-row block above
    return ~bad


def is_placeholder(pano_bgr, max_mean=2.0):
    """True for an all-black image served in place of a missing panorama."""
    return float(pano_bgr.mean()) < max_mean


def coverage_deg(mask):
    """(top, bottom) elevation in degrees of the valid band: the median over columns of each column's
    first and last valid row. Columns with no valid pixel (a stitching gap) are left out, so a gap
    does not move the edges. None if nothing is valid."""
    h = mask.shape[0]
    cols = mask.any(axis=0)
    if not cols.any():
        return None
    m = mask[:, cols]
    first = np.argmax(m, axis=0)
    last = h - np.argmax(m[::-1], axis=0)
    return 90 - float(np.median(first)) / h * 180, 90 - float(np.median(last)) / h * 180


def coverage_class(mask):
    """'full' (whole sphere), 'cut_bottom' (nadir missing), 'cut_top' (zenith missing) or 'band'
    (both missing), from the validity mask; None if nothing is valid."""
    cov = coverage_deg(mask)
    if cov is None:
        return None
    top_ok, bottom_ok = cov[0] >= 80, cov[1] <= -80
    return {(True, True): "full", (True, False): "cut_bottom", (False, True): "cut_top", (False, False): "band"}[(top_ok, bottom_ok)]


def screen_pano(pano_bgr, edge_black=True):
    """Per-panorama checks, run before any view is made. Returns (mask or None, info).
    A mis-stitched frame (warped, band in the wrong place) is not caught here; compare its coverage
    with its sequence afterwards (coverage_outliers)."""
    H, W = pano_bgr.shape[:2]
    info = {"pano_width": W, "pano_height": H, "placeholder": is_placeholder(pano_bgr)}
    if info["placeholder"]:
        return None, {**info, "pano_valid_share": 0.0, "coverage_top_deg": None, "coverage_bottom_deg": None,
                      "coverage_class": None, "broken": True}
    m = valid_mask(pano_bgr, edge_black=edge_black)
    cov = coverage_deg(m)
    share = float(m.mean())
    info.update(pano_valid_share=share, coverage_top_deg=None if cov is None else cov[0],
                coverage_bottom_deg=None if cov is None else cov[1], coverage_class=coverage_class(m),
                broken=share < PANO_MIN_VALID)
    return m, info


def coverage_outliers(top_deg, bottom_deg, tol=COVERAGE_TOL_DEG):
    """For the frames of one sequence (arrays of coverage_top_deg and coverage_bottom_deg from
    screen_pano): True where a frame's coverage is more than tol degrees from the sequence median,
    i.e. a mis-stitched panorama (75 KartaCam frames in 42 sequences of the Singapore collection).
    Meaningful for band and cut-bottom cameras; full spheres have no edge to compare."""
    t = np.asarray(top_deg, float)
    b = np.asarray(bottom_deg, float)
    out = (np.abs(t - np.nanmedian(t)) > tol) | (np.abs(b - np.nanmedian(b)) > tol)
    return out | np.isnan(t) | np.isnan(b)


def level_rotation(rotation_vec):
    """Leveling for a panorama whose camera is tilted, from its world-to-camera rotation as an
    angle-axis vector (Mapillary computed_rotation, OpenSfM convention: world east-north-up, camera
    x right, y down, z forward, the same axes as the views here).
    The leveled frame keeps the heading of the panorama center (camera forward projected onto the
    horizontal; its compass bearing equals Mapillary computed_compass_angle) and puts 'down' on
    gravity. Returns (C, info): C (3x3, leveled ray -> camera ray) to pass to views_from_pano as
    level, and the camera's pitch (forward axis above the horizon), roll (right axis above the
    horizon) and tilt (up axis from vertical) in degrees."""
    R, _ = cv2.Rodrigues(np.asarray(rotation_vec, float).reshape(3, 1))
    fwd = R[2].copy()
    fwd[2] = 0.0
    fwd /= np.linalg.norm(fwd)
    down = np.array([0.0, 0.0, -1.0])
    L = np.stack([np.cross(down, fwd), down, fwd])        # rows: leveled right, down, forward in world axes
    C = R @ L.T
    info = {"cam_pitch_deg": float(np.degrees(np.arcsin(np.clip(R[2, 2], -1, 1)))),
            "cam_roll_deg": float(np.degrees(np.arcsin(np.clip(R[0, 2], -1, 1)))),
            "cam_tilt_deg": float(np.degrees(np.arccos(np.clip(-R[1, 2], -1, 1))))}
    return C, info


@lru_cache(maxsize=64)
def _grid(W, H, yaw_deg, width, height, hfov_deg, pitch_down_deg, level=None):
    """Sampling grid (fixed-point maps for remap) and per-pixel resolution figures for one view.
    level: None, or the 9 entries of C from level_rotation (leveled ray -> camera ray)."""
    f = (width / 2) / np.tan(np.radians(hfov_deg) / 2)
    u, v = np.meshgrid(np.arange(width + 1) + 0.5, np.arange(height + 1) + 0.5)   # one extra row and column for differences
    x, y, z = (u - width / 2) / f, (v - height / 2) / f, np.ones_like(u)
    p = np.radians(pitch_down_deg)                  # pitch down: rotate about the x axis
    y2, z2 = y * np.cos(p) + z * np.sin(p), -y * np.sin(p) + z * np.cos(p)
    a = np.radians(yaw_deg)                         # yaw: rotate about the vertical axis, clockwise
    x3, z3 = x * np.cos(a) + z2 * np.sin(a), -x * np.sin(a) + z2 * np.cos(a)
    if level is not None:                           # leveled frame -> tilted camera frame
        c = np.asarray(level, float).reshape(3, 3)
        x3, y2, z3 = (c[0, 0] * x3 + c[0, 1] * y2 + c[0, 2] * z3, c[1, 0] * x3 + c[1, 1] * y2 + c[1, 2] * z3,
                      c[2, 0] * x3 + c[2, 1] * y2 + c[2, 2] * z3)
    lon = np.arctan2(x3, z3)
    lat = np.arctan2(-y2, np.hypot(x3, z3))
    mx = (lon / (2 * np.pi) + 0.5) * W - 0.5
    my = np.clip((0.5 - lat / np.pi) * H - 0.5, 0, H - 1)       # clamp at the poles, no vertical wrap
    # local scale: source pixels per view pixel along u and v (unwrap longitude jumps)
    dxu = np.diff(mx, axis=1)[:-1]
    dxu = (dxu + W / 2) % W - W / 2
    dyu = np.diff(my, axis=1)[:-1]
    dxv = np.diff(mx, axis=0)[:, :-1]
    dxv = (dxv + W / 2) % W - W / 2
    dyv = np.diff(my, axis=0)[:, :-1]
    J = np.stack([np.stack([dxu, dxv], -1), np.stack([dyu, dyv], -1)], -2)       # (h, w, 2, 2)
    smin = np.linalg.svd(J, compute_uv=False)[..., -1]          # < 1: the view enlarges the source here
    m1, m2 = cv2.convertMaps(mx[:-1, :-1].astype(np.float32), my[:-1, :-1].astype(np.float32), cv2.CV_16SC2)
    res = {"focal_px": float(f), "center_scale": float(W / (2 * np.pi) / f),
           "upsampled_share": float((smin < 1).mean()), "max_enlargement": float(1 / smin.min())}
    return m1, m2, res


def _prefilter(pano_bgr, width=VIEW_W, hfov_deg=HFOV_DEG):
    """Blur once per panorama when the view center reduces it, so detail finer than the view's
    pixel does not alias. Sigma follows the usual sqrt(s^2 - 1) / 2 rule for a reduction by s."""
    f = (width / 2) / np.tan(np.radians(hfov_deg) / 2)
    s = pano_bgr.shape[1] / (2 * np.pi) / f
    return cv2.GaussianBlur(pano_bgr, (0, 0), 0.5 * np.sqrt(s * s - 1)) if s > 1.05 else pano_bgr


def view_from_pano(pano_bgr, yaw_deg, mask=None, width=VIEW_W, height=VIEW_H, hfov_deg=HFOV_DEG,
                   pitch_down_deg=PITCH_DOWN_DEG, prefiltered=False, name=None, level=None):
    """One perspective view. yaw_deg is measured clockwise from the panorama center.
    level: None, or C from level_rotation (the view is then taken in the leveled frame).
    Returns (view_bgr, view_valid_mask, info)."""
    H, W = pano_bgr.shape[:2]
    lv = None if level is None else tuple(round(float(c), 5) for c in np.asarray(level).ravel())
    m1, m2, res = _grid(W, H, round(float(yaw_deg) % 360.0, 3), width, height, float(hfov_deg), float(pitch_down_deg), lv)
    src = pano_bgr if prefiltered else _prefilter(pano_bgr, width, hfov_deg)
    view = cv2.remap(src, m1, m2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    vm = np.ones((height, width), bool)
    if mask is not None:
        vm = cv2.remap(mask.astype(np.uint8), m1, m2, cv2.INTER_NEAREST, borderMode=cv2.BORDER_WRAP) > 0
    f, native_f = res["focal_px"], W / (2 * np.pi)
    share = float(vm.mean())
    info = {"view": name, "yaw_deg": float(yaw_deg), "pitch_down_deg": pitch_down_deg, "hfov_deg": hfov_deg,
            "focal_px": f, "cx": width / 2, "cy": height / 2, "width": width, "height": height,
            "pano_width": W, "native_px_per_rad": float(native_f),
            "effective_focal_px": float(min(f, native_f)),          # at the view center
            "center_scale": res["center_scale"],                    # source px per view px at the center
            "upsampled_share": res["upsampled_share"],              # share of the view enlarged from the source
            "max_enlargement": res["max_enlargement"],
            "upsampled": bool(native_f < f),                        # enlarged even at the center
            "valid_share": share, "usable": share >= VIEW_MIN_VALID}
    return view, vm, info


def views_from_pano(pano_bgr, travel_yaw_deg=0.0, yaws=YAWS, edge_black=True, level=None):
    """Screen one panorama and, unless it is a placeholder or broken, make all its views.
    travel_yaw_deg: direction of travel clockwise from the panorama center (travel_yaw()); 0 keeps
    the panorama center as 'front'. edge_black: see valid_mask (False for Mapillary).
    level: None, or C from level_rotation for a tilted camera (Mapillary).
    Returns (pano_info, {name: (view, valid mask, info)})."""
    mask, pinfo = screen_pano(pano_bgr, edge_black=edge_black)
    pinfo["travel_yaw_deg"] = float(travel_yaw_deg)
    if pinfo["broken"]:
        return pinfo, {}
    src = _prefilter(pano_bgr)
    return pinfo, {k: view_from_pano(src, travel_yaw_deg + a, mask, prefiltered=True, name=k, level=level)
                   for k, a in yaws.items()}


def read_mask(out_dir, row):
    """Validity mask (bool, True = valid) of one view, from a views.parquet row of the crop output:
    mask_file is the sequence's masks.zip and mask_member the PNG inside it (pack_masks.py), or
    a loose PNG when there is no mask_member. None for a view that was not written."""
    import zipfile
    mf = row["mask_file"]
    if mf is None or mf != mf:
        return None
    member = row.get("mask_member")
    if member is None or member != member:
        data = open(os.path.join(out_dir, mf), "rb").read()
    else:
        with zipfile.ZipFile(os.path.join(out_dir, mf)) as z:
            data = z.read(member)
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE) > 127


def crop_list(manifest, pano_dir=None):
    """Quality filter for the crop procedure: the panoramas to crop, from the Step 3 manifest
    (a path or a DataFrame built by check_download.py --pano). Keeps rows whose pano_status is
    CROP_STATUS and drops the rest (mis_stitched, black_at_source, failed, no_url, non-panoramas).
    Files of dropped panoramas are left on disk. With pano_dir, adds pano_path. Returns
    (kept, dropped counts by pano_status)."""
    import os

    import pandas as pd
    M = pd.read_parquet(manifest) if isinstance(manifest, str) else manifest
    if "pano_status" not in M:                       # manifest of a selection without panoramas: nothing to crop
        M = M.assign(pano_status=pd.Series(None, index=M.index, dtype=object))
    P = M[M.pano_status.notna()]
    keep = P.pano_status.eq(CROP_STATUS)
    K = P[keep].copy()
    if pano_dir:
        K["pano_path"] = [os.path.join(pano_dir, pf, s, f"{fid}.jpg") for pf, s, fid in zip(K.platform, K.sequence_id, K.frame_id)]
    return K, P[~keep].pano_status.value_counts().to_dict()


def configure(crop: dict) -> None:
    """Set the crop scheme from project.yaml (step3_download.crop): views, hfov_deg, pitch_down_deg, size,
    view_min_valid, pano_min_valid, coverage_tol_deg. Updates the module settings and the defaults of the
    functions that bind them, so every caller uses the project's scheme (added for the collect package)."""
    g = globals()
    w, h = crop.get("size", [g["VIEW_W"], g["VIEW_H"]])
    new = {"VIEW_W": int(w), "VIEW_H": int(h), "HFOV_DEG": float(crop.get("hfov_deg", g["HFOV_DEG"])),
           "PITCH_DOWN_DEG": float(crop.get("pitch_down_deg", g["PITCH_DOWN_DEG"])),
           "YAWS": {k: float(v) for k, v in (crop.get("views") or g["YAWS"]).items()},
           "VIEW_MIN_VALID": float(crop.get("view_min_valid", g["VIEW_MIN_VALID"])),
           "PANO_MIN_VALID": float(crop.get("pano_min_valid", g["PANO_MIN_VALID"])),
           "COVERAGE_TOL_DEG": float(crop.get("coverage_tol_deg", g["COVERAGE_TOL_DEG"]))}
    g.update(new)
    _prefilter.__defaults__ = (new["VIEW_W"], new["HFOV_DEG"])
    d = list(view_from_pano.__defaults__)            # mask, width, height, hfov_deg, pitch_down_deg, prefiltered, name, level
    d[1:5] = [new["VIEW_W"], new["VIEW_H"], new["HFOV_DEG"], new["PITCH_DOWN_DEG"]]
    view_from_pano.__defaults__ = tuple(d)
    d = list(views_from_pano.__defaults__)           # travel_yaw_deg, yaws, edge_black, level
    d[1] = new["YAWS"]
    views_from_pano.__defaults__ = tuple(d)
    coverage_outliers.__defaults__ = (new["COVERAGE_TOL_DEG"],)
    _grid.cache_clear()
