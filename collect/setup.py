"""Step 0: set up a project for a new study area (Phase 2, 18 Nov 2026).

  collect init  the boundary (an OSM place found with Nominatim, or a GeoJSON file) and the area settings derived
                from it: harvest and OSM boxes, a metric CRS (UTM zone of the center), the time zone (zone.tab, the
                country's zone nearest the center), driving side and country code -> <folder>/project.yaml,
                <folder>/00_setup/boundary.geojson. Everything else comes from collect/defaults.yaml.
  step 0a       the dated Geofabrik extract that contains the area: the smallest region of Geofabrik's index whose
                outline holds the boundary, its newest date-stamped file (<region>-YYMMDD.osm.pbf), MD5 checked
                -> 00_setup/osm/<file>, 00_setup/osm/extract.json (read by every OSM step unless osm.extract is set)
The user still has to set what cannot be derived: the road standard (lane and line widths), the storage budget,
the Mapillary token, and for KartaView bulk use the platform's approval.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import urllib.parse
import urllib.request

import yaml

from .config import service
from .net import Throttle, now

LEFT_HAND = set("AG AI AU BB BD BM BN BS BT BW CY DM FJ FK GB GD GG GY HK ID IE IM IN JE JM JP KE KI KN KY LC LK LS MO MS "
                "MT MU MV MW MY MZ NA NP NR NZ PG PK PN SB SC SG SH SR SZ TC TH TL TO TT TV TZ UG VC VG VI WS ZA ZM ZW".split())
_th = Throttle(1.0)   # Nominatim and Geofabrik: at most one request a second


def _get(url, timeout=120, cfg=None):
    """GET with the tool's User-Agent (platforms.user_agent), at most one request a second."""
    from .config import user_agent
    _th.wait()
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": user_agent(cfg)}), timeout=timeout) as r:
        return r.read()


def search_places(query: str, limit: int = 8) -> list[dict]:
    """Nominatim search: up to `limit` matches, each with an outline when OSM has one (a city, district, park ...)
    and always its bounding box, so a match that is only a point can still give an area."""
    q = urllib.parse.urlencode({"q": query, "format": "jsonv2", "polygon_geojson": 1, "polygon_threshold": 0.0005,
                                "addressdetails": 1, "limit": limit})
    out = []
    for r in json.loads(_get(f"{service(None, 'nominatim')}/search?{q}")):
        bb = [float(v) for v in r.get("boundingbox", [])]          # south, north, west, east
        g = r.get("geojson") or {}
        out.append({"name": r.get("display_name"), "kind": f"{r.get('category', r.get('class', ''))}/{r.get('type', '')}",
                    "osm": f"{r.get('osm_type')}/{r.get('osm_id')}",
                    "country_code": (r.get("address") or {}).get("country_code", "").upper(),
                    "has_outline": g.get("type") in ("Polygon", "MultiPolygon"),
                    "geometry": g if g.get("type") in ("Polygon", "MultiPolygon") else None,
                    "bbox": [bb[2], bb[0], bb[3], bb[1]] if len(bb) == 4 else None,       # west, south, east, north
                    "lon": float(r["lon"]), "lat": float(r["lat"])})
    return out


def bbox_polygon(b):
    w, s_, e, n = b
    return {"type": "Polygon", "coordinates": [[[w, s_], [e, s_], [e, n], [w, n], [w, s_]]]}


def find_place(query: str) -> dict:
    """The best match that has an outline; else the first match's bounding box (for a place OSM maps only as a point)."""
    res = search_places(query, 5)
    if not res:
        raise ValueError(f"no place found for {query!r}: try another spelling, add the country, or draw the area on the map")
    r = next((x for x in res if x["has_outline"]), None)
    if r is None:
        r = res[0]
        if not r["bbox"]:
            raise ValueError(f"no area outline found for {query!r}; draw the area on the map or give a GeoJSON file")
        r = {**r, "geometry": bbox_polygon(r["bbox"]), "name": f"{r['name']} (bounding box)"}
    return {"name": r["name"], "osm": r["osm"], "geometry": r["geometry"], "country_code": r["country_code"]}


def utm_epsg(lon, lat):
    zone = int((lon + 180) // 6) + 1
    return f"EPSG:{(32600 if lat >= 0 else 32700) + zone}"


def timezone_for(lon, lat, country):
    """The zone.tab zone of the country nearest the point (or of any country when the country is unknown)."""
    import tzdata
    path = os.path.join(os.path.dirname(tzdata.__file__), "zoneinfo", "zone.tab")
    best = None
    for line in open(path, encoding="utf8"):
        if line.startswith("#"):
            continue
        cc, coord, tz = line.split("\t")[:3]
        if country and cc != country:
            continue
        m = re.match(r"([+-])(\d{4,6})([+-])(\d{5,7})", coord)
        def dms(sign, s, n):
            v = int(s[:n]) + int(s[n:n + 2]) / 60 + (int(s[n + 2:]) / 3600 if len(s) > n + 2 else 0)
            return -v if sign == "-" else v
        zla, zlo = dms(m.group(1), m.group(2), 2), dms(m.group(3), m.group(4), 3)
        d = (zla - lat) ** 2 + ((zlo - lon) * math.cos(math.radians(lat))) ** 2
        if best is None or d < best[0]:
            best = (d, tz.strip(), cc)
    return best[1] if best else "UTC"


def country_for(lon, lat):
    """Country code of the zone.tab entry nearest the point (used when a boundary file gives no country)."""
    import tzdata
    path = os.path.join(os.path.dirname(tzdata.__file__), "zoneinfo", "zone.tab")
    best = None
    for line in open(path, encoding="utf8"):
        if line.startswith("#"):
            continue
        cc, coord = line.split("	")[:2]
        m = re.match(r"([+-])(\d{4,6})([+-])(\d{5,7})", coord)
        v = lambda sg, t, n: (-1 if sg == "-" else 1) * (int(t[:n]) + int(t[n:n + 2]) / 60)  # noqa: E731
        la, lo = v(m.group(1), m.group(2), 2), v(m.group(3), m.group(4), 3)
        d = (la - lat) ** 2 + ((lo - lon) * math.cos(math.radians(lat))) ** 2
        if best is None or d < best[0]:
            best = (d, cc)
    return best[1] if best else ""


def init_project(name, folder, place=None, boundary=None, platforms=("mapillary", "kartaview"), budget_gb=None, country=None):
    from shapely.geometry import mapping, shape
    os.makedirs(os.path.join(folder, "00_setup"), exist_ok=True)
    if place:
        p = find_place(place)
        geom, cc, src = shape(p["geometry"]), p["country_code"], {"nominatim": p["name"], "osm": p["osm"], "fetched_utc": now()}
    else:
        g = json.load(open(boundary, encoding="utf8"))
        feats = g["features"] if g.get("type") == "FeatureCollection" else [g]
        import shapely
        geom = shapely.union_all([shape(f.get("geometry", f)) for f in feats])
        c0 = geom.representative_point()
        if country and len(country) == 2:
            cc, src = country.upper(), {"file": os.path.abspath(boundary), "country_from": "the chosen place"}
        else:
            cc, src = country_for(c0.x, c0.y), {"file": os.path.abspath(boundary), "country_from": "nearest zone.tab entry"}
    bfile = os.path.join(folder, "00_setup", "boundary.geojson")
    json.dump({"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {"name": name, **src},
                                                          "geometry": mapping(geom)}]}, open(bfile, "w", encoding="utf8"))
    b = geom.bounds
    c = geom.representative_point()
    pad = lambda m: [round(b[0] - m, 4), round(b[1] - m, 4), round(b[2] + m, 4), round(b[3] + m, 4)]  # noqa: E731
    cfg = {"project": {"name": name, "folder": os.path.abspath(folder)},
           "area": {"boundary_file": "00_setup/boundary.geojson", "boundary_source": src,
                    "bbox_harvest": pad(0.005), "bbox_osm": pad(0.02), "metric_crs": utm_epsg(c.x, c.y),
                    "timezone": timezone_for(c.x, c.y, cc), "driving_side": "left" if cc in LEFT_HAND else "right",
                    "country_code": cc or "??"},
           "platforms": {p: {"enabled": p in platforms} for p in ("mapillary", "kartaview")}}
    if budget_gb:
        cfg["storage"] = {"budget_gb": budget_gb}
    path = os.path.join(folder, "project.yaml")
    with open(path, "w", encoding="utf8") as f:
        f.write("# Created by collect init. Check the derived area settings, then set what cannot be derived:\n"
                "#   step5_osm.road_standard (lane and line widths from the national standards; the defaults are generic),\n"
                "#   storage.budget_gb, the Mapillary token (collect token set), and for heavy KartaView use its approval.\n")
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    return path, cfg


# ---------------------------------------------------------------- step 0a: dated Geofabrik extract
def extract(cfg, log, offline=False, **_):
    import shapely
    from shapely.geometry import shape
    d = os.path.join(cfg.paths.step(0), "osm")
    os.makedirs(d, exist_ok=True)
    info_p = os.path.join(d, "extract.json")
    if cfg.get("osm.extract", None):
        log.info(f"osm.extract is set ({cfg.get('osm.extract')}); nothing to download")
        return [], {"extract": cfg.get("osm.extract")}
    if os.path.exists(info_p):
        info = json.load(open(info_p, encoding="utf8"))
        if os.path.exists(os.path.join(d, info["file"])):
            log.info(f"extract already here: {info['file']}")
            return [info_p], {"extract": info["file"], "date": info["date"]}
    if offline:
        raise RuntimeError("no OSM extract on disk and offline")
    idx_p = os.path.join(d, "geofabrik_index-v1.json")
    if not os.path.exists(idx_p):
        open(idx_p, "wb").write(_get(f"{service(cfg, 'geofabrik')}/index-v1.json", timeout=300, cfg=cfg))
    idx = json.load(open(idx_p, encoding="utf8"))
    area = cfg.boundary_ll
    best = None
    for f in idx["features"]:
        g = f.get("geometry")
        if not g:
            continue
        geom = shape(g)
        if geom.contains(area):
            a = geom.area
            if best is None or a < best[0]:
                best = (a, f["properties"])
    if best is None:
        raise RuntimeError("no Geofabrik region contains the whole boundary; set osm.extract to a file you provide")
    props = best[1]
    latest = props["urls"]["pbf"]
    folder_url, fname = latest.rsplit("/", 1)
    stem = fname.replace("-latest.osm.pbf", "")
    dated = []
    for page in (f"{folder_url}/{stem}.html", folder_url + "/"):     # the region page lists the dated files
        listing = _get(page, cfg=cfg).decode("utf8", "ignore")
        dated = sorted(set(re.findall(rf"{re.escape(stem)}-(\d{{6}})\.osm\.pbf", listing)))
        if dated:
            break
    if not dated:
        raise RuntimeError(f"no date-stamped file for {stem} at {folder_url}")
    ymd = dated[-1]
    fn = f"{stem}-{ymd}.osm.pbf"
    md5 = _get(f"{folder_url}/{fn}.md5", cfg=cfg).decode().split()[0]
    data = _get(f"{folder_url}/{fn}", timeout=1800, cfg=cfg)
    got = hashlib.md5(data).hexdigest()
    if got != md5:
        raise RuntimeError(f"MD5 mismatch for {fn}")
    open(os.path.join(d, fn), "wb").write(data)
    info = {"region": props.get("id"), "name": props.get("name"), "file": fn, "url": f"{folder_url}/{fn}",
            "date": f"20{ymd[:2]}-{ymd[2:4]}-{ymd[4:]}", "bytes": len(data), "md5": md5, "fetched_utc": now(),
            "license": "ODbL, © OpenStreetMap contributors (Geofabrik extract)"}
    json.dump(info, open(info_p, "w", encoding="utf8"), indent=1)
    log.info(f"extract {fn} ({len(data) / 1e6:.1f} MB, {info['date']}) from Geofabrik region {props.get('id')}")
    return [info_p], {"extract": fn, "date": info["date"], "mb": round(len(data) / 1e6, 1)}
