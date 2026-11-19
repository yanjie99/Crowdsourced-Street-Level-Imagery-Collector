"""Check that every web service the tool depends on still answers (19 Nov 2026): `collect doctor` and the Settings
page's "Check services" button.

One light request per service (about eight in all, one at a time), so it is safe to run before every large
collection. A failure names the setting that holds the address, so a moved or re-versioned service can be fixed in
the settings instead of the code. The Mapillary token is sent only in the request header or, for the coverage
tiles, in the tile address as Mapillary requires; it is never printed.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request

from .config import defaults, graph_url, service, user_agent


def _probe(url, headers=None, method="GET", timeout=20, read=True):
    t0 = time.time()
    req = urllib.request.Request(url, headers=headers or {}, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read() if read else b""
        return r.status, body, round((time.time() - t0) * 1000)


def _center(cfg):
    if cfg is None:
        return 103.85, 1.29                     # any populated place works for a liveness check
    try:
        c = cfg.harvest_ll.representative_point()   # the area Step 1 reads (the drawn area in selected-area mode)
    except Exception:  # noqa: BLE001
        c = cfg.boundary_ll.representative_point()
    return c.x, c.y


def _tile(lon, lat, z):
    import math
    n = 2 ** z
    x = int((lon + 180) / 360 * n)
    y = int((1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n)
    return x, y


def checks(cfg=None) -> list[dict]:
    """[{service, setting, host, ok, ms, detail}] for each service, in order."""
    g = (lambda k, d=None: cfg.get(k, d)) if cfg is not None else (
        lambda k, d=None: _dget(defaults(), k, d))
    ua = {"User-Agent": user_agent(cfg)}
    lon, lat = _center(cfg)
    out = []

    def run(name, setting, url, fn):
        host = urllib.parse.urlparse(url).netloc
        try:
            ok, ms, detail = fn()
        except urllib.error.HTTPError as e:
            ok, ms, detail = False, None, f"HTTP {e.code}"
        except Exception as e:  # noqa: BLE001
            ok, ms, detail = False, None, f"{type(e).__name__}: {str(e)[:120]}"
        out.append({"service": name, "setting": setting, "host": host, "ok": ok, "ms": ms, "detail": detail})

    # Mapillary
    from .secrets import get_token
    tok = get_token(required=False)
    gb = graph_url(cfg)
    if tok:
        def graph():
            # a small box: area searches are slow where captures are dense
            q = urllib.parse.urlencode({"bbox": f"{lon - .002},{lat - .002},{lon + .002},{lat + .002}", "limit": 1, "fields": "id"})
            st, body, ms = _probe(f"{gb}/images?{q}", {**ua, "Authorization": f"OAuth {tok}"}, timeout=60)
            return st == 200, ms, f"{len(json.loads(body).get('data', []))} image in a 1-image test"
        run("Mapillary Graph API", "services.mapillary_graph", gb, graph)
        turl = g("platforms.mapillary.tiles_url")

        def tiles():
            x, y = _tile(lon, lat, 14)
            st, body, ms = _probe(turl.format(z=14, x=x, y=y) + f"?access_token={tok}", ua)
            return st == 200, ms, f"coverage tile, {len(body):,} bytes"
        run("Mapillary coverage tiles", "platforms.mapillary.tiles_url", turl, tiles)
    else:
        out.append({"service": "Mapillary", "setting": "token", "host": urllib.parse.urlparse(gb).netloc, "ok": None,
                    "ms": None, "detail": "skipped: no Mapillary token (store one under Settings)"})

    # KartaView: the API, and one image link it gives out today (its storage host can change)
    api = str(g("platforms.kartaview.api")).rstrip("/")
    seen = {}

    cc = (g("area.country_code", None) if cfg is not None else None) or "SG"

    def kv_api():
        # the same kind of request as step 1d (an unfiltered listing is refused with "Restricted access")
        st, body, ms = _probe(f"{api}/sequence/?countryCode={cc}&itemsPerPage=5&page=1", ua)
        d = json.loads(body)
        code = (d.get("status") or {}).get("apiCode")
        rows = ((d.get("result") or {}).get("data") or [])
        seen["seqs"] = [r["id"] for r in rows]
        if code not in (None, 600):
            return False, ms, f"API code {code}: {(d.get('status') or {}).get('apiMessage')}"
        return st == 200, ms, f"sequence listing answers ({cc})"
    run("KartaView API", "platforms.kartaview.api", api, kv_api)
    if seen.get("seqs"):
        def kv_storage():
            # the first image of up to five listed sequences: some old files are gone from KartaView's own storage
            # even while the API lists them, so one dead link is not an outage
            good, hosts, ms = 0, set(), None
            for sid in seen["seqs"]:
                _, body, _ = _probe(f"{api}/sequence/{sid}/photos?itemsPerPage=1&page=1", ua)
                ph = ((json.loads(body).get("result") or {}).get("data") or [{}])[0]
                link = ph.get("fileurlLTh")
                if not link:
                    continue
                hosts.add(urllib.parse.urlparse(link).netloc)
                try:
                    st, data, t = _probe(link, ua)
                    if st == 200 and len(data) > 1000:
                        good, ms = good + 1, ms or t
                except urllib.error.HTTPError:
                    pass
            n = len(seen["seqs"])
            return good > 0, ms, f"{good} of {n} sampled image links work; hosts: {', '.join(sorted(hosts)) or 'none'}"
        run("KartaView image storage", "(links come from the API)", api, kv_storage)

    # OpenStreetMap services
    geo = service(cfg, "geofabrik")
    run("Geofabrik extracts", "services.geofabrik", geo,
        lambda: (lambda r: (r[0] == 200, r[2], "region index answers"))(_probe(f"{geo}/index-v1.json", ua, "HEAD", read=False)))
    nom = service(cfg, "nominatim")
    run("Nominatim place search", "services.nominatim", nom,
        lambda: (lambda r: (r[0] == 200, r[2], r[1].decode("utf8", "ignore")[:40]))(_probe(f"{nom}/status", ua)))
    oh = str(g("osm.history_api")).rstrip("/")
    run("ohsome OSM history", "osm.history_api", oh,
        lambda: (lambda r: (r[0] == 200, r[2], "metadata answers"))(_probe(f"{oh}/metadata", ua)))
    mt = service(cfg, "map_tiles")
    run("Background map tiles", "services.map_tiles", mt,
        lambda: (lambda r: (r[0] == 200, r[2], "tile 0/0/0"))(_probe(mt.format(z=0, x=0, y=0), ua)))
    return out


def _dget(d, key, default=None):
    for k in key.split("."):
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d
