"""Unit tests that run anywhere (no network, no Singapore data): python -m pytest"""
import json
import math
import os

import pytest
import yaml

from collect.config import Config, ConfigError, defaults, merge, us_names
from collect.ledger import Ledger, fingerprint
from collect.log import redact, register_secret
from collect.setup import timezone_for, utm_epsg
from collect.steps import STEPS, plan
from collect.steps import s5_osm


def make_project(tmp_path, **over):
    b = tmp_path / "boundary.geojson"
    b.write_text(json.dumps({"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {},
                 "geometry": {"type": "Polygon", "coordinates": [[[7.40, 43.72], [7.44, 43.72], [7.44, 43.76], [7.40, 43.76], [7.40, 43.72]]]}}]}))
    data = {"project": {"name": "t", "folder": str(tmp_path)},
            "area": {"boundary_file": str(b), "bbox_harvest": [7.39, 43.71, 7.45, 43.77], "bbox_osm": [7.38, 43.70, 7.46, 43.78],
                     "metric_crs": "EPSG:32632", "timezone": "Europe/Monaco", "country_code": "MC"},
            "platforms": {"mapillary": {"enabled": True}, "kartaview": {"enabled": False}}}
    data = merge(data, over)
    p = tmp_path / "project.yaml"
    p.write_text(yaml.safe_dump(data))
    return Config.load(str(p))


def test_defaults_merge_and_derived(tmp_path):
    c = make_project(tmp_path)
    assert c.get("step1_harvest.match.max_dist_m") == 20.0          # from defaults
    assert c.utc_offsets_h == [1.0, 2.0]                             # daylight saving
    assert c.selection_mode == "whole_area"
    assert "footway" not in c.drivable and "residential" in c.drivable
    x, y = c.to_m(7.42, 43.74)
    lon, lat = c.to_ll(x, y)
    assert abs(lon - 7.42) < 1e-7 and abs(lat - 43.74) < 1e-7
    assert 14 < c.boundary_m.area / 1e6 < 15                         # about 3.2 km x 4.4 km


def test_bad_settings_are_refused(tmp_path):
    with pytest.raises(ConfigError):
        make_project(tmp_path, area={"timezone": "Mars/Olympus"})
    with pytest.raises(ConfigError):
        make_project(tmp_path, area={"bbox_harvest": [8, 44, 7, 43]})
    with pytest.raises(ConfigError):
        make_project(tmp_path, step2_regions={"mode": "everything"}).selection_mode


def test_old_british_setting_names_still_load():
    d = us_names({"road_standard": {"lane_width_m": {"kerb_lane": 3.6}, "kerb_clearance_m": 0.4,
                                    "line_width_mm": {"centre_line": 150}}, "licence": "x"})
    assert d == {"road_standard": {"lane_width_m": {"curb_lane": 3.6}, "curb_clearance_m": 0.4,
                                   "line_width_mm": {"center_line": 150}}, "license": "x"}


def test_plan_follows_mode_and_platforms(tmp_path):
    c = make_project(tmp_path)
    p = plan(c)
    assert "2w" in p and "2a" not in p and "1c" not in p and p[0] == "0a"
    c2 = make_project(tmp_path, step2_regions={"mode": "strata"}, platforms={"kartaview": {"enabled": True}})
    p2 = plan(c2)
    assert "2a" in p2 and "2w" not in p2 and "1c" in p2
    assert all(s in STEPS for s in p2)


def test_timezone_and_utm():
    assert timezone_for(103.82, 1.35, "SG") == "Asia/Singapore"
    assert timezone_for(7.42, 43.74, "MC") == "Europe/Monaco"
    assert timezone_for(-74.0, 40.7, "US") == "America/New_York"
    assert timezone_for(151.2, -33.9, "AU") == "Australia/Sydney"
    assert utm_epsg(7.42, 43.74) == "EPSG:32632" and utm_epsg(151.2, -33.9) == "EPSG:32756"


def test_width_rules_singapore_standard():
    rs = {"lane_width_m": {"expressway": 3.7, "curb_lane": 3.7, "inner_lane": 3.4, "one_lane_oneway": 5.5,
                           "undivided_lane": 3.7, "slip_two_lane": 7.4, "service_road": 6.1}}
    w = lambda *a: s5_osm._width(rs, *a)[0]  # noqa: E731
    assert w("motorway", True, 4) == pytest.approx(14.8)
    assert w("primary", True, 3) == pytest.approx(10.8)               # 2 x 3.7 + 3.4
    assert w("primary_link", True, 1) == 5.5
    assert w("trunk_link", True, 2) == 7.4
    assert w("residential", False, 2) == pytest.approx(7.4)
    assert w("service", False, 2) == 6.1
    assert w("primary", False, 4) == pytest.approx(14.8)              # 2 + 2 lanes, each direction 2 x 3.7


def test_ledger_and_fingerprint(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("x")
    fp1 = fingerprint([str(f)], {"k": 1})
    assert fp1 == fingerprint([str(f)], {"k": 1}) and fp1 != fingerprint([str(f)], {"k": 2})
    led = Ledger(str(tmp_path / "l.sqlite"))
    rid = led.start("5a", fp1, [str(f)])
    led.finish(rid, ["out"], {"n": 3})
    last = led.last("5a")
    assert last["status"] == "ok" and json.loads(last["summary"]) == {"n": 3}


def test_secret_is_masked():
    register_secret("MLY|secret-token-123")
    assert redact("GET ...?access_token=MLY|secret-token-123&x") == "GET ...?access_token=***&x"


def test_defaults_have_no_singapore_values():
    d = json.dumps(defaults())
    for s in ("Asia/Singapore", "3414", "LTA", "HDB", "SG"):
        assert s not in d, s


def test_init_from_boundary_uses_given_country(tmp_path):
    from collect.setup import init_project
    b = tmp_path / "b.geojson"
    b.write_text(json.dumps({"type": "Polygon", "coordinates": [[[115.85, -31.96], [115.87, -31.96], [115.87, -31.94],
                                                                [115.85, -31.94], [115.85, -31.96]]]}))   # Perth CBD
    _, c = init_project("perth", str(tmp_path / "p"), boundary=str(b), country="au")
    a = c["area"]
    assert a["country_code"] == "AU" and a["timezone"] == "Australia/Perth" and a["driving_side"] == "left"
    assert a["metric_crs"] == "EPSG:32750"
    _, c2 = init_project("perth2", str(tmp_path / "p2"), boundary=str(b))           # no country: nearest zone entry
    assert c2["area"]["country_code"] == "AU" and c2["area"]["timezone"] == "Australia/Perth"


def test_selected_area_mode(tmp_path):
    from collect.diagnose import preflight
    c = make_project(tmp_path, step2_regions={"mode": "selected_area"})
    assert c.selection_mode == "selected_area" and c.selected_area_ll is None
    assert "2w" in plan(c) and "2a" not in plan(c)
    assert any("no area is drawn" in b["what"] for b in preflight(c)["blocking"])
    f = tmp_path / "00_setup" / "selected_area.geojson"
    f.parent.mkdir(exist_ok=True)
    f.write_text(json.dumps({"type": "Polygon", "coordinates": [[[7.41, 43.73], [7.43, 43.73], [7.43, 43.74], [7.41, 43.74], [7.41, 43.73]]]}))
    c = make_project(tmp_path, step2_regions={"mode": "selected_area"})
    assert c.selected_area_ll.contains(__import__("shapely").Point(7.42, 43.735))
    assert not any("no area is drawn" in b["what"] for b in preflight(c)["blocking"])
    # Step 1 harvests only the drawn area: its box is the area's bounds plus 0.005 deg, not area.bbox_harvest
    assert c.harvest_bbox == [7.405, 43.725, 7.435, 43.745]
    assert c.harvest_bbox_osm == [7.39, 43.71, 7.45, 43.76]
    assert abs(c.harvest_m.area / 1e6 - 1.78) < 0.1                  # 0.02 x 0.01 deg at 43.7 N
    assert c.selected_area_file in STEPS["1a"].inputs(c)              # redrawing the area reruns Step 1


def test_platforms_are_independent(tmp_path):
    """Selected area: no cross-platform pairing (1i), and an unticked platform's steps are left out and not required."""
    from collect.steps import needed
    both = make_project(tmp_path, platforms={"kartaview": {"enabled": True}})
    assert "1i" in plan(both)                                       # whole area: duplicate drives are still paired
    sel = make_project(tmp_path, platforms={"kartaview": {"enabled": True}}, step2_regions={"mode": "selected_area"})
    assert "1i" not in plan(sel) and {"1a", "1c", "1h", "2w"} <= set(plan(sel))
    kv = make_project(tmp_path, platforms={"mapillary": {"enabled": False}, "kartaview": {"enabled": True}})
    assert not {"1a", "1b", "1i"} & set(plan(kv)) and "1c" in plan(kv)
    assert not needed(kv, "1b") and needed(kv, "1f")                # 1h does not wait for Mapillary's step
    assert c_is_whole(both)


def c_is_whole(c):
    return c.harvest_bbox == c.get("area.bbox_harvest") and c.harvest_ll.equals(c.boundary_ll)


def test_pano_check_with_no_panoramas(tmp_path):
    """Step 3f on a project without panoramas finishes with zero counts instead of failing."""
    import logging
    from collect.steps import s3_download
    c = make_project(tmp_path)
    outputs, summary = s3_download.pano_check(c, logging.getLogger("t"), sequences=["none"])
    assert summary == {"panoramas": 0, "broken": 0, "mis_stitched": 0}
    assert os.path.exists(outputs[0])


def test_views_with_no_panoramas(tmp_path):
    """Steps 3h and 3i on a selection without 360-degree images finish with zero counts."""
    import logging

    import pandas as pd
    from collect.steps import s3_views
    c = make_project(tmp_path)
    os.makedirs(os.path.dirname(c.paths.manifest), exist_ok=True)
    pd.DataFrame({"platform": ["kartaview"], "sequence_id": ["1"], "frame_id": ["9"], "order": [0], "status": ["ok"],
                  "is_pano": [False], "device": ["x"]}).to_parquet(c.paths.manifest, index=False)   # old manifest: no pano_status
    log = logging.getLogger("t")
    assert s3_views.travel_yaw(c, log)[1] == {"vehicle_sequences": 0}
    out, summary = s3_views.views(c, log, out_root=str(tmp_path / "views"))
    assert summary["panoramas"] == 0 and summary["views"] == 0


def test_ohsome_cut_off_reply_is_retried_and_split(monkeypatch):
    """A 200 reply with a cut-off body counts as a failure; a large area is split into quarters and the
    ways found in more than one quarter are kept once."""
    import json as _json
    import logging

    import requests as _rq
    from shapely.geometry import box, shape
    from collect.steps import s5_osm
    monkeypatch.setattr(s5_osm.time, "sleep", lambda s: None)
    calls = []

    class Resp:
        def __init__(self, text):
            self.status_code, self.text, self.content = 200, text, text.encode()

        def json(self):
            return _json.loads(self.text)

    def post(url, data, headers, timeout):
        g = shape(_json.loads(data["bpolys"])["features"][0]["geometry"])
        calls.append(g.area)
        if g.area > 0.3:                                 # too big: ohsome's stream breaks off
            return Resp('{"type":"FeatureCollection","features":[{"type":"Feature"\n  "error": "timeout"')
        feats = [{"type": "Feature", "properties": {"@osmId": "way/1", "@validFrom": "2020"}, "geometry": None},
                 {"type": "Feature", "properties": {"@osmId": f"way/{round(g.centroid.x * 10)}", "@validFrom": "2020"}, "geometry": None}]
        return Resp(_json.dumps({"type": "FeatureCollection", "features": feats}))

    fake = type("R", (), {"post": staticmethod(post), "RequestException": _rq.RequestException})
    feats, nbytes, parts = s5_osm._history_area(fake, "u", box(0, 0, 1, 1), "c", {}, {}, logging.getLogger("t"), 0)
    assert parts == 4 and calls[:2] == [1.0, 1.0]       # two failed tries on the whole area, then 4 quarters
    ids = [f["properties"]["@osmId"] for f in feats]
    assert ids.count("way/1") == 1 and len(ids) == len(set(ids))


def test_cut_off_web_reply_is_not_called_a_damaged_file():
    from collect.diagnose import explain
    e = explain("5c", "requests.exceptions.JSONDecodeError: Expecting ',' delimiter: line 507 column 4")
    assert e["title"] == "A web service sent an incomplete reply"
    assert explain("2w", "pyarrow.lib.ArrowInvalid: Parquet magic bytes not found")["title"] == "A saved file is damaged"
