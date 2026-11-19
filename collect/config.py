"""Project configuration: one project.yaml per study area.

Config wraps the parsed YAML, checks the settings every step needs, and derives the values the step
scripts used to hard-code (metric CRS transformers, UTC offsets of the time zone, the drivable-way list).
Paths of a project's inputs and outputs are given by Paths, one folder per step under project.folder.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import cached_property
from zoneinfo import ZoneInfo

import yaml

REQUIRED = [
    "project.folder", "area.boundary_file", "area.bbox_harvest", "area.bbox_osm", "area.metric_crs",
    "area.timezone", "area.country_code", "osm.camera_highways", "osm.not_for_vehicles",
    "osm.street_highways", "step5_osm.match.max_dist_m", "step5_osm.road_standard.lane_width_m",
]


class ConfigError(ValueError):
    pass


def defaults() -> dict:
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "defaults.yaml"), encoding="utf8") as f:
        return yaml.safe_load(f)


# Setting names from before the switch to US English (18 Nov 2026); old project files keep working.
RENAMED = {"kerb_lane": "curb_lane", "kerb_clearance_m": "curb_clearance_m", "centre_line": "center_line",
           "licence": "license"}


def us_names(d):
    if isinstance(d, dict):
        return {RENAMED.get(k, k): us_names(v) for k, v in d.items()}
    if isinstance(d, list):
        return [us_names(v) for v in d]
    return d


def service(cfg, key: str) -> str:
    """A web service address from `services` (project.yaml, else the defaults), without a trailing slash."""
    v = cfg.get(f"services.{key}", None) if cfg is not None else None
    return str(v or defaults()["services"][key]).rstrip("/")


def graph_url(cfg) -> str:
    """Mapillary's Graph API base address (services.mapillary_graph)."""
    return service(cfg, "mapillary_graph")


def user_agent(cfg=None) -> str:
    """How the tool names itself to every server (platforms.user_agent; add a contact there, as Nominatim asks)."""
    v = cfg.get("platforms.user_agent", None) if cfg is not None else None
    return str(v or defaults()["platforms"]["user_agent"])


def merge(base: dict, over: dict) -> dict:
    """Recursive merge: dicts are merged key by key, everything else in `over` replaces `base`."""
    out = dict(base)
    for k, v in (over or {}).items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


class Config:
    def __init__(self, data: dict, path: str | None = None):
        self.data = data
        self.path = path
        missing = [k for k in REQUIRED if self.get(k, None) is None]
        if missing:
            raise ConfigError(f"project.yaml is missing: {', '.join(missing)}")
        for k in ("area.bbox_harvest", "area.bbox_osm"):
            b = self.get(k)
            if len(b) != 4 or not (b[0] < b[2] and b[1] < b[3]):
                raise ConfigError(f"{k} must be [lon_min, lat_min, lon_max, lat_max]")
        try:
            ZoneInfo(self.get("area.timezone"))
        except Exception as e:  # noqa: BLE001
            raise ConfigError(f"area.timezone is not a known time zone: {e}") from e

    @classmethod
    def load(cls, path: str) -> "Config":
        """project.yaml on top of the package defaults (collect/defaults.yaml): a project sets only what differs."""
        with open(path, encoding="utf8") as f:
            data = yaml.safe_load(f) or {}
        return cls(merge(us_names(defaults()), us_names(data)), os.path.abspath(path))

    def get(self, key: str, default=KeyError):
        node = self.data
        for part in key.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            elif default is KeyError:
                raise KeyError(f"setting {key} not in project.yaml")
            else:
                return default
        return node

    # ---- derived values ----
    @property
    def folder(self) -> str:
        return self.get("project.folder")

    def resolve(self, p: str) -> str:
        """A path from project.yaml: absolute as given, else relative to the project folder."""
        return p if os.path.isabs(p) else os.path.join(self.folder, p)

    @property
    def selection_mode(self) -> str:
        """step2_regions.mode: "whole_area" (every capture in the boundary), "selected_area" (only captures inside an
        area drawn within it, step2_regions.selected_area) or "strata" (squares per stratum, tuning and evaluation
        halves: the descriptor-validation design)."""
        m = self.get("step2_regions.mode", "whole_area")
        if m not in ("whole_area", "selected_area", "strata"):
            raise ConfigError("step2_regions.mode must be whole_area, selected_area or strata")
        return m

    @property
    def selected_area_file(self) -> str:
        return self.resolve(self.get("step2_regions.selected_area", None) or "00_setup/selected_area.geojson")

    @property
    def selected_area_ll(self):
        """The drawn download area (lon/lat), or None when none is saved."""
        p = self.selected_area_file
        if not os.path.exists(p):
            return None
        from shapely.geometry import shape
        import shapely
        with open(p, encoding="utf8") as f:
            g = json.load(f)
        feats = g["features"] if g.get("type") == "FeatureCollection" else [g]
        return shapely.union_all([shape(x.get("geometry", x)) for x in feats])

    # ---- the area Step 1 harvests (19 Nov 2026): the drawn area in selected_area mode, else the project boundary
    @property
    def harvest_ll(self):
        if self.selection_mode == "selected_area":
            g = self.selected_area_ll
            if g is None:
                raise ConfigError("Selection is 'selected area' but no area is saved: draw it under Settings > "
                                  f"What to download (expected {self.selected_area_file})")
            return g
        return self.boundary_ll

    @property
    def harvest_m(self):
        from shapely.ops import transform
        return transform(self.to_m, self.harvest_ll) if self.selection_mode == "selected_area" else self.boundary_m

    def _harvest_box(self, key: str, pad: float) -> list[float]:
        if self.selection_mode != "selected_area":
            return self.get(key)
        x0, y0, x1, y1 = self.harvest_ll.bounds   # padded like `collect init` pads the boundary
        return [round(x0 - pad, 4), round(y0 - pad, 4), round(x1 + pad, 4), round(y1 + pad, 4)]

    @property
    def harvest_bbox(self) -> list[float]:
        """Box of the platform harvest (area.bbox_harvest, or the drawn area plus 0.005 deg)."""
        return self._harvest_box("area.bbox_harvest", 0.005)

    @property
    def harvest_bbox_osm(self) -> list[float]:
        """Box of the Step 1 OSM segments (area.bbox_osm, or the drawn area plus 0.02 deg)."""
        return self._harvest_box("area.bbox_osm", 0.02)

    @property
    def osm_extract(self) -> str:
        """The dated OSM extract: osm.extract when set, else the file step 0a downloaded (00_setup/osm/extract.json)."""
        if self.get("osm.extract", None):
            return self.resolve(self.get("osm.extract"))
        info = os.path.join(self.paths.step(0), "osm", "extract.json")
        if os.path.exists(info):
            with open(info, encoding="utf8") as f:
                return os.path.join(os.path.dirname(info), json.load(f)["file"])
        raise ConfigError("no OSM extract yet: run step 0a, or set osm.extract")

    @property
    def osm_extract_date(self):
        if self.get("osm.extract_date", None):
            return self.get("osm.extract_date")
        info = os.path.join(self.paths.step(0), "osm", "extract.json")
        return json.load(open(info, encoding="utf8"))["date"] if os.path.exists(info) else None

    @cached_property
    def drivable(self) -> list[str]:
        nfv = set(self.get("osm.not_for_vehicles"))
        return [h for h in self.get("osm.camera_highways") if h not in nfv]

    @cached_property
    def utc_offsets_h(self) -> list[float]:
        """UTC offsets (hours) the area's time zone uses over a year: one value, or two with daylight saving."""
        tz = ZoneInfo(self.get("area.timezone"))
        start = datetime(datetime.now().year, 1, 1)   # this year's rules (from the tzdata package; keep it updated)
        offs = {(start + timedelta(days=d)).replace(tzinfo=tz).utcoffset().total_seconds() / 3600 for d in range(0, 366, 7)}
        return sorted(offs)

    @cached_property
    def to_m(self):
        from pyproj import Transformer
        return Transformer.from_crs(4326, self.get("area.metric_crs"), always_xy=True).transform

    @cached_property
    def to_ll(self):
        from pyproj import Transformer
        return Transformer.from_crs(self.get("area.metric_crs"), 4326, always_xy=True).transform

    @cached_property
    def boundary_ll(self):
        from shapely.geometry import shape
        with open(self.resolve(self.get("area.boundary_file")), encoding="utf8") as f:
            g = json.load(f)
        feats = g["features"] if g.get("type") == "FeatureCollection" else [g]
        geoms = [shape(f["geometry"] if "geometry" in f else f) for f in feats]
        import shapely
        return shapely.union_all(geoms)

    @cached_property
    def boundary_m(self):
        from shapely.ops import transform
        return transform(self.to_m, self.boundary_ll)

    def cache_dir(self, step: int, rel: str) -> str:
        """Where a step keeps its raw responses: <step folder>/<rel>, unless project.yaml's `caches` maps rel
        (e.g. "mapillary/tiles") to another folder, such as the cache of an earlier run."""
        override = (self.get("caches", {}) or {}).get(rel)
        p = self.resolve(override) if override else os.path.join(self.paths.step(step), *rel.split("/"))
        os.makedirs(p, exist_ok=True)
        return p

    @cached_property
    def paths(self) -> "Paths":
        return Paths(self)


@dataclass
class Paths:
    """Standard layout of a project folder. Every step reads and writes only here."""
    cfg: Config

    def step(self, n: int) -> str:
        names = {0: "00_setup", 1: "01_metadata", 2: "02_regions", 3: "03_download",
                 5: "05_osm", 6: "06_checks"}
        reuse = (self.cfg.get("reuse", {}) or {}).get(n) or (self.cfg.get("reuse", {}) or {}).get(str(n))
        if reuse:           # read another project's finished step (e.g. its Step 1 harvest) instead of running it here
            return self.cfg.resolve(reuse)
        p = os.path.join(self.cfg.folder, names[n])
        os.makedirs(p, exist_ok=True)
        return p

    @property
    def manifest(self) -> str:
        return os.path.join(self.step(3), "manifest.parquet")

    @property
    def ledger(self) -> str:
        return os.path.join(self.cfg.folder, "ledger.sqlite")

    @property
    def logs(self) -> str:
        p = os.path.join(self.cfg.folder, "logs")
        os.makedirs(p, exist_ok=True)
        return p
