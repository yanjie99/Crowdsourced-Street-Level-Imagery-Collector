"""Step registry: name -> (function, inputs, settings it reads). Each function is fn(cfg, log, **opts) and returns
(outputs, summary). The runner records every run in the project ledger."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .. import setup
from . import s1_kartaview, s1_mapillary, s1_osm, s2_regions, s3_download, s3_views, s5_osm, s6_checks


@dataclass
class Step:
    name: str
    title: str
    fn: Callable
    inputs: Callable          # cfg -> list of input paths (for the fingerprint)
    settings: tuple[str, ...]  # project.yaml keys that change the result
    needs: tuple[str, ...] = ()
    modes: tuple[str, ...] = ("whole_area", "selected_area", "strata")   # Step 2 selection modes this step belongs to


def _p(cfg, *parts):
    import os
    return os.path.join(cfg.paths.step(5), *parts)


def _s1(cfg, *parts):
    import os
    return os.path.join(cfg.paths.step(1), *parts)


def _s2(cfg, *parts):
    import os
    return os.path.join(cfg.paths.step(2), *parts)


def _s3(cfg, *parts):
    import os
    return os.path.join(cfg.paths.step(3), *parts)


_SEL = lambda c: [_s2(c, "selection", "frames.parquet")]  # noqa: E731
_BND = lambda c: [c.resolve(c.get("area.boundary_file"))]  # noqa: E731
# Step 1 reads the drawn area in selected_area mode: its file is an input, so redrawing the area reruns Step 1
_HARV = lambda c: _BND(c) + ([c.selected_area_file] if c.selection_mode == "selected_area" else [])  # noqa: E731
_M = ("step2_regions.mode",)


STEPS: dict[str, Step] = {s.name: s for s in [
    Step("0a", "Dated OSM extract (Geofabrik)", setup.extract, lambda c: _BND(c), ("osm.extract",)),
    Step("1a", "Mapillary coverage tiles", s1_mapillary.tiles, _HARV,
         ("area.bbox_harvest", "platforms.mapillary.tile_zoom") + _M),
    Step("1b", "Mapillary sequences and devices", s1_mapillary.sequences,
         lambda c: [_s1(c, "mapillary", "captures.parquet")],
         ("step1_harvest.move_m", "step1_harvest.min_moving_captures", "area.timezone"), ("1a",)),
    Step("1c", "KartaView grid sweep", s1_kartaview.sweep, _HARV,
         ("area.bbox_harvest", "platforms.kartaview.sweep_spacing_m", "platforms.kartaview.sweep_radius_m") + _M),
    Step("1d", "KartaView sequence discovery", s1_kartaview.discover,
         lambda c: [_s1(c, "kartaview", "sequences_raw.parquet")] + _HARV(c),
         ("area.country_code", "platforms.kartaview.listing_buffer_deg") + _M, ("1c",)),
    Step("1e", "KartaView photo lists", s1_kartaview.photos,
         lambda c: [_s1(c, "kartaview", "sequences.parquet")] + _HARV(c), ("existing.annotated_photo_pages",) + _M, ("1d",)),
    Step("1f", "KartaView sequence table", s1_kartaview.table, lambda c: [_s1(c, "kartaview", "photos.parquet")],
         ("step1_harvest.move_m", "step1_harvest.min_moving_captures", "area.timezone"), ("1e",)),
    Step("1g", "OSM road segments", s1_osm.segments, lambda c: [c.osm_extract] + _HARV(c),
         ("osm.camera_highways", "step1_harvest.segment_max_len_m", "area.bbox_osm", "area.metric_crs") + _M, ("0a",)),
    Step("1h", "Captures on segments, passes", s1_osm.passes,
         lambda c: [_s1(c, "osm", "segments.parquet"), _s1(c, "mapillary", "sequences.parquet"),
                    _s1(c, "kartaview", "sequence_table.parquet")],
         ("step1_harvest.match", "step1_harvest.vehicle", "step1_harvest.min_year", "step1_harvest.camera_reset_dates",
          "osm.not_for_vehicles", "area.timezone"), ("1b", "1f", "1g")),
    Step("1i", "Duplicate drives, combined passes", s1_osm.duplicates,
         lambda c: [_s1(c, "osm", "mapillary_capture_segments.parquet"), _s1(c, "osm", "kartaview_capture_segments.parquet")],
         ("step1_harvest.duplicates", "area.timezone"), ("1h",),
         modes=("whole_area", "strata")),   # selected_area: each platform is downloaded on its own, no cross-platform pairing
    Step("2a", "OSM features for strata", s2_regions.features, lambda c: [c.osm_extract],
         ("area.bbox_osm", "area.metric_crs", "step2_regions.landuse_kinds"), ("0a",), modes=("strata",)),
    Step("2b", "Cells: coverage and features", s2_regions.cells,
         lambda c: [_s1(c, "osm", "combined_segment_passes.parquet"), _s2(c, "osm")] + _BND(c),
         ("step2_regions.cell_m", "step2_regions.min_passes", "step2_regions.min_captures_per_cell", "step2_regions.rules",
          "step2_regions.tall_building", "step2_regions.building_types", "osm.street_highways"), ("1i", "2a"), modes=("strata",)),
    Step("2c", "Local footprint layer", s2_regions.layers, lambda c: [_s2(c, "cells.parquet")],
         ("step2_regions.local_layers", "step2_regions.tall_building", "step2_regions.building_types"), ("2b",), modes=("strata",)),
    Step("2d", "Strata and ranking", s2_regions.strata, lambda c: [_s2(c, "cells.parquet"), _s2(c, "cells_hdb.parquet")],
         ("step2_regions.strata",), ("2c",), modes=("strata",)),
    Step("2e", "Draft selection", s2_regions.draft, lambda c: [_s2(c, "cells_strata.parquet")],
         ("step2_regions.selection", "step2_regions.strata", "storage"), ("2d",), modes=("strata",)),
    Step("2f", "Tuning and evaluation halves", s2_regions.split,
         lambda c: [s2_regions.accepted(c), _s2(c, "cell_sequences.parquet")], ("step2_regions.split",), ("2e",), modes=("strata",)),
    Step("2g", "Representativeness, camera clocks", s2_regions.represent, lambda c: [_s2(c, "sequence_halves.parquet")],
         ("step1_harvest.clock_check", "area.timezone"), ("2f",), modes=("strata",)),
    Step("2h", "Frame and sequence lists", s2_regions.lists,
         lambda c: [_s2(c, "sequence_halves.parquet"), _s2(c, "clock_suspects.csv")],
         ("step2_regions.clip_buffer_frames", "existing.annotated_window_manifest", "storage.mb_per_frame"), ("2g",), modes=("strata",)),
    Step("2w", "Captures to download (whole or selected area)", s2_regions.whole_area,
         lambda c: [_s1(c, "mapillary", "captures.parquet"), _s1(c, "kartaview", "photos.parquet"), c.selected_area_file],
         ("step2_regions.whole_area", "step2_regions.mode", "existing.annotated_window_manifest", "storage",
          "step1_harvest.clock_check"), modes=("whole_area", "selected_area")),
    Step("3a", "Storage estimate per image size", s3_download.estimate, _SEL,
         ("step3_download.image_sizes", "step3_download.extra_sizes", "step3_download.estimate_sample"), ("2h|2w",)),
    Step("3b", "Download images and records", s3_download.download, _SEL,
         ("step3_download.image_sizes", "step3_download.extra_sizes", "step3_download.extra_sample"), ("2h|2w",)),
    Step("3c", "Working copies", s3_download.working, lambda c: [_s3(c, "status")], ("step3_download.working_copy",), ("3b",)),
    Step("3d", "Panoramas", s3_download.panoramas, lambda c: [_s3(c, "status")], ("step3_download.panoramas",), ("3b",)),
    Step("3e", "Screen images and panoramas", s3_download.screen, lambda c: [_s3(c, "status")], (), ("3d",)),
    Step("3f", "Screen panoramas, mark mis-stitched", s3_download.pano_check, lambda c: [_s3(c, "status")],
         ("step3_download.crop",), ("3d",)),
    Step("3g", "Manifest and completeness checks", s3_download.manifest, lambda c: [_s3(c, "status")] + _SEL(c), (), ("3b",)),
    Step("3h", "Travel yaw of vehicle sequences", s3_views.travel_yaw, lambda c: [c.paths.manifest],
         ("step3_download.capture_class", "step3_download.travel_yaw"), ("3g",)),
    Step("3i", "Perspective views", s3_views.views, lambda c: [c.paths.manifest, _s3(c, "travel_yaw.csv")],
         ("step3_download.crop", "step3_download.capture_class"), ("3h",)),
    Step("5a", "OSM drivable ways and junctions", s5_osm.ways,
         lambda c: [c.osm_extract, c.resolve(c.get("area.boundary_file"))],
         ("area.bbox_osm", "area.metric_crs", "osm.camera_highways", "osm.not_for_vehicles"), ("0a",)),
    Step("5b", "Carriageway widths", s5_osm.widths, lambda c: [_p(c, "ways.parquet")],
         ("step5_osm.road_standard", "step5_osm.lanes_default_min_tagged"), ("5a",)),
    Step("5c", "OSM history (ohsome)", s5_osm.history, lambda c: [c.paths.manifest],
         ("step5_osm.history_margin_m", "step5_osm.history_simplify_m", "osm.history_api"), ("3g",)),
    Step("5d", "Captures on the map of their date", s5_osm.match,
         lambda c: [c.paths.manifest, _p(c, "history"), _p(c, "ways.parquet")],
         ("step5_osm.match",), ("5a", "5c")),
    Step("6a", "Checks, storage, fetch log, run report", s6_checks.checks, lambda c: [c.paths.manifest, c.paths.ledger],
         ("storage.budget_gb",), ("3g",)),
]}


# Steps whose result depends on which platforms are ticked: the ticks are part of their fingerprint, so ticking or
# unticking a platform reruns them instead of keeping the other platform's data.
PLATFORM_STEPS = {"1h", "2w", "3a", "3b", "3d", "3g", "6a"}
PLATFORM_KEYS = ("platforms.mapillary.enabled", "platforms.kartaview.enabled")


def plan(cfg) -> list[str]:
    """The steps of this project, in order, for its Step 2 mode and enabled platforms."""
    mly, kv = cfg.get("platforms.mapillary.enabled", False), cfg.get("platforms.kartaview.enabled", False)
    out = []
    for name, st in STEPS.items():
        if cfg.selection_mode not in st.modes:
            continue
        if name in ("1a", "1b") and not mly or name in ("1c", "1d", "1e", "1f") and not kv:
            continue                 # an unticked platform's steps are left out entirely
        if name == "1i" and not (mly and kv):
            continue                 # pairing drives across platforms needs both
        out.append(name)
    return out


def needed(cfg, dep: str) -> bool:
    """Whether a prerequisite applies to this project: a step left out of its plan (an unticked platform's step,
    another Step 2 mode's step) is not required."""
    return dep in plan(cfg)
