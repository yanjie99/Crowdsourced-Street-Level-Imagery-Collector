# Crowdsourced Street-level Imagery Downloader: user guide

The tool's command is `collect`.

Version 1.0.0, November 2026.

`collect` gathers street-level imagery from **Mapillary** and **KartaView**, with **OpenStreetMap** context, for a
study area anywhere in the world. You give it an area; it finds the captures there, downloads the images you choose,
makes perspective views from 360° panoramas, matches every capture to the road map as it was on the capture date,
checks the result and writes a report. It runs as a local browser app or from the command line, and every step can
be stopped and resumed.

It is the general form of the Singapore collection made for descriptor validation in November 2026, and it
reproduces that collection exactly (see [Verification](#11-verification)).

Contents

1. [Install](#1-install)
2. [Quick start](#2-quick-start)
3. [The steps](#3-the-steps)
4. [Choosing what to download](#4-choosing-what-to-download)
5. [Settings you must set yourself](#5-settings-you-must-set-yourself)
6. [Review screens](#6-review-screens)
7. [What a project folder holds](#7-what-a-project-folder-holds)
8. [Resuming, offline runs and reuse](#8-resuming-offline-runs-and-reuse)
9. [Polite use, terms and credits](#9-polite-use-terms-and-credits)
10. [Troubleshooting](#10-troubleshooting)
11. [Verification](#11-verification)
12. [Settings reference](#12-settings-reference)

---

## 1. Install

You need **Python 3.11 or newer** (tested on 3.12, Windows 11). The package installs its libraries from PyPI:
PyYAML, Typer, pandas, PyArrow, NumPy, Shapely 2, pyproj, pyosmium, Requests, OpenCV (headless), mapbox-vector-tile,
Pillow and tzdata.

```bash
python -m venv collect-env
collect-env\Scripts\activate
pip install collect_street_imagery-1.0.0-py3-none-any.whl
collect --help
```

To build the wheel from the source folder: `pip wheel . --no-deps -w dist`.

**For long-term use, install the exact versions the tool was tested with:**
`pip install -r requirements.txt -c constraints.txt` (or install the wheel, then `pip install -c constraints.txt
-r requirements.txt`). Library releases can change behavior (pandas 3 did, twice, in Nov 2026); `constraints.txt`
pins the tested set. After upgrading libraries on purpose, run `python -m pytest`.

The interface page ships its map library (Leaflet 1.9.4) with the tool; only the background map tiles come from
OpenStreetMap (`services.map_tiles`), so the map needs an internet connection.

**Keeping the tool working for years.** Every web address the tool uses is a setting, not code: the `services`
block (Mapillary Graph API, Nominatim, Geofabrik, map tiles), `platforms.mapillary.tiles_url`,
`platforms.kartaview.api` and `osm.history_api`. If a service moves or releases a new version, change that one line
(in `defaults.yaml`, or per project in `project.yaml`). Check them with **`collect doctor`** (or Settings → **Check
services**) before a large collection: one light request per service, and a failure names the setting to change.
Set `platforms.user_agent` to include a contact (a project page or e-mail), as Nominatim's usage policy asks.
Keep every project folder, with its `raw` caches and `00_setup/osm`: a finished collection can then be rerun
offline even if a platform changes or closes.

**Disk.** Plan for the images: about 0.3–0.5 MB a frame at the default sizes, 1–2 MB for full panoramas, plus
working copies and views. Step 3a estimates it for your selection before anything is downloaded. On an exFAT disk
(large clusters), many small files waste space; the views' validity masks are therefore packed into one zip per
sequence.

## 2. Quick start

### In the browser

```bash
collect ui --workspace D:\my_projects
```

The page opens at `http://127.0.0.1:8765/` on the **Introduction**: what the tool does, how to use it, and its terms
of use (academic research only; no misuse of the data). Then:

1. **Projects → New project.** Choose the study area on the map, in any of these ways:
   - **Search** a place: OpenStreetMap returns several matches (e.g. a district, a station, a street). Use a match's
     outline, or its box when OpenStreetMap has it only as a point; *Show* zooms to it first, so you can check that it
     is the place you mean (a search for one city can return another).
   - **Draw rectangle** (press and drag) or **Draw polygon** (click the corners; double-click or click the first
     corner to finish).
   - **Use map view**: the area currently on screen (zoom in first).
   - **Load GeoJSON**: a boundary file.
   The page shows the chosen area and its size, and warns when it is large. Give a folder name, and under **Save in**
   keep the workspace or press **Browse…** to pick another folder (for example on a larger disk): the browser lists
   your drives and folders; go into folders, **↑ Up**, switch drive, **Create folder here** if needed, then **Use this
   folder** (typing a full path into the box works too); a project
   saved elsewhere is still listed on the Projects page (the workspace's `projects_elsewhere.json` remembers where).
   Press Create. The
   app derives the bounding boxes, a metric map projection (the UTM zone), the time zone, the driving side and the
   country code (from the chosen place, or, for a drawn area or a file, from the nearest entry of the time-zone
   table: check them on the Overview page).
2. **Overview.** Check the derived values and the boundary on the map. Warnings tell you what is still missing.
3. **Settings.** Store the Mapillary token, set the storage budget, choose what to download (section 4), and set the
   road standard (section 5). Edit, then press **Save changes**; nothing is written before.
4. **Run → Run all.** The steps run one after another as a separate process; you can close the page, or even stop
   and restart `collect ui`, and come back. The Run page shows:
   - **Before you run**: anything that blocks a run (no platform enabled, no Mapillary token, a nearly full disk) and
     warnings (placeholder road standard, KartaView not approved, a large area, a storage estimate over the budget).
   - **Progress**: the run as a whole (step k of N) and the current step: items done of the total, the rate, the time
     left and the time so far. A step that cannot count shows a moving bar. If a step reports nothing for 10 minutes,
     a warning says it may be waiting on a slow or unreachable server.
   - **What went wrong**, when a step fails: the problem in plain words, the likely cause and what to try, with a
     button to run it again and the technical details folded away. A step whose process ended without finishing
     (stopped, the computer slept) shows as *interrupted*; Run all resumes it.
   - The log, live.
5. **Review** at the gates (section 6), then **Run all** again: finished steps are skipped.
6. **Report** shows the storage estimate and the final run report.

### From the command line

```bash
collect init D:\my_projects\monaco --place "Monaco" --budget-gb 5
collect token set
collect check -p D:\my_projects\monaco\project.yaml
collect steps -p D:\my_projects\monaco\project.yaml
collect run all -p D:\my_projects\monaco\project.yaml
collect status -p D:\my_projects\monaco\project.yaml
```

`collect run 3a -p …` runs one step; `collect run 3b 3c -p …` several. Options: `--force` reruns a step whose inputs
and settings did not change, `--offline` uses only what is on disk (section 8). `collect check` also lists what
blocks a run and the warnings. When a step fails, the command prints the problem, its likely cause and what to try;
the explanation is also saved in `logs/last_error.json`, and the current step's progress in `logs/progress.json`.

## 3. The steps

Each step reads the outputs of earlier steps and the settings in `project.yaml`, writes into its own folder, and is
recorded in the project's run ledger. A step whose inputs and settings have not changed is skipped. `collect steps`
lists the steps of a project, which depend on the selection mode and the enabled platforms.

| Step | What it does | Network |
|---|---|---|
| **0a** | Finds the smallest Geofabrik region that contains the area and downloads its newest date-stamped OSM extract (MD5 checked). | Geofabrik |
| **1a** | Mapillary coverage tiles over the area: every public capture's position, time, sequence and contributor. | Mapillary (token) |
| **1b** | One row per Mapillary sequence; the camera of its first and last image. | Mapillary (token) |
| **1c** | KartaView: a grid of radius queries over the area, then each sequence's record. | KartaView |
| **1d** | KartaView sequences the grid missed: the country listing, each contributor's listing, sequences without a bounding box located by their first photo. | KartaView |
| **1e** | Every photo of every KartaView sequence. | KartaView |
| **1f** | One row per KartaView sequence over its photos inside the area. | – |
| **1g** | OSM road segments: ways a camera can travel, split at junctions, at most 100 m long. | – |
| **1h** | Each capture matched to a segment (nearest within 20 m, with a penalty when the travel direction is more than 45° off); passes per segment. | – |
| **1i** | Drives uploaded to both platforms (or twice to KartaView); combined passes counting each drive once. | – |
| **2w** | *Whole-area and selected-area modes.* Every capture inside the boundary (or inside the drawn download area, each ticked platform on its own), with an optional date range; the whole area can keep one copy per duplicated drive. Flags cameras whose clock is off by the UTC offset. | – |
| **2a–2h** | *Strata mode.* OSM features, square cells, an optional local building layer, strata, a draft selection of squares, tuning and evaluation halves, representativeness, frame lists. | – |
| **3a** | Storage estimate: a small sample fetched at each candidate image size, against the budget. | both platforms |
| **3b** | Images (size fallback per platform), Mapillary records and reconstruction files, KartaView records and raw sensor logs. | both platforms |
| **3c** | 1024 px working copies (EXIF orientation, Lanczos, JPEG 90). | – |
| **3d** | The full panorama of every 360° frame. | both platforms |
| **3e** | Black share, flat rows and mean of every image and panorama (faults platforms serve as ordinary files). | – |
| **3f** | Panorama screening; mis-stitched panoramas marked (review gate). | – |
| **3g** | The manifest: one row per selected frame with status, size, flags and panorama status; completeness checks. | – |
| **3h** | Travel yaw of vehicle sequences (direction of travel measured from the panorama center). | – |
| **3i** | Four perspective views per panorama (90° wide, 5° down, 1024 × 576); Mapillary panoramas leveled frame by frame; vehicle sequences turned so that "front" faces travel. | – |
| **5a** | Drivable OSM ways with their tags, and junction nodes. | – |
| **5b** | The carriageway width of every drivable way, from its lanes and the road standard. | – |
| **5c** | The edit history of the drivable ways around the captures (ohsome API). | ohsome |
| **5d** | Each capture matched to the map as it was on its capture date, and to today's map. | – |
| **6a** | Checks of the whole project, storage against the budget, the fetch logs, and the run report. | – |

## 4. Choosing what to download

**Selection mode** (`step2_regions.mode`):

- `whole_area` (default): every capture inside the boundary. Narrow it with `whole_area.date_from` / `date_to`
  (local dates, inclusive) and `whole_area.platforms`. `drop_duplicate_drives` keeps one copy of a drive uploaded
  twice.
- `selected_area`: every capture inside an area you draw within the boundary (Settings → **Download area**: draw a
  rectangle or polygon, or press **Same as the project's area** to use the whole boundary, then press **Save area**). It is saved as `00_setup/selected_area.geojson`
  (`step2_regions.selected_area`). In this mode:
  - **Step 1 reads only the drawn area** (the platform harvest, the KartaView listings and the OSM segments), so a
    small area in a large city stays quick. Redrawing the area makes Step 1 run again for the new area.
  - **Each ticked platform is handled on its own.** Every Mapillary capture and every KartaView photo inside the area
    is downloaded; no drive is dropped because it is also on the other platform (step 1i, which pairs drives across
    platforms, does not run, and `drop_duplicate_drives` is ignored).
  - **An unticked platform is skipped entirely**: its Step 1 steps are left out of the plan, no later step waits for
    them, and only the ticked platform's captures are listed and downloaded.
  - The date range is an optional filter: leave both dates empty for every date. If no capture falls in the range,
    step 2w says so and lists the months that do have captures.

  Ticking or unticking a platform reruns the steps that combine platforms (1h, 2w, 3a, 3b, 3d, 3g, 6a), so the other
  platform's data never stays in the selection.
- `strata`: the descriptor-validation design. The area is cut into squares (1 km by default), squares are described
  by OSM and coverage features, tagged with strata by rules you write as expressions (e.g.
  `"tall_non_res >= 5 and hdb_blocks < 20"`), a few squares per stratum are picked, and they are split into a tuning
  and an evaluation half so that no drive is in both. Three plan rules decide which squares are usable (both
  platforms have their own passes, enough street length has several passes, no contributor dominates); in a sparsely
  covered area you may need to relax them in `step2_regions.rules`, and should record why.

**Image sizes** (`step3_download.image_sizes`): a fallback order per platform; the first size that is available is
kept, and each frame records which one it got.

| Platform | Sizes |
|---|---|
| Mapillary | `thumb_256`, `thumb_1024`, `thumb_2048`, `original` |
| KartaView | `th`, `lth` (1280 px), `proc`, `original` |
| KartaView panoramas | `wrapped_lth`, `wrapped_proc` |
| Mapillary panoramas | `original` |

Example: `kartaview: [original, proc, lth]` takes originals where they exist and falls back otherwise (many older
KartaView originals are no longer served). `extra_sizes` plus `extra_sample` (0–1) also saves other sizes for a
seeded share of frames, as `<frame>_<size>.jpg`, for instance originals for an audit sample.

**Storage.** Step 3a fetches `estimate_sample` frames (20 by default) per platform at every candidate size and
reports MB a frame and the estimated total against `storage.budget_gb`. Step 2w and 2e also estimate from
`storage.mb_per_frame`. Step 6a measures what is actually on disk.

**Panoramas and views.** `step3_download.panoramas.download` turns the panorama download and the views on or off.
The crop scheme (`step3_download.crop`: view directions, field of view, pitch, size, validity thresholds) is the
one set for Singapore; change it only with a reason. A panorama size that gives fewer than about 400 px of detail per
view makes poor views.

## 5. Settings you must set yourself

`collect init` derives the area settings; check them on the Overview page. Four things cannot be derived:

1. **The Mapillary token.** Create one at Mapillary's developer dashboard, then `collect token set` (or Settings →
   Mapillary token). It is kept in Windows Credential Manager under `collect:mapillary`, read only when a request
   needs it, sent only in the request header (and, for coverage tiles, in the tile URL as Mapillary requires), and
   masked in every log. On other systems, or if you prefer, set the environment variable `MAPILLARY_TOKEN`.
   A stored token takes precedence; when none is stored, the tool uses `MAPILLARY_TOKEN` if this computer has it. The
   Settings page says which one is in use, and the Run page's "Before you run" panel warns when the token comes from
   the environment variable rather than from Settings (it is easy to have one set and forget it).
2. **The road standard** (`step5_osm.road_standard`): lane widths per road type, shoulders, curb clearance and line
   widths, from the national road-design and road-marking standards. A new project starts from generic placeholder
   values, marked `GENERIC PLACEHOLDER` in `source`, and the Overview page warns until you change them. For
   Singapore they come from LTA's Civil Design Criteria (2010), Code of Practice on Street Work Proposals (2019) and
   Standard Details of Road Elements, chapter 8 (2026); see `examples/singapore.yaml`.
3. **The storage budget** (`storage.budget_gb`).
4. **KartaView bulk use.** Tick "KartaView bulk use approved" (`platforms.kartaview.bulk_access_approved`) only if the
   platform has approved it for you. Without it, contributor listings stop at 50 pages and requests stay light.

Also worth checking: `step1_harvest.camera_reset_dates` (dates cameras write after a reset; captures on them are left
out of pass counts), `step1_harvest.min_year`, and the capture-class rules (section 6).

## 6. Review screens

Three decisions need a person. The Review page shows each with the evidence; the command-line equivalent is the file
named below.

| Gate | When | What you decide | Written to |
|---|---|---|---|
| **Region selection** (strata mode) | after 2e | Which squares to use, per stratum. Click squares on the map. | `02_regions/selection_accepted.csv`; then run from 2f |
| **Mis-stitched panoramas** | after 3f | Panoramas whose picture band is far from the sequence's usual edge are flagged. Untick those that look fine. | `03_download/mis_stitched_accepted.csv`; then rerun 3f and 3g onwards |
| **Capture class** | after 3g | For each panorama sequence: vehicle, two-wheeler or on foot, from sample panoramas (bonnet, rider, person at the bottom). | `step3_download.capture_class.overrides`; then rerun 3h and 3i (delete that sequence's views first) |

Without a decision, strata mode uses the draft selection, every flagged panorama is marked, and the capture-class
rules apply (`step3_download.capture_class.rules`, tried in order: by default KartaView is vehicle, Mapillary GoPro
cameras two-wheeler, other Mapillary on foot).

## 7. What a project folder holds

```
project.yaml            the project's settings (only what differs from the defaults)
ledger.sqlite           every step run: status, times, inputs, outputs, counts
logs/<step>.log         each step's log; logs/ui_job.log for runs started from the page
00_setup/               boundary.geojson; osm/<region>-YYMMDD.osm.pbf and extract.json
01_metadata/            mapillary/, kartaview/ (tables and raw responses), osm/ (segments, matches, passes)
02_regions/             selection/frames.parquet and sequences.csv (what to download); strata tables in strata mode
03_download/
  data/<platform>/<sequence>/<frame>.jpg      images, with records/ (raw API responses)
  data_pano/…                                  panoramas
  data_1024/…                                  working copies
  data_views/<platform>/<sequence>/            views (JPEG), masks.zip, views.parquet, panos.parquet
  status/<platform>/<sequence>_frames.csv      download status of every frame (and _pano.csv)
  manifest.parquet                             one row per selected frame
05_osm/                 ways, junctions, way_widths, history/, capture_map_match.parquet
06_checks/              checks.json, checks_problems.csv, report.md
```

Key tables:

- **`manifest.parquet`**: platform, sequence, frame, file, status (`ok`, `black_at_source`, `failed`, `no_url`, …),
  the size kept, bytes, pixel size, selection reason, clock-suspect flag, panorama status, quality flag.
- **`views.parquet`**: one row per view: direction, yaw from the panorama center and from travel, valid share,
  usable, leveled, camera tilt, intrinsics, file and mask. Read a mask with `collect.pano.read_mask(views_dir, row)`.
- **`capture_map_match.parquet`**: per capture, the way it matches on its capture date and today, distances, angles,
  and whether the way changed or disappeared since.

## 8. Resuming, offline runs and reuse

- **Resume.** Stop a run at any time (Ctrl+C, or Stop on the page). Raw responses are cached before use and finished
  sequences are skipped, so rerunning continues where it stopped.
- **Offline** (`--offline`, or the checkbox on the Run page): nothing is requested; anything not on disk is reported
  as missing. Useful to rebuild tables from caches, or to prove a run is reproducible.
- **Reuse another project's step** (`reuse: {1: D:\other_project\01_metadata}`): the step's folder is read from the
  other project and not run here. Handy for trying a second selection on the same harvest.
- **Point at existing data** (`caches`): map `images`, `panoramas`, `working`, `views`, `mapillary/tiles`,
  `mapillary/graph_devices` or `kartaview/raw` to another folder. Status files and tables are still written in the
  project.

## 9. Polite use, terms and credits

- Requests are throttled: KartaView 2 a second to start (see below), Mapillary Graph 3 a second and files 8 a second,
  Nominatim and Geofabrik 1 a second, ohsome one request at a time with pauses. Retries back off on errors.
- **KartaView's rate adapts** (`platforms.kartaview.adaptive`). It starts at `per_s` (2 a second). After every 60
  quick, error-free replies it rises by 0.5, up to `max_per_s` (5) — but only when **bulk use is approved** for the
  project; without approval 2 a second is also the ceiling. It falls at once when the server shows strain: halved
  on a "too many requests" (429), a server error (5xx) or a timeout, down to 0.5 a second if the server keeps
  struggling; one step down (not below 2) when replies take more than twice the usual time; and it waits as long as
  a `Retry-After` header asks. Every change is written to the step's log, and the Run page shows the current rate
  (e.g. "KartaView 3.5 req/s"). KartaView publishes no rate limit, so the cap of 5 is this tool's own polite choice.
- **Mapillary and KartaView run at the same time** (`platforms.parallel`, on by default when both are ticked): Step
  1's platform steps (1a–1b and 1c–1f) and steps 3b and 3d run as two parallel lanes, one per platform. Each
  platform keeps its own rate, so neither receives more requests than before; a run just takes about as long as
  the slower platform instead of both added together. The Run page shows a bar per lane; each lane logs to
  `logs/<step>.<platform>.log` (Step 3) and `logs/progress_<platform>.json`. If one lane fails, the other finishes
  its work and the step reports which lane failed. Untick `parallel` (or `collect run --sequential`) to run them
  one after the other.
- Follow each platform's terms. Mapillary: register an app for the token, and no unapproved data mining, unblurring
  or re-identification. KartaView: heavy use needs the platform's approval (section 5).
- Credit the sources in anything you publish (the run report lists them):
  images © Mapillary, CC BY-SA 4.0; images © Grab and KartaView Contributors, CC BY-SA 4.0; map data © OpenStreetMap
  contributors, ODbL (Geofabrik extract; history via ohsome).

## 10. Troubleshooting

The Run page and the command line explain most failures by themselves (problem, cause, what to try). The rules they
use are in `collect/diagnose.py`; the table below lists the main ones.

| Message or symptom | Cause and fix |
|---|---|
| A service fails in `collect doctor` | It moved, has a new version, or is down. If its address changed, set the new one in the setting the message names; otherwise try again later. |
| KartaView images fail with 404 / unknown host | Step 3b fetches the sequence's photo list again and retries with the links KartaView gives now (noted "retried with a refreshed link"; fresh pages kept in `records/photos_refreshed_*`). If the API still gives the same link, the file is gone from KartaView's own storage (seen for some 2016 sequences): the note says so, and no retry can help. |
| A step shows *interrupted* | Its process ended without finishing (stopped, closed, the computer slept). Run all resumes it. |
| "No progress for N minutes" | The step waits on a slow or unreachable server (it retries with pauses), or works on a very large item. Check the log; stop and run again if nothing changes. |
| "Mapillary refused the token" | The token is wrong or expired: store a new one under Settings. |
| "No connection to the server" | Check the internet connection, VPN or proxy, then run again: the step resumes. |
| "The platform is limiting requests" (HTTP 429) | Wait and run again; lower the platform's rate in Settings if it repeats. |
| "A file is open in another program" | Close the program holding the file (often Excel), then run again. |
| "A saved file is damaged" | Rerun the earlier step with *force rerun*. |
| "A web service sent an incomplete reply" (often step 5c) | The service stopped part-way, usually its time limit. Step 5c retries and, when an area keeps failing, splits it into quarters by itself; just run the step again. If it repeats, the service may be busy: try later. |
| `step 3b needs step 2h or 2w to have run successfully first` | Run the earlier step (or `collect run all`). |
| `no OSM extract yet: run step 0a, or set osm.extract` | Run step 0a, or point `osm.extract` at a file you have. |
| `no Geofabrik region contains the whole boundary` | The area crosses region borders: download a larger extract yourself and set `osm.extract`. |
| `No Mapillary token` | `collect token set`, or Settings → Mapillary token. |
| `no square of the N meets all three plan rules` | Strata mode in a sparsely covered area. Relax `step2_regions.rules` or `min_passes` (and record why), or use `whole_area`. |
| `no split of the selected squares meets the balance rules` | Review the selection, or widen `step2_regions.split.balance`. |
| `OVER the … GB budget` | Narrow the dates or platforms, choose smaller image sizes, or raise the budget. |
| The boundary includes sea | OSM country boundaries include territorial waters. Harmless (a few empty requests), or draw a land-only area. |
| A place name is not found, or the wrong place is found | Add the city or country to the search, pick another match, or draw the area on the map. |
| Many captures unmatched in step 5d | Only drivable roads are candidates, so captures on footpaths stay unmatched; their distance shows it. |
| `step X belongs to Step 2 mode strata` | The step is not part of this project's mode; `collect steps` lists the right ones. |
| A setting is not saved on the page | It is saved only if the whole project still validates; the message says why. The previous file is kept as `project.yaml.bak`. |

## 11. Verification

The package was checked against the Singapore collection (D:\CaseStudy_SG, November 2026), rerun offline from the
saved responses: every Step 1 table (15), every Step 2 table (17, including the 436,966 frames to download), Step 3
statuses, screening, manifest, travel yaw and views (JPEGs and masks byte-identical) on 20 sequences, and Step 5
(ways, junctions, widths, history, date matches). The only difference is a fixed tie-break: a capture exactly as far
from two ways now goes to the way closer to its travel direction (3 of 4,573 matches). The comparison scripts are in
`tests/verification/`; they need the Singapore data.

A second area, Monaco (time zone with daylight saving, right-hand traffic, UTM 32N), was run end to end from
`collect init` with only derived settings and a date range.

`python -m pytest` runs the unit tests, which need neither the network nor the Singapore data.

## 12. Settings reference

`collect/defaults.yaml` holds every setting with its default; a project's `project.yaml` sets only what differs. The
groups:

| Group | Holds |
|---|---|
| `project` | `name`, `folder` |
| `area` | `boundary_file`, `bbox_harvest`, `bbox_osm`, `metric_crs`, `timezone`, `driving_side`, `country_code` (all derived by `collect init`) |
| `platforms` | per platform: `enabled`, rates (`per_s`, `graph_per_s`, `files_per_s`), `workers`, `license`; KartaView `bulk_access_approved`, `adaptive` (`max_per_s`, `floor_per_s`, `min_per_s`, `step`, `every`), sweep spacing and radius; `parallel` (platform lanes) |
| `osm` | `extract` (optional), the highway kinds a camera can travel, those not for vehicles, the street kinds used for coverage |
| `step1_harvest` | `min_year`, `camera_reset_dates`, moving-capture and short-sequence thresholds, the matching rule, vehicle detection, duplicate-drive rule, camera clock check |
| `step2_regions` | `mode`, `whole_area` options; for strata: cell size, minimum passes, clip buffer, local layers, building types, strata rules, selection and split rules |
| `storage` | `budget_gb`, `mb_per_frame` |
| `step3_download` | `image_sizes`, `extra_sizes`, `extra_sample`, `estimate_sample`, `workers`, `working_copy`, `panoramas`, `crop`, `capture_class`, `travel_yaw` |
| `step5_osm` | history margin and pause, the matching rule, `road_standard` (lane widths, shoulders, curb clearance, line widths and patterns, source) |
| `existing` | optional data to reuse (annotated photo pages, an annotated-window manifest and folder) |
| `caches`, `reuse` | other folders to read raw responses, data or whole steps from (section 8) |

Two complete examples: `examples/singapore.yaml` (strata mode, every setting written out) and `examples/monaco.yaml`
(a project as `collect init` makes it, plus a date range).

---

© 2026 Crowdsourced Street-Level Imagery (CSI) Capture and Processing, Department of Geography, National University of Singapore. For non-commercial academic research only; see the Introduction page for the terms of use.
