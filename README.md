# Crowdsourced Street-Level Imagery Collector

**Version 1.0.0** · Python 3.11+ · Command: `collect` · Package: `collect-street-imagery`

`collect` gathers street-level imagery from **Mapillary** and **KartaView**, with **OpenStreetMap** context, for any
study area in the world. You give it an area; it finds the captures there, downloads the images you choose, makes
perspective views from 360° panoramas, matches every capture to the road map as it was on the capture date, checks
the result and writes a report. It runs as a local browser app or from the command line, and every step can be
stopped and resumed.

It is the general form of the Singapore descriptor-validation collection (November 2026) and reproduces that
collection exactly (see the [user guide](docs/USER_GUIDE.md#11-verification)).

## Features

- **Any study area**: draw it on a map, upload a GeoJSON boundary, or pick an OpenStreetMap administrative area.
- **Two platforms at once**: Mapillary and KartaView run in parallel lanes, each at its own polite request rate.
- **Panoramas to views**: 360° panoramas become perspective views, and mis-stitched panoramas are flagged.
- **Map matching by date**: each capture is matched to the OSM road network as it was on the capture date (via ohsome).
- **Resumable**: a rerun continues where it stopped, and offline runs can reuse cached responses.
- **Checks and reports**: storage estimates, review screens, and a run report with counts, checks and credits.
- **Safe token handling**: the Mapillary token is kept in Windows Credential Manager (or the `MAPILLARY_TOKEN`
  environment variable), sent only in the `Authorization` header and masked in logs. It is never written to project files.

## Install

Download the wheel from the [latest release](../../releases/latest), then:

```bash
python -m venv collect-env
collect-env\Scripts\activate            # on macOS/Linux: source collect-env/bin/activate
pip install collect_street_imagery-1.0.0-py3-none-any.whl
collect --help
```

Or install from source:

```bash
git clone https://github.com/yanjie99/Crowdsourced-Street-Level-Imagery-Collector.git
cd Crowdsourced-Street-Level-Imagery-Collector
pip install -r requirements.txt -c constraints.txt   # exact tested versions (recommended)
pip install .
```

`constraints.txt` pins the library versions the tool was verified with, so a future library release cannot change
the results unnoticed.

## Quick start

```bash
collect token set                         # store your Mapillary token once
collect init my_area --place "Monaco"     # create a project for a study area
collect ui --workspace .                  # open the browser interface at http://127.0.0.1:8765/
```

From the command line instead:

```bash
collect check  -p my_area/project.yaml
collect run all -p my_area/project.yaml
collect status -p my_area/project.yaml
collect doctor                            # check that every web service answers
```

**Before your first run**, set your own contact details in `platforms.user_agent` (in `collect/defaults.yaml` or in
your `project.yaml`). Nominatim and other services ask every client to name a contact.

See the **[user guide](docs/USER_GUIDE.md)** for the workflow, every setting, the outputs and troubleshooting.
`examples/` holds two complete project files (`singapore.yaml`, `monaco.yaml`).

## Repository layout

| Path | Contents |
|---|---|
| `collect/` | the package: CLI (`cli.py`), configuration, networking, ledger and the pipeline steps (`steps/`) |
| `collect/ui/` | the local browser interface (server and single-page app; Leaflet included) |
| `docs/USER_GUIDE.md` | full user guide |
| `examples/` | example project files |
| `tests/unit/` | unit tests (`python -m pytest`) |
| `tests/verification/` | scripts that compare a rerun with the original Singapore collection |

## Development

```bash
pip install -e ".[test]"
python -m pytest
pip wheel . --no-deps -w dist             # build the wheel
```

Every web address the tool uses is a setting, not code (the `services` block and the platform API URLs in
`defaults.yaml`). If a service moves, change that one line, and run `collect doctor` to confirm.

## Terms of use

- **License**: [PolyForm Noncommercial License 1.0.0](LICENSE). You may use, copy and modify the tool for
  non-commercial purposes; commercial use needs the authors' permission.
- **Academic research only**: the tool is intended for non-commercial academic research.
- **No misuse of data**: do not use the tool or its data to identify, track or profile individuals, to de-blur or
  recover faces, licence plates or other personal information, or for surveillance. Obtain the ethics approval your
  institution requires.
- **Platform terms**: follow each platform's terms. Mapillary does not allow unapproved data mining, unblurring or
  re-identification; heavy KartaView use needs the platform's approval.

## Data credits

Credit the sources in anything you publish: images © Mapillary (CC BY-SA 4.0); images © Grab and KartaView
Contributors (CC BY-SA 4.0); map data © OpenStreetMap contributors (ODbL), with extracts by Geofabrik, history via
ohsome (HeiGIT) and place search by Nominatim. The bundled map library is [Leaflet](https://leafletjs.com/)
(BSD-2-Clause, see `collect/ui/static/leaflet/LICENSE`).

---

© 2026 Crowdsourced Street-Level Imagery (CSI) Capture and Processing, Department of Geography, National University
of Singapore.
