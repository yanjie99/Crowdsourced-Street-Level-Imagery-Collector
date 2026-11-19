"""Command line: collect <command> -p project.yaml ...

  collect init <folder> --place "Monaco" create a project for a study area (boundary and derived settings)
  collect check -p project.yaml          validate the settings and show derived values
  collect status -p project.yaml         last run of every step
  collect ui [--workspace runs]          the local browser interface
  collect steps -p project.yaml          the steps this project runs (whole-area or strata selection)
  collect run 5a 5b -p project.yaml      run steps (skips steps whose inputs and settings did not change)
  collect run all -p project.yaml        run every step of the project in order
  collect token set | test | source      the Mapillary token, kept in Windows Credential Manager
"""
from __future__ import annotations

import getpass
import json

import typer

from .config import Config
from .ledger import Ledger
from .runner import run_step
from .steps import STEPS, plan

app = typer.Typer(add_completion=False, no_args_is_help=True, help="Crowdsourced Street-level Imagery Downloader (command: collect)")
token_app = typer.Typer(no_args_is_help=True, help="Mapillary token (Windows Credential Manager)")
app.add_typer(token_app, name="token")

P = typer.Option(..., "-p", "--project", help="project.yaml")


@app.command()
def check(project: str = P):
    """Validate project.yaml and print the values derived from it."""
    cfg = Config.load(project)
    b = cfg.boundary_ll
    typer.echo(f"project folder  {cfg.folder}")
    typer.echo(f"boundary        {cfg.get('area.boundary_file')} ({b.geom_type}, bounds {tuple(round(v, 3) for v in b.bounds)})")
    typer.echo(f"metric CRS      {cfg.get('area.metric_crs')}  area {cfg.boundary_m.area / 1e6:,.0f} km2")
    typer.echo(f"time zone       {cfg.get('area.timezone')}  UTC offsets {cfg.utc_offsets_h} h")
    typer.echo(f"drivable ways   {', '.join(cfg.drivable)}")
    typer.echo(f"platforms       " + ", ".join(k for k in ("mapillary", "kartaview") if cfg.get(f"platforms.{k}.enabled", False)))
    from .diagnose import preflight
    pf = preflight(cfg)
    for b in pf["blocking"]:
        typer.echo(f"BLOCKING  {b['what']} -> {b['fix']}")
    for w in pf["warnings"]:
        typer.echo(f"warning   {w['what']} -> {w['fix']}")
    if pf["ok"] and not pf["warnings"]:
        typer.echo("ready     nothing blocks a run")


@app.command()
def doctor(project: str = typer.Option("", "-p", "--project", help="project.yaml (optional: its settings and area)")):
    """Check that every web service the tool uses still answers (one light request each)."""
    from .doctor import checks
    cfg = Config.load(project) if project else None
    bad = 0
    for c in checks(cfg):
        mark = "ok  " if c["ok"] else ("skip" if c["ok"] is None else "FAIL")
        bad += c["ok"] is False
        ms = f"{c['ms']} ms" if c["ms"] is not None else ""
        typer.echo(f"{mark}  {c['service']:26} {c['host']:32} {ms:>8}  {c['detail']}")
        if c["ok"] is False:
            typer.echo(f"      -> if the service moved or has a new version, change {c['setting']} in the settings")
    raise typer.Exit(1 if bad else 0)


@app.command()
def init(folder: str = typer.Argument(..., help="new project folder"),
         place: str = typer.Option("", help="a place name to look up on OpenStreetMap (Nominatim), e.g. 'Monaco'"),
         boundary: str = typer.Option("", help="or a GeoJSON file with the study-area outline"),
         name: str = typer.Option("", help="project name (default: the folder name)"),
         platforms: str = typer.Option("mapillary,kartaview", help="platforms to collect from"),
         budget_gb: float = typer.Option(0, help="storage budget for images, GB")):
    """Create a project for a study area: boundary and derived area settings in <folder>/project.yaml."""
    import os
    from .setup import init_project
    if not place and not boundary:
        raise typer.BadParameter("give --place or --boundary")
    path, c = init_project(name or os.path.basename(os.path.abspath(folder)), folder, place or None, boundary or None,
                           tuple(p for p in platforms.split(",") if p), budget_gb or None)
    a = c["area"]
    typer.echo(f"wrote {path}")
    for k in ("country_code", "timezone", "driving_side", "metric_crs", "bbox_harvest"):
        typer.echo(f"  {k:14} {a[k]}")
    typer.echo("Check these, set the road standard and budget in project.yaml, then: collect run all -p " + path)


@app.command()
def ui(workspace: str = typer.Option("runs", help="folder that holds the projects (one sub-folder each)"),
       port: int = typer.Option(8765, help="port on 127.0.0.1"),
       browser: bool = typer.Option(True, help="open the page in the default browser")):
    """Start the local browser interface (http://127.0.0.1:<port>/)."""
    from .ui.server import serve
    serve(workspace, port, browser)


@app.command("steps")
def steps_cmd(project: str = P):
    """The steps this project runs, in order (depends on step2_regions.mode and the enabled platforms)."""
    cfg = Config.load(project)
    typer.echo(f"Step 2 mode: {cfg.selection_mode}")
    for n in plan(cfg):
        typer.echo(f"{n:4} {STEPS[n].title}")


@app.command()
def status(project: str = P):
    """Last run of every step."""
    cfg = Config.load(project)
    rows = {r["step"]: r for r in Ledger(cfg.paths.ledger).all_last()}
    for name, st in STEPS.items():
        r = rows.get(name)
        line = f"{name:4} {st.title:38} "
        line += "not run" if not r else f"{r['status']:7} {r['finished_utc'] or r['started_utc']}  {r['summary'] or ''}"
        typer.echo(line)


@app.command()
def run(steps: list[str], project: str = P, force: bool = typer.Option(False, help="rerun even if up to date"),
        regions: str = typer.Option("", help="comma-separated region squares to restrict steps 3b-3i, 5c and 5d to"),
        offline: bool = typer.Option(False, help="use only responses already cached on disk; never send a request"),
        sequences: str = typer.Option("", help="comma-separated sequence ids to restrict Step 3 to"),
        platform: str = typer.Option("", help="run only this platform's part of a step (used by the parallel lanes)"),
        sequential: bool = typer.Option(False, help="run Mapillary and KartaView one after the other, not in parallel")):
    """Run one or more steps in order (Mapillary and KartaView in parallel lanes where they are independent)."""
    from .runner import LANE_STEPS, parallel_ok, run_lanes
    cfg = Config.load(project)
    opts = {"regions": [r for r in regions.split(",") if r]} if regions else {}
    if offline:
        opts["offline"] = True
    if sequences:
        opts["sequences"] = [x for x in sequences.split(",") if x]
    if platform:
        opts["platform"] = platform
    if sequential:
        cfg.data.setdefault("platforms", {})["parallel"] = False
    if steps == ["all"]:
        steps = plan(cfg)
    for s in steps:
        if s not in STEPS:
            raise typer.BadParameter(f"unknown step {s}; known: {', '.join(STEPS)}")
    lane_of = {s: pf for pf, ss in LANE_STEPS.items() for s in ss}
    i = 0
    while i < len(steps):
        s = steps[i]
        if s in lane_of and parallel_ok(cfg, opts):
            # a run of Step 1 platform steps: Mapillary's and KartaView's go to two parallel lanes
            j = i
            while j < len(steps) and steps[j] in lane_of:
                j += 1
            groups = {pf: [x for x in steps[i:j] if lane_of[x] == pf] for pf in LANE_STEPS}
            if all(groups.values()):
                try:
                    run_lanes(cfg, groups, force=force, **opts)
                except Exception as e:  # noqa: BLE001
                    typer.echo(f"Step 1 lanes: error: {e}", err=True)
                    raise typer.Exit(1) from None
                i = j
                continue
        i += 1
        try:
            res = run_step(cfg, s, force=force, **opts)
        except Exception as e:  # noqa: BLE001  one clear explanation instead of a traceback
            from .diagnose import explain
            d = explain(s, f"{type(e).__name__}: {e}")
            nl = "\n  -> "
            typer.echo(f"{s}: error: {e}{nl}{d['title']}. {d['cause']}{nl}What to try: {d['fix']}", err=True)
            raise typer.Exit(1) from None
        typer.echo(f"{s}: {json.dumps(res, default=str)}")


@token_app.command("set")
def token_set():
    """Store the Mapillary token in Windows Credential Manager (typed, not echoed)."""
    from .secrets import store_token
    tok = getpass.getpass("Mapillary token: ").strip()
    if not tok:
        raise typer.Exit(1)
    store_token(tok)
    typer.echo("stored in Windows Credential Manager as collect:mapillary")


@token_app.command("source")
def token_source():
    """Where the token would be read from (never prints the token)."""
    from .secrets import token_source as ts
    typer.echo(ts())


@token_app.command("test")
def token_test(project: str = P):
    """One small Graph API request near the center of the area."""
    from .secrets import test_token
    cfg = Config.load(project)
    c = cfg.boundary_ll.representative_point()
    ok, msg = test_token([c.x - 0.005, c.y - 0.005, c.x + 0.005, c.y + 0.005])
    typer.echo(msg)
    raise typer.Exit(0 if ok else 1)


if __name__ == "__main__":
    app()
