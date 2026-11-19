"""Phase 1 check: the ported Step 2 reproduces the Singapore Step 2 tables (step2_regions, 13 Nov 2026).
Usage: python -I compare_step2.py <original step2_regions dir> <check project 02_regions dir>
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_step1 import compare, read  # noqa: E402


def main():
    o, n = sys.argv[1:3]
    ok = True
    T = [("osm/way_attrs.parquet", ["way_id"], ()), ("osm/buildings.parquet", ["osm_id", "from_way"], ()),
         ("osm/landuse.parquet", ["osm_id", "from_way"], ()), ("osm/trees.parquet", ["osm_id"], ()),
         ("cells.parquet", ["cell"], ()), ("cell_sequences.parquet", ["cell", "sequence_id", "platform"], ()),
         ("cells_hdb.parquet", ["cell"], ()), ("hdb_blocks.parquet", ["postal", "x", "y"], ()),
         ("cells_strata.parquet", ["cell"], ()), ("strata_summary.csv", ["stratum"], ()),
         ("draft_selection.csv", ["cell"], ()), ("region_halves.csv", ["cell"], ()),
         ("sequence_halves.parquet", ["cell", "sequence_id", "platform"], ()),
         ("clock_suspects.csv", ["platform", "contributor", "device"], ("captures_selected",)),
         ("representativeness_summary.csv", ["dimension", "platform", "unit"], ()),
         ("selection/frames.parquet", ["platform", "frame_id"], ()),
         ("selection/sequences.csv", ["platform", "sequence_id"], ())]
    for f, key, skip in T:
        pb = os.path.join(n, f)
        if not os.path.exists(pb):
            print(f"FAIL {f}: not produced")
            ok = False
            continue
        ok &= compare(f, read(os.path.join(o, f)), read(pb), key, skip)
    print("ALL SAME" if ok else "DIFFERENCES FOUND")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
