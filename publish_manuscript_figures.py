"""Copy manuscript figures and tables into one flat folder per experiment run.

Each run folder gets only the publication figures and the table, named with the run as prefix
(`30-1` -> `30_1_fraction_dd.pdf`, `30_1_alpha_side_by_side.pdf`, ...). Reports, manifests and
plotted data stay in the export folders under `outputs/`, which are the source here.

The table is the unfiltered authentic agreement table: every clean PRN drawn in the figures must
have a row, so the error-cutoff (filtered) table is not published.

    uv run python publish_manuscript_figures.py outputs/aoa_figure_exports/sp-partial-28-09/120s \
        --output-root ~/Desktop/AOA_Figures_sp-partial-28-09
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

from aoa_figure_report import AGREEMENT_TABLE_STEM, FIGURE_STEMS

NAMES = dict(zip(FIGURE_STEMS, ("fraction_dd", "alpha_ephemeris", "alpha_proposed_best_m", "alpha_side_by_side")))
NAMES[f"{AGREEMENT_TABLE_STEM}_unfiltered"] = "authentic_agreement"
EXTENSIONS = ("png", "pdf", "svg", "eps", "csv", "tex")


FIGURE_DATA = ("fractional_double_difference_cycles", "alpha_i_ephemeris_deg", "alpha_i_proposed_best_m_deg")


def check_table_covers_figures(export_dir: Path) -> None:
    """Raise if a clean PRN drawn in any figure has no row in the agreement table."""
    spoof = set(json.loads((export_dir / "report_manifest.json").read_text())["declared_spoof_svids"])
    drawn = set()
    for name in FIGURE_DATA:
        frame = pd.read_csv(export_dir / "data" / f"{name}.csv", index_col="epoch_index")
        drawn |= {int(prn) for prn in frame.columns[frame.notna().any()]}
    table = pd.read_csv(export_dir / f"{AGREEMENT_TABLE_STEM}_unfiltered.csv")
    rows = {int(prn) for prn in table["target_prn"] if prn != "Overall"}
    missing = sorted(drawn - spoof - rows)
    if missing:
        raise ValueError(f"{export_dir}: clean PRNs {missing} are drawn in the figures but missing from the table.")


def publish(export_dir: Path, destination: Path) -> list[Path]:
    check_table_covers_figures(export_dir)
    prefix = export_dir.name.replace("-", "_")
    sources = [(export_dir / f"{stem}.{extension}", destination / f"{prefix}_{name}.{extension}")
               for stem, name in NAMES.items() for extension in EXTENSIONS]
    sources = [(source, target) for source, target in sources if source.is_file()]
    if not all(any(source.stem == stem for source, _ in sources) for stem in FIGURE_STEMS):
        raise FileNotFoundError(f"{export_dir} is missing publication figures.")
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    for source, target in sources:
        shutil.copy2(source, target)
    return [target for _, target in sources]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("export_root", type=Path, help="folder with one export folder per run (e.g. 30-1/)")
    parser.add_argument("--runs", nargs="+", help="default: every export folder with a report_manifest.json")
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.export_root.expanduser().resolve()
    runs = args.runs or sorted(path.name for path in root.iterdir() if (path / "report_manifest.json").is_file())
    for run in runs:
        files = publish(root / run, args.output_root.expanduser().resolve() / run)
        print(f"{run}: {len(files)} files")


if __name__ == "__main__":
    main()
