"""Data check for a two-receiver UBX capture before running the DD-method AoA flow.

Command-line wrapper around ``aoa_pipeline.probe_capture`` (the same check the
``AOA_Paper.ipynb`` data-check cell runs). Reports, per receiver: message inventory
(RXM-RAWX / NAV-PVT / NAV-SAT / NAV-SIG), RAWX epochs and time span, per-PRN RAWX rows,
rows with usable carrier phase (cpValid=1 and halfCyc=1), lock time, C/N0 and NAV-SAT
svUsed. Then the synchronized overlap of the two receivers and, for the spoof group, the
epochs where every spoof PRN has valid phase in both receivers (what the partial-spoof flow
needs) plus each spoof PRN's valid-phase timeline. Exit status 1 when the capture is
unusable.

    uv run python probe_dataset.py dataset/sp-partial-28-08/v9 --spoof 15 18 24 29
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from aoa_pipeline import probe_capture


def receiver_paths(folder: Path) -> tuple[Path, Path]:
    rx1 = sorted(folder.glob("COM3*.ubx"))
    rx2 = sorted(folder.glob("COM4*.ubx"))
    if not rx1 or not rx2:
        raise FileNotFoundError(f"Expected COM3*.ubx and COM4*.ubx in {folder}")
    return rx1[0], rx2[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", type=Path, help="capture folder holding COM3*.ubx and COM4*.ubx")
    parser.add_argument("--spoof", type=int, nargs="*", default=[], help="spoof PRN group")
    args = parser.parse_args()
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 200)

    rx1, rx2 = receiver_paths(args.folder)
    probe = probe_capture(rx1, rx2, spoof_svids=args.spoof)

    print("== UBX message inventory")
    print(probe.message_inventory.to_string())
    if not probe.receiver_svid_summary.empty:
        print("== per receiver and PRN (usable_phase_rows: cpValid=1 and halfCyc=1)")
        print(probe.receiver_svid_summary.round(1).to_string(index=False))
    if probe.synced is not None:
        print(f"== synchronized epochs: {len(probe.synced.epochs)}; common valid-phase epochs per PRN:")
        print(probe.common_valid_phase.sum().to_string())
    if args.spoof:
        print("== spoof PRNs")
        print(probe.spoof_phase_summary.to_string(index=False))
        print(
            f"== epochs with all usable spoof PRNs valid in both receivers: "
            f"{probe.all_spoof_valid_epoch_count}; longest continuous run: {probe.longest_all_spoof_valid_run}"
        )
    for warning in probe.warnings:
        print(f"WARNING: {warning}")
    for problem in probe.problems:
        print(f"PROBLEM: {problem}")
    raise SystemExit(1 if probe.problems else 0)


if __name__ == "__main__":
    main()
