#!/usr/bin/env python3
"""Plot mixed-dataset fractional DD tracks using PRN 23 as reference."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path


OUTPUT_DIR = Path(__file__).resolve().parent
REPO_ROOT = OUTPUT_DIR.parents[4]
sys.path.insert(0, str(REPO_ROOT))

from aoa_pipeline import (  # noqa: E402
    DatasetConfig,
    plot_fractional_dd,
    run_paper_scenario,
)


REFERENCE_PRN = 23
SPOOF_PRNS = (6, 9, 11, 20)
SPOOF_PRNS_ = (32)

DATA_DIR = OUTPUT_DIR / "data"
FIGURE_DIR = OUTPUT_DIR / "figures"


def main() -> None:
    config = DatasetConfig(
        name="sp-partial-19-06/18-06",
        receiver1_path=REPO_ROOT
        / "dataset/sp-partial-19-06/18-06/COM3___9600_260618_095042.ubx",
        receiver2_path=REPO_ROOT
        / "dataset/sp-partial-19-06/18-06/COM4___9600_260618_095054.ubx",
        alpha_r_deg=None,
        reference_svid=REFERENCE_PRN,
        selected_svids=SPOOF_PRNS,
        excluded_reference_svids=SPOOF_PRNS_,
        reference_required_svids=SPOOF_PRNS,
        start_index=0,
        length=500,
        sync_tow_decimals=0,
        use_navpvt_time_correction=True,
        single_difference_order="rx2-rx1",
        navsat_azimuth_bias_deg=0.0,
        output_root=REPO_ROOT / "outputs/aoa_paper",
    )

    scenario = run_paper_scenario(
        config,
        partial_spoof_enabled=True,
        spoof_svids=SPOOF_PRNS,
        convergence_threshold_deg=5.0,
        save_outputs=True,
    )
    if scenario.all_satellite_result.reference_svid != REFERENCE_PRN:
        raise ValueError(
            "Fractional DD reference changed: expected PRN "
            f"{REFERENCE_PRN}, got {scenario.all_satellite_result.reference_svid}."
        )

    fractional_dd = (
        scenario.all_satellite_result.fractional_dd_cycles
        .dropna(axis=1, how="all")
        .sort_index()
        .sort_index(axis=1)
    )
    fractional_dd = fractional_dd.drop(columns=[REFERENCE_PRN], errors="ignore")
    available_spoof = tuple(
        prn for prn in SPOOF_PRNS if prn in fractional_dd.columns
    )
    if available_spoof != SPOOF_PRNS:
        raise ValueError(
            f"Expected spoof PRNs {SPOOF_PRNS}, got {available_spoof}."
        )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fractional_dd.to_csv(DATA_DIR / "fractional_double_difference_cycles.csv")
    fractional_dd.to_csv(
        DATA_DIR / "fractional_double_difference_cycles_all_prns_ref23.csv"
    )

    plot_result = replace(
        scenario.all_satellite_result,
        fractional_dd_cycles=fractional_dd,
    )
    fig = plot_fractional_dd(
        plot_result,
        highlight_svids=SPOOF_PRNS,
        title=None,
    )
    for stem in (
        "fraction_dd_ref_prn23",
        "figure_01_fractional_double_difference_mixed",
    ):
        fig.savefig(
            FIGURE_DIR / f"{stem}.png",
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )
        fig.savefig(
            FIGURE_DIR / f"{stem}.pdf",
            bbox_inches="tight",
            facecolor="white",
        )

    spoof_prns = [prn for prn in fractional_dd if prn in SPOOF_PRNS]
    authentic_prns = [prn for prn in fractional_dd if prn not in SPOOF_PRNS]
    print(f"Reference PRN: {REFERENCE_PRN}")
    print(f"Epochs: {len(fractional_dd)}")
    print(f"Spoof PRNs: {spoof_prns}")
    print(f"Authentic PRNs: {authentic_prns}")


if __name__ == "__main__":
    main()
