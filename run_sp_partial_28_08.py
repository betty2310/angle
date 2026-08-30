"""Run the AOA_Paper.ipynb partial-spoof flow on dataset/sp-partial-28-08.

This mirrors the notebook's configuration cell with the partial-spoof flow: data check
(``probe_capture``), rig check (``fit_baseline_azimuth``), ``run_paper_scenario`` with the
joint-source branch selection, and the per-PRN detection table that lists both Eq. 21
branches next to the selected one and how it was chosen (``chosen_by``).

Example::

    uv run python run_sp_partial_28_08.py --version v4 --spoof 10 23 24 --prior 30
    uv run python run_sp_partial_28_08.py --version v9 --spoof 15 18 24 29 --order rx1-rx2
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from aoa_pipeline import (
    BRANCH_METRICS,
    DatasetConfig,
    compute_navsat_geometry_by_synced_epoch,
    compute_single_difference,
    fit_baseline_azimuth,
    probe_capture,
    run_paper_scenario,
    summarize_pairwise_single_difference_collapse,
)

DATASET_ROOT = Path("dataset/sp-partial-28-08")
OUTPUT_ROOT = Path("outputs/aoa_paper")


def receiver_paths(version: str) -> tuple[Path, Path]:
    folder = DATASET_ROOT / version
    rx1 = sorted(folder.glob("COM3*.ubx"))
    rx2 = sorted(folder.glob("COM4*.ubx"))
    if not rx1 or not rx2:
        raise FileNotFoundError(f"Expected COM3/COM4 UBX logs in {folder}")
    return rx1[0], rx2[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", required=True, help="dataset folder, e.g. v5")
    parser.add_argument("--spoof", type=int, nargs="+", default=[10, 23, 24, 25])
    parser.add_argument("--reference", type=int, default=None, help="manual DD reference PRN (default: automatic non-spoof)")
    parser.add_argument("--bias", type=float, default=0.0, help="navsat_azimuth_bias_deg (baseline azimuth)")
    parser.add_argument("--order", default="rx2-rx1", choices=["rx1-rx2", "rx2-rx1"])
    parser.add_argument("--k", type=float, default=1.0, help="baseline_over_wavelength (D / lambda)")
    parser.add_argument("--threshold", type=float, default=5.0, help="convergence threshold in degrees")
    parser.add_argument("--detection-threshold", type=float, default=10.0, help="paper AoA-difference threshold in degrees")
    parser.add_argument("--prior", type=float, default=None, help="declared prior AoA in degrees that breaks the branch tie (default: report unresolved)")
    parser.add_argument("--metric", default="direction_cosine", choices=list(BRANCH_METRICS), help="DP branch metric")
    parser.add_argument("--tie-tolerance", type=float, default=None, help="override the automatic tie tolerance (direction cosine)")
    parser.add_argument("--suffix", default="", help="output folder suffix")
    parser.add_argument(
        "--no-halfcyc-gate",
        action="store_true",
        help="diagnostic only: accept RAWX phase with cpValid=1 even when halfCyc=0 "
        "(phase may carry unresolved 0.5-cycle errors)",
    )
    args = parser.parse_args()

    if args.no_halfcyc_gate:
        import aoa_pipeline

        _strict_pivot = aoa_pipeline._pivot_rawx

        def _relaxed_pivot(rawx, value_col, epoch_index, require_valid_phase=False):
            if require_valid_phase and "half_cycle" in rawx.columns:
                rawx = rawx.assign(half_cycle=np.nan)
            return _strict_pivot(rawx, value_col, epoch_index, require_valid_phase)

        aoa_pipeline._pivot_rawx = _relaxed_pivot
        print("WARNING: halfCyc gate disabled; carrier phase may contain unresolved half-cycle errors.")

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 80)
    pd.set_option("display.max_colwidth", 120)

    rx1, rx2 = receiver_paths(args.version)
    print(f"== {args.version}: {rx1.name} / {rx2.name}")

    print("\n-- data check")
    probe = probe_capture(rx1, rx2, spoof_svids=args.spoof)
    print(probe.message_inventory.to_string())
    print(probe.spoof_phase_summary[["svid", "rx1_valid_epochs", "rx2_valid_epochs", "common_valid_epochs", "longest_common_run", "usable", "common_segments"]].to_string(index=False))
    print(f"  epochs with all usable spoof PRNs valid = {probe.all_spoof_valid_epoch_count}; longest continuous run = {probe.longest_all_spoof_valid_run}")
    for warning in probe.warnings:
        print(f"  WARNING: {warning}")
    probe.raise_for_problems()
    spoof = probe.usable_spoof_svids

    print("\n-- rig check (authentic PRNs, branch-free)")
    single_difference = compute_single_difference(probe.carrier_phase, order=args.order)
    elevation, azimuth = compute_navsat_geometry_by_synced_epoch(
        probe.synced.epochs, probe.observations[0].navsat, epoch_tow_column="rcvTow_rx1", navsat_tow_decimals=0
    )
    rig = fit_baseline_azimuth(
        single_difference, elevation, azimuth, exclude_svids=spoof,
        baseline_over_wavelength=args.k, configured_azimuth_deg=args.bias,
    )
    print(rig.best_per_ratio.round(4).to_string(index=False))
    print(f"  per-PRN median |residual| at the fit: {rig.per_svid_residual_cycles.round(3).to_dict()}")
    print(("  OK: " if rig.consistent else "  WARNING: ") + rig.recommendation)
    collapse = summarize_pairwise_single_difference_collapse(single_difference, spoof)
    print("-- spoof-group common-source collapse over all common epochs, median |wrap(SD_i - SD_j)| (cycles)")
    print(collapse.round(4).to_string(index=False))

    config = DatasetConfig(
        name=f"sp-partial-28-08/{args.version}{args.suffix}",
        receiver1_path=rx1,
        receiver2_path=rx2,
        alpha_r_deg=None,
        reference_svid=args.reference,
        selected_svids=spoof,
        excluded_reference_svids=spoof,
        reference_required_svids=spoof,
        excluded_analysis_svids=(),
        start_index=0,
        length=None,
        sync_tow_decimals=0,
        use_navpvt_time_correction=True,
        single_difference_order=args.order,
        baseline_over_wavelength=args.k,
        branch_metric=args.metric,
        navsat_azimuth_bias_deg=args.bias,
        output_root=OUTPUT_ROOT,
    )
    scenario = run_paper_scenario(
        config,
        partial_spoof_enabled=True,
        spoof_svids=spoof,
        convergence_threshold_deg=args.threshold,
        detection_threshold_deg=args.detection_threshold,
        prior_aoa_deg=args.prior,
        tie_tolerance=args.tie_tolerance,
        save_outputs=True,
    )
    result = scenario.analysis_result

    print("\n-- run summary")
    print(scenario.run_summary.T.to_string())
    print("\n-- reference ranking (probe run, non-spoof candidates)")
    print(scenario.probe_result.reference_selection.head(8).to_string())
    window = scenario.reference_alpha_by_epoch.loc[
        scenario.reference_alpha_by_epoch["epoch_index"].isin(result.epochs["source_epoch_index"])
    ]
    print("\n-- reference-angle anchor inside the analysis window (Receiver 1 NAV-SAT)")
    print(window[["epoch_index", "reference_svid", "reference_elev_deg", "reference_azim_deg", "reference_alpha_deg"]].describe().loc[["count", "mean", "min", "max"]].to_string())
    print("\n-- analysis window epochs (source index, TOW)")
    print(result.epochs[["epoch_index", "source_epoch_index", "rcvTow_rx1"]].to_string())

    print("\n-- fractional DD (cycles) of spoof PRNs in the window")
    print(result.fractional_dd_cycles.round(4).to_string())
    print("\n-- joint-source DD-method AoA (deg) of spoof PRNs")
    print(scenario.joint_source_aoa_deg.round(2).to_string())
    print("\n-- joint-source selected branch m")
    print(scenario.joint_source_branch.to_string())
    print("\n-- joint-source branch ambiguity (per epoch)")
    print(scenario.joint_source_ambiguity.round(4).to_string(index=False))
    print("\n-- per-PRN joint-source AoA summary")
    print(scenario.joint_source_aoa_deg.agg(["count", "median", "mean", "std", "min", "max"]).round(2).to_string())
    print("\n-- pairwise convergence")
    print(scenario.pairwise_convergence.round(3).to_string())
    print("\n-- cluster summary")
    print(scenario.cluster_summary.round(3).to_string())
    print("\n-- collapse metrics (fractional DD, window)")
    print(result.collapse_metrics.round(4).to_string(index=False))

    print("\n-- Eq. 21 valid branch candidates per spoof PRN (all window epochs)")
    print(scenario.spoof_branch_candidates.round(3).to_string(index=False))
    print(f"\n-- detection table (threshold {args.detection_threshold:.0f} deg; prior {args.prior})")
    print(scenario.spoof_branch_detection.round(2).to_string(index=False))

    print("\n-- authentic PRNs: DP DD-method AoA vs ephemeris AoA (all-satellite run)")
    eq = scenario.aoa_method_equivalence_summary
    print(eq.loc[eq["method"].eq("dp_smoothness")].round(2).to_string())

    out_dir = config.output_dir
    scenario.spoof_branch_candidates.to_csv(out_dir / "sp28_spoof_branch_alternatives.csv", index=False)
    ambiguity = scenario.joint_source_ambiguity.iloc[0] if not scenario.joint_source_ambiguity.empty else None
    report = {
        "version": args.version,
        "spoof_svids": spoof,
        "reference_svid": result.reference_svid,
        "reference_alpha_deg": float(scenario.run_summary["reference_alpha_deg"].iloc[0]),
        "navsat_azimuth_bias_deg": args.bias,
        "baseline_over_wavelength": args.k,
        "single_difference_order": args.order,
        "branch_metric": args.metric,
        "prior_aoa_deg": args.prior,
        "halfcyc_gate": not args.no_halfcyc_gate,
        "rig_fitted_azimuth_deg": rig.fitted_azimuth_deg,
        "rig_consistent": rig.consistent,
        "window_epochs": len(result.epochs),
        "window_source_start": int(result.epochs["source_epoch_index"].iloc[0]),
        "chosen_by": None if ambiguity is None else str(ambiguity["chosen_by"]),
        "ambiguous": None if ambiguity is None else bool(ambiguity["ambiguous"]),
        "selected_branches": None if ambiguity is None else str(ambiguity["selected_branches"]),
        "alternative_branches": None if ambiguity is None else str(ambiguity["alternative_branches"]),
        "cluster_svids": scenario.cluster_summary["cluster_svids"].iloc[0],
        "cluster_median_aoa_deg": float(scenario.cluster_summary["median_aoa_deg"].iloc[0]),
        "cluster_median_spread_deg": float(scenario.cluster_summary["median_spread_deg"].iloc[0]),
    }
    detection = scenario.spoof_branch_detection.set_index("svid")
    for svid in spoof:
        report[f"prn{svid}_selected_aoa_deg"] = float(detection.loc[svid, "selected_aoa_deg"])
        report[f"prn{svid}_alternative_aoa_deg"] = float(detection.loc[svid, "alternative_aoa_deg"])
        report[f"prn{svid}_ephemeris_aoa_deg"] = float(detection.loc[svid, "ephemeris_aoa_deg"])
        report[f"prn{svid}_flag_all_branches"] = bool(detection.loc[svid, "spoof_flag_all_branches"])
    pd.DataFrame([report]).to_csv(out_dir / "sp28_report.csv", index=False)
    print(f"\n-- report written to {out_dir / 'sp28_report.csv'}")


if __name__ == "__main__":
    main()
