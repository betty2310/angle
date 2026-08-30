"""Run the AOA_Paper.ipynb partial-spoof flow on dataset/sp-partial-28-08.

This mirrors the notebook's scenario-configuration cell with
``PARTIAL_SPOOF_ENABLED = True`` and reports the DD-method AoA of the spoof
PRNs (joint-source selection, pairwise convergence, cluster median) together
with the Eq. 21 branch alternatives, so the one-wavelength branch ambiguity is
visible next to the selected value.

Example::

    uv run python run_sp_partial_28_08.py --version v5 --spoof 10 23 24 25
    uv run python run_sp_partial_28_08.py --version v4 --spoof 10 23 24 --bias 0
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from aoa_pipeline import (
    DatasetConfig,
    build_carrier_phase_table,
    longest_valid_epoch_segment,
    read_ubx_observations,
    run_paper_scenario,
    sync_receiver_epochs,
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


def usable_spoof_svids(rx1: Path, rx2: Path, spoof: tuple[int, ...]) -> tuple[int, ...]:
    """Drop spoof PRNs that never have common valid carrier phase in both receivers."""
    obs1 = read_ubx_observations(rx1, 0, 0)
    obs2 = read_ubx_observations(rx2, 0, 0)
    synced = sync_receiver_epochs(obs1, obs2, tow_round_decimals=0)
    tables = build_carrier_phase_table(synced, use_navpvt_time_correction=True, navpvt_tow_decimals=0)
    common = tables.rx1_cycles.notna() & tables.rx2_cycles.notna()
    usable = []
    for svid in spoof:
        count = int(common[svid].sum()) if svid in common.columns else 0
        print(f"  spoof PRN {svid}: common valid carrier-phase epochs = {count}")
        if count > 0:
            usable.append(svid)
    all_valid = common.reindex(columns=usable).all(axis=1) if usable else pd.Series(False, index=common.index)
    segment = longest_valid_epoch_segment(all_valid)
    print(f"  epochs with all usable spoof PRNs valid = {int(all_valid.sum())}; longest continuous run = {len(segment)}")
    return tuple(usable)


def branch_alternatives(candidates: pd.DataFrame, svids: tuple[int, ...]) -> pd.DataFrame:
    valid = candidates.loc[candidates["svid"].isin(svids) & candidates["candidate_aoa_deg"].notna()]
    return (
        valid.groupby(["svid", "branch"])
        .agg(
            epoch_count=("epoch_index", "nunique"),
            median_aoa_deg=("candidate_aoa_deg", "median"),
            mean_aoa_deg=("candidate_aoa_deg", "mean"),
            std_aoa_deg=("candidate_aoa_deg", "std"),
            median_cos_alpha=("candidate_cos_alpha", "median"),
        )
        .reset_index()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", required=True, help="dataset folder, e.g. v5")
    parser.add_argument("--spoof", type=int, nargs="+", default=[10, 23, 24, 25])
    parser.add_argument("--reference", type=int, default=None, help="manual DD reference PRN (default: automatic non-spoof)")
    parser.add_argument("--bias", type=float, default=0.0, help="navsat_azimuth_bias_deg (baseline azimuth)")
    parser.add_argument("--order", default="rx2-rx1", choices=["rx1-rx2", "rx2-rx1"])
    parser.add_argument("--k", type=float, default=1.0, help="baseline_over_wavelength (D / lambda)")
    parser.add_argument("--threshold", type=float, default=5.0, help="convergence threshold in degrees")
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

    rx1, rx2 = receiver_paths(args.version)
    print(f"== {args.version}: {rx1.name} / {rx2.name}")
    spoof = usable_spoof_svids(rx1, rx2, tuple(args.spoof))
    if not spoof:
        raise SystemExit("No spoof PRN has common valid carrier phase; nothing to estimate.")
    if spoof != tuple(args.spoof):
        print(f"  NOTE: spoof group reduced to {spoof} (others have no common valid phase)")

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
        navsat_azimuth_bias_deg=args.bias,
        output_root=OUTPUT_ROOT,
    )
    scenario = run_paper_scenario(
        config,
        partial_spoof_enabled=True,
        spoof_svids=spoof,
        convergence_threshold_deg=args.threshold,
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
    print("\n-- per-PRN joint-source AoA summary")
    print(scenario.joint_source_aoa_deg.agg(["count", "median", "mean", "std", "min", "max"]).round(2).to_string())
    print("\n-- pairwise convergence")
    print(scenario.pairwise_convergence.round(3).to_string())
    print("\n-- cluster summary")
    print(scenario.cluster_summary.round(3).to_string())

    alternatives = branch_alternatives(result.candidate_aoa_deg, spoof)
    print("\n-- Eq. 21 valid branch candidates per spoof PRN (all window epochs)")
    print(alternatives.round(3).to_string())

    eph = scenario.ephemeris_aoa_rx1_deg.reindex(columns=[s for s in spoof if s in scenario.ephemeris_aoa_rx1_deg.columns])
    print("\n-- ephemeris-based AoA (Receiver 1 NAV-SAT) of spoof PRNs in the window")
    print(eph.agg(["count", "median", "min", "max"]).round(2).to_string())

    print("\n-- authentic PRNs: DP DD-method AoA vs ephemeris AoA (all-satellite run)")
    eq = scenario.aoa_method_equivalence_summary
    print(eq.loc[eq["method"].eq("dp_smoothness")].round(2).to_string())

    out_dir = config.output_dir
    alternatives.to_csv(out_dir / "sp28_spoof_branch_alternatives.csv", index=False)
    report = {
        "version": args.version,
        "spoof_svids": spoof,
        "reference_svid": result.reference_svid,
        "reference_alpha_deg": float(scenario.run_summary["reference_alpha_deg"].iloc[0]),
        "navsat_azimuth_bias_deg": args.bias,
        "baseline_over_wavelength": args.k,
        "single_difference_order": args.order,
        "halfcyc_gate": not args.no_halfcyc_gate,
        "window_epochs": len(result.epochs),
        "window_source_start": int(result.epochs["source_epoch_index"].iloc[0]),
        "cluster_svids": scenario.cluster_summary["cluster_svids"].iloc[0],
        "cluster_median_aoa_deg": float(scenario.cluster_summary["median_aoa_deg"].iloc[0]),
        "cluster_median_spread_deg": float(scenario.cluster_summary["median_spread_deg"].iloc[0]),
    }
    for svid in spoof:
        report[f"prn{svid}_joint_median_aoa_deg"] = float(scenario.joint_source_aoa_deg[svid].median())
        if svid in eph.columns:
            report[f"prn{svid}_ephemeris_median_aoa_deg"] = float(eph[svid].median())
    pd.DataFrame([report]).to_csv(out_dir / "sp28_report.csv", index=False)
    print(f"\n-- report written to {out_dir / 'sp28_report.csv'}")


if __name__ == "__main__":
    main()
