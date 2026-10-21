"""Fit the antenna-baseline azimuth (and D/lambda) from the authentic satellites.

Command-line wrapper around ``aoa_pipeline.fit_baseline_azimuth`` (the ``AOA_Paper.ipynb``
rig-check cell). For authentic PRNs i, j the wrapped inter-satellite double difference must
equal wrap(k * (cos a_i - cos a_j)) with cos a = cos(el) * cos(az - az_b), where el/az come
from Receiver 1 NAV-SAT and az_b is the baseline azimuth. The fit grid-searches az_b over
[0, 360) and k = D/lambda over a small set, using the single differences as computed with
``--order`` (default rx2-rx1, the notebook default).

How to read it: a fitted az_b near the configured ``--bias`` confirms the configuration. A
fitted az_b near bias + 180 means the receivers are wired the other way round: either use
``navsat_azimuth_bias_deg=<fit>`` or keep the bias and swap ``single_difference_order``.
The 08-30 afternoon captures (v8, v9) fit 176-182 deg with rx2-rx1; the 08-28 captures fit
352-356 deg.

Also prints, for the spoof group, the branch-invariant common-source collapse
(median |wrap(SD_i - SD_j)| between spoof PRNs), which needs no anchor at all.

    uv run python calibrate_baseline_azimuth.py dataset/sp-partial-28-08/v9 --spoof 15 18 24 29
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from aoa_pipeline import (
    compute_navsat_geometry_by_synced_epoch,
    compute_single_difference,
    fit_baseline_azimuth,
    probe_capture,
    summarize_pairwise_single_difference_collapse,
)


def receiver_paths(folder: Path) -> tuple[Path, Path]:
    rx1 = sorted(folder.glob("COM3*.ubx"))
    rx2 = sorted(folder.glob("COM4*.ubx"))
    if not rx1 or not rx2:
        raise FileNotFoundError(f"Expected COM3*.ubx and COM4*.ubx in {folder}")
    return rx1[0], rx2[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--spoof", type=int, nargs="*", default=[], help="spoof PRNs (excluded from the fit)")
    parser.add_argument("--min-epochs", type=int, default=100, help="minimum valid epochs for an authentic PRN")
    parser.add_argument("--k", type=float, nargs="*", default=[0.9, 0.95, 1.0, 1.05, 1.1], help="D/lambda grid")
    parser.add_argument("--ratio", type=float, default=1.0, help="D/lambda used for the reported fit")
    parser.add_argument("--order", default="rx2-rx1", choices=["rx1-rx2", "rx2-rx1"], help="single-difference order")
    parser.add_argument("--bias", type=float, default=0.0, help="configured navsat_azimuth_bias_deg to test")
    args = parser.parse_args()
    pd.set_option("display.width", 250)

    rx1, rx2 = receiver_paths(args.folder)
    probe = probe_capture(rx1, rx2, spoof_svids=args.spoof)
    probe.raise_for_problems()
    single_difference = compute_single_difference(probe.carrier_phase, order=args.order)
    elevation, azimuth = compute_navsat_geometry_by_synced_epoch(
        probe.synced.epochs, probe.observations[0].navsat, epoch_tow_column="rcvTow_rx1", navsat_tow_decimals=0
    )
    fit = fit_baseline_azimuth(
        single_difference,
        elevation,
        azimuth,
        exclude_svids=args.spoof,
        min_epochs=args.min_epochs,
        baseline_over_wavelength=args.ratio,
        ratio_grid=args.k,
        configured_azimuth_deg=args.bias,
    )
    print(f"authentic PRNs used: {list(fit.authentic_svids)}")
    print("best baseline azimuth per D/lambda:")
    print(fit.best_per_ratio.round(4).to_string(index=False))
    print(
        f"k={fit.baseline_over_wavelength}: best az_b={fit.fitted_azimuth_deg:.0f} deg "
        f"(median resid {fit.fitted_residual_cycles:.3f} cyc); configured az_b={fit.configured_azimuth_deg:.0f} -> "
        f"{fit.configured_residual_cycles:.3f}; az_b={fit.configured_azimuth_deg + 180:.0f} -> "
        f"{fit.opposite_residual_cycles:.3f} (random data gives about 0.25)"
    )
    print(f"per-PRN median |resid| at the best fit: {fit.per_svid_residual_cycles.round(3).to_dict()}")
    print(("OK: " if fit.consistent else "WARNING: ") + fit.recommendation)

    collapse = summarize_pairwise_single_difference_collapse(single_difference, args.spoof)
    if not collapse.empty:
        print("spoof-group common-source collapse, median |wrap(SD_i - SD_j)| in cycles (n epochs):")
        for row in collapse.itertuples():
            print(f"   {row.pair}: {row.median_abs_wrapped_sd_diff_cycles:.4f} ({row.epoch_count})")


if __name__ == "__main__":
    main()
