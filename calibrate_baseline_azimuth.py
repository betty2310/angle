"""Fit the antenna-baseline azimuth (and D/lambda) from the authentic satellites.

For authentic PRNs i, j the wrapped inter-satellite double difference must equal
wrap(k * (cos a_i - cos a_j)) with cos a = cos(el) * cos(az - az_b), where el/az come
from Receiver 1 NAV-SAT and az_b is the baseline azimuth. This script grid-searches
az_b over [0, 360) and k = D/lambda over a small set using the single-difference
order rx2-rx1 (the notebook default) and reports the fit.

How to read it: a fitted az_b near 0 confirms the notebook default
(single_difference_order="rx2-rx1", navsat_azimuth_bias_deg=0). A fitted az_b near
180 means the receivers are wired the other way round: either use
navsat_azimuth_bias_deg=<fit> or keep bias 0 and set single_difference_order="rx1-rx2".
The 08-30 afternoon captures (v8, v9) fit 176-182 deg; the 08-28 captures fit 352-356 deg.

Also prints, for the spoof group, the branch-invariant common-source collapse
(median |wrap(SD_i - SD_j)| between spoof PRNs), which needs no anchor at all.

    uv run python calibrate_baseline_azimuth.py dataset/sp-partial-28-08/v9 --spoof 15 18 24 29
"""
from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from aoa_pipeline import (
    _drop_invalid_navsat_geometry,
    _navsat_epoch_merge,
    build_carrier_phase_table,
    read_ubx_observations,
    sync_receiver_epochs,
    wrap_cycles,
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
    args = parser.parse_args()
    spoof = set(args.spoof)

    rx1, rx2 = receiver_paths(args.folder)
    obs1 = read_ubx_observations(rx1, 0, 0)
    obs2 = read_ubx_observations(rx2, 0, 0)
    synced = sync_receiver_epochs(obs1, obs2, tow_round_decimals=0)
    epoch_index = pd.Index(synced.epochs["epoch_index"], name="epoch_index")
    tables = build_carrier_phase_table(synced, use_navpvt_time_correction=True, navpvt_tow_decimals=0)
    single_difference = tables.rx2_cycles - tables.rx1_cycles  # notebook order rx2-rx1

    nav = _navsat_epoch_merge(synced.epochs, obs1.navsat, epoch_tow_column="rcvTow_rx1", navsat_tow_decimals=0)
    nav = _drop_invalid_navsat_geometry(nav.dropna(subset=["svid", "elev", "azim"]))
    nav["svid"] = nav["svid"].astype(int)
    elev = nav.pivot_table(index="epoch_index", columns="svid", values="elev", aggfunc="first").reindex(epoch_index)
    azim = nav.pivot_table(index="epoch_index", columns="svid", values="azim", aggfunc="first").reindex(epoch_index)

    authentic = [
        int(s) for s in single_difference.columns
        if int(s) not in spoof and int(s) in elev.columns
        and single_difference[s].notna().sum() >= args.min_epochs
    ]
    if len(authentic) < 3:
        raise SystemExit(f"Need at least 3 authentic PRNs with >= {args.min_epochs} epochs; found {authentic}")
    print(f"authentic PRNs used: {authentic}")

    sd = single_difference[authentic].to_numpy(dtype=float)
    el = np.deg2rad(elev[authentic].to_numpy(dtype=float))
    az = np.deg2rad(azim[authentic].to_numpy(dtype=float))
    pairs = list(combinations(range(len(authentic)), 2))
    measured = np.stack([sd[:, i] - sd[:, j] for i, j in pairs], axis=1)

    def residuals(az_b_deg: float, k: float) -> np.ndarray:
        cos_alpha = np.cos(el) * np.cos(az - np.deg2rad(az_b_deg))
        expected = np.stack([cos_alpha[:, i] - cos_alpha[:, j] for i, j in pairs], axis=1)
        return np.abs(wrap_cycles(measured - k * expected))

    rows = []
    for k in args.k:
        for az_b in np.arange(0.0, 360.0, 1.0):
            r = residuals(az_b, k)
            rows.append({"k": k, "az_b_deg": az_b, "median_abs_resid_cycles": float(np.nanmedian(r)),
                         "pct_le_0p05": float(100 * np.nanmean(r <= 0.05))})
    grid = pd.DataFrame(rows)
    best_per_k = grid.loc[grid.groupby("k")["median_abs_resid_cycles"].idxmin()].reset_index(drop=True)
    print("best baseline azimuth per D/lambda:")
    print(best_per_k.round(4).to_string(index=False))
    k1 = grid[grid["k"].eq(1.0)]
    best = k1.loc[k1["median_abs_resid_cycles"].idxmin()]
    at0 = k1.loc[k1["az_b_deg"].eq(0.0)].iloc[0]
    at180 = k1.loc[k1["az_b_deg"].eq(180.0)].iloc[0]
    print(f"k=1: best az_b={best.az_b_deg:.0f} deg (median resid {best.median_abs_resid_cycles:.3f} cyc); "
          f"az_b=0 -> {at0.median_abs_resid_cycles:.3f}; az_b=180 -> {at180.median_abs_resid_cycles:.3f} "
          f"(random data gives about 0.25)")
    r_best = residuals(float(best.az_b_deg), 1.0)
    per_prn = {}
    for idx, s in enumerate(authentic):
        cols = [p for p, (i, j) in enumerate(pairs) if idx in (i, j)]
        per_prn[s] = round(float(np.nanmedian(r_best[:, cols])), 3)
    print(f"per-PRN median |resid| at the best k=1 fit: {per_prn}")

    az_fit = float(best.az_b_deg)
    if min(az_fit, 360 - az_fit) <= 30:
        print("recommendation: notebook default is fine (single_difference_order='rx2-rx1', navsat_azimuth_bias_deg=0); "
              f"for a few degrees more accuracy use navsat_azimuth_bias_deg={az_fit:.0f}.")
    elif abs(az_fit - 180) <= 30:
        print(f"recommendation: receivers are wired the other way round. Use navsat_azimuth_bias_deg={az_fit:.0f} "
              f"with 'rx2-rx1', or keep bias 0 and set single_difference_order='rx1-rx2'.")
    else:
        print(f"recommendation: baseline is not north-south; use navsat_azimuth_bias_deg={az_fit:.0f} with 'rx2-rx1'.")

    present = [s for s in args.spoof if s in single_difference.columns and single_difference[s].notna().any()]
    if len(present) >= 2:
        print("spoof-group common-source collapse, median |wrap(SD_i - SD_j)| in cycles (n epochs):")
        for a, b in combinations(present, 2):
            d = wrap_cycles(single_difference[a] - single_difference[b]).abs().dropna()
            print(f"   {a}-{b}: {d.median():.4f} ({len(d)})" if len(d) else f"   {a}-{b}: no common epochs")


if __name__ == "__main__":
    main()
