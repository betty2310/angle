"""Side-by-side manuscript figures for the authentic dataset (no titles, no captions).

Reproduces archive figures from ``AoA_Estimate_archive/AOA_Paper.ipynb`` (cell 2,
start index 4000, reference = first tracked SV = PRN 10) as two wide images:

* ``dd_authentic_side_by_side``  - DD carrier phase, full window | first 200 s
  (manuscript Figs. 5-6).
* ``aoa_authentic_side_by_side`` - Ephemeris-based AoA | DD-method AoA
  (manuscript Figs. 7-8).

Each is also written as ``*_paper`` in the ``aoa_figure_style.json`` theme used by
``AOA_Export_Figures.ipynb`` (shared top legend, serif font, no panel captions).

    uv run python export_authentic_dd_side_by_side.py
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pyubx2 import UBXReader

from aoa_figure_report import draw_prn_traces, load_figure_design, prn_styles, save_figure

DATA_DIR = Path("AoA_Estimate_archive/Authentic dataset")
OUT_DIR = Path("Plot/AOA/authentic_dd_side_by_side")
EXCLUDED_SVIDS = {12}

# Archive cell 2 parameters (manuscript Figs. 7-8).
ALPHA_R_DEG = 53.89629
BIAS_ANGLE_DEG = 16
AOA_SVIDS = [15, 18, 23, 24, 25, 28, 32]
BRANCH_SHIFT = {15: 0, 23: 0, 24: 0, 32: 0, 18: -1, 25: -1, 28: -1}


def read_ubx(path: Path):
    raws, pvts, sats = [], [], []
    with open(path, "rb") as fh:
        for _, msg in UBXReader(fh, protfilter=2):
            if msg is None:
                continue
            if msg.identity == "RXM-RAWX":
                raws.append(msg)
            elif msg.identity == "NAV-PVT":
                pvts.append(msg)
            elif msg.identity == "NAV-SAT":
                sats.append(msg)
    return raws, pvts, sats


def l1_carrier(rawx, pvt) -> np.ndarray:
    """Same per-epoch quantity as the archive ``calc_pseudorange``."""
    ps = np.zeros(32)
    for i in range(1, rawx.numMeas + 1):
        if getattr(rawx, f"gnssId_{i:02}") == 0 and getattr(rawx, f"sigId_{i:02}") == 0:
            sv = getattr(rawx, f"svId_{i:02}")
            ps[sv - 1] = getattr(rawx, f"cpMes_{i:02}") + float(pvt.nano) * 1e-9 * float(
                getattr(rawx, f"doMes_{i:02}")
            )
    return ps


def sync_by_tow(a, b):
    """Archive ``sync_by_tow``: trims RAWX/PVT only (NAV-SAT is left untouched)."""
    offset = int(float(a[0][0].rcvTow) - float(b[0][0].rcvTow))
    if offset > 0:
        b = (b[0][offset:], b[1][offset:], b[2])
    elif offset < 0:
        a = (a[0][-offset:], a[1][-offset:], a[2])
    return a, b


def compute_dps(d1, d2, start: int, length: int):
    """Archive ``compute_dps`` (rx1 - rx2): DD vs. first-seen SV, nearest integer removed."""
    dps = np.zeros((length, 32))
    sv_to_col: dict[int, int] = {}
    for k in range(length):
        j = start + k
        if j >= len(d1[0]) or j >= len(d2[0]):
            break
        ps1 = l1_carrier(d1[0][j], d1[1][j])
        ps2 = l1_carrier(d2[0][j], d2[1][j])
        for i in range(32):
            if ps1[i] != 0 and ps2[i] != 0:
                sv_to_col.setdefault(i + 1, len(sv_to_col))
                dps[k, sv_to_col[i + 1]] = ps1[i] - ps2[i]
        if sv_to_col:
            dps[k, :] -= dps[k, 0]
            dps[k, :] -= np.round(dps[k, :])
    # Unassigned columns pick up -ref after the subtraction; keep only real SVs, minus the reference.
    col_to_sv = {c: s for s, c in sv_to_col.items()}
    cols = [c for c in sorted(col_to_sv) if np.any(dps[:, c] != 0)]
    return dps[:, cols], [col_to_sv[c] for c in cols], col_to_sv[0]


def ephemeris_aoa(navsats, start: int, length: int) -> np.ndarray:
    """Archive ephemeris-based AoA: acos(cos(elev) * cos(azim - bias)), NaN where SV absent."""
    out = np.full((length, len(AOA_SVIDS)), np.nan)
    for k, msg in enumerate(navsats[start : start + length]):
        for i in range(1, msg.numSvs + 1):
            sv = getattr(msg, f"svId_{i:02d}")
            if getattr(msg, f"gnssId_{i:02d}") == 0 and sv in AOA_SVIDS:
                el = math.radians(getattr(msg, f"elev_{i:02d}"))
                az = math.radians(getattr(msg, f"azim_{i:02d}") - BIAS_ANGLE_DEG)
                out[k, AOA_SVIDS.index(sv)] = math.degrees(math.acos(math.cos(el) * math.cos(az)))
    return out


def load(args):
    if args.cache.exists():
        z = np.load(args.cache)
        return {k: z[k] for k in z.files}
    d1, d2 = sync_by_tow(read_ubx(DATA_DIR / "raw_data_1.ubx"), read_ubx(DATA_DIR / "raw_data_2.ubx"))
    dd, svids, ref = compute_dps(d1, d2, args.start, args.full_length)
    data = {
        "dd": dd,
        "svids": np.array(svids),
        "ref": np.array(ref),
        "tow": np.array([float(r.rcvTow) for r in d1[0][args.start : args.start + len(dd)]]),
        "eph": ephemeris_aoa(d1[2], args.start, args.full_length),
        "eph_itow": np.array([m.iTOW for m in d1[2][args.start : args.start + args.full_length]]) / 1000.0,
    }
    np.savez(args.cache, **data)
    return data


def dd_method_aoa(dd: np.ndarray, svids: list[int]) -> np.ndarray:
    cos_r = math.cos(math.radians(ALPHA_R_DEG))
    out = np.full((len(dd), len(AOA_SVIDS)), np.nan)
    for j, sv in enumerate(AOA_SVIDS):
        cos_i = dd[:, svids.index(sv)] + cos_r + BRANCH_SHIFT[sv]
        out[:, j] = np.degrees(np.arccos(np.clip(cos_i, -1, 1)))
    return out


def save(fig, name: str) -> None:
    fig.tight_layout()
    for ext in ("png", "pdf", "eps"):
        fig.savefig(OUT_DIR / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    print(f"saved -> {OUT_DIR}/{name}.{{png,pdf,eps}}")


def plot_dd_panel(ax, t, dps, svids):
    for i, sv in enumerate(svids):
        if sv not in EXCLUDED_SVIDS and np.any(dps[:, i] != 0):
            ax.plot(t, dps[:, i], label=f"SV {sv}", linestyle="-", alpha=0.9)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Cycle")
    ax.set_ylim(-0.5, 0.5)
    ax.legend(ncol=4, fontsize=8, loc="upper right")


def plot_aoa_panel(ax, t, aoa, legend_loc):
    for j, sv in enumerate(AOA_SVIDS):
        ax.plot(t, aoa[:, j], label=f"SV {sv}", linestyle="-", alpha=0.9)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"$\alpha_i$ (in degree)")
    ax.legend(ncol=4, fontsize=8, loc=legend_loc)


def export_paper_theme(t, t_eph, dd_plot, dd_svids, eph, aoa_dd, zoom: int, aoa_len: int) -> None:
    """Same panels in the AOA_Export_Figures.ipynb theme; every PRN is authentic (no spoof set)."""
    design = load_figure_design()
    keep = [sv for sv in dd_svids if sv not in EXCLUDED_SVIDS]
    dd_frame = pd.DataFrame(dd_plot[:, [dd_svids.index(sv) for sv in keep]], columns=keep)
    eph_frame = pd.DataFrame(eph[:aoa_len], columns=AOA_SVIDS)
    aoa_frame = pd.DataFrame(aoa_dd, columns=AOA_SVIDS)
    xlabel = design["elapsed_axis_label"]
    with plt.rc_context(design["rc_params"]):
        styles = prn_styles(keep, (), design)
        fig, axes = plt.subplots(1, 2, figsize=design["comparison_figsize"], sharey=True)
        for ax, n in zip(axes, [len(t), zoom]):
            draw_prn_traces(ax, t[:n], dd_frame.iloc[:n], declared=(), styles=styles, design=design,
                            xlabel=xlabel, break_wraps=True)
            ax.set_ylim(-0.55, 0.55)
            ax.set_yticks(np.arange(-0.5, 0.51, 0.25))
        axes[0].set_ylabel("Fractional double difference (cycles)")
        save_figure(fig, axes, OUT_DIR, "dd_authentic_side_by_side_paper", design, show=False)

        styles = prn_styles(AOA_SVIDS, (), design)
        fig, axes = plt.subplots(1, 2, figsize=design["comparison_figsize"], sharex=True, sharey=True)
        for ax, x, frame in [(axes[0], t_eph[:aoa_len], eph_frame), (axes[1], t[:aoa_len], aoa_frame)]:
            draw_prn_traces(ax, x, frame, declared=(), styles=styles, design=design, xlabel=xlabel)
            ax.set_ylim(0, 180)
            ax.set_yticks(np.arange(0, 181, 30))
        axes[0].set_ylabel(r"$\alpha_i$ (degree)")
        save_figure(fig, axes, OUT_DIR, "aoa_authentic_side_by_side_paper", design, show=False)
    print(f"saved -> {OUT_DIR}/{{dd,aoa}}_authentic_side_by_side_paper.{{{','.join(design['formats'])}}}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start", type=int, default=4000)
    p.add_argument("--full-length", type=int, default=2000)
    p.add_argument("--zoom-length", type=int, default=200)
    p.add_argument("--aoa-length", type=int, default=1000)
    p.add_argument("--cache", type=Path, default=OUT_DIR / "authentic_cache.npz")
    args = p.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    d = load(args)
    dd, svids = d["dd"], [int(s) for s in d["svids"]]
    t = d["tow"] - d["tow"][0]
    t_eph = d["eph_itow"] - d["eph_itow"][0]
    print(f"reference SV {int(d['ref'])}; SVs {svids}; epochs {len(dd)}; "
          f"RAWX step {np.median(np.diff(t)):.3f} s, NAV-SAT step {np.median(np.diff(t_eph)):.3f} s")

    # Figs. 5-6: manuscript DD plots are rx2 - rx1; re-wrap so values stay in [-0.5, 0.5).
    dd_plot = -dd
    dd_plot -= np.round(dd_plot)
    sns.set_style("darkgrid")
    fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(16, 5))
    plot_dd_panel(ax_full, t, dd_plot, svids)
    n = args.zoom_length
    plot_dd_panel(ax_zoom, t[:n], dd_plot[:n], svids)
    save(fig, "dd_authentic_side_by_side")

    # Figs. 7-8: archive DD-method AoA uses rx1 - rx2 (compute_dps as written).
    n = args.aoa_length
    sns.set_style("whitegrid")
    fig, (ax_eph, ax_dd) = plt.subplots(1, 2, figsize=(16, 5), sharey=True)
    plot_aoa_panel(ax_eph, t_eph[:n], d["eph"][:n], "upper left")
    aoa_dd = dd_method_aoa(dd[:n], svids)
    plot_aoa_panel(ax_dd, t[:n], aoa_dd, "lower left")
    ax_dd.tick_params(labelleft=True)
    save(fig, "aoa_authentic_side_by_side")
    plt.close("all")

    export_paper_theme(t, t_eph, dd_plot, svids, d["eph"], aoa_dd, args.zoom_length, n)


if __name__ == "__main__":
    main()
