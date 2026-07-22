#!/usr/bin/env python3
"""Build paper-ready mixed-spoof AoA tables, figures, and aligned data.

The inputs are the saved outputs of AOA_Paper.ipynb for the
sp-partial-19-06/18-06 scenario.  The exported proposed-method AoA exactly
matches ``scenario.best_m_all_satellite_aoa_deg``: authentic PRNs use their
ephemeris-selected constant best-m branch, while known spoof PRNs retain the
DP branch so their common-source collapse is preserved.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUTPUT_DIR = Path(__file__).resolve().parent
REPO_ROOT = OUTPUT_DIR.parents[4]
SOURCE_DIR = REPO_ROOT / "outputs/aoa_paper/sp-partial-19-06/18-06"

DATA_DIR = OUTPUT_DIR / "data"
FIGURE_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"

REFERENCE_PRN = 23
SPOOF_PRNS = (6, 9, 11, 20)
THRESHOLDS_DEG = (5.0, 7.0, 10.0)

SPOOF_COLORS = {
    6: "#d62728",
    9: "#ff7f0e",
    11: "#9467bd",
    20: "#8c564b",
}
AUTHENTIC_COLORS = {
    10: "#1f77b4",
    12: "#17becf",
    13: "#bcbd22",
    15: "#2ca02c",
    18: "#0b7285",
    24: "#4c78a8",
    32: "#54a24b",
}


def read_wide_csv(name: str) -> pd.DataFrame:
    frame = pd.read_csv(SOURCE_DIR / name, index_col=0)
    frame.index = pd.Index(frame.index.astype(int), name="epoch_index")
    frame.columns = [int(column) for column in frame.columns]
    return frame.sort_index().sort_index(axis=1)


def aligned_pair(
    dd_aoa: pd.DataFrame, ephemeris_aoa: pd.DataFrame, prn: int
) -> pd.DataFrame:
    return pd.concat(
        [
            dd_aoa[prn].rename("dd_aoa_deg"),
            ephemeris_aoa[prn].rename("ephemeris_aoa_deg"),
        ],
        axis=1,
    ).dropna()


def format_percentage(value: float) -> str:
    if value < 1.0 and value != 0.0:
        return f"{value:.2f}%"
    return f"{value:.1f}%"


def build_threshold_table(
    dd_aoa: pd.DataFrame,
    ephemeris_aoa: pd.DataFrame,
    prns: Sequence[int],
    thresholds_deg: Sequence[float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    display_rows: list[dict[str, object]] = []
    numeric_rows: list[dict[str, object]] = []

    for threshold_deg in thresholds_deg:
        block_identical = 0
        block_total = 0
        for prn in prns:
            pair = aligned_pair(dd_aoa, ephemeris_aoa, prn)
            differences = (pair["dd_aoa_deg"] - pair["ephemeris_aoa_deg"]).abs()
            identical = int(differences.le(threshold_deg).sum())
            total = int(len(differences))
            percentage = 100.0 * identical / total if total else np.nan
            block_identical += identical
            block_total += total

            numeric_rows.append(
                {
                    "prn": prn,
                    "threshold_deg": threshold_deg,
                    "identical_epoch_count": identical,
                    "valid_epoch_count": total,
                    "percentage": percentage,
                }
            )
            display_rows.append(
                {
                    "PRN": str(prn),
                    "Threshold": f"{threshold_deg:g} deg",
                    "Number of Identical/total valid epochs": f"{identical}/{total}",
                    "Percentage": format_percentage(percentage),
                }
            )

        aggregate_percentage = (
            100.0 * block_identical / block_total if block_total else np.nan
        )
        numeric_rows.append(
            {
                "prn": "Average",
                "threshold_deg": threshold_deg,
                "identical_epoch_count": block_identical,
                "valid_epoch_count": block_total,
                "percentage": aggregate_percentage,
            }
        )
        display_rows.append(
            {
                "PRN": "Average",
                "Threshold": f"{threshold_deg:g} deg",
                "Number of Identical/total valid epochs": (
                    f"{block_identical}/{block_total}"
                ),
                "Percentage": format_percentage(aggregate_percentage),
            }
        )

    return pd.DataFrame(display_rows), pd.DataFrame(numeric_rows)


def build_mixed_decision_table(
    dd_aoa: pd.DataFrame,
    ephemeris_aoa: pd.DataFrame,
    authentic_prns: Sequence[int],
    spoof_prns: Sequence[int],
    threshold_deg: float = 10.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    display_rows: list[dict[str, object]] = []
    numeric_rows: list[dict[str, object]] = []
    correct_total = 0
    valid_total = 0

    for known_class, prns in (("Authentic", authentic_prns), ("Spoof", spoof_prns)):
        for prn in prns:
            pair = aligned_pair(dd_aoa, ephemeris_aoa, prn)
            differences = (pair["dd_aoa_deg"] - pair["ephemeris_aoa_deg"]).abs()
            classified_authentic = differences.le(threshold_deg)
            correct = int(
                classified_authentic.sum()
                if known_class == "Authentic"
                else (~classified_authentic).sum()
            )
            total = int(len(differences))
            percentage = 100.0 * correct / total if total else np.nan
            correct_total += correct
            valid_total += total

            numeric_rows.append(
                {
                    "prn": prn,
                    "known_class": known_class,
                    "threshold_deg": threshold_deg,
                    "correct_epoch_count": correct,
                    "valid_epoch_count": total,
                    "correct_percentage": percentage,
                }
            )
            display_rows.append(
                {
                    "PRN": str(prn),
                    "Known class": known_class,
                    "Threshold": f"{threshold_deg:g} deg",
                    "Number correct/total valid epochs": f"{correct}/{total}",
                    "Correct percentage": format_percentage(percentage),
                }
            )

    aggregate_percentage = 100.0 * correct_total / valid_total if valid_total else np.nan
    numeric_rows.append(
        {
            "prn": "Average",
            "known_class": "Mixed",
            "threshold_deg": threshold_deg,
            "correct_epoch_count": correct_total,
            "valid_epoch_count": valid_total,
            "correct_percentage": aggregate_percentage,
        }
    )
    display_rows.append(
        {
            "PRN": "Average",
            "Known class": "Mixed",
            "Threshold": f"{threshold_deg:g} deg",
            "Number correct/total valid epochs": f"{correct_total}/{valid_total}",
            "Correct percentage": format_percentage(aggregate_percentage),
        }
    )
    return pd.DataFrame(display_rows), pd.DataFrame(numeric_rows)


def markdown_table(frame: pd.DataFrame) -> str:
    headers = [str(column) for column in frame.columns]
    rows = [[str(value) for value in row] for row in frame.itertuples(index=False)]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]

    def render_row(values: Iterable[str]) -> str:
        return "| " + " | ".join(
            value.ljust(width) for value, width in zip(values, widths)
        ) + " |"

    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    return "\n".join([render_row(headers), separator, *(render_row(row) for row in rows)])


def latex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def write_latex_table(
    frame: pd.DataFrame,
    path: Path,
    *,
    caption: str,
    label: str,
) -> None:
    alignment = "c" * len(frame.columns)
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        f"\\begin{{tabular}}{{{alignment}}}",
        r"\hline",
        " & ".join(latex_escape(column) for column in frame.columns) + r" \\",
        r"\hline",
    ]
    for row in frame.itertuples(index=False):
        lines.append(" & ".join(latex_escape(value) for value in row) + r" \\")
    lines.extend(
        [
            r"\hline",
            r"\end{tabular}",
            f"\\caption{{{latex_escape(caption)}}}",
            f"\\label{{{latex_escape(label)}}}",
            r"\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_table_bundle(
    stem: str,
    display_frame: pd.DataFrame,
    numeric_frame: pd.DataFrame,
    *,
    caption: str,
    label: str,
) -> None:
    display_frame.to_csv(TABLE_DIR / f"{stem}.csv", index=False)
    numeric_frame.to_csv(TABLE_DIR / f"{stem}_numeric.csv", index=False)
    (TABLE_DIR / f"{stem}.md").write_text(
        markdown_table(display_frame) + "\n", encoding="utf-8"
    )
    write_latex_table(
        display_frame,
        TABLE_DIR / f"{stem}.tex",
        caption=caption,
        label=label,
    )


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def color_for_prn(prn: int) -> str:
    return SPOOF_COLORS.get(prn, AUTHENTIC_COLORS.get(prn, "#7f7f7f"))


def draw_tracks(
    ax: plt.Axes,
    frame: pd.DataFrame,
    prns: Sequence[int],
    *,
    include_class_in_label: bool = True,
) -> None:
    for prn in prns:
        if prn not in frame.columns or not frame[prn].notna().any():
            continue
        spoof = prn in SPOOF_PRNS
        suffix = " (spoof)" if spoof and include_class_in_label else ""
        ax.plot(
            frame.index,
            frame[prn],
            label=f"PRN {prn}{suffix}",
            color=color_for_prn(prn),
            linewidth=2.0 if spoof else 1.25,
            alpha=0.95 if spoof else 0.8,
            marker="o" if spoof else None,
            markersize=2.4,
            markevery=3,
        )


def finish_axis(ax: plt.Axes, *, ylabel: str, ylim: tuple[float, float]) -> None:
    ax.set_xlabel("Time index")
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper center", ncol=4, frameon=True)


def save_figure(
    fig: plt.Figure,
    stem: str,
    *,
    aliases: Sequence[str] = (),
) -> None:
    for output_stem in (stem, *aliases):
        fig.savefig(FIGURE_DIR / f"{output_stem}.png", facecolor="white")
        fig.savefig(FIGURE_DIR / f"{output_stem}.pdf", facecolor="white")
    plt.close(fig)


def make_figures(
    fractional_dd: pd.DataFrame,
    best_m_aoa: pd.DataFrame,
    ephemeris_aoa: pd.DataFrame,
    comparable_prns: Sequence[int],
) -> None:
    configure_plot_style()

    fractional_prns = [
        prn
        for prn in sorted(int(column) for column in fractional_dd.columns)
        if prn != REFERENCE_PRN and fractional_dd[prn].notna().any()
    ]
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    draw_tracks(ax, fractional_dd, fractional_prns)
    finish_axis(ax, ylabel="Fractional DD (cycles)", ylim=(-0.5, 0.5))
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=4,
        frameon=True,
    )
    fig.tight_layout()
    save_figure(
        fig,
        "figure_01_fractional_double_difference_mixed",
        aliases=("fraction_dd_ref_prn23",),
    )

    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    draw_tracks(ax, ephemeris_aoa, comparable_prns)
    finish_axis(ax, ylabel=r"$\alpha_i$ (degree)", ylim=(0.0, 180.0))
    fig.tight_layout()
    save_figure(fig, "figure_02_alpha_i_ephemeris_mixed")

    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    draw_tracks(ax, best_m_aoa, comparable_prns)
    finish_axis(ax, ylabel=r"$\alpha_i$ (degree)", ylim=(0.0, 180.0))
    fig.tight_layout()
    save_figure(fig, "figure_03_alpha_i_proposed_best_m_mixed")

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), sharex=True, sharey=True)
    draw_tracks(axes[0], ephemeris_aoa, comparable_prns)
    axes[0].set_xlabel("Time index")
    axes[0].set_ylabel(r"$\alpha_i$ (degree)")
    axes[0].grid(True, alpha=0.25)
    draw_tracks(axes[1], best_m_aoa, comparable_prns)
    axes[1].set_xlabel("Time index")
    axes[1].grid(True, alpha=0.25)
    axes[0].set_ylim(0.0, 180.0)
    handles, labels = axes[1].get_legend_handles_labels()
    for axis in axes:
        legend = axis.get_legend()
        if legend is not None:
            legend.remove()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=5,
        frameon=True,
    )
    fig.tight_layout(rect=(0.0, 0.13, 1.0, 1.0))
    save_figure(fig, "figure_04_alpha_i_side_by_side_best_m")


def build_long_aligned_data(
    best_m_aoa: pd.DataFrame,
    ephemeris_aoa: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    all_prns = sorted(set(best_m_aoa.columns) | set(ephemeris_aoa.columns))
    for prn in all_prns:
        frame = pd.concat(
            [
                best_m_aoa.get(prn, pd.Series(dtype=float)).rename(
                    "dd_aoa_best_m_deg"
                ),
                ephemeris_aoa.get(prn, pd.Series(dtype=float)).rename(
                    "ephemeris_aoa_rx1_deg"
                ),
            ],
            axis=1,
        )
        frame.index.name = "epoch_index"
        frame = frame.reset_index()
        frame.insert(1, "prn", prn)
        if prn == REFERENCE_PRN:
            known_class = "Reference authentic"
        elif prn in SPOOF_PRNS:
            known_class = "Spoof"
        else:
            known_class = "Authentic"
        frame.insert(2, "known_class", known_class)
        frame["abs_diff_best_m_vs_ephemeris_deg"] = (
            frame["dd_aoa_best_m_deg"] - frame["ephemeris_aoa_rx1_deg"]
        ).abs()
        rows.append(frame)
    return pd.concat(rows, ignore_index=True).sort_values(["epoch_index", "prn"])


def average_row(numeric_table: pd.DataFrame, threshold_deg: float) -> pd.Series:
    return numeric_table.loc[
        numeric_table["prn"].eq("Average")
        & numeric_table["threshold_deg"].eq(threshold_deg)
    ].iloc[0]


def main() -> None:
    for directory in (DATA_DIR, FIGURE_DIR, TABLE_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    stale_paths = [
        DATA_DIR / "alpha_i_dd_method_dp_deg.csv",
        DATA_DIR / "alpha_i_dd_method_notebook_best_m_deg.csv",
        DATA_DIR / "best_m_branch_by_prn.csv",
        DATA_DIR / "spoof_cluster_summary.csv",
        DATA_DIR / "spoof_pairwise_convergence.csv",
        FIGURE_DIR / "figure_03_alpha_i_dd_method_dp_mixed.png",
        FIGURE_DIR / "figure_03_alpha_i_dd_method_dp_mixed.pdf",
        FIGURE_DIR / "figure_04_alpha_i_side_by_side_paper_dp.png",
        FIGURE_DIR / "figure_04_alpha_i_side_by_side_paper_dp.pdf",
        FIGURE_DIR / "figure_05_alpha_i_notebook_best_m_diagnostic.png",
        FIGURE_DIR / "figure_05_alpha_i_notebook_best_m_diagnostic.pdf",
    ]
    for stem in (
        "table_1_like_authentic_dp_thresholds",
        "table_1_like_authentic_notebook_best_m_thresholds",
        "table_2_like_spoof_dp_10deg",
        "table_3_mixed_decisions_dp_10deg",
    ):
        stale_paths.extend(
            [
                TABLE_DIR / f"{stem}.csv",
                TABLE_DIR / f"{stem}_numeric.csv",
                TABLE_DIR / f"{stem}.md",
                TABLE_DIR / f"{stem}.tex",
            ]
        )
    for path in stale_paths:
        path.unlink(missing_ok=True)

    run_summary = pd.read_csv(SOURCE_DIR / "paper_scenario_run_summary.csv")
    reference_alpha = pd.read_csv(SOURCE_DIR / "paper_reference_alpha_by_epoch.csv")
    selected_epochs = pd.read_csv(SOURCE_DIR / "epochs.csv")
    fractional_dd = read_wide_csv(
        "paper_all_satellite_fractional_dd_cycles.csv"
    ).drop(columns=[REFERENCE_PRN], errors="ignore")
    best_m_aoa = read_wide_csv("paper_all_satellite_dd_method_aoa_best_m_deg.csv")
    ephemeris_aoa = read_wide_csv("paper_all_satellite_ephemeris_aoa_rx1_deg.csv")

    source_reference_prn = int(run_summary.loc[0, "reference_svid"])
    source_spoof_prns = tuple(
        sorted(int(value) for value in ast.literal_eval(run_summary.loc[0, "spoof_svids"]))
    )
    if source_reference_prn != REFERENCE_PRN:
        raise ValueError(
            f"Expected reference PRN {REFERENCE_PRN}, got {source_reference_prn}."
        )
    if source_spoof_prns != tuple(sorted(SPOOF_PRNS)):
        raise ValueError(f"Expected spoof PRNs {SPOOF_PRNS}, got {source_spoof_prns}.")

    comparable_prns = [
        prn
        for prn in sorted(set(best_m_aoa.columns) & set(ephemeris_aoa.columns))
        if prn != REFERENCE_PRN and ephemeris_aoa[prn].notna().any()
    ]
    authentic_prns = [prn for prn in comparable_prns if prn not in SPOOF_PRNS]
    spoof_prns = [prn for prn in comparable_prns if prn in SPOOF_PRNS]
    omitted_no_ephemeris = [
        prn
        for prn in sorted(set(ephemeris_aoa.columns) - {REFERENCE_PRN})
        if prn not in comparable_prns
    ]

    if spoof_prns != list(SPOOF_PRNS):
        raise ValueError(f"Comparable spoof PRNs are {spoof_prns}, expected {SPOOF_PRNS}.")

    authentic_best_m_display, authentic_best_m_numeric = build_threshold_table(
        best_m_aoa, ephemeris_aoa, authentic_prns, THRESHOLDS_DEG
    )
    spoof_best_m_display, spoof_best_m_numeric = build_threshold_table(
        best_m_aoa, ephemeris_aoa, spoof_prns, (10.0,)
    )
    mixed_display, mixed_numeric = build_mixed_decision_table(
        best_m_aoa,
        ephemeris_aoa,
        authentic_prns,
        spoof_prns,
        threshold_deg=10.0,
    )

    write_table_bundle(
        "table_1_like_authentic_best_m_thresholds",
        authentic_best_m_display,
        authentic_best_m_numeric,
        caption=(
            "AoA comparison between the ephemeris-based and proposed best-m "
            "methods for authentic PRNs under mixed conditions."
        ),
        label="tab:mixed-authentic-best-m",
    )
    write_table_bundle(
        "table_2_like_spoof_best_m_10deg",
        spoof_best_m_display,
        spoof_best_m_numeric,
        caption=(
            "AoA comparison between the ephemeris-based and proposed best-m "
            "methods for spoof PRNs under mixed conditions."
        ),
        label="tab:mixed-spoof-best-m",
    )
    write_table_bundle(
        "table_3_mixed_decisions_best_m_10deg",
        mixed_display,
        mixed_numeric,
        caption=(
            "Per-epoch decisions in the mixed dataset using the proposed best-m "
            "method and a 10 degree AoA difference threshold."
        ),
        label="tab:mixed-decisions-best-m",
    )

    best_m_aoa.to_csv(DATA_DIR / "alpha_i_dd_method_best_m_deg.csv")
    ephemeris_aoa.to_csv(DATA_DIR / "alpha_i_ephemeris_rx1_deg.csv")
    fractional_dd.to_csv(DATA_DIR / "fractional_double_difference_cycles.csv")
    analysis_epoch_count = int(run_summary.loc[0, "analysis_epoch_count"])
    source_epoch_start = int(run_summary.loc[0, "source_epoch_start"])
    reference_alpha_window = reference_alpha.loc[
        reference_alpha["epoch_index"].between(
            source_epoch_start,
            source_epoch_start + analysis_epoch_count - 1,
        )
    ].copy()
    reference_alpha_window.to_csv(
        DATA_DIR / "reference_prn_23_alpha_selected_window.csv", index=False
    )
    build_long_aligned_data(best_m_aoa, ephemeris_aoa).to_csv(
        DATA_DIR / "alpha_i_aligned_long.csv", index=False
    )

    make_figures(
        fractional_dd,
        best_m_aoa,
        ephemeris_aoa,
        comparable_prns,
    )

    reference_alpha_deg = float(run_summary.loc[0, "reference_alpha_deg"])
    window_reference_alpha_deg = float(
        reference_alpha_window["reference_alpha_deg"].median()
    )
    if not np.isclose(reference_alpha_deg, window_reference_alpha_deg, atol=1e-9):
        raise ValueError(
            "The run-summary reference alpha does not match the selected reference "
            f"window ({reference_alpha_deg} vs {window_reference_alpha_deg})."
        )
    spoof_valid_epoch_count = int(
        average_row(spoof_best_m_numeric, 10.0)["valid_epoch_count"]
        / len(spoof_prns)
    )
    authentic_best_m_10 = average_row(authentic_best_m_numeric, 10.0)
    spoof_best_m_10 = average_row(spoof_best_m_numeric, 10.0)
    mixed_10 = mixed_numeric.loc[mixed_numeric["prn"].eq("Average")].iloc[0]
    selected_time_span_s = float(
        selected_epochs["rcvTow_rx1"].max() - selected_epochs["rcvTow_rx1"].min()
    )
    synchronized_log_epoch_count = int(len(reference_alpha))
    synchronized_log_time_span_s = float(
        reference_alpha["rcvTow_rx1"].max()
        - reference_alpha["rcvTow_rx1"].min()
    )
    first_spoof_pair = aligned_pair(best_m_aoa, ephemeris_aoa, spoof_prns[0])
    paired_timestamps = selected_epochs.set_index("epoch_index").loc[
        first_spoof_pair.index, "rcvTow_rx1"
    ]
    paired_time_span_s = float(paired_timestamps.max() - paired_timestamps.min())

    metadata = {
        "dataset": str(run_summary.loc[0, "dataset"]),
        "source_notebook": "AOA_Paper.ipynb",
        "reference_prn": REFERENCE_PRN,
        "reference_alpha_deg": reference_alpha_deg,
        "spoof_prns": list(SPOOF_PRNS),
        "authentic_comparable_prns": authentic_prns,
        "omitted_prns_without_matched_ephemeris": omitted_no_ephemeris,
        "analysis_epoch_count": analysis_epoch_count,
        "analysis_window_time_span_s": selected_time_span_s,
        "paired_ephemeris_epoch_count_per_spoof_prn": spoof_valid_epoch_count,
        "paired_ephemeris_time_span_s": paired_time_span_s,
        "synchronized_log_epoch_count": synchronized_log_epoch_count,
        "synchronized_log_time_span_s": synchronized_log_time_span_s,
        "receiver1_raw_log_time_span_s": 498.0,
        "receiver2_raw_log_time_span_s": 488.0,
        "proposed_estimator": (
            "notebook best-m policy: authentic PRNs use ephemeris-selected "
            "constant best-m; known spoof PRNs retain DP"
        ),
        "authentic_best_m_within_10deg_percentage": float(
            authentic_best_m_10["percentage"]
        ),
        "spoof_best_m_identical_within_10deg_percentage": float(
            spoof_best_m_10["percentage"]
        ),
        "mixed_best_m_correct_decision_percentage": float(
            mixed_10["correct_percentage"]
        ),
    }
    (OUTPUT_DIR / "experiment_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    captions = f"""# Caption-ready text

## Figure 1 - Fractional double difference

Fractional double-difference carrier-phase measurements for all valid spoof and authentic PRNs in {analysis_epoch_count} consecutive epochs. Spoof PRNs 6, 9, 11, and 20 are highlighted. Satellite PRN {REFERENCE_PRN} is selected as a reference and is therefore not plotted as a DD track.

## Figure 2 - Ephemeris-based AoA

Values of $\\alpha_i$ computed by ephemeris-based information over the selected {analysis_epoch_count} consecutive epochs.

## Figure 3 - Proposed best-m AoA

Values of $\\alpha_i$ computed by the proposed method using the best-m branch in {analysis_epoch_count} consecutive epochs. Satellite PRN {REFERENCE_PRN} is selected as a reference.

## Figure 4 - Side-by-side comparison

Values of $\\alpha_i$ computed by ephemeris-based information (left) and by the proposed best-m method (right) over the selected {analysis_epoch_count} consecutive epochs. Satellite PRN {REFERENCE_PRN} is selected as a reference.

## Table 1-like - Authentic PRNs

AoA comparison between the ephemeris-based and proposed best-m methods for authentic PRNs in the mixed-tracking experiment. Counts use only epochs with both AoA values available. The reference PRN {REFERENCE_PRN} is excluded because its DD-method AoA is anchored by construction. PRN {', '.join(str(prn) for prn in omitted_no_ephemeris)} is omitted because no matched Receiver 1 ephemeris AoA is available in the selected window.

## Table 2-like - Spoof PRNs

AoA comparison between the ephemeris-based and proposed best-m methods for spoof PRNs 6, 9, 11, and 20 under mixed conditions, using a 10 degree allowable difference. Each spoof PRN has {spoof_valid_epoch_count} epochs with both AoA values available.
"""
    (OUTPUT_DIR / "captions.md").write_text(captions, encoding="utf-8")

    readme = f"""# Mixed-case AoA manuscript package

This folder extracts the mixed-tracking results from `AOA_Paper.ipynb` in the same form used for the manuscript's clean and full-spoof sections. The proposed-method figures and tables exactly use `scenario.best_m_all_satellite_aoa_deg`: authentic PRNs use ephemeris-selected best-m, while known spoof PRNs 6, 9, 11, and 20 retain the DP branch so their common-source collapse is preserved. Plot titles are intentionally omitted; copy-ready captions are in `captions.md`.

## Experiment identity

- Dataset: `{run_summary.loc[0, 'dataset']}`
- Receiver logs: `dataset/sp-partial-19-06/18-06/COM3___9600_260618_095042.ubx` and `COM4___9600_260618_095054.ubx`
- Double-difference reference: authentic **PRN {REFERENCE_PRN}**
- Reference angle: median $\\alpha_r={reference_alpha_deg:.6f}^\\circ$ from Receiver 1 NAV-SAT
- Spoof PRNs: **6, 9, 11, 20**
- Selected DD window: **{analysis_epoch_count} one-second epochs spanning {selected_time_span_s:g} seconds**
- Paired DD/ephemeris observations for each spoof PRN: **{spoof_valid_epoch_count} epochs spanning {paired_time_span_s:g} seconds**
- Complete synchronized log overlap: **{synchronized_log_epoch_count} epochs spanning {synchronized_log_time_span_s:g} seconds**
- Comparable authentic PRNs: **{', '.join(str(prn) for prn in authentic_prns)}**
- PRN {', '.join(str(prn) for prn in omitted_no_ephemeris)} is present in DD output but omitted from AoA comparison because Receiver 1 has no matched NAV-SAT AoA in this window.

## Paper-ready files

- `figures/figure_01_fractional_double_difference_mixed.*`
- `figures/figure_02_alpha_i_ephemeris_mixed.*`
- `figures/figure_03_alpha_i_proposed_best_m_mixed.*`
- `figures/figure_04_alpha_i_side_by_side_best_m.*`
- `tables/table_1_like_authentic_best_m_thresholds.*`
- `tables/table_2_like_spoof_best_m_10deg.*`
- `captions.md`

Every figure is provided as a 300-dpi PNG and a vector PDF. Every table is provided as display CSV, numeric CSV, Markdown, and copy-ready LaTeX.

## Best-m comparison result

- At the manuscript's 10 degree comparison threshold, ephemeris and best-m AoA are identical for **{format_percentage(float(spoof_best_m_10['percentage']))}** of spoof observations ({int(spoof_best_m_10['identical_epoch_count'])}/{int(spoof_best_m_10['valid_epoch_count'])}).
- For authentic PRNs, best-m AoA is within 10 degrees for **{format_percentage(float(authentic_best_m_10['percentage']))}** of paired observations ({int(authentic_best_m_10['identical_epoch_count'])}/{int(authentic_best_m_10['valid_epoch_count'])}).
- The overall per-epoch mixed decision accuracy at 10 degrees is **{format_percentage(float(mixed_10['correct_percentage']))}** ({int(mixed_10['correct_epoch_count'])}/{int(mixed_10['valid_epoch_count'])}).

## Why the plot shows about 30 seconds

The receiver files are not only 30 seconds long. Receiver 1 spans 498 seconds, Receiver 2 spans 488 seconds, and their synchronized overlap spans {synchronized_log_time_span_s:g} seconds. The notebook selected the longest continuous segment in which all four requested spoof PRNs were simultaneously valid: {analysis_epoch_count} samples at 1 Hz, from time index 0 to {analysis_epoch_count - 1}, which spans {selected_time_span_s:g} seconds. The threshold table has only {spoof_valid_epoch_count} paired samples per spoof PRN because ephemeris AoA is missing for the first four selected epochs; those paired samples span {paired_time_span_s:g} seconds.

## Best-m interpretation

This is the notebook's hybrid best-m policy. Authentic PRNs use the constant branch selected against ephemeris AoA. Known spoof PRNs 6, 9, 11, and 20 keep their DP branch, which preserves their common-source collapse near 101 degrees. Therefore the authentic rows are post-branch-resolution comparisons rather than ephemeris-independent detection accuracy measurements.

## Rebuild

Run from the repository root:

```bash
uv run python outputs/aoa_paper/sp-partial-19-06/18-06/manuscript_mixed_case/build_mixed_case_package.py
```
"""
    (OUTPUT_DIR / "README.md").write_text(readme, encoding="utf-8")

    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
