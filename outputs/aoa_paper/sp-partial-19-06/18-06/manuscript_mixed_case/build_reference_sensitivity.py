#!/usr/bin/env python3
"""Build paper-ready mixed-case artifacts for DD reference sensitivity.

The experiment holds the receiver logs, selected 34-epoch window, target PRNs,
baseline convention, and branch-selection policy fixed while changing only the
double-difference reference satellite.  Two labelled-authentic references
(PRN 23 and PRN 12) and two known-spoof references (PRN 9 and PRN 11) are used.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd


OUTPUT_DIR = Path(__file__).resolve().parent
REPO_ROOT = OUTPUT_DIR.parents[4]
sys.path.insert(0, str(REPO_ROOT))

from aoa_pipeline import DatasetConfig, PaperScenarioResult, run_paper_scenario  # noqa: E402
from build_mixed_case_package import (  # noqa: E402
    color_for_prn,
    configure_plot_style,
    format_percentage,
    save_figure,
    write_table_bundle,
)


DATA_DIR = OUTPUT_DIR / "data"
FIGURE_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"

SPOOF_PRNS = (6, 9, 11, 20)
AUTHENTIC_PRNS = (10, 12, 15, 18, 23, 24, 32)
TARGET_PRNS = (*SPOOF_PRNS, *AUTHENTIC_PRNS)
THRESHOLD_DEG = 10.0
AUTHENTIC_CLASS = "Authentic"
SPOOF_CLASS = "Spoof"
REFERENCE_SECTION_START = "<!-- reference-sensitivity:start -->"
REFERENCE_SECTION_END = "<!-- reference-sensitivity:end -->"


@dataclass(frozen=True)
class ReferenceCase:
    prn: int
    known_class: str


REFERENCE_CASES = (
    ReferenceCase(23, AUTHENTIC_CLASS),
    ReferenceCase(12, AUTHENTIC_CLASS),
    ReferenceCase(9, SPOOF_CLASS),
    ReferenceCase(11, SPOOF_CLASS),
)
REFERENCE_PRNS = tuple(case.prn for case in REFERENCE_CASES)
COMMON_AUTHENTIC_COMPARISON_PRNS = tuple(
    prn for prn in AUTHENTIC_PRNS if prn not in REFERENCE_PRNS
)
COMMON_SPOOF_COMPARISON_PRNS = tuple(
    prn for prn in SPOOF_PRNS if prn not in REFERENCE_PRNS
)


def build_config(reference_prn: int) -> DatasetConfig:
    return DatasetConfig(
        name=f"_reference_sensitivity/ref_{reference_prn}",
        receiver1_path=REPO_ROOT
        / "dataset/sp-partial-19-06/18-06/COM3___9600_260618_095042.ubx",
        receiver2_path=REPO_ROOT
        / "dataset/sp-partial-19-06/18-06/COM4___9600_260618_095054.ubx",
        alpha_r_deg=None,
        reference_svid=reference_prn,
        selected_svids=SPOOF_PRNS,
        excluded_reference_svids=SPOOF_PRNS,
        reference_required_svids=SPOOF_PRNS,
        start_index=0,
        length=500,
        sync_tow_decimals=0,
        use_navpvt_time_correction=True,
        single_difference_order="rx2-rx1",
        navsat_azimuth_bias_deg=0.0,
        output_root=REPO_ROOT / "outputs/aoa_paper",
    )


def run_cases() -> dict[int, PaperScenarioResult]:
    results: dict[int, PaperScenarioResult] = {}
    for case in REFERENCE_CASES:
        results[case.prn] = run_paper_scenario(
            build_config(case.prn),
            partial_spoof_enabled=True,
            spoof_svids=SPOOF_PRNS,
            convergence_threshold_deg=5.0,
            save_outputs=False,
        )
    return results


def reference_alpha_deg(result: PaperScenarioResult) -> float:
    return float(result.run_summary.loc[0, "reference_alpha_deg"])


def verify_cases(results: dict[int, PaperScenarioResult]) -> None:
    expected = {case.prn for case in REFERENCE_CASES}
    if set(results) != expected:
        raise ValueError(f"Reference cases are {sorted(results)}, expected {sorted(expected)}.")

    canonical_tow: np.ndarray | None = None
    canonical_fractional = results[23].all_satellite_result.fractional_dd_cycles
    for case in REFERENCE_CASES:
        result = results[case.prn]
        if result.all_satellite_result.reference_svid != case.prn:
            raise ValueError(
                f"Requested reference PRN {case.prn}, got "
                f"{result.all_satellite_result.reference_svid}."
            )
        tow = result.all_satellite_result.epochs["rcvTow_rx1"].to_numpy(float)
        if len(tow) != 34 or not np.isclose(tow[-1] - tow[0], 33.0):
            raise ValueError(
                f"PRN {case.prn} uses {len(tow)} epochs spanning {tow[-1] - tow[0]:g} s."
            )
        if canonical_tow is None:
            canonical_tow = tow
        elif not np.array_equal(tow, canonical_tow):
            raise ValueError(f"PRN {case.prn} does not use the canonical time window.")

        raw = result.all_satellite_dd_method_aoa_deg
        reference_track = raw[case.prn].dropna()
        if reference_track.empty or not np.allclose(
            reference_track.to_numpy(float), reference_alpha_deg(result), atol=1e-12
        ):
            raise ValueError(f"PRN {case.prn} is not fixed to its reference-angle anchor.")

        fractional = result.all_satellite_result.fractional_dd_cycles
        common_index = canonical_fractional.index.intersection(fractional.index)
        common_columns = canonical_fractional.columns.intersection(fractional.columns)
        predicted = canonical_fractional.loc[common_index, common_columns].subtract(
            canonical_fractional.loc[common_index, case.prn], axis=0
        )
        predicted = predicted - predicted.round()
        residual = (
            predicted - fractional.loc[common_index, common_columns]
        ).abs().stack().dropna()
        if residual.empty or float(residual.max()) > 1e-12:
            raise ValueError(
                f"Fractional-DD reference transform failed for PRN {case.prn}."
            )


def target_class(prn: int) -> str:
    return SPOOF_CLASS if prn in SPOOF_PRNS else AUTHENTIC_CLASS


def build_aligned_long(results: dict[int, PaperScenarioResult]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    class_by_reference = {case.prn: case.known_class for case in REFERENCE_CASES}
    for reference_prn, result in results.items():
        raw = result.all_satellite_dd_method_aoa_deg
        hybrid = result.best_m_all_satellite_aoa_deg
        ephemeris = result.ephemeris_aoa_rx1_deg
        fractional = result.all_satellite_result.fractional_dd_cycles
        branch = result.all_satellite_result.selected_branch
        epochs = result.all_satellite_result.epochs.set_index("epoch_index")
        alpha_r = reference_alpha_deg(result)

        for epoch_index in raw.index:
            for prn in TARGET_PRNS:
                rows.append(
                    {
                        "reference_prn": reference_prn,
                        "reference_class": class_by_reference[reference_prn],
                        "reference_alpha_deg": alpha_r,
                        "epoch_index": int(epoch_index),
                        "rcvTow_rx1": float(epochs.loc[epoch_index, "rcvTow_rx1"]),
                        "target_prn": prn,
                        "target_class": target_class(prn),
                        "is_reference": prn == reference_prn,
                        "fractional_dd_cycles": fractional.at[epoch_index, prn]
                        if prn in fractional.columns
                        else np.nan,
                        "selected_branch": branch.at[epoch_index, prn]
                        if prn in branch.columns
                        else np.nan,
                        "raw_dp_aoa_deg": raw.at[epoch_index, prn]
                        if prn in raw.columns
                        else np.nan,
                        "hybrid_best_m_aoa_deg": hybrid.at[epoch_index, prn]
                        if prn in hybrid.columns
                        else np.nan,
                        "ephemeris_aoa_deg": ephemeris.at[epoch_index, prn]
                        if prn in ephemeris.columns
                        else np.nan,
                    }
                )
    frame = pd.DataFrame(rows)
    frame["raw_abs_error_deg"] = (
        frame["raw_dp_aoa_deg"] - frame["ephemeris_aoa_deg"]
    ).abs()
    frame["hybrid_abs_error_deg"] = (
        frame["hybrid_best_m_aoa_deg"] - frame["ephemeris_aoa_deg"]
    ).abs()
    return frame.sort_values(
        ["reference_prn", "epoch_index", "target_prn"]
    ).reset_index(drop=True)


def valid_group(
    long: pd.DataFrame,
    reference_prn: int,
    known_class: str,
    estimator_column: str,
) -> pd.DataFrame:
    return long.loc[
        long["reference_prn"].eq(reference_prn)
        & long["target_class"].eq(known_class)
        & ~long["target_prn"].isin(REFERENCE_PRNS)
    ].dropna(subset=[estimator_column, "ephemeris_aoa_deg"])


def build_run_summary(
    results: dict[int, PaperScenarioResult], long: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    class_by_reference = {case.prn: case.known_class for case in REFERENCE_CASES}
    for reference_prn, result in results.items():
        run = long.loc[long["reference_prn"].eq(reference_prn)]
        spoof_tracks = run.loc[
            run["target_prn"].isin(COMMON_SPOOF_COMPARISON_PRNS)
        ].dropna(subset=["raw_dp_aoa_deg"])
        epoch_spread = spoof_tracks.groupby("epoch_index")["raw_dp_aoa_deg"].agg(
            lambda values: float(values.max() - values.min())
            if len(values) > 1
            else np.nan
        )
        authentic_raw = valid_group(
            long, reference_prn, AUTHENTIC_CLASS, "raw_dp_aoa_deg"
        )
        authentic_hybrid = valid_group(
            long, reference_prn, AUTHENTIC_CLASS, "hybrid_best_m_aoa_deg"
        )
        spoof_hybrid = valid_group(
            long, reference_prn, SPOOF_CLASS, "hybrid_best_m_aoa_deg"
        )
        rows.append(
            {
                "reference_prn": reference_prn,
                "reference_class": class_by_reference[reference_prn],
                "reference_alpha_deg": reference_alpha_deg(result),
                "epoch_count": int(run["epoch_index"].nunique()),
                "time_span_s": float(run["rcvTow_rx1"].max() - run["rcvTow_rx1"].min()),
                "median_common_spoof_aoa_deg": float(spoof_tracks["raw_dp_aoa_deg"].median()),
                "median_common_spoof_epoch_range_deg": float(epoch_spread.median()),
                "authentic_raw_valid_count": int(len(authentic_raw)),
                "authentic_raw_acceptance_percent": float(
                    100.0 * authentic_raw["raw_abs_error_deg"].le(THRESHOLD_DEG).mean()
                ),
                "authentic_hybrid_valid_count": int(len(authentic_hybrid)),
                "authentic_hybrid_acceptance_percent": float(
                    100.0
                    * authentic_hybrid["hybrid_abs_error_deg"].le(THRESHOLD_DEG).mean()
                ),
                "spoof_hybrid_valid_count": int(len(spoof_hybrid)),
                "spoof_rejection_percent": float(
                    100.0 * spoof_hybrid["hybrid_abs_error_deg"].gt(THRESHOLD_DEG).mean()
                ),
            }
        )
    numeric = pd.DataFrame(rows)
    display = pd.DataFrame(
        {
            "Reference": numeric.apply(
                lambda row: f"PRN {int(row.reference_prn)}", axis=1
            ),
            "Known class": numeric["reference_class"],
            "Anchor": numeric["reference_alpha_deg"].map(lambda value: f"{value:.2f} deg"),
            "Common-spoof AoA": numeric["median_common_spoof_aoa_deg"].map(
                lambda value: f"{value:.2f} deg"
            ),
            "Spoof spread": numeric["median_common_spoof_epoch_range_deg"].map(
                lambda value: f"{value:.2f} deg"
            ),
            "Authentic accepted (raw)": numeric["authentic_raw_acceptance_percent"].map(
                format_percentage
            ),
            "Authentic accepted (hybrid)": numeric[
                "authentic_hybrid_acceptance_percent"
            ].map(format_percentage),
            "Spoof rejected": numeric["spoof_rejection_percent"].map(format_percentage),
        }
    )
    return display, numeric


def build_target_medians(long: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    medians = long.pivot_table(
        index=["target_prn", "target_class"],
        columns="reference_prn",
        values="raw_dp_aoa_deg",
        aggfunc="median",
    )
    medians = medians.reindex(TARGET_PRNS, level="target_prn")
    numeric = medians.reset_index()
    reference_columns = [case.prn for case in REFERENCE_CASES]
    range_values = numeric[reference_columns].copy()
    for reference_prn in reference_columns:
        range_values.loc[numeric["target_prn"].eq(reference_prn), reference_prn] = np.nan
    numeric["reference_range_excluding_self_deg"] = (
        range_values.max(axis=1) - range_values.min(axis=1)
    )
    display_columns: dict[str, pd.Series] = {
        "Target": numeric["target_prn"].map(lambda value: f"PRN {int(value)}"),
        "Known class": numeric["target_class"],
    }
    for reference_prn in reference_columns:
        display_columns[f"Ref PRN {reference_prn}"] = numeric.apply(
            lambda row: (
                f"{row[reference_prn]:.2f} deg"
                + ("*" if int(row["target_prn"]) == reference_prn else "")
            ),
            axis=1,
        )
    display_columns["Range excluding self"] = numeric[
        "reference_range_excluding_self_deg"
    ].map(
        lambda value: f"{value:.2f} deg"
    )
    display = pd.DataFrame(display_columns)
    return display, numeric


def build_pairwise_sensitivity(long: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    for left, right, pair_class in (
        (23, 12, AUTHENTIC_CLASS),
        (9, 11, SPOOF_CLASS),
    ):
        left_frame = long.loc[long["reference_prn"].eq(left), [
            "rcvTow_rx1",
            "target_prn",
            "target_class",
            "raw_dp_aoa_deg",
        ]].rename(columns={"raw_dp_aoa_deg": "left_aoa_deg"})
        right_frame = long.loc[long["reference_prn"].eq(right), [
            "rcvTow_rx1",
            "target_prn",
            "raw_dp_aoa_deg",
        ]].rename(columns={"raw_dp_aoa_deg": "right_aoa_deg"})
        paired = left_frame.merge(right_frame, on=["rcvTow_rx1", "target_prn"])
        paired = paired.loc[~paired["target_prn"].isin((left, right))].dropna(
            subset=["left_aoa_deg", "right_aoa_deg"]
        )
        paired["abs_change_deg"] = (
            paired["left_aoa_deg"] - paired["right_aoa_deg"]
        ).abs()
        for target_group in (AUTHENTIC_CLASS, SPOOF_CLASS):
            values = paired.loc[
                paired["target_class"].eq(target_group), "abs_change_deg"
            ]
            rows.append(
                {
                    "reference_pair": f"PRN {left} vs PRN {right}",
                    "reference_pair_class": pair_class,
                    "target_group": target_group,
                    "matched_observation_count": int(len(values)),
                    "median_abs_change_deg": float(values.median()),
                    "p95_abs_change_deg": float(values.quantile(0.95)),
                    "max_abs_change_deg": float(values.max()),
                    "within_1deg_percent": float(100.0 * values.le(1.0).mean()),
                    "within_5deg_percent": float(100.0 * values.le(5.0).mean()),
                }
            )
    numeric = pd.DataFrame(rows)
    display = pd.DataFrame(
        {
            "Reference pair": numeric["reference_pair"],
            "Pair class": numeric["reference_pair_class"],
            "Target group": numeric["target_group"],
            "Matched observations": numeric["matched_observation_count"],
            "Median change": numeric["median_abs_change_deg"].map(
                lambda value: f"{value:.2f} deg"
            ),
            "95th percentile": numeric["p95_abs_change_deg"].map(
                lambda value: f"{value:.2f} deg"
            ),
            "Maximum": numeric["max_abs_change_deg"].map(
                lambda value: f"{value:.2f} deg"
            ),
        }
    )
    return display, numeric


def figure_spoof_tracks(
    results: dict[int, PaperScenarioResult], run_numeric: pd.DataFrame
) -> None:
    configure_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.2), sharex=True, sharey=True)
    panel_labels = ("(a)", "(b)", "(c)", "(d)")
    class_by_reference = {case.prn: case.known_class for case in REFERENCE_CASES}

    for ax, panel_label, case in zip(axes.flat, panel_labels, REFERENCE_CASES):
        raw = results[case.prn].all_satellite_dd_method_aoa_deg
        alpha_r = float(
            run_numeric.loc[
                run_numeric["reference_prn"].eq(case.prn), "reference_alpha_deg"
            ].iloc[0]
        )
        ax.axhline(
            alpha_r,
            color="#303030",
            linestyle=":",
            linewidth=1.3,
            alpha=0.8,
            label=r"Reference anchor $\alpha_r$",
        )
        for prn in SPOOF_PRNS:
            is_reference = prn == case.prn
            ax.plot(
                raw.index,
                raw[prn],
                color=color_for_prn(prn),
                linewidth=2.4 if is_reference else 1.8,
                linestyle="--" if is_reference else "-",
                marker="s" if is_reference else "o",
                markersize=3.0 if is_reference else 2.2,
                markevery=3,
                label=f"PRN {prn}",
            )
        ax.set_title(
            f"{panel_label} {class_by_reference[case.prn]} reference PRN {case.prn} "
            f"($\\alpha_r$={alpha_r:.2f}$^\\circ$)"
        )
        ax.set_ylim(40.0, 130.0)
        ax.grid(True, alpha=0.25)

    for ax in axes[1, :]:
        ax.set_xlabel("Time index")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"Raw DP DD-method $\alpha_i$ (degree)")
    legend_handles = [
        Line2D([0], [0], color=color_for_prn(prn), linewidth=2.0, marker="o", markersize=3)
        for prn in SPOOF_PRNS
    ] + [
        Line2D(
            [0],
            [0],
            color="#303030",
            linewidth=2.2,
            linestyle="--",
            marker="s",
            markersize=3,
        ),
        Line2D([0], [0], color="#303030", linewidth=1.3, linestyle=":"),
    ]
    legend_labels = [f"PRN {prn}" for prn in SPOOF_PRNS] + [
        "Spoof PRN used as reference",
        r"Reference anchor $\alpha_r$",
    ]
    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=6,
        frameon=True,
    )
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))
    save_figure(fig, "figure_05_reference_choice_spoof_tracks")


def figure_target_heatmap(target_numeric: pd.DataFrame) -> None:
    configure_plot_style()
    reference_prns = [case.prn for case in REFERENCE_CASES]
    values = target_numeric.set_index("target_prn").loc[TARGET_PRNS, reference_prns]
    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    image = ax.imshow(values.to_numpy(float), aspect="auto", cmap="viridis", vmin=40, vmax=130)
    ax.set_xticks(range(len(reference_prns)))
    ax.set_xticklabels(
        [
            f"PRN {case.prn}\n({case.known_class.lower()})"
            for case in REFERENCE_CASES
        ]
    )
    ax.set_yticks(range(len(TARGET_PRNS)))
    ax.set_yticklabels(
        [f"PRN {prn} ({'S' if prn in SPOOF_PRNS else 'A'})" for prn in TARGET_PRNS]
    )
    ax.set_xlabel("Double-difference reference")
    ax.set_ylabel("Target satellite")

    for row, prn in enumerate(TARGET_PRNS):
        for column, reference_prn in enumerate(reference_prns):
            value = float(values.loc[prn, reference_prn])
            text_color = "white" if value < 70 or value > 112 else "black"
            ax.text(column, row, f"{value:.1f}", ha="center", va="center", color=text_color)
            if prn == reference_prn:
                ax.add_patch(
                    Rectangle(
                        (column - 0.48, row - 0.48),
                        0.96,
                        0.96,
                        fill=False,
                        edgecolor="#d62728",
                        linewidth=2.0,
                    )
                )
    ax.axhline(len(SPOOF_PRNS) - 0.5, color="white", linewidth=2.0)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.03)
    colorbar.set_label("Median raw DP DD-method AoA (degree)")
    fig.tight_layout()
    save_figure(fig, "figure_06_reference_choice_target_median_heatmap")


def figure_decision_rates(run_numeric: pd.DataFrame) -> None:
    configure_plot_style()
    ordered = run_numeric.set_index("reference_prn").loc[
        [case.prn for case in REFERENCE_CASES]
    ]
    x = np.arange(len(ordered))
    width = 0.24
    fig, ax = plt.subplots(figsize=(8.2, 4.5))
    series = (
        ("Authentic accepted (raw DP)", "authentic_raw_acceptance_percent", "#4c78a8"),
        ("Authentic accepted (hybrid)", "authentic_hybrid_acceptance_percent", "#59a14f"),
        ("Spoof rejected", "spoof_rejection_percent", "#e15759"),
    )
    for offset, (label, column, color) in zip((-width, 0.0, width), series):
        values = ordered[column].to_numpy(float)
        bars = ax.bar(x + offset, values, width, label=label, color=color)
        ax.bar_label(bars, labels=[f"{value:.1f}" for value in values], padding=2, fontsize=8)
    ax.axhline(100.0, color="#404040", linewidth=0.8, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"PRN {case.prn}\n{case.known_class} ref." for case in REFERENCE_CASES]
    )
    ax.set_ylabel("Rate at 10 degree threshold (%)")
    ax.set_ylim(0.0, 108.0)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=3, frameon=True)
    fig.tight_layout()
    save_figure(fig, "figure_07_reference_choice_detection_rates")


def replace_marked_section(
    path: Path,
    content: str,
    *,
    initial_text: str = "",
    start_marker: str = REFERENCE_SECTION_START,
    end_marker: str = REFERENCE_SECTION_END,
) -> None:
    if not start_marker or not end_marker or start_marker == end_marker:
        raise ValueError("Section markers must be non-empty and distinct")
    existing = path.read_text(encoding="utf-8") if path.exists() else initial_text
    start_count = existing.count(start_marker)
    end_count = existing.count(end_marker)
    if start_count != end_count or start_count > 1:
        raise ValueError(
            f"Malformed marked section in {path}: "
            f"found {start_count} start marker(s) and {end_count} end marker(s)"
        )
    if start_count == 1 and existing.index(start_marker) > existing.index(end_marker):
        raise ValueError(f"Malformed marked section in {path}: end marker precedes start")
    block = (
        f"{start_marker}\n"
        f"{content.strip()}\n"
        f"{end_marker}"
    )
    if start_count == 1:
        prefix, remainder = existing.split(start_marker, 1)
        _, suffix = remainder.split(end_marker, 1)
        updated = prefix.rstrip() + "\n\n" + block
        if suffix.strip():
            updated += "\n\n" + suffix.strip()
    else:
        updated = existing.rstrip() + "\n\n" + block
    path.write_text(updated.rstrip() + "\n", encoding="utf-8")


def write_captions() -> None:
    captions = """# Reference-sensitivity caption-ready text

## Figure 5 - Spoof-track AoA under four reference choices

Raw dynamic-programming DD-method AoA tracks for known spoof PRNs 6, 9, 11, and 20 under four double-difference reference choices: authentic PRNs 23 and 12 and spoof PRNs 9 and 11. All panels use the same 34 consecutive epochs. Dotted horizontal lines indicate the receiver-reported reference-angle anchor. A dashed satellite track denotes a spoof PRN that is itself the reference; that self-reference track is anchored by construction.

## Figure 6 - Target-by-reference median AoA

Median raw dynamic-programming DD-method AoA for every comparable target PRN under the four reference choices. S and A identify known spoof and labelled-authentic targets, respectively. Red boxes identify self-reference cells, whose DD-method AoA equals the reference-angle anchor by construction. The horizontal divider separates spoof and authentic target groups.

## Figure 7 - Reference-dependent decision rates

Authentic-satellite acceptance and spoof-satellite rejection rates at the 10 degree DD-versus-ephemeris threshold for each reference choice. Every case uses the same comparison population: authentic PRNs 10, 15, 18, 24, and 32 and spoof PRNs 6 and 20. The raw estimator uses only the dynamic-programming smoothness branch. The hybrid estimator uses the ephemeris-aligned constant branch for labelled-authentic PRNs while retaining the dynamic-programming branch for spoof PRNs.

## Table 4 - Four-reference run summary

Reference-angle anchor, spoof-cluster location and spread, and 10 degree decision rates for the two authentic-reference and two spoof-reference cases. All cases use the same fixed comparison populations, excluding all four reference candidates.

## Table 5 - Per-target median AoA

Median raw dynamic-programming DD-method AoA for each target PRN under all four reference choices. An asterisk marks a self-reference value that equals the reference-angle anchor by construction. The final column reports the range across reference cases after excluding that self-reference cell.

## Table 6 - Pairwise reference sensitivity

Matched-epoch absolute changes in raw dynamic-programming DD-method AoA when switching between the two authentic references and between the two spoof references. Both reference PRNs in a pair are excluded from that pair's target metrics.
"""
    caption_path = OUTPUT_DIR / "reference_sensitivity_captions.md"
    caption_path.write_text(captions, encoding="utf-8")

    main_path = OUTPUT_DIR / "captions.md"
    appendix = captions.replace("# Reference-sensitivity caption-ready text", "## Reference-sensitivity figures and tables")
    replace_marked_section(
        main_path,
        appendix,
        initial_text="# Caption-ready text\n",
    )


def update_readme() -> None:
    readme_path = OUTPUT_DIR / "README.md"
    section = """## Reference-satellite sensitivity package

The mixed-tracking reference-sensitivity experiment repeats the same 34-epoch window with authentic references **PRN 23** and **PRN 12** and spoof references **PRN 9** and **PRN 11**. Raw DP AoA is used to measure reference sensitivity; the hybrid best-m output is retained only for the paper's 10 degree decision-rate comparison. Every case uses the same scored populations: authentic PRNs **10, 15, 18, 24, 32** and spoof PRNs **6, 20**. All four reference candidates are excluded from scoring because a reference's own DD AoA is anchored by construction.

Paper-ready figures:

- `figures/figure_05_reference_choice_spoof_tracks.*`
- `figures/figure_06_reference_choice_target_median_heatmap.*`
- `figures/figure_07_reference_choice_detection_rates.*`

Paper-ready tables:

- `tables/table_4_reference_choice_summary.*`
- `tables/table_5_reference_choice_target_medians.*`
- `tables/table_6_reference_choice_pairwise_sensitivity.*`

Supporting files:

- `data/reference_choice_aligned_long.csv`
- `reference_sensitivity_metadata.json`
- `reference_sensitivity_captions.md`

Rebuild from the repository root:

```bash
uv run python outputs/aoa_paper/sp-partial-19-06/18-06/manuscript_mixed_case/build_reference_sensitivity.py
```
"""
    replace_marked_section(readme_path, section)


def main() -> None:
    for directory in (DATA_DIR, FIGURE_DIR, TABLE_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    results = run_cases()
    verify_cases(results)
    long = build_aligned_long(results)
    run_display, run_numeric = build_run_summary(results, long)
    target_display, target_numeric = build_target_medians(long)
    pair_display, pair_numeric = build_pairwise_sensitivity(long)

    if len(run_numeric) != 4 or len(target_numeric) != len(TARGET_PRNS) or len(pair_numeric) != 4:
        raise ValueError("Reference-sensitivity summary dimensions are incomplete.")

    long.to_csv(DATA_DIR / "reference_choice_aligned_long.csv", index=False)
    write_table_bundle(
        "table_4_reference_choice_summary",
        run_display,
        run_numeric,
        caption=(
            "Reference-angle anchors, spoof-cluster behaviour, and 10 degree "
            "decision rates under four DD reference choices."
        ),
        label="tab:mixed-reference-choice-summary",
    )
    write_table_bundle(
        "table_5_reference_choice_target_medians",
        target_display,
        target_numeric,
        caption=(
            "Median raw DP DD-method AoA for each mixed-dataset target under "
            "four reference choices; an asterisk marks a self-reference value."
        ),
        label="tab:mixed-reference-choice-target-medians",
    )
    write_table_bundle(
        "table_6_reference_choice_pairwise_sensitivity",
        pair_display,
        pair_numeric,
        caption=(
            "Matched-epoch raw DP AoA sensitivity for the authentic-reference "
            "and spoof-reference pairs."
        ),
        label="tab:mixed-reference-choice-pairwise",
    )

    figure_spoof_tracks(results, run_numeric)
    figure_target_heatmap(target_numeric)
    figure_decision_rates(run_numeric)
    write_captions()
    update_readme()

    metadata = {
        "dataset": "sp-partial-19-06/18-06",
        "reference_cases": run_numeric.to_dict(orient="records"),
        "spoof_prns": list(SPOOF_PRNS),
        "authentic_prns": list(AUTHENTIC_PRNS),
        "common_authentic_comparison_prns": list(COMMON_AUTHENTIC_COMPARISON_PRNS),
        "common_spoof_comparison_prns": list(COMMON_SPOOF_COMPARISON_PRNS),
        "analysis_epoch_count": 34,
        "analysis_time_span_s": 33.0,
        "threshold_deg": THRESHOLD_DEG,
        "primary_estimator": "raw DP-selected DD-method AoA",
        "decision_estimator": (
            "raw DP for raw authentic acceptance; hybrid best-m for hybrid "
            "authentic acceptance; spoof PRNs retain raw DP in both"
        ),
        "all_reference_candidates_excluded_from_metrics": True,
        "fractional_dd_reference_transform_verified": True,
    }
    (OUTPUT_DIR / "reference_sensitivity_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
