#!/usr/bin/env python3
"""Build one self-contained mixed-case manuscript package per DD reference."""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import pandas as pd


OUTPUT_DIR = Path(__file__).resolve().parent
REPO_ROOT = OUTPUT_DIR.parents[4]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(OUTPUT_DIR))

from aoa_pipeline import PaperScenarioResult  # noqa: E402
from build_mixed_case_package import (  # noqa: E402
    build_mixed_decision_table,
    build_threshold_table,
    configure_plot_style,
    draw_tracks,
    finish_axis,
    format_percentage,
    markdown_table,
    write_latex_table,
)
from build_reference_sensitivity import (  # noqa: E402
    AUTHENTIC_PRNS,
    REFERENCE_CASES,
    SPOOF_PRNS,
    ReferenceCase,
    replace_marked_section,
    run_cases,
    verify_cases,
)


REFERENCE_CASES_DIR = OUTPUT_DIR / "reference_cases"
THRESHOLDS_DEG = (5.0, 7.0, 10.0)
PACKAGE_SECTION_START = "<!-- reference-case-packages:start -->"
PACKAGE_SECTION_END = "<!-- reference-case-packages:end -->"


def case_slug(case: ReferenceCase) -> str:
    return f"reference_prn_{case.prn}_{case.known_class.lower()}"


def case_directory(case: ReferenceCase) -> Path:
    return REFERENCE_CASES_DIR / case_slug(case)


def composite_rgba_on_white(
    rgba: tuple[float, float, float, float],
) -> tuple[float, ...]:
    alpha = rgba[3]
    return tuple(alpha * channel + (1.0 - alpha) for channel in rgba[:3]) + (1.0,)


def make_transparency_eps_safe(fig: plt.Figure) -> None:
    for artist in fig.findobj():
        get_alpha = getattr(artist, "get_alpha", None)
        alpha = get_alpha() if get_alpha is not None else None
        if alpha is None or alpha == 1.0:
            continue
        if isinstance(artist, Line2D):
            artist.set_color(
                composite_rgba_on_white(mcolors.to_rgba(artist.get_color(), alpha))
            )
            artist.set_alpha(1.0)
        elif isinstance(artist, Patch):
            facecolor = composite_rgba_on_white(artist.get_facecolor())
            edgecolor = composite_rgba_on_white(artist.get_edgecolor())
            artist.set_alpha(1.0)
            artist.set_facecolor(facecolor)
            artist.set_edgecolor(edgecolor)
        else:
            artist.set_alpha(1.0)


def save_case_figure(fig: plt.Figure, directory: Path, stem: str) -> None:
    fig.savefig(directory / f"{stem}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(
        directory / f"{stem}.pdf",
        bbox_inches="tight",
        facecolor="white",
        metadata={"CreationDate": None, "ModDate": None},
    )
    make_transparency_eps_safe(fig)

    previous_source_date_epoch = os.environ.get("SOURCE_DATE_EPOCH")
    os.environ["SOURCE_DATE_EPOCH"] = "0"
    try:
        fig.savefig(
            directory / f"{stem}.eps",
            format="eps",
            bbox_inches="tight",
            facecolor="white",
        )
    finally:
        if previous_source_date_epoch is None:
            os.environ.pop("SOURCE_DATE_EPOCH", None)
        else:
            os.environ["SOURCE_DATE_EPOCH"] = previous_source_date_epoch
    plt.close(fig)


def make_case_figures(
    result: PaperScenarioResult,
    case: ReferenceCase,
    figure_dir: Path,
) -> list[str]:
    configure_plot_style()
    plt.rcParams["ps.fonttype"] = 42
    reference_prn = case.prn
    fractional = (
        result.all_satellite_result.fractional_dd_cycles
        .drop(columns=[reference_prn], errors="ignore")
        .dropna(axis=1, how="all")
        .sort_index(axis=1)
    )
    ephemeris = result.ephemeris_aoa_rx1_deg.sort_index(axis=1)
    best_m = result.best_m_all_satellite_aoa_deg.sort_index(axis=1)
    comparable_prns = [
        prn
        for prn in sorted(set(best_m.columns) & set(ephemeris.columns))
        if prn != reference_prn and ephemeris[prn].notna().any()
    ]
    fractional_prns = [
        int(prn) for prn in fractional.columns if fractional[prn].notna().any()
    ]

    stems: list[str] = []
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    draw_tracks(ax, fractional, fractional_prns)
    finish_axis(ax, ylabel="Fractional DD (cycles)", ylim=(-0.5, 0.5))
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=4,
        frameon=True,
    )
    fig.tight_layout()
    stem = "figure_01_fractional_double_difference_mixed"
    save_case_figure(fig, figure_dir, stem)
    stems.append(stem)

    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    draw_tracks(ax, ephemeris, comparable_prns)
    finish_axis(ax, ylabel=r"$\alpha_i$ (degree)", ylim=(0.0, 180.0))
    fig.tight_layout()
    stem = "figure_02_alpha_i_ephemeris_mixed"
    save_case_figure(fig, figure_dir, stem)
    stems.append(stem)

    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    draw_tracks(ax, best_m, comparable_prns)
    finish_axis(ax, ylabel=r"$\alpha_i$ (degree)", ylim=(0.0, 180.0))
    fig.tight_layout()
    stem = "figure_03_alpha_i_proposed_best_m_mixed"
    save_case_figure(fig, figure_dir, stem)
    stems.append(stem)

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), sharex=True, sharey=True)
    draw_tracks(axes[0], ephemeris, comparable_prns)
    axes[0].set_xlabel("Time index")
    axes[0].set_ylabel(r"$\alpha_i$ (degree)")
    axes[0].grid(True, alpha=0.25)
    draw_tracks(axes[1], best_m, comparable_prns)
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
    stem = "figure_04_alpha_i_side_by_side_best_m"
    save_case_figure(fig, figure_dir, stem)
    stems.append(stem)
    return stems


def write_case_table_bundle(
    table_dir: Path,
    stem: str,
    display: pd.DataFrame,
    numeric: pd.DataFrame,
    *,
    caption: str,
    label: str,
) -> None:
    display.to_csv(table_dir / f"{stem}.csv", index=False)
    numeric.to_csv(table_dir / f"{stem}_numeric.csv", index=False)
    (table_dir / f"{stem}.md").write_text(
        markdown_table(display) + "\n", encoding="utf-8"
    )
    write_latex_table(
        display,
        table_dir / f"{stem}.tex",
        caption=caption,
        label=label,
    )


def average_row(numeric: pd.DataFrame, threshold_deg: float = 10.0) -> pd.Series:
    return numeric.loc[
        numeric["prn"].eq("Average")
        & numeric["threshold_deg"].eq(threshold_deg)
    ].iloc[0]


def build_case_tables(
    result: PaperScenarioResult,
    case: ReferenceCase,
    table_dir: Path,
) -> dict[str, float | int]:
    reference_prn = case.prn
    best_m = result.best_m_all_satellite_aoa_deg
    ephemeris = result.ephemeris_aoa_rx1_deg
    comparable_prns = [
        int(prn)
        for prn in sorted(set(best_m.columns) & set(ephemeris.columns))
        if prn != reference_prn and ephemeris[prn].notna().any()
    ]
    authentic_prns = [prn for prn in comparable_prns if prn not in SPOOF_PRNS]
    spoof_prns = [prn for prn in comparable_prns if prn in SPOOF_PRNS]

    authentic_display, authentic_numeric = build_threshold_table(
        best_m,
        ephemeris,
        authentic_prns,
        THRESHOLDS_DEG,
    )
    spoof_display, spoof_numeric = build_threshold_table(
        best_m,
        ephemeris,
        spoof_prns,
        (10.0,),
    )
    mixed_display, mixed_numeric = build_mixed_decision_table(
        best_m,
        ephemeris,
        authentic_prns,
        spoof_prns,
        threshold_deg=10.0,
    )
    mixed_display = mixed_display.rename(
        columns={
            "Number correct/total valid epochs": "Agreement count/total valid epochs",
            "Correct percentage": "Agreement percentage",
        }
    )
    mixed_numeric = mixed_numeric.rename(
        columns={
            "correct_epoch_count": "agreement_epoch_count",
            "correct_percentage": "agreement_percentage",
        }
    )

    write_case_table_bundle(
        table_dir,
        "table_1_authentic_best_m_thresholds",
        authentic_display,
        authentic_numeric,
        caption=(
            f"Authentic-PRN AoA comparison with PRN {reference_prn} as the "
            "double-difference reference."
        ),
        label=f"tab:mixed-ref-{reference_prn}-authentic",
    )
    write_case_table_bundle(
        table_dir,
        "table_2_spoof_best_m_10deg",
        spoof_display,
        spoof_numeric,
        caption=(
            f"Spoof-PRN AoA comparison with PRN {reference_prn} as the "
            "double-difference reference."
        ),
        label=f"tab:mixed-ref-{reference_prn}-spoof",
    )
    write_case_table_bundle(
        table_dir,
        "table_3_mixed_decisions_best_m_10deg",
        mixed_display,
        mixed_numeric,
        caption=(
            f"Post-selection mixed-label agreement at 10 degrees with PRN "
            f"{reference_prn} as the double-difference reference. Labelled-"
            "authentic branch selection is ephemeris-assisted; this is not an "
            "independent detection-accuracy result."
        ),
        label=f"tab:mixed-ref-{reference_prn}-decisions",
    )

    authentic_10 = average_row(authentic_numeric)
    spoof_10 = average_row(spoof_numeric)
    mixed_10 = mixed_numeric.loc[mixed_numeric["prn"].eq("Average")].iloc[0]
    return {
        "comparable_authentic_prn_count": len(authentic_prns),
        "comparable_spoof_prn_count": len(spoof_prns),
        "authentic_within_10deg_count": int(authentic_10["identical_epoch_count"]),
        "authentic_valid_count": int(authentic_10["valid_epoch_count"]),
        "authentic_within_10deg_percent": float(authentic_10["percentage"]),
        "spoof_within_10deg_count": int(spoof_10["identical_epoch_count"]),
        "spoof_valid_count": int(spoof_10["valid_epoch_count"]),
        "spoof_within_10deg_percent": float(spoof_10["percentage"]),
        "post_selection_agreement_count": int(mixed_10["agreement_epoch_count"]),
        "post_selection_valid_count": int(mixed_10["valid_epoch_count"]),
        "post_selection_agreement_percent": float(mixed_10["agreement_percentage"]),
    }


def build_aligned_data(
    result: PaperScenarioResult,
    case: ReferenceCase,
) -> pd.DataFrame:
    raw = result.all_satellite_dd_method_aoa_deg
    best_m = result.best_m_all_satellite_aoa_deg
    ephemeris = result.ephemeris_aoa_rx1_deg
    rows: list[pd.DataFrame] = []
    for prn in sorted(set(raw.columns) | set(best_m.columns) | set(ephemeris.columns)):
        frame = pd.concat(
            [
                raw.get(prn, pd.Series(dtype=float)).rename("raw_dp_aoa_deg"),
                best_m.get(prn, pd.Series(dtype=float)).rename("hybrid_best_m_aoa_deg"),
                ephemeris.get(prn, pd.Series(dtype=float)).rename("ephemeris_aoa_deg"),
            ],
            axis=1,
        )
        frame.index.name = "epoch_index"
        frame = frame.reset_index()
        frame.insert(1, "target_prn", int(prn))
        frame.insert(2, "target_class", "Spoof" if prn in SPOOF_PRNS else "Authentic")
        frame.insert(3, "is_reference", int(prn) == case.prn)
        frame["raw_abs_error_deg"] = (
            frame["raw_dp_aoa_deg"] - frame["ephemeris_aoa_deg"]
        ).abs()
        frame["hybrid_abs_error_deg"] = (
            frame["hybrid_best_m_aoa_deg"] - frame["ephemeris_aoa_deg"]
        ).abs()
        rows.append(frame)
    return pd.concat(rows, ignore_index=True).sort_values(["epoch_index", "target_prn"])


def write_case_data(
    result: PaperScenarioResult,
    case: ReferenceCase,
    data_dir: Path,
) -> None:
    result.all_satellite_result.fractional_dd_cycles.drop(
        columns=[case.prn], errors="ignore"
    ).to_csv(data_dir / "fractional_double_difference_cycles.csv")
    result.all_satellite_dd_method_aoa_deg.to_csv(
        data_dir / "alpha_i_dd_method_raw_dp_deg.csv"
    )
    result.best_m_all_satellite_aoa_deg.to_csv(
        data_dir / "alpha_i_dd_method_best_m_deg.csv"
    )
    result.ephemeris_aoa_rx1_deg.to_csv(data_dir / "alpha_i_ephemeris_rx1_deg.csv")
    result.all_satellite_result.selected_branch.to_csv(data_dir / "selected_branch.csv")
    build_aligned_data(result, case).to_csv(data_dir / "alpha_i_aligned_long.csv", index=False)
    result.reference_alpha_by_epoch.loc[
        result.reference_alpha_by_epoch["epoch_index"].between(0, 33)
    ].to_csv(data_dir / "reference_alpha_selected_window.csv", index=False)


def captions_text(case: ReferenceCase, alpha_r_deg: float) -> str:
    return f"""# Caption-ready text - reference PRN {case.prn}

Reference PRN {case.prn} is labelled {case.known_class.lower()} and uses the receiver-reported median reference-angle anchor $\\alpha_r={alpha_r_deg:.6f}^\\circ$. The reference PRN is excluded from DD tracks and threshold scoring because its own DD is zero by construction.

## Figure 1 - Fractional double difference

Fractional double-difference carrier-phase measurements for all valid non-reference PRNs over the selected 34 consecutive epochs, using PRN {case.prn} as the double-difference reference.

## Figure 2 - Ephemeris-based AoA

Values of $\\alpha_i$ computed from Receiver 1 ephemeris information over the selected 34 consecutive epochs. Reference PRN {case.prn} is omitted.

## Figure 3 - Proposed best-m AoA

Values of $\\alpha_i$ computed by the proposed hybrid best-m method over the selected 34 consecutive epochs, using PRN {case.prn} as the double-difference reference. Labelled-authentic PRNs use the ephemeris-aligned constant branch; known spoof PRNs retain the dynamic-programming branch.

## Figure 4 - Side-by-side comparison

Values of $\\alpha_i$ computed from ephemeris information (left) and by the proposed hybrid best-m method (right), using PRN {case.prn} as the double-difference reference.

## Tables 1-3

Table 1 reports authentic-PRN agreement at 5, 7, and 10 degrees. Table 2 reports spoof-PRN agreement at 10 degrees. Table 3 reports post-selection mixed-label agreement at 10 degrees. Because labelled-authentic branch selection uses ephemeris alignment, Table 3 is an ephemeris-assisted agreement result, not independent detection accuracy. All tables exclude reference PRN {case.prn}.
"""


def write_case_readme(
    case: ReferenceCase,
    alpha_r_deg: float,
    metrics: dict[str, float | int],
    case_dir: Path,
) -> None:
    text = f"""# Mixed-case package - reference PRN {case.prn}

- Reference PRN: **{case.prn}**
- Reference class: **{case.known_class}**
- Median reference-angle anchor: **{alpha_r_deg:.6f} deg**
- Dataset: `sp-partial-19-06/18-06`
- Selected window: **34 epochs spanning 33 seconds**
- Reference self-observations: excluded from figures' DD tracks and all tables
- Figure formats: **PNG, PDF, and EPS**

## Figures

- `figures/figure_01_fractional_double_difference_mixed.*`
- `figures/figure_02_alpha_i_ephemeris_mixed.*`
- `figures/figure_03_alpha_i_proposed_best_m_mixed.*`
- `figures/figure_04_alpha_i_side_by_side_best_m.*`

## Tables

- `tables/table_1_authentic_best_m_thresholds.*`
- `tables/table_2_spoof_best_m_10deg.*`
- `tables/table_3_mixed_decisions_best_m_10deg.*`

## Results at 10 degrees

- Authentic agreement: **{format_percentage(float(metrics['authentic_within_10deg_percent']))}** ({metrics['authentic_within_10deg_count']}/{metrics['authentic_valid_count']})
- Spoof agreement: **{format_percentage(float(metrics['spoof_within_10deg_percent']))}** ({metrics['spoof_within_10deg_count']}/{metrics['spoof_valid_count']})
- Post-selection mixed-label agreement: **{format_percentage(float(metrics['post_selection_agreement_percent']))}** ({metrics['post_selection_agreement_count']}/{metrics['post_selection_valid_count']})

The mixed-label result is **not independent detection accuracy**: labelled-authentic branch selection uses ephemeris alignment. Also, because each case excludes its own reference, compare cross-reference rates using the fixed-population sensitivity tables in the parent manuscript package rather than these per-case totals.

Copy-ready text is in `captions.md`; numeric provenance is in `data/` and `metadata.json`.
"""
    (case_dir / "README.md").write_text(text, encoding="utf-8")


def build_case(
    result: PaperScenarioResult,
    case: ReferenceCase,
) -> dict[str, object]:
    case_dir = case_directory(case)
    figure_dir = case_dir / "figures"
    if case_dir.exists():
        shutil.rmtree(case_dir)
    figure_dir.mkdir(parents=True)

    figure_stems = make_case_figures(result, case, figure_dir)
    return {
        "reference_prn": case.prn,
        "reference_class": case.known_class,
        "figures": figure_stems,
    }


def write_manifest(metadata_by_case: Sequence[dict[str, object]]) -> None:
    rows = []
    for metadata in metadata_by_case:
        rows.append(
            "| "
            + " | ".join(
                [
                    f"PRN {metadata['reference_prn']}",
                    str(metadata["reference_class"]),
                    f"{float(metadata['reference_alpha_deg']):.2f} deg",
                    f"{format_percentage(float(metadata['authentic_within_10deg_percent']))} "
                    f"({metadata['authentic_within_10deg_count']}/{metadata['authentic_valid_count']})",
                    f"{format_percentage(float(metadata['spoof_within_10deg_percent']))} "
                    f"({metadata['spoof_within_10deg_count']}/{metadata['spoof_valid_count']})",
                    f"{format_percentage(float(metadata['post_selection_agreement_percent']))} "
                    f"({metadata['post_selection_agreement_count']}/{metadata['post_selection_valid_count']})",
                ]
            )
            + " |"
        )
    text = """# Per-reference mixed-case manuscript packages

Each directory is a self-contained copy-ready package with four PNG/PDF/EPS figures, three CSV/Markdown/LaTeX table bundles, aligned data, captions, metadata, and a README.

| Reference | Class | Anchor | Authentic agreement | Spoof agreement | Post-selection label agreement |
| --- | --- | --- | --- | --- | --- |
""" + "\n".join(rows) + "\n\n"
    text += """The selected reference is excluded from its own case, so authentic-reference and spoof-reference folders have different authentic/spoof target populations. These per-folder rates are not like-for-like cross-reference comparisons. Use the parent package's fixed-population reference-sensitivity Table 4 for that comparison. The post-selection metric is ephemeris-assisted and is not independent detection accuracy.\n\n"""
    for case in REFERENCE_CASES:
        text += f"- [`{case_slug(case)}/`]({case_slug(case)}/)\n"
    text += """
## Rebuild

From the repository root:

```bash
uv run python outputs/aoa_paper/sp-partial-19-06/18-06/manuscript_mixed_case/build_reference_case_packages.py
```
"""
    REFERENCE_CASES_DIR.mkdir(parents=True, exist_ok=True)
    (REFERENCE_CASES_DIR / "README.md").write_text(text, encoding="utf-8")

    main_section = """## Separate per-reference packages

Four self-contained case folders reproduce the manuscript figure and table sequence independently for each selected reference. Every figure is available in PNG, PDF, and EPS:

- `reference_cases/reference_prn_23_authentic/`
- `reference_cases/reference_prn_12_authentic/`
- `reference_cases/reference_prn_9_spoof/`
- `reference_cases/reference_prn_11_spoof/`

See `reference_cases/README.md` for metrics and rebuild instructions.
"""
    replace_marked_section(
        OUTPUT_DIR / "README.md",
        main_section,
        start_marker=PACKAGE_SECTION_START,
        end_marker=PACKAGE_SECTION_END,
    )


def main() -> None:
    if REFERENCE_CASES_DIR.exists():
        shutil.rmtree(REFERENCE_CASES_DIR)
    REFERENCE_CASES_DIR.mkdir(parents=True)

    results = run_cases()
    verify_cases(results)
    generated_cases = [
        build_case(results[case.prn], case) for case in REFERENCE_CASES
    ]
    print(json.dumps(generated_cases, indent=2))


if __name__ == "__main__":
    main()
