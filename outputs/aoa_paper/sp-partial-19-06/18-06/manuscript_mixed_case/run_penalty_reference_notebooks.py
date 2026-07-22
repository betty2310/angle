#!/usr/bin/env python3
"""Execute the common-direction-penalty notebook for several DD references.

The source notebook remains unchanged. Each requested reference is injected into
an in-memory copy, executed with the repository kernel, and saved beneath the
generated reference-penalty output directory. The per-reference pair results
are then median-aggregated across all valid reference choices.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import nbformat
import numpy as np
import pandas as pd
from nbclient import NotebookClient


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[4]
SOURCE_NOTEBOOK = REPO_ROOT / "AOA_Paper.ipynb"
OUTPUT_ROOT = REPO_ROOT / "outputs/aoa_paper"
RUN_OUTPUT_DIR = SCRIPT_DIR / "reference_penalty_runs"

DEFAULT_EXECUTED_REFERENCES = (12, 11, 9)
ALL_REFERENCE_ORDER = (12, 23, 11, 9)
SPOOF_SVIDS = (6, 9, 11, 20)
AUTHENTIC_SVIDS = (10, 12, 15, 18, 23, 24, 32)
PAPER_SVIDS = tuple(sorted((*SPOOF_SVIDS, *AUTHENTIC_SVIDS)))
GROUND_TRUTH_BY_SVID = {
    svid: "Counterfeit" if svid in SPOOF_SVIDS else "Authentic"
    for svid in PAPER_SVIDS
}

PAIRWISE_THRESHOLD = 0.20
COMMON_DIRECTION_THRESHOLD = 0.02
SCORE_THRESHOLD = 0.50
METHOD_PREFIX = "hybrid_best_m"


@dataclass(frozen=True)
class ReferenceOutputs:
    summary: Path
    scores: Path
    pairs: Path
    q_matrix: Path
    c_matrix: Path
    net_matrix: Path
    notebook_heatmap: Path

    def all_paths(self) -> tuple[Path, ...]:
        return (
            self.summary,
            self.scores,
            self.pairs,
            self.q_matrix,
            self.c_matrix,
            self.net_matrix,
            self.notebook_heatmap,
        )


def scenario_name(reference_svid: int) -> str:
    return f"sp-partial-19-06/18-06-prn{reference_svid}-pairwise"


def scenario_output_dir(reference_svid: int) -> Path:
    return OUTPUT_ROOT / scenario_name(reference_svid)


def parameterize_notebook(reference_svid: int) -> nbformat.NotebookNode:
    notebook = nbformat.read(SOURCE_NOTEBOOK, as_version=4)
    notebook.cells = [
        cell
        for cell in notebook.cells
        if "reference-aggregate-only" not in cell.metadata.get("tags", [])
    ]
    reference_class = GROUND_TRUTH_BY_SVID[reference_svid].lower()
    authentic_target_count = len(
        set(AUTHENTIC_SVIDS).difference({reference_svid})
    )
    counterfeit_target_count = len(
        set(SPOOF_SVIDS).difference({reference_svid})
    )

    config_cells = [
        cell
        for cell in notebook.cells
        if cell.cell_type == "code"
        and "HARDCODED_REFERENCE_SVID =" in cell.source
        and "SCENARIO_CONFIG = DatasetConfig" in cell.source
    ]
    if len(config_cells) != 1:
        raise ValueError(f"Expected one scenario configuration cell, found {len(config_cells)}")
    config_cell = config_cells[0]
    config_cell.source, reference_replacements = re.subn(
        r"HARDCODED_REFERENCE_SVID = \d+[^\n]*",
        (
            f"HARDCODED_REFERENCE_SVID = {reference_svid}  "
            f"# Parameterized {reference_class} reference run."
        ),
        config_cell.source,
        count=1,
    )
    if reference_replacements != 1:
        raise ValueError("Failed to parameterize the reference PRN")

    for cell in notebook.cells:
        if cell.cell_type != "markdown":
            continue
        if cell.source.startswith("# GNSS Spoofing Detection by AoA Comparison"):
            cell.source = re.sub(
                r"The current partial-spoof experiment uses authentic PRN 23 as the DD reference",
                (
                    "This executed partial-spoof experiment uses "
                    f"{reference_class} PRN {reference_svid} as the DD reference"
                ),
                cell.source,
            )
        elif cell.source.startswith("## Scenario Configuration"):
            cell.source = re.sub(
                r"This scenario hard-codes authentic PRN 23",
                f"This executed scenario hard-codes {reference_class} PRN {reference_svid}",
                cell.source,
            )
        elif cell.source.startswith("## Pairwise Threshold Classifier"):
            paragraphs = cell.source.split("\n\n")
            paragraphs = [
                paragraph
                for paragraph in paragraphs
                if not paragraph.startswith("A fixed DD reference has no DD reconstruction")
            ]
            paragraphs.append(
                "A fixed DD reference has no DD reconstruction of itself. "
                f"PRN {reference_svid} is consequently excluded, leaving "
                f"{authentic_target_count} authentic and {counterfeit_target_count} "
                "counterfeit targets (nine possible partners per target)."
            )
            cell.source = "\n\n".join(paragraphs)
    penalty_cells = [
        cell
        for cell in notebook.cells
        if cell.cell_type == "code"
        and "COMMON_DIRECTION_THRESHOLD = 0.02" in cell.source
    ]
    if len(penalty_cells) != 1:
        raise ValueError(f"Expected one penalty cell, found {len(penalty_cells)}")

    notebook.cells.insert(
        0,
        nbformat.v4.new_markdown_cell(
            "# Executed reference-sensitivity run\n\n"
            f"Reference PRN: **{reference_svid}** ({reference_class}).  "
            "Generated by `run_penalty_reference_notebooks.py`; the source notebook "
            "was not modified."
        ),
    )
    return notebook


def execute_reference(reference_svid: int) -> Path:
    notebook = parameterize_notebook(reference_svid)
    print(f"Executing notebook for reference PRN {reference_svid}...", flush=True)
    client = NotebookClient(
        notebook,
        timeout=1800,
        kernel_name="python3",
        resources={"metadata": {"path": str(REPO_ROOT)}},
    )
    client.execute(cwd=str(REPO_ROOT))
    RUN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    executed_path = (
        RUN_OUTPUT_DIR
        / f"AOA_Paper_Refactored_ref_prn{reference_svid}.executed.ipynb"
    )
    nbformat.write(notebook, executed_path)
    print(f"Saved executed notebook: {executed_path}", flush=True)
    return executed_path


def required_reference_paths(reference_svid: int) -> ReferenceOutputs:
    output_dir = scenario_output_dir(reference_svid)
    return ReferenceOutputs(
        summary=output_dir / f"{METHOD_PREFIX}_common_direction_summary.csv",
        scores=output_dir / f"{METHOD_PREFIX}_adjusted_classification_scores.csv",
        pairs=output_dir / f"{METHOD_PREFIX}_adjusted_pair_details.csv",
        q_matrix=output_dir / f"{METHOD_PREFIX}_residual_matrix_direction_cosine.csv",
        c_matrix=output_dir / f"{METHOD_PREFIX}_common_direction_residual_matrix.csv",
        net_matrix=output_dir / f"{METHOD_PREFIX}_adjusted_pair_contribution_matrix.csv",
        notebook_heatmap=output_dir / f"{METHOD_PREFIX}_common_direction_matrices.png",
    )


def verify_reference_outputs(reference_svid: int) -> ReferenceOutputs:
    paths = required_reference_paths(reference_svid)
    missing = [str(path) for path in paths.all_paths() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Reference PRN {reference_svid} is missing outputs: {missing}"
        )
    summary = pd.read_csv(paths.summary)
    if len(summary) != 1 or int(summary.loc[0, "reference_svid"]) != reference_svid:
        raise ValueError(f"Reference PRN {reference_svid} summary is inconsistent")
    if int(summary.loc[0, "N_classified_satellites"]) != 10:
        raise ValueError(f"Reference PRN {reference_svid} did not classify ten targets")
    if int(summary.loc[0, "N_minus_1_partners_per_satellite"]) != 9:
        raise ValueError(f"Reference PRN {reference_svid} did not use nine partners")
    return paths


def summarize_adjusted_scores(scores: pd.DataFrame) -> dict[str, object]:
    authentic = scores["ground_truth"].eq("Authentic")
    counterfeit = scores["ground_truth"].eq("Counterfeit")
    return {
        "adjusted_correct": int(scores["adjusted_correct"].sum()),
        "adjusted_overall_accuracy_pct": float(
            100 * scores["adjusted_correct"].mean()
        ),
        "adjusted_authentic_acceptance_pct": float(
            100
            * scores.loc[authentic, "adjusted_decision"].eq("Authentic").mean()
        ),
        "adjusted_counterfeit_detection_pct": float(
            100
            * scores.loc[counterfeit, "adjusted_decision"].eq("Counterfeit").mean()
        ),
        "authentic_correct": int(scores.loc[authentic, "adjusted_correct"].sum()),
        "authentic_total": int(authentic.sum()),
        "counterfeit_correct": int(
            scores.loc[counterfeit, "adjusted_correct"].sum()
        ),
        "counterfeit_total": int(counterfeit.sum()),
        "misclassified_prns": ", ".join(
            map(str, scores.loc[~scores["adjusted_correct"], "svid"].astype(int))
        )
        or "None",
    }


def reference_summary(reference_svid: int, paths: ReferenceOutputs) -> pd.DataFrame:
    summary = pd.read_csv(paths.summary)
    scores = pd.read_csv(paths.scores)
    summary.insert(
        1,
        "reference_class",
        GROUND_TRUTH_BY_SVID[reference_svid],
    )
    for column, value in summarize_adjusted_scores(scores).items():
        summary[column] = value
    return summary


def symmetric_matrix(
    pairs: pd.DataFrame,
    value_column: str,
    *,
    diagonal_value: float = 0.0,
) -> pd.DataFrame:
    matrix = pd.DataFrame(
        np.nan,
        index=PAPER_SVIDS,
        columns=PAPER_SVIDS,
        dtype=float,
    )
    for row in pairs.itertuples():
        left = int(row.left_svid)
        right = int(row.right_svid)
        value = float(getattr(row, value_column))
        matrix.loc[left, right] = value
        matrix.loc[right, left] = value
    for svid in PAPER_SVIDS:
        matrix.loc[svid, svid] = diagonal_value
    matrix.index.name = "svid"
    matrix.columns.name = "svid"
    return matrix


def aggregate_reference_pairs(
    paths_by_reference: dict[int, ReferenceOutputs],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[str, pd.DataFrame],
    pd.DataFrame,
    pd.DataFrame,
]:
    pair_frames = []
    for reference_svid, paths in paths_by_reference.items():
        frame = pd.read_csv(paths.pairs)
        frame.insert(0, "reference_svid", reference_svid)
        pair_frames.append(
            frame[
                [
                    "reference_svid",
                    "left_svid",
                    "right_svid",
                    "aggregated_pairwise_residual",
                    "aggregated_common_direction_residual",
                ]
            ]
        )
    by_reference = pd.concat(pair_frames, ignore_index=True)
    aggregated = (
        by_reference.groupby(["left_svid", "right_svid"], as_index=False)
        .agg(
            aggregated_pairwise_residual=(
                "aggregated_pairwise_residual",
                "median",
            ),
            aggregated_common_direction_residual=(
                "aggregated_common_direction_residual",
                "median",
            ),
            reference_count=("reference_svid", "nunique"),
            contributing_references=(
                "reference_svid",
                lambda values: ", ".join(map(str, sorted(set(map(int, values))))),
            ),
        )
        .sort_values(["left_svid", "right_svid"])
        .reset_index(drop=True)
    )
    expected_pair_count = len(PAPER_SVIDS) * (len(PAPER_SVIDS) - 1) // 2
    if len(aggregated) != expected_pair_count:
        raise ValueError(
            f"Aggregated {len(aggregated)} pairs, expected {expected_pair_count}"
        )
    if int(aggregated["reference_count"].min()) < 2:
        raise ValueError("Every aggregated pair must have at least two valid references")

    aggregated["journal_consistent"] = aggregated[
        "aggregated_pairwise_residual"
    ].le(PAIRWISE_THRESHOLD)
    aggregated["common_direction_pair"] = aggregated[
        "aggregated_common_direction_residual"
    ].le(COMMON_DIRECTION_THRESHOLD)
    aggregated["adjusted_pair_contribution"] = (
        aggregated["journal_consistent"].astype(int)
        - aggregated["common_direction_pair"].astype(int)
    )

    q_matrix = symmetric_matrix(aggregated, "aggregated_pairwise_residual")
    c_matrix = symmetric_matrix(
        aggregated,
        "aggregated_common_direction_residual",
    )
    a_matrix = q_matrix.le(PAIRWISE_THRESHOLD).astype(int)
    b_matrix = c_matrix.le(COMMON_DIRECTION_THRESHOLD).astype(int)
    for svid in PAPER_SVIDS:
        a_matrix.loc[svid, svid] = 0
        b_matrix.loc[svid, svid] = 0
    net_matrix = a_matrix - b_matrix

    score_rows = []
    denominator = len(PAPER_SVIDS) - 1
    for svid in PAPER_SVIDS:
        journal_count = int(a_matrix.loc[svid].sum())
        common_count = int(b_matrix.loc[svid].sum())
        raw_score = (journal_count - common_count) / denominator
        adjusted_score = max(0.0, raw_score)
        decision = "Authentic" if adjusted_score >= SCORE_THRESHOLD else "Counterfeit"
        ground_truth = GROUND_TRUTH_BY_SVID[svid]
        score_rows.append(
            {
                "svid": svid,
                "ground_truth": ground_truth,
                "journal_consistent_pair_count": journal_count,
                "common_direction_pair_count": common_count,
                "raw_adjusted_score": raw_score,
                "adjusted_consistency_score": adjusted_score,
                "adjusted_decision": decision,
                "adjusted_correct": decision == ground_truth,
            }
        )
    scores = pd.DataFrame(score_rows)
    score_summary = summarize_adjusted_scores(scores)
    journal_decisions = np.where(
        a_matrix.sum(axis=1).div(denominator).ge(SCORE_THRESHOLD),
        "Authentic",
        "Counterfeit",
    )
    journal_correct = sum(
        decision == GROUND_TRUTH_BY_SVID[int(svid)]
        for svid, decision in zip(a_matrix.index, journal_decisions)
    )
    summary = pd.DataFrame(
        [
            {
                "method": "median_across_references_q_minus_c",
                "reference_svid": "Aggregated",
                "reference_class": "Multiple references",
                "N_classified_satellites": len(PAPER_SVIDS),
                "N_minus_1_partners_per_satellite": denominator,
                "unique_pair_count": expected_pair_count,
                "tau_q_direction_cosine": PAIRWISE_THRESHOLD,
                "tau_c_direction_cosine": COMMON_DIRECTION_THRESHOLD,
                "eta": SCORE_THRESHOLD,
                "journal_correct": journal_correct,
                **score_summary,
                "minimum_references_per_pair": int(
                    aggregated["reference_count"].min()
                ),
                "maximum_references_per_pair": int(
                    aggregated["reference_count"].max()
                ),
            }
        ]
    )
    matrices = {"q": q_matrix, "c": c_matrix, "net": net_matrix}
    return by_reference, aggregated, matrices, scores, summary


def read_matrix(path: Path) -> pd.DataFrame:
    matrix = pd.read_csv(path, index_col=0)
    matrix.index = matrix.index.astype(int)
    matrix.columns = matrix.columns.astype(int)
    return matrix


def plot_reference_heatmaps(
    paths_by_reference: dict[int, ReferenceOutputs],
    aggregate_matrices: dict[str, pd.DataFrame],
) -> tuple[Path, Path]:
    cases: list[tuple[str, dict[str, pd.DataFrame]]] = []
    for reference_svid in ALL_REFERENCE_ORDER:
        paths = paths_by_reference[reference_svid]
        cases.append(
            (
                f"Reference PRN {reference_svid}\n"
                f"({GROUND_TRUTH_BY_SVID[reference_svid].lower()})",
                {
                    "q": read_matrix(paths.q_matrix),
                    "c": read_matrix(paths.c_matrix),
                    "net": read_matrix(paths.net_matrix),
                },
            )
        )
    cases.append(("Median aggregation\n(4 references)", aggregate_matrices))

    q_max = max(
        float(np.nanmax(matrices["q"].to_numpy()))
        for _, matrices in cases
    )
    fig, axes = plt.subplots(
        len(cases),
        3,
        figsize=(15.5, 4.0 * len(cases)),
        constrained_layout=True,
    )
    image_by_column = {}
    for row_index, (row_label, matrices) in enumerate(cases):
        specs = (
            ("q", r"Pairwise residual $q_{ij}$", "magma_r", 0.0, q_max),
            (
                "c",
                r"Common-direction residual $c_{ij}$ (clipped)",
                "magma_r",
                0.0,
                0.05,
            ),
            (
                "net",
                r"Net contribution $A_{ij}-B_{ij}$",
                "RdBu_r",
                -1.0,
                1.0,
            ),
        )
        for column_index, (key, title, cmap, vmin, vmax) in enumerate(specs):
            ax = axes[row_index, column_index]
            matrix = matrices[key]
            image = ax.imshow(
                matrix.to_numpy(float),
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                aspect="equal",
            )
            image_by_column[column_index] = image
            labels = list(map(int, matrix.columns))
            ax.set_xticks(range(len(labels)), labels, rotation=45)
            ax.set_yticks(range(len(matrix.index)), list(map(int, matrix.index)))
            if row_index == 0:
                ax.set_title(title)
            if column_index == 0:
                ax.set_ylabel(f"{row_label}\nPRN")
            else:
                ax.set_ylabel("PRN")
            ax.set_xlabel("PRN")
    for column_index in range(3):
        fig.colorbar(
            image_by_column[column_index],
            ax=axes[:, column_index],
            fraction=0.025,
            pad=0.02,
        )
    fig.suptitle(
        "Common-direction penalty across DD reference choices\n"
        rf"$\tau_q={PAIRWISE_THRESHOLD:.2f}$, "
        rf"$\tau_c={COMMON_DIRECTION_THRESHOLD:.2f}$, "
        rf"$\eta={SCORE_THRESHOLD:.2f}$",
        fontsize=16,
    )
    RUN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = RUN_OUTPUT_DIR / "figure_penalty_reference_heatmaps.png"
    pdf_path = RUN_OUTPUT_DIR / "figure_penalty_reference_heatmaps.pdf"
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def write_consolidated_outputs(
    paths_by_reference: dict[int, ReferenceOutputs],
) -> tuple[pd.DataFrame, Path, Path]:
    RUN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    reference_summaries = [
        reference_summary(reference_svid, paths_by_reference[reference_svid])
        for reference_svid in ALL_REFERENCE_ORDER
    ]
    by_reference, aggregated, matrices, aggregate_scores, aggregate_summary = (
        aggregate_reference_pairs(paths_by_reference)
    )
    consolidated_summary = pd.concat(
        [*reference_summaries, aggregate_summary],
        ignore_index=True,
        sort=False,
    )

    by_reference.to_csv(
        RUN_OUTPUT_DIR / "penalty_pairs_by_reference.csv",
        index=False,
    )
    aggregated.to_csv(
        RUN_OUTPUT_DIR / "penalty_pairs_aggregated_across_references.csv",
        index=False,
    )
    aggregate_scores.to_csv(
        RUN_OUTPUT_DIR / "penalty_aggregated_classification_scores.csv",
        index=False,
    )
    matrices["q"].to_csv(
        RUN_OUTPUT_DIR / "penalty_aggregated_q_residual_matrix.csv"
    )
    matrices["c"].to_csv(
        RUN_OUTPUT_DIR / "penalty_aggregated_c_residual_matrix.csv"
    )
    matrices["net"].to_csv(
        RUN_OUTPUT_DIR / "penalty_aggregated_net_contribution_matrix.csv"
    )
    summary_path = RUN_OUTPUT_DIR / "penalty_reference_summary.csv"
    consolidated_summary.to_csv(summary_path, index=False)
    png_path, pdf_path = plot_reference_heatmaps(paths_by_reference, matrices)
    print("\nConsolidated penalty results:")
    print(
        consolidated_summary[
            [
                "reference_svid",
                "reference_class",
                "authentic_correct",
                "authentic_total",
                "adjusted_authentic_acceptance_pct",
                "counterfeit_correct",
                "counterfeit_total",
                "adjusted_counterfeit_detection_pct",
                "adjusted_correct",
                "N_classified_satellites",
                "adjusted_overall_accuracy_pct",
                "misclassified_prns",
            ]
        ].to_string(index=False)
    )
    return consolidated_summary, png_path, pdf_path


def parse_references(values: Iterable[str]) -> tuple[int, ...]:
    references = tuple(int(value) for value in values)
    invalid = sorted(set(references).difference(ALL_REFERENCE_ORDER))
    if invalid:
        raise ValueError(
            f"Unsupported reference PRNs: {invalid}; "
            f"choose from {list(ALL_REFERENCE_ORDER)}"
        )
    return references


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--references",
        nargs="+",
        default=list(map(str, DEFAULT_EXECUTED_REFERENCES)),
        choices=list(map(str, ALL_REFERENCE_ORDER)),
        help=(
            "Fixed-workflow reference PRNs to execute (default: 12 11 9). "
            "Consolidation always uses existing outputs for 12, 23, 11, and 9."
        ),
    )
    parser.add_argument(
        "--skip-execution",
        action="store_true",
        help="Reuse existing per-reference notebook outputs and rebuild summaries/figures.",
    )
    args = parser.parse_args()
    references = parse_references(args.references)

    if not args.skip_execution:
        for reference_svid in references:
            execute_reference(reference_svid)

    paths_by_reference = {
        reference_svid: verify_reference_outputs(reference_svid)
        for reference_svid in ALL_REFERENCE_ORDER
    }
    _, png_path, pdf_path = write_consolidated_outputs(paths_by_reference)
    print(f"\nConsolidated summary: {RUN_OUTPUT_DIR / 'penalty_reference_summary.csv'}")
    print(f"Heatmap PNG: {png_path}")
    print(f"Heatmap PDF: {pdf_path}")


if __name__ == "__main__":
    main()
