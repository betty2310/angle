"""Compare centered wrapped-DD of two recordings of one set-up angle, side by side.

The layout is the 90-degree v8/v9 manuscript figure (export_centered_wrapped_dd_90deg.py):
(a) recording 1 and (b) recording 2, a legend above each panel and a caption below it. Each
recording is replayed from its saved experiment.json through the same export as its report,
so the plotted values, epoch selection and PRN colors equal that report's Figure 1.

Recordings follow the cold-start split naming `<angle>-1` / `<angle>-2`
(split_ubx_at_cold_start.py), each with one solved window:

    uv run python export_centered_wrapped_dd_comparison.py outputs/aoa_experiments/sp-partial-28-09 \
        --angles 30 60 90 --display-duration-s 120 --extend-authentic
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib.ticker import MaxNLocator

from aoa_figure_report import DESIGN_PATH, break_at_wraps, export_saved_experiment, file_sha256, load_figure_design, prn_styles

PROJECT = Path(__file__).resolve().parent
FIGURE_WIDTH_IN, BASE_HEIGHT_IN, LEGEND_ROW_IN = 7.2, 2.8, 0.13


def draw_wrapped_traces(ax, x, frame: pd.DataFrame, *, declared, styles, design) -> dict[int, np.ndarray]:
    """draw_prn_traces as in the report's Figure 1: no line across a +-0.5 wrap (spoof PRNs; wrapping
    authentic PRNs are already excluded by the export)."""
    step = max(1, round(len(x) / design.get("spoof_marker_count", len(x))))
    originals = {}
    for prn in frame:
        is_spoof = prn in declared
        color, marker = styles[prn]
        xs, ys, positions = break_at_wraps(np.asarray(x), frame[prn].to_numpy(dtype=float))
        markevery = None
        if is_spoof:
            markevery = np.zeros(len(xs), dtype=bool)
            markevery[positions[declared.index(prn) * step // len(declared)::step]] = True
        ax.plot(xs, ys, color=color, lw=design["spoof_linewidth"] if is_spoof else design["authentic_linewidth"],
                ls="-" if is_spoof else "--", marker=marker, markersize=design["markersize"], markevery=markevery,
                label=f"PRN {prn}" + (" (spoof)" if is_spoof else ""), zorder=2 if is_spoof else 3)
        originals[prn] = positions
    ax.grid(True, color=design["grid_color"], linewidth=0.5, alpha=1.0)
    ax.margins(x=0.025)
    ax.set_axisbelow(True)
    return originals


def replay_recording(experiment_file: Path, export_dir: Path, *, compact_epochs, display_duration_s,
                     extend_authentic, authentic_max_error_deg, filter_figures) -> dict:
    manifest = export_saved_experiment(
        experiment_file, export_dir, compact_epochs=compact_epochs, display_duration_s=display_duration_s,
        extend_authentic=extend_authentic,
        authentic_max_error_deg=authentic_max_error_deg, filter_figures=filter_figures,
        include_diagnostics=False, show=False,
    )
    values = pd.read_csv(export_dir / "data/fractional_double_difference_cycles.csv", index_col="epoch_index",
                         float_precision="round_trip")
    values.columns = values.columns.astype(int)
    epochs = pd.read_csv(export_dir / "data/selected_epoch_map.csv", float_precision="round_trip")
    assert values.index.tolist() == epochs["epoch_index"].tolist()
    assert manifest["reference_svid"] not in values.columns
    finite = values.to_numpy()[np.isfinite(values.to_numpy())]
    assert np.all(np.abs(finite) <= 0.5), "Expected centered wrapped DD in cycles"
    return {"values": values, "epochs": epochs, "manifest": manifest, "export_dir": export_dir}


def align_recordings(recordings: list[dict], design: dict) -> dict:
    """Give both panels the same PRN group, order and styles so each PRN can be compared side by side.

    Only PRNs drawn in both recordings are kept (a PRN missing, without phase, or wrap-excluded in
    either recording is dropped from both). One style table is shared, so a PRN has the same color
    and marker in (a) and (b). Recordings must share the reference satellite; otherwise their
    DDs are not comparable.
    """
    references = {run["manifest"]["reference_svid"] for run in recordings}
    if len(references) != 1:
        raise ValueError(f"Recordings use different reference satellites {sorted(references)}; DDs are not comparable.")
    declared = tuple(sorted(set().union(*(run["manifest"]["declared_spoof_svids"] for run in recordings))))
    common = sorted(set.intersection(*(set(run["values"].columns) for run in recordings)))
    for run in recordings:
        run["dropped_prns"] = sorted(set(run["values"].columns) - set(common))
        run["values"] = run["values"][common]
    styles = prn_styles(common, declared, design)
    for run in recordings:
        run["styles"] = styles
    return {"declared": declared, "common_prns": common, "styles": styles}


def export_comparison(recordings: list[dict], captions: tuple[str, str], output_dir: Path, stem: str) -> list[Path]:
    design = load_figure_design()
    aligned = align_recordings(recordings, design)
    compacted = recordings[0]["manifest"]["plot_axis_mode"] == "compacted_1hz_samples"
    duration = recordings[0]["manifest"].get("display_duration_s")
    legend_rows = max(-(-len(run["values"].columns) // 3) for run in recordings)
    height = BASE_HEIGHT_IN + LEGEND_ROW_IN * max(0, legend_rows - 3)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    with plt.rc_context(design["rc_params"]):
        fig, axes = plt.subplots(1, 2, figsize=(FIGURE_WIDTH_IN, height), sharex=True, sharey=True)
        fig.subplots_adjust(**design["subplot_margins"])
        legends, caption_texts = [], []
        for ax, run, caption in zip(axes, recordings, captions, strict=True):
            x = run["epochs"]["plot_time_s"].to_numpy(dtype=float)
            run["positions"] = draw_wrapped_traces(ax, x, run["values"], declared=aligned["declared"],
                                                   styles=aligned["styles"], design=design)
            if compacted:
                step = int(design["compacted_tick_step"])
                ax.set_xlabel(design["compacted_axis_label"])
                ax.set_xticks(sorted({1, len(x), *range(step, len(x) + 1, step)}))
            else:
                ax.set_xlabel(design["elapsed_axis_label"])
                ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
                if duration is not None:
                    ax.set_xlim(-0.025 * duration, 1.025 * duration)
                    ax.set_xticks(np.linspace(0, duration, 7))
            ax.set_ylim(-0.5, 0.5)
            ax.set_yticks(np.arange(-0.4, 0.41, 0.2))
            center = (ax.get_position().x0 + ax.get_position().x1) / 2
            handles, labels = ax.get_legend_handles_labels()
            legends.append(fig.legend(
                handles, labels, loc="upper center", bbox_to_anchor=(center, 1.0),
                ncol=3, fontsize=7.0, frameon=False, columnspacing=0.85,
                handlelength=1.8, handletextpad=0.4, labelspacing=0.35,
            ))
            caption_texts.append(fig.text(center, 0.10 / height, caption, ha="center", va="bottom", fontsize=10.0))
        axes[0].set_ylabel(r"Fractional DD, $y_{ir}$ (cycles)")
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()

        def bounds(artist):
            return artist.get_window_extent(renderer).transformed(fig.transFigure.inverted())

        top = min(bounds(legend).y0 for legend in legends) - design["legend_gap_in"] / height
        axis_label_height = max(
            ax.get_position().y0 - ax.get_tightbbox(renderer).transformed(fig.transFigure.inverted()).y0
            for ax in axes
        )
        bottom = max(bounds(text).y1 for text in caption_texts) + 0.10 / height + axis_label_height
        fig.subplots_adjust(top=top, bottom=bottom)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        for ax, run, legend, text in zip(axes, recordings, legends, caption_texts, strict=True):
            box = ax.get_position()
            assert box.x0 <= bounds(text).x0 and bounds(text).x1 <= box.x1
            assert box.x0 <= bounds(legend).x0 and bounds(legend).x1 <= box.x1
            assert bounds(legend).y0 > box.y1 and bounds(text).y0 > 0
            assert bounds(text).y1 < ax.get_tightbbox(renderer).transformed(fig.transFigure.inverted()).y0
            for line, prn in zip(ax.lines, run["values"], strict=True):
                positions = run["positions"][prn]
                np.testing.assert_array_equal(np.asarray(line.get_xdata())[positions], run["epochs"]["plot_time_s"])
                np.testing.assert_array_equal(np.asarray(line.get_ydata())[positions], run["values"][prn])
                assert np.isnan(np.delete(np.asarray(line.get_ydata(), dtype=float), positions)).all()
        for extension in design["formats"]:
            path = output_dir / f"{stem}.{extension}"
            fig.savefig(path, format=extension, dpi=600, facecolor="white")
            assert path.stat().st_size > 0
            paths.append(path)
        plt.close(fig)
    sources = [Path(__file__), DESIGN_PATH]
    for run in recordings:
        sources += [Path(run["manifest"]["experiment_file"]), run["export_dir"] / "data/fractional_double_difference_cycles.csv",
                    run["export_dir"] / "data/selected_epoch_map.csv"]
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "panel_captions": captions, "figure_size_in": [FIGURE_WIDTH_IN, height], "png_dpi": 600,
        "time_axis_policy": ("compacted 1 Hz samples at Epoch positions" if compacted
                             else f"actual elapsed receiver time from each recording's window start, fixed 0-{duration:g} s axis "
                                  "(first samples of a longer window; a shorter one leaves the rest empty)" if duration is not None
                             else "actual elapsed receiver time from each recording's window start"),
        "display_duration_s": duration,
        "measurement_policy": "Saved fractional DD in cycles; no smoothing or interpolation; NaNs remain gaps",
        "wrap_policy": "as the report's Figure 1: authentic PRNs that cross the +-0.5 wrap are excluded; spoof lines are broken (not drawn) across the wrap; every plotted sample is kept",
        "prn_policy": "both panels show the same PRNs (drawn in both recordings), in the same order; PRNs drawn in "
                      "only one recording are left out of both",
        "common_prns": aligned["common_prns"],
        "reference_svid": recordings[0]["manifest"]["reference_svid"],
        "color_policy": "one shared color and marker per PRN in both panels (assigned over the common PRNs)",
        "validation": "Plotted values equal the replayed export CSVs; captions and legends fit without overlap",
        "recordings": [
            {"dataset": run["manifest"]["dataset"], "experiment_file": run["manifest"]["experiment_file"],
             "experiment_replay_verified": run["manifest"]["experiment_replay_verified"],
             "tow_start_s": run["manifest"]["tow_start_s"], "tow_end_s": run["manifest"]["tow_end_s"],
             "analysis_window_epochs": run["manifest"]["analysis_window_epochs"],
             "plotted_epochs": len(run["values"]), "plotted_span_s": run["manifest"]["elapsed_span_s"], "source_epoch_indices": run["manifest"]["selected_source_epoch_indices"],
             "reference_svid": run["manifest"]["reference_svid"], "displayed_prns": run["values"].columns.tolist(),
             "dropped_prns_not_in_other_recording": run["dropped_prns"],
             "chosen_by": run["manifest"]["chosen_by"], "group_median_aoa_deg": run["manifest"]["group_median_aoa_deg"],
             "authentic_max_error_deg": run["manifest"]["authentic_max_error_deg"],
             "authentic_filter_applies_to_figures": run["manifest"]["authentic_filter_applies_to_figures"],
             "authentic_after_window": run["manifest"].get("authentic_after_window")}
            for run in recordings
        ],
        "sources_sha256": {str(path.resolve()): file_sha256(path) for path in sources},
        "figure_files": [str(path.resolve()) for path in paths],
    }
    (output_dir / f"{stem}.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiment_root", type=Path, help="folder holding <angle>-1/ and <angle>-2/ experiment runs")
    parser.add_argument("--angles", nargs="+", required=True)
    parser.add_argument("--window", default="window_01")
    parser.add_argument("--compact-epochs", type=int)
    parser.add_argument("--display-duration-s", type=float, help="fixed 0..N s time axis for the manuscript")
    parser.add_argument("--extend-authentic", action="store_true",
                        help="a shorter window: authentic PRNs continue to N s, spoof PRNs end with the window")
    parser.add_argument("--authentic-max-error-deg", type=float, default=7.0)
    parser.add_argument("--filter-figures", action="store_true", help="drop authentic PRNs over the cutoff, as the report can")
    parser.add_argument("--replay-root", type=Path, help="default: outputs/aoa_figure_exports/<capture group>")
    parser.add_argument("--output-dir", type=Path, help="default: ~/Desktop/AOA_Figures_<capture group>/centered_wrapped_dd")
    args = parser.parse_args()
    root = args.experiment_root.expanduser().resolve()
    variant = (f"{args.compact_epochs}_epochs" if args.compact_epochs
               else f"{args.display_duration_s:g}s" if args.display_duration_s else "full_window")
    replay_root = (args.replay_root or PROJECT / "outputs/aoa_figure_exports" / root.name).expanduser().resolve() / variant
    output_dir = (args.output_dir or Path.home() / "Desktop" / f"AOA_Figures_{root.name}" / "centered_wrapped_dd").expanduser().resolve() / variant
    for angle in args.angles:
        recordings = [
            replay_recording(root / f"{angle}-{part}" / args.window / "experiment.json", replay_root / f"{angle}-{part}",
                             compact_epochs=args.compact_epochs, display_duration_s=args.display_duration_s,
                             extend_authentic=args.extend_authentic,
                             authentic_max_error_deg=args.authentic_max_error_deg,
                             filter_figures=args.filter_figures)
            for part in (1, 2)
        ]
        captions = tuple(f"({letter}) centered wrapped-DD, {angle}°, recording {part}" for letter, part in (("a", 1), ("b", 2)))
        for path in export_comparison(recordings, captions, output_dir, f"centered_wrapped_dd_{angle}deg_recording_1_2"):
            print(path)


if __name__ == "__main__":
    main()
