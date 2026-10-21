"""Export one manuscript figure comparing centered wrapped-DD in v8 and v9.

Read the existing report samples so measurements, missing values, epoch selection,
and each recording's PRN colors match the previously exported figures.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from aoa_figure_report import DESIGN_NOTEBOOK, file_sha256, load_figure_design


PROJECT = Path(__file__).resolve().parent
DEFAULT_REPORT = PROJECT / "AOA_Figure_Report_sp-partial-28-08_12epochs_compacted_authentic_le10deg"
STEM = "figure_07_centered_wrapped_dd_90deg_v8_v9"
CAPTIONS = (
    "(a) centered wrapped-DD, 90°, recording 1",
    "(b) centered wrapped-DD, 90°, recording 2",
)


def load_recording(report_root: Path, version: str) -> dict:
    folder = report_root / version
    manifest_path = folder / "report_manifest.json"
    values_path = folder / "data" / "fractional_double_difference_cycles.csv"
    epochs_path = folder / "data" / "selected_epoch_map.csv"
    palette_path = folder / "data" / "analysis_window" / "fractional_double_difference_cycles.csv"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    values = pd.read_csv(values_path, index_col="epoch_index", float_precision="round_trip")
    values.columns = values.columns.astype(int)
    epochs = pd.read_csv(epochs_path, float_precision="round_trip")
    original = pd.read_csv(palette_path, index_col="epoch_index", float_precision="round_trip")
    original = original.reindex(values.index).dropna(axis=1, how="all")
    assert manifest["plot_axis_mode"] == "compacted_1hz_samples"
    assert manifest["dataset"].endswith(f"/{version}")
    assert values.index.tolist() == epochs["epoch_index"].tolist()
    assert values.columns.tolist() == manifest["fractional_dd_prns"]
    assert manifest["reference_svid"] not in values.columns
    assert len(values) == manifest["epochs"]
    np.testing.assert_array_equal(epochs["plot_time_s"], np.arange(1, len(values) + 1))
    finite_values = values.to_numpy()[np.isfinite(values.to_numpy())]
    assert np.all(np.abs(finite_values) <= 0.5), "Expected centered wrapped DD in cycles"
    return {
        "version": version, "values": values, "epochs": epochs, "manifest": manifest,
        "palette_prns": sorted(original.columns.astype(int)),
        "source_paths": [manifest_path, values_path, epochs_path, palette_path],
    }


def export_comparison(report_root: Path, output_dir: Path) -> list[Path]:
    recordings = [load_recording(report_root, version) for version in ("v8", "v9")]
    np.testing.assert_array_equal(recordings[0]["epochs"]["plot_time_s"], recordings[1]["epochs"]["plot_time_s"])
    design = load_figure_design()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    with plt.rc_context(design["rc_params"]):
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharex=True, sharey=True)
        fig.subplots_adjust(**design["subplot_margins"])
        legends, captions = [], []
        for ax, recording, caption in zip(axes, recordings, CAPTIONS, strict=True):
            values = recording["values"]
            manifest = recording["manifest"]
            source_design = manifest["figure_design"]
            spoof = set(manifest["declared_spoof_svids"])
            colors, markers = {}, {}
            for is_spoof, palette_key in ((False, "authentic_colors"), (True, "spoof_colors")):
                members = [prn for prn in recording["palette_prns"] if (prn in spoof) == is_spoof]
                for position, prn in enumerate(members):
                    palette = source_design[palette_key]
                    colors[prn] = palette[position % len(palette)]
                    markers[prn] = source_design["spoof_markers"][position % len(source_design["spoof_markers"])] if is_spoof else None
            for prn in values:
                is_spoof = prn in spoof
                ax.plot(
                    recording["epochs"]["plot_time_s"], values[prn], color=colors[prn],
                    lw=design["spoof_linewidth"] if is_spoof else design["authentic_linewidth"],
                    ls="-" if is_spoof else "--", marker=markers[prn], markersize=design["markersize"],
                    label=f"PRN {prn}" + (" (spoof)" if is_spoof else ""),
                    zorder=3 if is_spoof else 2,
                )
            ax.set_ylim(-0.5, 0.5)
            ax.set_yticks(np.arange(-0.4, 0.41, 0.2))
            step = int(design["compacted_tick_step"])
            ax.set_xticks(sorted({1, len(values), *range(step, len(values) + 1, step)}))
            ax.set_xlabel(design["compacted_axis_label"])
            ax.grid(True, color=design["grid_color"], linewidth=0.5, alpha=1.0)
            ax.margins(x=0.025)
            ax.set_axisbelow(True)
            center = (ax.get_position().x0 + ax.get_position().x1) / 2
            handles, labels = ax.get_legend_handles_labels()
            legends.append(fig.legend(
                handles, labels, loc="upper center", bbox_to_anchor=(center, 1.0),
                ncol=3, fontsize=7.0, frameon=False, columnspacing=0.85,
                handlelength=1.8, handletextpad=0.4, labelspacing=0.35,
            ))
            captions.append(fig.text(
                center, 0.10 / fig.get_figheight(), caption,
                ha="center", va="bottom", fontsize=10.0,
            ))
        axes[0].set_ylabel(r"Fractional DD, $y_{ir}$ (cycles)")
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()

        def bounds(artist):
            return artist.get_window_extent(renderer).transformed(fig.transFigure.inverted())

        top = min(bounds(legend).y0 for legend in legends) - design["legend_gap_in"] / fig.get_figheight()
        caption_top = max(bounds(caption).y1 for caption in captions)
        axis_label_height = max(
            ax.get_position().y0 - ax.get_tightbbox(renderer).transformed(fig.transFigure.inverted()).y0
            for ax in axes
        )
        bottom = caption_top + 0.10 / fig.get_figheight() + axis_label_height
        fig.subplots_adjust(top=top, bottom=bottom)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        for ax, recording, legend, caption in zip(axes, recordings, legends, captions, strict=True):
            assert bounds(caption).x0 >= ax.get_position().x0
            assert bounds(caption).x1 <= ax.get_position().x1
            assert bounds(legend).x0 >= ax.get_position().x0
            assert bounds(legend).x1 <= ax.get_position().x1
            assert bounds(legend).y0 > ax.get_position().y1
            assert bounds(caption).y0 > 0
            assert bounds(caption).y1 < ax.get_tightbbox(renderer).transformed(fig.transFigure.inverted()).y0
            for line, prn in zip(ax.lines, recording["values"], strict=True):
                np.testing.assert_array_equal(line.get_xdata(), recording["epochs"]["plot_time_s"])
                np.testing.assert_array_equal(line.get_ydata(), recording["values"][prn])
        for extension in design["formats"]:
            path = output_dir / f"{STEM}.{extension}"
            fig.savefig(path, format=extension, dpi=600, facecolor="white")
            assert path.stat().st_size > 0
            paths.append(path)
        plt.close(fig)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "panel_order": ["v8", "v9"], "panel_captions": CAPTIONS,
        "figure_size_in": [7.2, 2.8], "png_dpi": 600,
        "time_axis_policy": "Original chronological 1 Hz samples at compacted Epoch positions; no new selection",
        "measurement_policy": "Saved fractional DD in cycles; no smoothing or interpolation; NaNs remain gaps",
        "cohort_policy": "Use the existing whole-PRN authentic filter from each source report",
        "color_policy": "Keep each recording's original PRN colors and spoof markers",
        "validation": "Plotted values equal source CSVs; captions and legends fit without overlap",
        "recordings": [
            {"version": run["version"], "source_epoch_indices": run["manifest"]["selected_source_epoch_indices"],
             "reference_svid": run["manifest"]["reference_svid"], "displayed_prns": run["values"].columns.tolist(),
             "authentic_max_error_deg": run["manifest"]["authentic_max_error_deg"]}
            for run in recordings
        ],
        "sources_sha256": {
            str(path.resolve()): file_sha256(path)
            for path in [Path(__file__), DESIGN_NOTEBOOK, *(path for run in recordings for path in run["source_paths"])]
        },
        "figure_files": [str(path.resolve()) for path in paths],
    }
    (output_dir / f"{STEM}.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--output-dir", type=Path, default=Path.home() / "Desktop" / "AOA_Centered_Wrapped_DD_90deg_v8_v9")
    args = parser.parse_args()
    for path in export_comparison(args.report_root.expanduser().resolve(), args.output_dir.expanduser().resolve()):
        print(path)


if __name__ == "__main__":
    main()
