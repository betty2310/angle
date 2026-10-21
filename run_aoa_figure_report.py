"""Run the experiment notebook, then the complete figure-export notebook.

Example: .venv/bin/python run_aoa_figure_report.py --dataset-dir dataset/sp-partial-16-09 --update-notebooks
Legacy capture presets: --versions v4 v5 v8 v9
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import time

import nbformat
from nbclient import NotebookClient
import pandas as pd

from aoa_experiment import ExperimentConfig
from aoa_figure_report import CAPTURE_PRESETS, FIGURE_STEMS, capture_report_preset

PROJECT = Path(__file__).resolve().parent


def receiver_paths(folder):
    rx1, rx2 = sorted(folder.glob("COM3*.ubx")), sorted(folder.glob("COM4*.ubx"))
    if len(rx1) != 1 or len(rx2) != 1:
        raise ValueError(f"Expected one COM3*.ubx and one COM4*.ubx in {folder}.")
    return rx1[0], rx2[0]


def experiment_cell(config):
    keys = ["name", "receiver1_path", "receiver2_path", "spoof_svids",
            "single_difference_order", "baseline_azimuth_deg", "baseline_over_wavelength",
            "reference_svid", "spoof_branch_m", "prior_aoa_deg"]
    defaults = ExperimentConfig(config.receiver1_path, config.receiver2_path, config.spoof_svids)
    keys.extend(key for key in config.to_dict() if key not in keys and getattr(config, key) != getattr(defaults, key))
    lines = ["CONFIG = ExperimentConfig("]
    for key in keys:
        value = getattr(config, key)
        if isinstance(value, Path):
            try:
                value = value.relative_to(PROJECT)
            except ValueError:
                pass
            literal = f"Path({str(value)!r})"
        else:
            literal = repr(value)
        lines.append(f"    {key}={literal},")
    return "\n".join([*lines, ")", ""])


def execute_notebook(template, tag, parameters, output, timeout):
    notebook = nbformat.read(template, as_version=4)
    inputs = [cell for cell in notebook.cells if tag in cell.metadata.get("tags", [])]
    if len(inputs) != 1:
        raise ValueError(f"Expected one {tag!r} input cell in {template}.")
    inputs[0].source = parameters
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Executing {template.name} -> {output.parent}", flush=True)

    def progress(cell_index, **kwargs):
        print(f"  {template.name}: cell {cell_index + 1}/{len(notebook.cells)} complete", flush=True)

    client = NotebookClient(notebook, timeout=timeout, kernel_name="python3",
                            resources={"metadata": {"path": str(PROJECT)}}, on_cell_executed=progress)
    try:
        client.execute()
    except Exception:
        nbformat.write(notebook, output.with_name(output.stem + "_failed.ipynb"))
        raise
    nbformat.write(notebook, output)
    return notebook


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=PROJECT / "dataset/sp-partial-16-09")
    parser.add_argument("--versions", nargs="+", choices=tuple(CAPTURE_PRESETS))
    parser.add_argument("--spoof", type=int, nargs="+")
    parser.add_argument("--single-difference-order", choices=("rx1-rx2", "rx2-rx1"))
    parser.add_argument("--baseline-azimuth-deg", type=float, default=0.0)
    parser.add_argument("--baseline-over-wavelength", type=float, default=1.0)
    parser.add_argument("--reference", type=int)
    parser.add_argument("--prior", type=float)
    parser.add_argument("--spoof-branch-m", type=int)
    parser.add_argument("--angle-reference", choices=("configured", "opposite", "closest_to_expected"), default="configured")
    parser.add_argument("--expected-aoa-deg", type=float)
    parser.add_argument("--analysis-root", type=Path, default=PROJECT / "outputs/aoa_experiments")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--cell-timeout", type=int, default=600)
    parser.add_argument("--min-valid-spoof-prns", type=int)
    parser.add_argument("--max-gap-epochs", type=int, default=0)
    parser.add_argument("--min-window-epochs", type=int, default=2)
    parser.add_argument("--compact-epochs", type=int)
    parser.add_argument("--display-duration-s", type=float, help="fixed time axis 0..N s (first N s of the window)")
    parser.add_argument("--extend-authentic", action="store_true", help="draw authentic PRNs after a shorter spoof window up to N s")
    parser.add_argument("--authentic-max-error-deg", type=float)
    parser.add_argument("--filter-figures", action="store_true")
    parser.add_argument("--no-diagnostics", action="store_true")
    parser.add_argument("--update-notebooks", action="store_true")
    args = parser.parse_args()
    if args.versions and args.update_notebooks and len(args.versions) != 1:
        parser.error("--update-notebooks requires one capture.")
    folders = [PROJECT / "dataset/sp-partial-28-08" / v for v in args.versions] if args.versions else [args.dataset_dir.resolve()]
    group_name = "sp-partial-28-08" if args.versions else args.dataset_dir.name
    output_root = (args.output_root or Path.home() / "Desktop" / f"AOA_Figures_{group_name}").expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    started, rows = time.monotonic(), []
    for folder in folders:
        rx1, rx2 = receiver_paths(folder)
        if args.spoof is None:
            preset = capture_report_preset(folder)
        else:
            preset = {"spoof_svids": tuple(args.spoof), "single_difference_order": "rx2-rx1", "spoof_branch_m": None}
            if folder.parent.name == "sp-partial-28-08":
                preset.update(CAPTURE_PRESETS.get(folder.name, {}))
            preset["spoof_svids"] = tuple(args.spoof)
        try:
            name = folder.relative_to(PROJECT / "dataset").as_posix()
        except ValueError:
            name = folder.name
        config = ExperimentConfig(
            name=name, receiver1_path=rx1, receiver2_path=rx2,
            spoof_svids=tuple(preset["spoof_svids"]), output_root=args.analysis_root,
            single_difference_order=args.single_difference_order or preset["single_difference_order"],
            baseline_azimuth_deg=args.baseline_azimuth_deg,
            baseline_over_wavelength=args.baseline_over_wavelength,
            reference_svid=args.reference, prior_aoa_deg=args.prior,
            angle_reference=args.angle_reference, expected_aoa_deg=args.expected_aoa_deg,
            spoof_branch_m=args.spoof_branch_m if args.spoof_branch_m is not None else (None if args.prior is not None else preset["spoof_branch_m"]),
            min_valid_spoof_prns=args.min_valid_spoof_prns, max_gap_epochs=args.max_gap_epochs,
            min_window_epochs=args.min_window_epochs,
        )
        run_dir = output_root / (folder.name if args.versions or folder.parent.name != "dataset" else "original")
        experiment = execute_notebook(
            PROJECT / "AOA_Paper.ipynb", "experiment-config", experiment_cell(config),
            config.output_dir / "AOA_Paper_executed.ipynb", args.cell_timeout,
        )
        run_dir.mkdir(parents=True, exist_ok=True)
        nbformat.write(experiment, run_dir / "AOA_Paper_executed.ipynb")
        if args.update_notebooks:
            nbformat.write(experiment, PROJECT / "AOA_Paper.ipynb")
        windows = pd.read_csv(config.output_dir / "windows.csv")
        solved = windows.loc[windows["status"].eq("solved"), "window"].astype(int).tolist()
        if not solved:
            raise RuntimeError(f"No spoof-group window of {name} was solved; see windows.csv.")
        for window in solved:
            window_name = f"window_{window:02d}"
            export_dir = run_dir / window_name
            parameters = (
                f"EXPERIMENT_RUN = Path({str(config.output_dir / window_name / 'experiment.json')!r})\n"
                f"EXPORT_DIR = Path({str(export_dir)!r})\n"
                "FORMATS = ('png', 'pdf', 'eps')\n"
                f"INCLUDE_DIAGNOSTICS = {not args.no_diagnostics!r}\n\n"
                f"COMPACT_EPOCHS = {args.compact_epochs!r}\n"
                f"DISPLAY_DURATION_S = {args.display_duration_s!r}\n"
                f"EXTEND_AUTHENTIC = {args.extend_authentic!r}\n"
                f"AUTHENTIC_MAX_ERROR_DEG = {args.authentic_max_error_deg!r}\n"
                f"FILTER_FIGURES = {args.filter_figures!r}\n"
            )
            exported = execute_notebook(
                PROJECT / "AOA_Export_Figures.ipynb", "export-config", parameters,
                export_dir / "AOA_Export_Figures_executed.ipynb", args.cell_timeout,
            )
            if args.update_notebooks:
                nbformat.write(exported, PROJECT / "AOA_Export_Figures.ipynb")
            report = json.loads((export_dir / "report_manifest.json").read_text())
            if report["figure_count"] != len(FIGURE_STEMS) or not report["experiment_replay_verified"]:
                raise RuntimeError("Incomplete or unverified publication export.")
            rows.append({"dataset": name, "window": window, "spoof_svids": list(config.spoof_svids),
                         "epochs": report["epochs"], "reference_svid": report["reference_svid"],
                         "group_median_aoa_deg": report["group_median_aoa_deg"],
                         "chosen_by": report["chosen_by"], "publication_figures": report["figure_count"],
                         "angle_reference": report["angle_reference"]["selected_reference"],
                         "reference_selection": report["angle_reference"]["selected_by"],
                         "diagnostic_figures": report["diagnostic_figure_count"],
                         "total_figure_files": report["total_figure_file_count"],
                         "analysis_dir": str(config.output_dir / window_name), "export_dir": str(export_dir)})
            print(f"Complete: {name} {window_name}; {report['total_figure_file_count']} figure files", flush=True)
    pd.DataFrame(rows).to_csv(output_root / "report_summary.csv", index=False)
    execution = {"status": "completed", "created_at_utc": datetime.now(timezone.utc).isoformat(),
                 "duration_seconds": round(time.monotonic() - started, 2), "runs": rows,
                 "packages": {name: importlib.metadata.version(name) for name in
                              ["numpy", "pandas", "matplotlib", "pyubx2", "nbclient", "nbformat"]}}
    (output_root / "execution_manifest.json").write_text(json.dumps(execution, indent=2) + "\n")
    (output_root / "README.md").write_text(
        "# AoA experiment and complete figure export\n\n"
        "AOA_Paper.ipynb contains the experiment. AOA_Export_Figures.ipynb contains figure export. "
        "Each capture folder contains the executed experiment notebook and one window_NN folder per "
        "solved spoof-group window with its executed export notebook, Figures 1-4 in PNG/PDF/SVG/EPS, "
        "plotted data and branch metadata, and optional diagnostic figures in separate folders.\n\n"
        "The export replays the saved experiment and checks its DD and branch tables. "
        "See report_summary.csv for results and file counts.\n\n"
        "## Material Passport\n\n- Origin Skill: academic-research-suite / experiment-agent\n"
        "- Origin Mode: run\n- Verification Status: both notebook stages completed and replay checks passed\n"
        "- Version Label: separated_experiment_and_export_v1\n", encoding="utf-8",
    )


if __name__ == "__main__":
    main()
