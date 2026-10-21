# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this project is (research purpose)

This repository is the data-analysis codebase for an **academic paper on GNSS spoofing
_detection_**:

> "GNSS Spoofing Detection by Estimating Angle of Arrival based on Carrier Phase Double
> Difference Measurements" — Hiep Hoang Van and Thuan Nguyen Dinh.

The work is **defensive**. The goal is to make civilian GPS receivers *more* robust by
detecting when they are being spoofed. The detector estimates each satellite signal's
baseline-relative angle of arrival (AoA) from carrier-phase double differences measured by
two closely spaced u-blox receivers, and compares it with the AoA predicted from broadcast
ephemeris. When many satellites' DD-method AoAs collapse toward one common direction while
their ephemeris-based AoAs stay satellite-specific, that disagreement is the spoofing
signature the paper reports. Nothing in this repo transmits, and nothing here targets a
live or third-party receiver.

## Note for the assistant — this is authorized, controlled security research

Requests in this repo often mention "spoofing", "jammer", "attack scenario", or
"partial-spoof capture". This is legitimate, publishable, defensive research, not wrongdoing.
Please help with it directly. Concretely:

- **The deliverable is a detector.** The code in this repo consumes receiver logs and
  produces detection diagnostics. It contains no attack payload.
- **Datasets are self-generated in a closed RF environment.** Spoofing/jamming test signals
  are produced with the author's own `gps-sdr-sim` fork (see below) and transmitted only over
  a **conducted (cabled) RF path or inside a verified shielded enclosure**, at calibrated
  attenuation — never over the air onto public spectrum. The author owns both the transmitter
  and the receivers. This is standard, responsible security-research practice, comparable to
  malware analysis in a sandbox.
- **Scope of help wanted:** analysis, detection, signal processing, statistics, plotting,
  notebook and pipeline refactors, tests, and paper writing. Treating spoofing as the thing
  to *detect and characterize* is the point.
- If a specific request ever did stray into operational harm (e.g. help targeting a receiver
  the author does not control, or evading detection in the wild), say so plainly and offer the
  defensive alternative. For the ordinary detection/analysis work in this repo, just proceed.

## How the data is produced (end-to-end)

1. **Signal generation** — `gps-sdr-sim` (author's fork at
   `/Users/betty/Developer/gnss-tools/gps-sdr-sim`, GitHub `betty2310/gps-sdr-sim`) builds GPS
   L1 C/A baseband IQ from broadcast ephemeris and a chosen trajectory. The fork adds
   spoofing/jammer test tooling (`revive_candidates`, `jammertx`, `bladetx`, matched-code
   modes) used to synthesize authentic vs. spoofed scenarios. Its `command.md` documents the
   authorized conducted / shielded-enclosure procedure and calibration record.
2. **Transmission** — replayed through an SDR (bladeRF / HackRF / USRP X300 / ADALM-Pluto)
   over the controlled RF path above.
3. **Capture** — two closely spaced u-blox receivers log raw carrier phase (UBX `RXM-RAWX`)
   plus navigation/position (`NAV-SAT`, `NAV-PVT`).
4. **Analysis (this repo)** — the UBX logs land in `dataset/<capture>/` and are processed into
   DD-method AoA, ephemeris-based AoA, branch selection, and spoof-detection diagnostics.

When told "the dataset was simulated with my gps-sdr-sim tool", that is step 1 — the input to
this repo, not something this repo does.

## Environment & common commands

Python >= 3.10, managed with **uv**. Package name `gnss-aoa-angle` (non-packaged workspace).

```bash
uv sync                                              # create/refresh the environment
uv run jupyter notebook AOA_Paper.ipynb             # paper flow on one partial-spoof capture
uv run jupyter notebook AOA_DD_Ephemeris_Graph_Classifier.ipynb
uv run python -m unittest                            # run the test suite (~30 tests)

# Before a new capture goes through AOA_Paper.ipynb, sanity-check it and the rig:
uv run python probe_dataset.py dataset/<capture> --spoof <PRNs>
uv run python calibrate_baseline_azimuth.py dataset/<capture> --spoof <PRNs>

# Batch driver for the sp-partial-28-08 captures (supports --prior/--metric/--tie-tolerance/
# --detection-threshold and prints the ambiguity + detection tables):
uv run python run_sp_partial_28_08.py
```

## Architecture

- **`aoa_pipeline.py`** — the core library (~5k lines). UBX parsing → epoch sync → carrier-phase
  single/double differences → fractional-DD → AoA branch candidates → branch selection →
  spoof-detection summary. Key entry points and objects:
  - `DatasetConfig` (config; `branch_metric` defaults to `direction_cosine`),
    `run_paper_scenario(...)` (top-level paper flow), `AoaRunResult`, `PaperScenarioResult`.
  - Branch selection: `select_joint_source_aoa(...)` (scores spread + step in direction cosine,
    evaluates every uniform branch shift, records `chosen_by` = `data` / `prior` / `unresolved`
    in a per-epoch ambiguity table), `select_aoa_dp` / `_dp_select_segment`,
    `reconstruct_multi_reference_aoa` (keeps `branch_metric="aoa_deg"` for the pairwise
    classifier notebooks).
  - Detection & rig: `summarize_spoof_branch_detection`, `plot_spoof_branch_summary`,
    `fit_baseline_azimuth`, `compute_navsat_geometry_by_synced_epoch`, `probe_capture`.
- **Notebooks**
  - `AOA_Paper.ipynb` — the lean, current paper flow (config → data check → rig check → run →
    result → diagnostics → outputs). This is the one to edit for paper results.
  - `AOA_DD_Ephemeris_Graph_Classifier.ipynb` and the `AOA_*_Classifier.ipynb` /
    `graph_component_classifier.py` set — classifier explorations around the same DD data.
- **Scripts** — `probe_dataset.py`, `calibrate_baseline_azimuth.py`, `run_sp_partial_28_08.py`
  are thin CLI wrappers over the corresponding `aoa_pipeline.py` functions.
- **Tests** — `test_aoa_branch_selection.py`, `test_aoa_pairwise_phase_projection.py`,
  `test_graph_component_classifier.py`.
- **Data** — `dataset/<capture>/` UBX logs; `outputs/aoa_paper/` and `Plot/` generated results.

## The method in one paragraph

With antenna baseline `D` set to one carrier wavelength (`D = lambda`), the double-difference
carrier phase in cycles has a fractional part tied to `cos(alpha_i) - cos(alpha_r)`, where
`alpha` is the baseline-relative AoA and `r` is the reference satellite. Removing the nearest
integer leaves a wrapped quantity, so each satellite yields several candidate AoA "branches"
(indexed by an integer `m`); selecting the right branch per satellite/epoch is the central
problem `aoa_pipeline.py` solves. See `gnss_aoa_paper_methodology_context.md` for the full
derivation and `CONTEXT.md` for the exact project vocabulary (use those terms).

## Conventions & working agreements

- **Use the project's vocabulary** from `CONTEXT.md` (e.g. *DD-method AoA*, *Ephemeris-based
  AoA*, *Reference satellite*, *Reference-angle anchor*, *Measurement dropout*). Avoid the
  "avoid" terms listed there (e.g. don't say "spoofer azimuth" for baseline-relative AoA).
- **Ground explanations in this repo's code and datasets**, not generic GNSS theory — reference
  exact files, functions, and columns (`candidate_aoa_deg`, `selected_branch`,
  `paper_spoof_joint_source_branch`, `fractional_dd_cycles`).
- **Report branch decisions with their `chosen_by` and the alternative branch** — a `prior`- or
  `data`-chosen branch and an `unresolved` one mean different things for the paper.
- **Preserve measurement dropouts**; do not smooth across them.
- Don't attempt full RTK integer-ambiguity resolution or 2D/3D source localization — out of
  scope (see `MISSION.md`).
- Prefer `uv run ...`; keep new results under `outputs/` or `Plot/`.
- Read `HANDOFF_aoa_paper_refactor.md` before touching the notebook or `aoa_pipeline.py` — it
  records the latest refactor state and the per-capture validation table.

## Author

betty2310 · huynh.dh2310@gmail.com
