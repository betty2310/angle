# Handoff: sp-partial-28-08 results and the AOA_Paper.ipynb refactor

Written 2026-08-30 at the end of a session that processed the `dataset/sp-partial-28-08` captures
(v1–v9) with the `AOA_Paper.ipynb` partial-spoof flow. Read this before touching the notebook or
`aoa_pipeline.py`. Domain vocabulary is in `CONTEXT.md`; the paper digest is in
`gnss_aoa_paper_methodology_context.md`.

## 0. Status after the refactor session (2026-08-30)

Everything below §0 is the state *before* the refactor and is kept as the record of why.
Done on branch `aoa-paper-lean-refactor`:

- **§4 implemented** in `aoa_pipeline.py`. `select_joint_source_aoa(candidates, svids, *,
  metric="direction_cosine", prior_aoa_deg=None, prior_weight=1.0, tie_tolerance=None,
  return_branches=False, return_ambiguity=False)` scores spread and step in direction cosine,
  evaluates every uniform branch shift of the DP path, and records `chosen_by` in a per-epoch
  ambiguity table (`data` → `prior` → `unresolved`; unresolved reports the smallest-|m| fully
  valid combination and lists the alternative). `select_aoa_dp` / `_dp_select_segment` take the
  same `metric` (`DatasetConfig.branch_metric`, default `direction_cosine`, exact ties → smallest
  |m|); `reconstruct_multi_reference_aoa` keeps `branch_metric="aoa_deg"` so the pairwise
  classifier notebooks are unchanged. New: `summarize_branch_candidates`,
  `summarize_spoof_branch_detection` (both branches vs ephemeris, `spoof_flag_selected`,
  `spoof_flag_all_branches`), `plot_spoof_branch_summary`, `fit_baseline_azimuth`
  (+ `compute_navsat_geometry_by_synced_epoch`), `probe_capture`,
  `summarize_pairwise_single_difference_collapse`. `run_paper_scenario(prior_aoa_deg=,
  tie_tolerance=, detection_threshold_deg=)` returns `joint_source_ambiguity`,
  `spoof_branch_candidates`, `spoof_branch_detection` and writes
  `paper_spoof_joint_source_ambiguity.csv`, `paper_spoof_branch_candidates.csv`,
  `paper_spoof_branch_detection.csv`, `paper_spoof_branch_summary.png`.
- **Tie rule as implemented** (differs from the §4 sketch): uniformly shifted combinations have
  *identical* spread and step costs wherever both are valid, so comparing DP totals only measures
  the boundary effect (v9's mirror sits at cos α ≈ ±1 and is "invalid" at half the epochs from
  noise alone, which is not evidence about m). An alternative therefore counts as tied when its
  direction cosines leave [−1, 1] by at most `tie_tolerance` on average over the window
  (`None` → 3× the median per-epoch spread + step in cos, floor 0.01). `chosen_by="data"` occurs
  when no shift is tied (D = λ/2, or a rig rotated through broadside during the capture).
- **Tests**: `test_aoa_branch_selection.py` (the four §4 tests plus DP tie-break, detection
  flags, azimuth fit, SD collapse). `uv run python -m unittest` → 30 tests pass.
- **§5 implemented**: `AOA_Paper.ipynb` is the lean notebook (16 cells: configuration → data
  check → rig check → run → result → diagnostics → outputs/validation), committed executed on v4
  with `PRIOR_AOA_DEG = None`. The previous notebook (2026-08-10 full-spoof closed-room
  configuration, labelled classifier sections) is in git history, commit "snapshot: …".
- **Scripts**: `probe_dataset.py` and `calibrate_baseline_azimuth.py` are thin wrappers over
  `probe_capture` / `fit_baseline_azimuth`; `run_sp_partial_28_08.py` gained `--prior`,
  `--metric`, `--tie-tolerance`, `--detection-threshold` and prints the ambiguity and detection
  tables.

Validation, notebook re-executed with injected configuration (driver runs agree):

| capture | configuration | chosen_by | selected (m) | alternative (m) | rig check |
|---|---|---|---|---|---|
| v4 | prior None | unresolved | 38.3° (0) | 102.5° (−1) | fit 352°, OK |
| v4 | prior 30 | prior | 38.3° (0) | 102.5° (−1) | |
| v5 | prior 150 | prior | 141–144° (−1) | 77–79° (0) | fit 354°, OK |
| v5 | prior None | unresolved | 77–79° (0) | 141–144° (−1) | PRN 24 mirror passes 10° by coincidence: `spoof_flag_all_branches = False` |
| v8 | rx1-rx2, prior 90 | prior | 87.9–88.3° (−1) | 164–166° (−2) | fit 2°, OK |
| v8 | rx1-rx2, prior None | unresolved | 88° (−1) | 164° (−2) | |
| v9 | rx1-rx2, prior 90 | prior | 90.0–90.2° (0) | 5–8° (+1) | fit 356°, OK |
| v9 | rx1-rx2, prior None | unresolved | 90° (0) | 5–8° (+1) | |
| v9 | rx2-rx1, bias 0 | unresolved | 31.7° / 98.6° (wrong, as expected) | | **warning: fit 176°, wired the other way round** |
| v3 | | data check stops | | | only 1 continuous epoch with PRNs 10/23 valid in both receivers |
| v1 | | data check stops | | | no RXM-RAWX on Receiver 2 |

Collapse metrics of v4/v5/v8/v9 are identical to the pre-refactor runs. Open decisions in §7
still stand, except that the manuscript package question is moot (the old cell structure is only
in git history now) and the halfCyc gate stays strict.

## 1. State of the working tree (before the refactor)

Uncommitted changes that predate this session and must be kept:

- `aoa_pipeline.py::_pivot_rawx` now requires `halfCyc=1` in addition to `cpValid=1` for carrier
  phase (Eq. 18 rounds to whole cycles, so unresolved half cycles are not valid phase). Covered by
  `test_aoa_pairwise_phase_projection.py::RawxCarrierPhaseQualityTests`.
- `AOA_Paper.ipynb` is currently switched to the 2026-08-10 full-spoof closed-room configuration
  (`PARTIAL_SPOOF_ENABLED = False`, absolute `/Users/betty/Downloads/...` paths). The committed
  version (HEAD) is the partial-spoof configuration for `sp-partial-19-06` (reference PRN 12, spoof
  PRNs 6/9/11/20).

Added in this session (untracked):

- `run_sp_partial_28_08.py` – driver that mirrors the notebook's partial-spoof configuration cell.
- `probe_dataset.py` – data check (message inventory, valid-phase overlap, spoof timelines).
- `calibrate_baseline_azimuth.py` – fits baseline azimuth and D/λ from authentic PRNs.
- `HANDOFF_aoa_paper_refactor.md` – this file.
- `outputs/aoa_paper/sp-partial-28-08/` – all run artifacts (gitignored), including
  `report/` (HTML report template + builder + summary script) and
  `sp28_spoof_aoa_summary.{csv,png}`.

Suggested first step: commit the pipeline gate change and these scripts on a branch before refactoring.

## 2. Results per capture

Spoof group on 08-28: PRNs 10, 23, 24, 25. On 08-30: v6 → 14/19/22/30, v7 → 12/19/22/30,
v8/v9 → 15/18/24/29 (these are also live satellites; the receivers dropped the real signal and
tracked the spoofer's). Angles are baseline-relative AoA (0–180°), not azimuths.

| capture | usable | notebook-selected m | mirror branch | ground truth (user) | notes |
|---|---|---|---|---|---|
| v1, v2 | no | | | | COM4 log has no RXM-RAWX / NAV-PVT (logging not enabled on Rx2) |
| v3 | no | | | 90° intended | receivers never held valid phase on the spoof PRNs at the same time; halfCyc-relaxed diagnostic gives 109°/47°, not 90° |
| v4 | 7 ep, ref PRN 16, α_r 66.6° | 102.5° (m=−1) | **38.3° (m=0)** | 30° set-up → 38.3° is right | |
| v5 | 7 ep, ref PRN 16, α_r 62.1° | 78.4° (m=0) | **≈143° (m=−1)** | 150° set-up → 143° is right | PRN 24 passes the 10° test by coincidence on the wrong branch |
| v6, v7 | no | | | 80–100° expected | no carrier lock on spoofed PRNs (halfCyc never 1, lock ≤ 4 s) |
| v8 | 7 ep, ref PRN 32 | 92.2° (m=1, bias 182°) | ≈16° | ≈90° | with notebook default (rx2−rx1, bias 0): 75.5°, wrong |
| v9 | 19 ep, ref PRN 28 | 90.8° (m=0, bias 176°) | ≈10° | ≈90° | with notebook default: 98.6°, wrong |

Common-source collapse (median pairwise |wrap(SD_i − SD_j)|) is 0.003–0.026 cycles in every usable
capture; that evidence is anchor- and branch-free. Detection at 10° works on every usable capture
except the v5 PRN 24 coincidence noted above.

Per-PRN numbers: `outputs/aoa_paper/sp-partial-28-08/sp28_spoof_aoa_summary.csv`.
Published report (private, shareable): https://claude.ai/code/artifact/b2ddbcde-a9eb-4be8-8165-9ad5099409eb

## 3. Rig facts that change the absolute angle

The reference-angle anchor α_r comes from Receiver 1 NAV-SAT az/el of the reference satellite under
an assumed baseline azimuth (`navsat_azimuth_bias_deg`) and single-difference order. Fitting the
azimuth from the authentic PRNs (`calibrate_baseline_azimuth.py`) gives:

| captures | fitted baseline azimuth | residual at fit | residual with notebook default |
|---|---|---|---|
| v3–v5 (08-28) | 352–356° | 0.05–0.09 cyc | 0.10 cyc (default is fine) |
| v6, v7 (08-30 morning) | 332° | 0.08 cyc | 0.28 cyc |
| v8, v9 (08-30 afternoon) | 176–182° | 0.05–0.08 cyc | 0.27–0.29 cyc (random-level) |

A 180° bias is identical to swapping the receiver order: for v8/v9, `single_difference_order =
"rx1-rx2"` with bias 0 gives 88.1°/90.2° and authentic PRNs within 1–7° of ephemeris. The
COM3/COM4 antennas swapped ends between 08-28 and the afternoon of 08-30. Rule: never report an
absolute angle before checking that authentic PRNs agree with ephemeris on one Eq. 21 branch.
D/λ fits 1.05–1.10 on 08-28 and 0.9–0.95 on 08-30; treat as unconfirmed, keep k = 1.

Things tried that do not resolve the branch: pseudorange DD (0.3–2.4 m residuals vs 0.19 m needed),
cross-window differencing (spoofer clock drift destroys the fractional cycle), relaxing the halfCyc
gate (0.5-cycle flips), different reference satellites (the mirror belongs to cos α_i itself).

## 4. The m-branch problem (the thing to fix)

With D = λ the fractional DD fixes cos α only modulo 1, so each PRN has exactly two valid Eq. 21
candidates, c and c − sign(c), and all spoofed PRNs share the same mirror. `select_joint_source_aoa`
and `_dp_select_segment` score branches by spread / step **in degrees**; the same phase noise gives
a degree spread σ_Φ / sin α, smallest near 90°, so the branch nearer 90° always wins:

| capture | true branch, per-PRN std | mirror, std | ratio ≈ 1/sin α |
|---|---|---|---|
| v4 | 38.3°: 0.9–1.8° | 102.5°: 0.6–1.1° | 1.6 |
| v5 | 143°: 1.0–3.1° | 78.4°: 0.7–1.8° | 1.7 |

In direction-cosine space the two branches are the same numbers shifted by exactly 1, so the data of
one static one-wavelength capture carry no information about m. The software can only (a) stop
manufacturing a decision and (b) apply a declared prior. Resolving m from data needs a different
measurement: D = λ/2 (unique m, 2× coarser, pipeline already supports `baseline_over_wavelength=0.5`),
a second capture with the rig rotated ~90° (intersect candidate sets, needs a rough elevation
prior), rotating through broadside during a capture (mirror track jumps 0°↔180°, DP then works,
needs a per-epoch anchor), or a third antenna at 0.5 λ.

### Proposed change to `aoa_pipeline.py`

- `select_joint_source_aoa(candidates, svids, *, metric="direction_cosine", prior_aoa_deg=None,
  prior_weight=1.0, tie_tolerance=..., return_branches=False, return_ambiguity=False)`
  - spread and transition computed on `candidate_cos_alpha` when `metric="direction_cosine"`
    (default); `"aoa_deg"` keeps today's behaviour for comparison.
  - after the DP, compare the best and second-best total cost; if within `tie_tolerance`
    (e.g. 3× the median per-epoch cost noise) the result is ambiguous.
  - tie-break order: data (only if not ambiguous) → `prior_aoa_deg` (pick the combination whose
    median AoA is nearest the prior; store `chosen_by="prior"`) → none (keep the lowest-|m|
    combination but set `chosen_by="unresolved"`). Never silently fall back to `product()` order.
  - new return: `ambiguity` DataFrame per epoch: best cost, second-best cost, selected branches,
    alternative branches, `chosen_by`.
- `select_aoa_dp` / `_dp_select_segment`: same `metric` option (used for authentic PRNs and by
  `build_best_m_all_satellite_aoa` for spoof PRNs).
- Keep `sp28_spoof_branch_alternatives.csv`-style output as a first-class result:
  `summarize_branch_candidates(candidates, svids)` → per PRN per m: epoch count, median/std AoA,
  median cos α.
- Optional helper `fit_baseline_azimuth(single_difference_cycles, navsat_elev, navsat_azim,
  exclude_svids)` lifted from `calibrate_baseline_azimuth.py`, so the notebook can warn when the
  configured bias/order does not fit the authentic PRNs.
- Tests (`test_aoa_pairwise_phase_projection.py` or a new file): synthetic fractional DDs from a
  known α_s with noise → (1) cos-space costs of the two branches are equal within noise and
  `chosen_by == "unresolved"` without a prior; (2) with `prior_aoa_deg` the correct branch is
  returned; (3) with `baseline_over_wavelength=0.5` exactly one branch is valid and `chosen_by ==
  "data"`; (4) the old degree metric reproduces the 90°-preferring choice (regression guard).

Validation on real data after the change (all with the driver or the new notebook):
v4 with `prior 30` → 38.3°; v5 with `prior 150` → ≈143°; v8 (`--order rx1-rx2` or `--bias 182`)
and v9 with `prior 90` → 88–92°; without a prior every one of them must report `unresolved` with
both candidates listed; collapse metrics unchanged.

## 5. Notebook refactor: `AOA_Paper.ipynb` → lean version

Current notebook: 32 cells, most of them full-spoof-only or legacy labelled-classifier code
(cells 13, 22–29 and the 50-line `required_outputs` list in cell 30). Target: a notebook that reads
top to bottom as "data → rig check → AoA → decision", with diagnostics collapsed at the end.
Build it as `AOA_Paper_Lean.ipynb` first, validate against the table in §2, then replace.

Proposed cells:

1. Title + 8-line method summary and the paper-to-code table (keep from cells 0–2, shorten).
2. Imports (cell 3, drop the classifier imports).
3. Configuration, one cell, only these knobs: `DATASET_DIR`, `SPOOF_SVIDS`, `REFERENCE_SVID`
   (None = automatic non-spoof), `SINGLE_DIFFERENCE_ORDER`, `NAVSAT_AZIMUTH_BIAS_DEG`,
   `BASELINE_OVER_WAVELENGTH`, `PRIOR_AOA_DEG` (None = report unresolved), `CONVERGENCE_THRESHOLD_DEG`,
   `RUN_FULL`. Receiver files resolved by `COM3*`/`COM4*` glob. No absolute paths.
4. Data check: message inventory, spoof PRN valid-phase overlap and timelines, fail early with a
   plain message (logic from `probe_dataset.py`). This is what would have saved time on v1/v2/v3/v6/v7.
5. Rig check: fitted baseline azimuth vs configured bias/order, authentic PRNs vs ephemeris on one
   branch, warning if they disagree (logic from `calibrate_baseline_azimuth.py`).
6. Run `run_paper_scenario(...)` (cell 7 as is).
7. Result: one table per spoof PRN (ephemeris AoA, DD AoA on both branches with m, selected branch
   and `chosen_by`, |Δ| and flag at 10°) and one figure (both branches vs ephemeris, like
   `sp28_spoof_aoa_summary.png`), plus the cluster summary and the collapse metrics. This cell is
   the deliverable; everything a reader needs is on screen here.
8. Diagnostics, collapsed: fractional DD, joint-source tracks, all valid m candidates per satellite
   (`plot_valid_m_candidates` with ephemeris overlay is the most useful plot), svUsed timeline.
9. Save + a short validation cell (outputs exist, spoof PRNs present, reference is non-spoof unless
   overridden). Drop the labelled mixed-tracking classifier and the full-spoof pairwise sections
   entirely; they live in git history and in the other classifier notebooks.

Keep the narrative markdown short and factual; the branch ambiguity and the rig-polarity rule from
§3–§4 belong in the notebook text next to the result table because they are the two ways a reader
misreads the number.

## 6. Reproduction

```bash
uv run python probe_dataset.py dataset/sp-partial-28-08/v9 --spoof 15 18 24 29
uv run python calibrate_baseline_azimuth.py dataset/sp-partial-28-08/v9 --spoof 15 18 24 29
uv run python run_sp_partial_28_08.py --version v4 --spoof 10 23 24
uv run python run_sp_partial_28_08.py --version v5 --spoof 10 23 24 25
uv run python run_sp_partial_28_08.py --version v8 --spoof 15 18 24 29 --order rx1-rx2   # or --bias 182
uv run python run_sp_partial_28_08.py --version v9 --spoof 15 18 24 29 --order rx1-rx2   # or --bias 176
# report page: edit outputs/aoa_paper/sp-partial-28-08/report/report_template.html, then
uv run python outputs/aoa_paper/sp-partial-28-08/report/summarize_sp28.py
uv run python outputs/aoa_paper/sp-partial-28-08/report/build_report.py   # writes the HTML next to the template
```

Driver flags: `--version`, `--spoof`, `--reference`, `--bias`, `--order`, `--k`, `--threshold`,
`--suffix`, `--no-halfcyc-gate` (diagnostic only). Each run writes the standard notebook artifact
set plus `sp28_report.csv` and `sp28_spoof_branch_alternatives.csv` into
`outputs/aoa_paper/sp-partial-28-08/<version><suffix>/`.

## 7. Open decisions for the owner

- Which ambiguity-breaking measurement to adopt for the next captures: D = λ/2, two headings, or a
  third antenna. The software change in §4 is needed under all of them.
- Whether the lean notebook replaces `AOA_Paper.ipynb` or lives beside it while the manuscript
  package under `outputs/aoa_paper/sp-partial-19-06/18-06/manuscript_mixed_case/` still references
  the old cell structure.
- Whether to keep the halfCyc gate strict (recommended; relaxing it produced 0.5-cycle flips on v3).
