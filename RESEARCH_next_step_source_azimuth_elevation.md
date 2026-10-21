# Next research step: azimuth and elevation of the spoof-group source

Written 2026-10-12. Research note, no code changes. Vocabulary follows `CONTEXT.md`; where the
note goes beyond the single-baseline method it introduces three new terms (section 2.1).
Numbers come from (a) the sp-partial-28-09 experiment outputs already in `outputs/aoa_experiments/`,
(b) the HANDOFF validation tables, and (c) the Monte Carlo in `simulate_multibaseline_direction.py`
(results in `outputs/direction_feasibility/sim_results.txt`). Literature is in section 8.
Theory write-up with the full derivation and an interactive solver (2026-10-13):
`source-direction-theory.html`, published privately at https://claude.ai/artifact/Y6LFiDyGyLCaEexcoLnDq4. Its section 5 is a drag-to-orbit 3-D construction of the solution (each fractional DD a plane cutting the unit sphere; two planes a line; branches parallel planes spaced 1/k; a vertical arm a horizontal plane), built up in seven steps, preceded by the equations in plain coordinates (sphere, dot product, cone, plane, line, piercing points) and followed by a live table that rewrites them with the current numbers.

## 0. Summary

- **What one baseline gives.** The current method measures one direction cosine of the source,
  `cos alpha = b_hat . u_s`, the projection of the source direction `u_s` onto the baseline unit
  vector `b_hat`. All directions on a cone around the baseline share it. The TAES manuscript
  (Zotero `M5DA6HM8`, Section III-D) already states this: "a single baseline observes only the
  projection of the source direction onto that baseline ... the method does not recover a unique
  azimuth–elevation pair." Azimuth and elevation therefore need a second, non-collinear baseline.
  That is the right next step and the literature (section 8) confirms three non-collinear antennas
  is the minimum for an unambiguous direction.
- **The triangle itself does not remove the branch ambiguity; the baseline length does.** With
  `D = lambda` every baseline's cosine is known only modulo 1, i.e. two branches per baseline
  (the `m` problem of HANDOFF section 4). Two `D = lambda` baselines give 2 x 2 = 4 candidate
  directions, of which 3 or 4 are physically valid over most of the sky (Monte Carlo, table 4.1:
  0 % of the sky has a unique solution). The third side of the triangle is the loop closure of the
  other two and carries no new information. So "add one receiver and make a triangle with
  `D = lambda`" would reproduce today's `chosen_by = unresolved` situation in 2-D.
- **Recommended geometry.** A right-angle "L" of three independent ZED-F9P receivers with both
  baselines at `D = lambda/2` (centre spacing 9.5 cm). Every direction in the upper hemisphere then
  has exactly one candidate (`chosen_by = data` without any prior); the pipeline already supports
  `baseline_over_wavelength = 0.5`. The price is 2x the direction-cosine noise, which at the
  measured 0.012 cycle per PRN per epoch is still only about 1.4 deg of azimuth per single
  measurement and below 0.1 deg over a 100 epoch, 4 PRN window. Fallback if 9.5 cm spacing is
  physically impossible with the Tallysman housings: `D = lambda` North plus `D = lambda/2` East
  (two candidates, far apart; a weak elevation prior resolves 89 % of low-elevation cases).
- **Elevation is the hard part, not azimuth.** A horizontal array measures `cos(el)`; near the
  horizon `d(el) = -d|u_h| / sin(el)` blows up. For a ground spoofer at el <= 15 deg a horizontal
  array gives azimuth to 2 to 5 deg but elevation to only 10 to 20 deg, and a 0.03 cycle rig bias
  (the level of the current authentic-PRN residual after calibration) pushes the elevation estimate
  onto the horizon. Good elevation near the horizon needs a non-coplanar fourth antenna (vertical
  component about `lambda/2`), which also makes the array 3-D. Decide early whether the next paper
  claims azimuth only, or azimuth and elevation.
- **Side benefit for detection.** A 10 deg cone band around one baseline covers up to 17 % of the
  sky; a 10 deg cap around a 2-D direction covers 0.8 %. The v5 PRN 24 coincidence (ephemeris AoA
  within 10 deg of the mirror branch) essentially disappears with two baselines.
- **No new hardware pilot.** Rotating the existing rig by 90 deg between two captures with the
  transmitter fixed gives two synthetic baselines. It validates the 2-D solver and the
  baseline-vector calibration, but with `D = lambda` it keeps the 3 to 4 way ambiguity (table 4.1,
  row I). It cannot replace the third receiver for the ambiguity claim.

## 1. Where the method stands (facts used below)

| item | value | source |
|---|---|---|
| receivers | 2 x u-blox ZED-F9P, independent clocks, Tallysman antennas | manuscript V-A, `gnss-aoa-dd/README.md` |
| baseline | `D = lambda_L1 = 0.19 m`, pointing north, `k = D/lambda = 1` | manuscript Eq. 53–55 |
| observable | centered wrapped DD, `fractional_dd_cycles` | `fractional_round_dd` (Eq. 18) |
| candidate | `cos alpha_i = (f_i + m)/k + cos alpha_r`, `m in {-2..2}` | `generate_aoa_candidates` (Eq. 21) |
| ephemeris cosine | `cos alpha = cos(el) cos(az - az_b)`, horizontal baseline | `compute_navsat_aoa`, `compute_reference_alpha_by_epoch` |
| branch choice | `select_joint_source_aoa`, `chosen_by` = data / prior / unresolved | HANDOFF section 0 |
| per-PRN fractional-DD std, spoof group | 0.0105–0.0130 cyc (28-09 30-1, 60-1, 90-1 window 1); 0.019–0.020 cyc (90 window 2) | `outputs/aoa_experiments/sp-partial-28-09/*/window_01/fractional_dd_cycles.csv` |
| within-group dispersion (PRN-specific part) | median 0.006 cyc, p90 0.009 cyc (90-1) | `collapse_metrics.csv` |
| authentic PRNs vs ephemeris, rig fit residual | 0.05–0.09 cyc median abs | HANDOFF section 3 |
| fitted `D/lambda` | 1.05–1.10 (08-28), 0.90–0.95 (08-30), kept at 1 | HANDOFF section 3 |
| results | 38.3 / ~143 / 88–92 deg (28-08), 40.1 / 68.5 / 94.8 deg (28-09) for set-ups 30 / 60(150) / 90 | HANDOFF |

Two consequences matter for the 2-D step. First, the window noise is small (0.012 cyc), so
precision is not the problem. Second, the systematic part is large (0.05–0.09 cyc on authentic
PRNs, of which a 5–10 % `D/lambda` error alone is up to 0.1 cyc at `|cos alpha| = 1`), so the
2-D method lives or dies on baseline-vector calibration and antenna phase-centre behaviour.

## 2. Geometry of the 2-D problem

### 2.1 Terms to add to `CONTEXT.md`

- **Source direction**: the unit vector `u_s` (or the pair azimuth `az_s`, elevation `el_s`, ENU
  frame, azimuth clockwise from north) of the common spoofing source. This is the new target; keep
  "baseline-relative AoA" for the per-baseline cone angle.
- **Baseline vector**: `b_j = D_j b_hat_j`, the vector from the first to the second antenna of
  pair `j`, in ENU. Today only its azimuth (`navsat_azimuth_bias_deg`) and `k = D/lambda` exist.
- **Direction candidate**: one `(az, el)` obtained from one combination of per-baseline branches.
  Generalises "AoA branch candidate" (`candidate_aoa_deg`) to two baselines.
- **Loop closure**: for three antennas `SD_12 + SD_23 + SD_31 = 0` for every satellite and epoch,
  up to integer cycles. A consistency check, not new information.

### 2.2 Equations

With `u = (cos el sin az, cos el cos az, sin el)` in ENU and baseline unit vector `b_hat_j`:

```text
c_j(i)   = b_hat_j . u_i                                   direction cosine of signal i on baseline j
f_j(i)   = wrap( k_j (c_j(i) - c_j(r)) ) + noise            Eq. 18 on baseline j, reference r
c_j(i)   = (f_j(i) + m_j) / k_j + c_j(r)                   Eq. 21 on baseline j
```

For a horizontal baseline at azimuth `a_j`, `c_j = cos(el) cos(az - a_j)`, which is exactly what
`compute_navsat_aoa` computes today. The generalisation is to allow a tilt: `b_hat_j` is any unit
vector and `c_j(r)` is `b_hat_j . u_r` with `u_r` from NAV-SAT azimuth/elevation of the reference.

Two non-collinear baselines fix the component of `u` in their plane:

```text
[ b_hat_1^T ; b_hat_2^T ] u_par = [ c_1 ; c_2 ]           2 x 3 system, solve in the baseline plane
|u_par|^2 = (c_1^2 + c_2^2 - 2 c_1 c_2 cos gamma) / sin^2 gamma     (gamma = angle between baselines)
u = u_par +- sqrt(1 - |u_par|^2) n_hat                     n_hat = normal of the baseline plane
```

For a horizontal array `n_hat` is "up" and the minus solution is below the horizon, so it is
dropped. Then `el = arccos |u_par|` and `az = atan2(u_E, u_N)`. Combinations with `|u_par| > 1`
are physically impossible and are rejected, which is the 2-D form of today's "arccos argument
outside [-1, 1]" rule in `generate_aoa_candidates`.

Three or more antennas in one plane give extra baselines but only two independent components;
with four antennas (three independent baselines in a plane) the third equation is a consistency
check that can reject integer combinations, and a non-coplanar fourth antenna makes the system
rank 3 and determines `u` without the mirror (then `|u| = 1` becomes the check).

### 2.3 The constrained integer problem, and its name in the literature

Per epoch and spoof group, the unknowns are `u_s` with `|u_s| = 1` and one integer per baseline
(shared by all spoofed PRNs because they share `u_s`). This is the GNSS-compass problem with the
roles swapped: Teunissen's constrained integer least squares (C-LAMBDA) fixes the baseline length
and solves for its direction from many satellites; here the baseline vectors are known and the
direction of one "satellite" (the source) is solved from several baselines. The 2026 Aerospace
paper by Chen et al. (section 8) uses exactly this constrained-LAMBDA framing for a three-antenna
spoofing detector. Because `k <= 1`, the integer set is tiny (at most 2 per baseline) and brute
force enumeration, as in `select_joint_source_aoa`, remains the right tool. Full RTK ambiguity
resolution stays out of scope (`MISSION.md`).

## 3. Why the length matters: candidates per baseline

`c_j in [-1, 1]`. The wrapped observation fixes `c_j` modulo `1/k_j`:

| `k = D/lambda` | candidates for `c_j` | comment |
|---|---|---|
| 1 (today) | `c` and `c - sign(c)` | always 2, the mirror of HANDOFF section 4 |
| 0.75 | `c`, and `c -+ 4/3` only if `|c| >= 1/3` | 1 or 2 |
| 0.5 | `c` only (`c +- 2` is outside `[-1, 1]`) | always unique |
| 2 (long) | `c`, `c +- 0.5`, `c +- 1`, ... | 4 to 5, needs a short baseline to resolve |

Longer baselines sharpen the cosine (`sigma_c = sigma_f / k`) and multiply the candidates; the
classic interferometer answer is a short unambiguous baseline plus a long precise one (Yang et al.
2023 use exactly this with a six-element array). With the measured noise, precision is not the
bottleneck, so the shortest unambiguous length `k = 0.5` is attractive, and that is what the
Monte Carlo confirms.

## 4. Monte Carlo (script: `simulate_multibaseline_direction.py`)

Noise model: fractional DD noise `sigma_f = 0.012 cyc` per PRN per epoch on each baseline (the
measured 28-09 value), reference direction known, far field. Candidates are enumerated as in
section 2.2 with a consistency tolerance of 0.04 cyc (about 3 sigma) for redundant baselines.

### 4.1 Ambiguity: share of the upper hemisphere with exactly one physical direction candidate

Grid: azimuth 0–359 deg step 5, elevation 2–88 deg step 2. "Prior" column: true elevation
<= 15 deg and only candidates with elevation <= 25 deg kept (a "ground spoofer" prior).

| geometry (receivers) | unique | 2 | 3 | >= 4 | unique, el <= 15 | unique with prior |
|---|---|---|---|---|---|---|
| A. one baseline `lambda` (2, today) | 0 % | 98 % | 2 % | 0 % | 0 % | n/a |
| B. L-shape, both `lambda` (3) | 0 % | 11 % | 55 % | 34 % | 0 % | 67 % |
| C. equilateral triangle, side `lambda` (3) | 10 % | 14 % | 76 % | 0 % | 0 % | 76 % |
| D. L-shape, `lambda` N + `lambda/2` E (3) | 21 % | 79 % | 0 % | 0 % | 31 % | 89 % |
| **E. L-shape, both `lambda/2` (3)** | **100 %** | 0 % | 0 % | 0 % | 100 % | 100 % |
| F. L-shape, both `0.75 lambda` (3) | 26 % | 62 % | 10 % | 2 % | 0 % | 68 % |
| G. L with `lambda/2` and collinear `lambda` on each arm (4) | 100 % | 0 % | 0 % | 0 % | 100 % | 100 % |
| H. L-shape `lambda` + vertical `lambda/2` (4, 3-D) | 80 % | 19 % | 1 % | 0 % | 83 % | 83 % |
| I. two headings of today's rig, 90 deg apart (2, two captures) | 0 % | 11 % | 55 % | 34 % | 0 % | 67 % |
| J. three headings 0/60/120 deg (2, three captures) | 10 % | 14 % | 76 % | 0 % | 0 % | 76 % |
| K. square side `lambda` with diagonal (4) | 0 % | 11 % | 55 % | 34 % | 0 % | 67 % |
| L. vertical L: N `lambda/2` + up `lambda/2` (3) | 3 % | 97 % | 0 % | 0 % | 3 % | 3 % |
| M. equilateral `lambda/2` tilted 45 deg (3) | 49 % | 51 % | 0 % | 0 % | 49 % | 83 % |
| O. tilted L: N `lambda/2` + 45 deg-up `lambda/2` (3) | 49 % | 51 % | 0 % | 0 % | 49 % | 83 % |
| **N. L-shape `lambda/2` + vertical `lambda/2` (4, 3-D)** | **100 %** | 0 % | 0 % | 0 % | 100 % | 100 % |

Reading: any geometry whose baselines are all `D = lambda` (B, C, I, J, K) stays ambiguous almost
everywhere, whatever its shape. Only `lambda/2` baselines (E, G, N) are unique everywhere. The
vertical L (row L) has a left/right mirror about its plane; a tilted plane (M, O) halves the
ambiguity but does not remove it.

Worked example, source at az 40 deg, el 10 deg: geometry B returns four candidates, (236, 64),
(111, 47), (334, 33), (40, 10) deg. Geometry D returns (111, 47) and (40, 10): the wrong one is
37 deg higher, so an elevation prior separates them. Geometry E returns (40, 10) only.

### 4.2 Noise only, correct candidate, 1 PRN, 1 epoch (`sigma_f = 0.012 cyc`): 1 sigma az / el in deg

| geometry | el 5 | el 15 | el 30 | el 45 | el 60 |
|---|---|---|---|---|---|
| B. L `lambda`,`lambda` | 0.7 / 4.6 | 0.7 / 2.8 | 0.8 / 1.4 | 1.0 / 1.0 | 1.4 / 0.8 |
| D. L `lambda`,`lambda/2` | 1.1 / 5.6 | 1.1 / 4.8 | 1.2 / 2.2 | 1.6 / 1.6 | 2.2 / 1.3 |
| E. L `lambda/2`,`lambda/2` | 1.4 / 6.3 | 1.4 / 6.1 | 1.6 / 2.8 | 1.9 / 2.0 | 2.8 / 1.6 |
| H. L `lambda` + vertical `lambda/2` | 0.7 / 1.4 | 0.7 / 1.3 | 0.8 / 1.3 | 1.0 / 1.1 | 1.4 / 0.9 |
| N. L `lambda/2` + vertical `lambda/2` | 1.4 / 1.4 | 1.4 / 1.4 | 1.6 / 1.4 | 2.0 / 1.4 | 2.8 / 1.4 |

Averaged over a 4 PRN, 100 epoch window (independent noise) every entry drops below 0.1 deg in
azimuth and below 0.9 deg in elevation. Noise is not the limit; the rows differ in the next table.

### 4.3 A common 0.03 cycle bias on every baseline (does not average out): worst-case shift az / el in deg

| geometry | el 5 | el 15 | el 30 | el 45 | el 60 |
|---|---|---|---|---|---|
| B. L `lambda`,`lambda` | 2.4 / 12.5 | 2.5 / 15.0* | 2.8 / 5.3 | 3.4 / 3.5 | 4.9 / 2.8 |
| D. L `lambda`,`lambda/2` | 3.9 / 16.7 | 4.0 / 15.0* | 4.4 / 8.9 | 5.4 / 5.7 | 7.7 / 4.5 |
| E. L `lambda/2`,`lambda/2` | 4.9 / 19.3 | 5.0 / 15.0* | 5.6 / 12.0 | 6.8 / 7.4 | 9.7 / 5.8 |
| H. L `lambda` + vertical `lambda/2` | 2.4 / 3.8 | 2.5 / 4.0 | 2.8 / 4.2 | 3.4 / 4.2 | 4.9 / 3.9 |
| N. L `lambda/2` + vertical `lambda/2` | 4.9 / 4.2 | 5.0 / 4.9 | 5.6 / 5.6 | 6.8 / 6.0 | 9.7 / 6.0 |

`*` the estimate collapses onto the horizon (`|u_par| > 1` clipped). A 0.03 cyc bias is 5.7 mm of
path, the size of a patch-antenna phase-centre variation or a 3 % `D/lambda` error at `|c| = 1`;
the present rig shows 0.05–0.09 cyc before calibration. Horizontal arrays therefore give good
azimuth and unreliable elevation for a low source; only the 3-D rows keep elevation within 4–6 deg.

## 5. Error budget specific to this rig

1. **PRN-specific tracking noise**, 0.006 cyc (within-group dispersion). Averages over PRNs and
   epochs. Harmless.
2. **Reference-satellite single-difference error**, roughly `sqrt(0.012^2 - 0.006^2)` = 0.010 cyc.
   Common to every spoofed PRN (the manuscript's Eq. 51 discussion says the same), averages over
   epochs only, and its multipath part varies over minutes. Choose a high-elevation reference, or
   fuse several authentic references (`reconstruct_multi_reference_aoa` already exists).
3. **Baseline-vector error**: azimuth, tilt, and `D/lambda` of each baseline. Pure bias. Today
   `fit_baseline_azimuth` fits azimuth and `D/lambda` on a grid from authentic inter-satellite
   wrapped DDs; extend it to the tilt (three parameters per baseline) and run it per pair. With
   three antennas the two fitted vectors must satisfy the loop closure with the third pair.
4. **Antenna phase-centre variation**: depends on the direction of arrival, identical for all
   spoofed PRNs, so a pure bias of a few mm (0.02–0.05 cyc). The 2023 and 2025 Remote Sensing
   papers (section 8) identify this as the dominant error of low-cost short-baseline direction
   finding and correct it with a per-antenna az/el map. Calibrate it from the authentic satellites
   (residual of fit 3 as a function of az/el) and apply the correction at the estimated source
   direction. Keep the three antennas of the same model and the same orientation.
5. **NAV-SAT quantisation**: azimuth and elevation are integer degrees, up to 0.5 deg each, i.e.
   up to 0.009 cyc in `c_j(r)`. Acceptable; computing the reference direction from the broadcast
   ephemeris (RXM-SFRBX) and the NAV-PVT position removes it if needed.
6. **Mutual coupling at 9.5 cm spacing**: unknown for the Tallysman housings; shows up as a
   direction-dependent phase-centre shift (item 4). Measure it with the authentic-satellite fit
   before trusting any `lambda/2` result; this is the main risk of the recommended geometry.
7. **Far field**: 0.19 m aperture against a transmitter several metres away is far field; keep the
   transmitter distance much larger than the array as the manuscript already does.

## 6. Options compared

| option | hardware | ambiguity | elevation near horizon | motion / prior needed | comment |
|---|---|---|---|---|---|
| keep one baseline + rotate rig through broadside during a capture | turntable | resolved by continuity (cosine passes 0, mirror jumps 0 to 180 deg) | poor (horizontal) | platform motion, heading per epoch | contradicts the manuscript's "no platform motion" claim; mechanical complexity |
| two captures, rig rotated 90 deg (row I) | none | 3–4 candidates | poor | prior or known geometry | good pilot for the solver and calibration only |
| 3 receivers, L or triangle with `D = lambda` (B, C) | +1 ZED-F9P + antenna | 3–4 candidates | poor | prior | same `unresolved` problem as today, now in 2-D |
| **3 receivers, L with both `D = lambda/2` (E)** | +1 ZED-F9P + antenna | **unique everywhere** | poor | **none** | recommended first experiment; 2x noise, coupling risk |
| 3 receivers, L with `lambda` N + `lambda/2` E (D) | +1 | 2 candidates, far apart | poor | weak elevation prior | fallback if one `lambda/2` arm is enough physically |
| 4 receivers, `lambda/2` L + raised antenna (N) | +2 | unique everywhere | good (4–6 deg under bias) | none | the configuration for an azimuth-and-elevation claim |
| 4 receivers, `lambda` L + raised antenna (H) | +2 | 80 % unique | good | prior for 20 % | best precision, keeps `lambda` arms |

Practical notes on the raised antenna: do not stack it directly above another patch (ground-plane
shadowing of the lower antenna's sky view, which kills the authentic reference). Mount it on a
short mast at one corner, about `lambda/2` higher and laterally offset; any non-coplanar fourth
antenna gives the vertical component, and `fit_baseline_vector` recovers the actual vector.

## 7. What changes in the repo (design, not implemented)

- **Capture**: three logs per capture (`COM3/COM4/COM5` or similar). `probe_capture` runs on each
  pair; the data check must require valid phase on the spoof PRNs in all three receivers.
- **Sync**: `sync_receiver_epochs` is pairwise; sync `rx1–rx2` and `rx1–rx3`, intersect epochs,
  keep dropouts (never bridge them).
- **Per-baseline AoA**: run the existing pair pipeline twice with the same `reference_svid`,
  `k_j`, and `single_difference_order`; the `candidate_cos_alpha` table of each run is `c_j`.
- **Baseline vectors**: `fit_baseline_azimuth` becomes `fit_baseline_vector(az, tilt, D/lambda)`
  per pair, plus a loop-closure check across the three pairs. `compute_navsat_aoa` and
  `compute_reference_alpha_by_epoch` take a unit vector instead of an azimuth bias.
- **Direction solver** (new module, e.g. `aoa_direction.py`): per epoch, enumerate the product of
  per-baseline branch sets for the spoof group, solve section 2.2, drop candidates with
  `|u_par| > 1` or below the horizon, apply the loop-closure/redundancy consistency check, then
  select jointly over the group and the window in direction-cosine space as
  `select_joint_source_aoa` does today. Keep `chosen_by` with the same meaning: `data` when one
  candidate survives, `prior` when an azimuth/elevation (or elevation-only) prior picks, else
  `unresolved` with every alternative listed. Report the alternatives in the result table.
- **Detection**: `summarize_spoof_branch_detection` compares angles on one cone; the 2-D version
  compares the angular distance between the DD-method direction and the ephemeris direction of each
  PRN (great-circle angle), same 10 deg threshold, with the sky-coverage argument of section 0.
- **Tests**: synthetic three-receiver fractional DDs from a known `u_s` and known baseline vectors;
  (1) `lambda/2` L returns one candidate, `chosen_by = data`; (2) `lambda` L returns 3–4 and
  `unresolved` without a prior; (3) a 0.03 cyc bias moves elevation but not azimuth near the horizon;
  (4) loop closure rejects a wrong integer on one pair.
- **Vocabulary**: add the four terms of section 2.1 to `CONTEXT.md`; keep "baseline-relative AoA"
  for per-baseline cone angles and never call the 1-D angle an azimuth.

## 8. Literature (checked 2026-10-12)

Group's own work
- Hoang Van, Duong Huu, Mai Xuan, Nguyen Van, Pham Ngoc, Nguyen Dinh, La The, "Baseline-Referenced
  GNSS Spoofing-Source AoA Estimation Using Carrier-Phase Double Differences From Two Independent
  Receivers", TAES manuscript, Zotero `M5DA6HM8`. States the 1-D limitation (Section III-D) and
  that reference-related errors do not average over PRNs (after Eq. 51). The next paper is its
  direct 2-D extension.
- Hoang et al., "Enhancing GNSS spoofing detection by carrier-smoothing pseudorange double
  difference measurements", GPS Solutions, 2025, Zotero `JEVNHCSI`.

Two-antenna carrier-phase methods (detection, no direction)
- Psiaki, O'Hanlon, Powell, Bhatti, Humphreys, Schofield, "GNSS Spoofing Detection using
  Two-Antenna Differential Carrier Phase", ION GNSS+ 2014,
  https://radionavlab.ae.utexas.edu/images/stories/files/papers/Psiaki_WROD_2014.pdf. Spoofed
  model: one direction of arrival, single differences identical up to integers; baseline 0.74
  wavelength; the non-spoofed fit is "closely related to CDGPS attitude determination" and yields the
  baseline direction as a by-product.
- Borio, Gioia, "A Dual-antenna Spoofing Detection System Using GNSS Commercial Receivers", ION
  GNSS+ 2015, https://www.researchgate.net/publication/282099794 ; Borio, Gioia, "A Dual Antenna GNSS
  Spoofing Detector Based on the Dispersion of Double Difference Measurements",
  https://ieeexplore.ieee.org/document/8642705/ . Low-cost receivers, DD dispersion, no synchronisation.
- Rothmaier, Chen, Lo, Walter, "GNSS spoofing detection through spatial processing", NAVIGATION 2021,
  https://doi.org/10.1002/navi.420 .
- Zhang, Ding, Xia, Liu, Yao, "INS-Aided Multi-Antenna GNSS Carrier Phase Double Difference Spoofing
  Detection", IEEE Access 2023, https://doi.org/10.1109/access.2023.3247968 . Names the dual-antenna
  "spatial ambiguity" explicitly; detection coverage 95.2 % of the sky at a 1-wavelength baseline and
  99.3 % at 5 wavelengths, with INS supplying the attitude.

Three-antenna and array methods (closest prior art for the next step)
- Chen, Wang, Fang, Jiang, Gao, Xu, "A Real-Time Spoofing Detection Method Using Three Low-Cost
  Antennas in Satellite Navigation", Electronics 2024, https://doi.org/10.3390/electronics13061134 .
  Three collinear antennas on a common oscillator, baselines d12, d23, d13 with d23 - d12 below half
  a wavelength to keep the integer ambiguity consistent, 1 m baseline. Collinear, so still a cone;
  shows the long-plus-short trick on one axis.
- Chen, Wang, Fang, Gao, Xu, "High-Precision Spoofing Detection Using an Auxiliary Baseline
  Three-Antenna Configuration for GNSS Systems", Aerospace 2026, https://doi.org/10.3390/aerospace13040339 .
  "One-primary-two-auxiliary" antennas, constrained LAMBDA with the rigid baseline length,
  chi-square SSR test, minimum detectable direction deviation 0.33 deg, single epoch, commercial
  receivers. The constrained-integer formulation of section 2.3.
- Yang, Jin, Xu, Che, Zhen, "Satellite Navigation Spoofing Interference Detection and Direction
  Finding Based on Array Antenna", Sensors 2023, https://doi.org/10.3390/s23031604 . Six-element
  array, long- and short-baseline ambiguity search over the full circle, direction within 2 deg.
- "A GNSS Spoofing Detection and Direction-Finding Method Based on Low-Cost Commercial Board
  Components", Remote Sensing 2023, https://doi.org/10.3390/rs15112781 . Four low-cost boards and
  antennas (A, B, C an equilateral triangle plus D), about 5 deg accuracy; identifies unstable
  phase centres and inconsistent sampling times across boards as the main problems; cites Zhang et al.
  (J. Natl. Univ. Def. Technol. 2014) for "three non-collinear antennas is the minimum to resolve the
  DOA ambiguity".
- Jin et al., "A Spoofing Detection and Direction-Finding Approach for GNSS Signals Using
  Off-the-Shelf Anti-Jamming Antennas", Remote Sensing 2025, https://doi.org/10.3390/rs17050864 .
  Differential phase-centre correction for coupled array elements, direction finding to about 1 deg.
- Meurer, Konovaltsev, Cuntz, Hättich, "Robust Joint Multi-Antenna Spoofing Detection and Attitude
  Estimation using Direction Assisted Multiple Hypotheses RAIM", ION GNSS 2012,
  https://elib.dlr.de/79200 ; Appel, Konovaltsev, Meurer, "Robust Spoofing Detection and Mitigation
  based on Direction of Arrival Estimation", ION GNSS+ 2015, https://elib.dlr.de/99721 . DOA from a
  2 x 2 array (GALANT) compared with ephemeris-predicted DOA, attitude estimated jointly.
- Lehmann, Larsen, Elsholm, Olesen, "Real-World Jammer and Spoofer Localization Using a Low-Cost
  Array-Based Software-Defined Radio", NAVIGATION 2026, https://doi.org/10.33012/navi.735 , Zotero
  `9FA72Z95`. Five-channel coherent KrakenSDR, uniform circular array of radius 8 cm (9.4 cm spacing,
  about half a wavelength), MUSIC, scans in the horizontal plane assuming low elevation; repeat
  bearings differ by a few degrees. A coherent-sampling alternative to independent receivers.
- Esswein, Psiaki, "Classification of Authentic and Spoofed GNSS Signals Using a Calibrated Antenna
  Array", NAVIGATION 2025, https://doi.org/10.33012/navi.675 ; van der Merwe, Lipka, Rügamer,
  "Enhanced Spatial Spoofing Detection With and Without Direction of Arrival Estimation", IEEE TAES
  2023, https://doi.org/10.1109/taes.2023.3262455 .

Two-dimensional direction with motion or inertial aiding
- Zhang, Wang, Wu, "Two-Dimensional Directions Determination for GNSS Spoofing Source Based on
  MEMS-Based Dual-GNSS/INS Integration", Remote Sensing 2024, https://doi.org/10.3390/rs16234568 .
  Dual antenna plus MEMS IMU to get azimuth and elevation of the source; the closest paper to the
  goal, and the one the next paper must differentiate from (no INS, no motion, third receiver instead).
- Liu, Yang, Chen, Liao, Zhu, "Localization of GNSS Spoofing Interference Source Based on a Moving
  Array Antenna", Remote Sensing 2023, https://doi.org/10.3390/rs15235497 . DD carrier phase to
  separate spoofed signals, then direct position estimation from single differences along a path.
- "GNSS Spoofing Detection via the Intersection Angle between Two Directions of Arrival in a Single
  Rotating Antenna", Sensors 2024, https://doi.org/10.3390/s24041116 . Rotation-based; detection only.

Multi-receiver localisation (pseudorange / clock based, for context)
- Wen, Li, Lu, "A Flexible GNSS Spoofer Localization System", NAVIGATION 2022,
  https://doi.org/10.33012/navi.511 (asynchronous receivers, pseudorange DD discrimination).
- Chen, Ni, Cheng, Lei, Jia, Fu, "Detection and Orientation of GNSS Spoofing Based on Positioning
  Solutions of Three Receivers", IEEE Access 2023, https://doi.org/10.1109/access.2023.3262996
  (direction from single-difference clock offsets, synchronized receivers).

Theory
- Teunissen, "Integer least-squares theory for the GNSS compass", J. Geodesy 2010,
  https://doi.org/10.1007/s00190-010-0380-8 ; Teunissen, "The affine constrained GNSS attitude model
  and its multivariate integer least-squares solution", J. Geodesy 2012,
  https://doi.org/10.1007/s00190-011-0538-z . C-LAMBDA / MC-LAMBDA, the dual of section 2.3.
- Psiaki, Humphreys, "GNSS Spoofing and Detection", Proc. IEEE 2016, Zotero `JF92RLG5` (survey).

## 9. Proposed plan

1. **Decide the claim**: azimuth only (3 receivers) or azimuth and elevation (4 receivers, or
   accept 10–20 deg elevation error for a low source). This fixes the hardware.
2. **Phase 0, no hardware** (1 day): implement `fit_baseline_vector` and the direction solver with
   synthetic data and tests; run the 90-deg-rotated two-capture pilot on the existing rig to debug
   calibration and the candidate enumeration (expect `unresolved` with 3–4 candidates, correct one
   among them).
3. **Phase 1, third receiver** (first new capture): horizontal L, both arms `lambda/2`, same
   Tallysman model, baseline vectors fitted from authentic PRNs before the spoof window; transmitter
   at measured azimuth and elevation (place it at 20–40 deg elevation for at least one capture so
   the elevation estimate can be checked outside the ill-conditioned horizon). Repeat the 30/60/90
   set-ups as azimuth set-ups. Acceptance: `chosen_by = data` in every window, azimuth within 5 deg
   of the set-up, authentic PRNs' 2-D directions within 10 deg of ephemeris.
4. **Phase 2, if elevation is claimed**: raise a fourth antenna by about `lambda/2` on a corner mast;
   verify the loop closure and the fitted vertical component; repeat Phase 1 captures.
5. **Paper framing**: "from one direction cosine to a source direction with three independent
   receivers, no common clock, no INS, no motion"; differentiate from Zhang–Wang–Wu 2024 (INS),
   Chen et al. 2024/2026 (common oscillator, collinear, detection-only), and array/SDR approaches
   (calibrated coherent channels).

Open questions for the owner: physical minimum centre spacing of the Tallysman antennas; whether
a fourth ZED-F9P is available; whether the transmitter can be raised for an elevation test; whether
the shielded or over-the-air site allows a 0.19 m by 0.19 m L plus a mast without new multipath.
