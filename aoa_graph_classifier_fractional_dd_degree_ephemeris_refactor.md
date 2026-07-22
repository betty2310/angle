# Fractional-DD / Degree-Ephemeris Graph Refactor Note

## Purpose

This note records the agreed refactor for
`AOA_Three_Scenario_DD_Ephemeris_Graph_Classifier.ipynb`.

The graph method will combine two separate pairwise tests:

1. Fractional carrier-phase DD collapse, evaluated in cycles.
2. Ephemeris-based absolute AoA separation, evaluated in degrees.

The two tests do not need the same unit because they are not subtracted from one
another. They are independent Boolean conditions combined to assign the graph
relation for a satellite pair.

## Refactor Decision

For satellites `i` and `j`, use the synchronized inter-receiver single
differences

$$
s_{i,k}=\phi_{2,i,k}-\phi_{1,i,k}
$$

to calculate the direct fractional pairwise DD:

$$
f_{ij,k}=\operatorname{wrap}(s_{i,k}-s_{j,k}),
$$

where

$$
\operatorname{wrap}(x)=\left((x+\tfrac12)\bmod 1\right)-\tfrac12.
$$

The fractional-DD collapse statistic is

$$
C_{ij}^{DD}
=\operatorname{median}_{k}\left|f_{ij,k}\right|
\quad [\text{cycles}].
$$

This quantity is used directly. The graph notebook does not reconstruct an
absolute DD-method AoA and does not perform integer-branch selection or dynamic
programming.

The existing explanation that constructs two DDs against a selected reference
satellite and then subtracts them is redundant for this notebook. The
implementation already forms `s_i - s_j` directly. Remove the `DD_ir - DD_jr`
derivation and avoid presenting reference-satellite elimination as a separate
algorithmic step.

## Ephemeris-Based AoA

Continue calculating the absolute baseline-relative ephemeris AoA from satellite
azimuth and elevation using the established geometry.

For a horizontal antenna baseline with azimuth convention `beta`,

$$
\cos\alpha_{i,k}^{eph}
=\cos E_{i,k}\cos(A_{i,k}-\beta),
$$

$$
\alpha_{i,k}^{eph}
=\arccos\!\left[
\cos E_{i,k}\cos(A_{i,k}-\beta)
\right]
\quad [^\circ].
$$

For each satellite pair, calculate the absolute ephemeris-AoA separation:

$$
a_{ij,k}^{eph}
=\left|
\alpha_{i,k}^{eph}-\alpha_{j,k}^{eph}
\right|
\quad [^\circ],
$$

and aggregate it over common valid epochs:

$$
A_{ij}^{eph}
=\operatorname{median}_{k}a_{ij,k}^{eph}
\quad [^\circ].
$$

Do not convert this ephemeris-AoA separation into predicted carrier cycles for
the primary graph rule. Do not wrap the ephemeris separation modulo one cycle.

## Graph Relations

Use two separately calibrated thresholds:

$$
\tau_c\quad [\text{cycles}]
$$

for fractional-DD collapse, and

$$
\tau_\alpha\quad [^\circ]
$$

for ephemeris-AoA separation.

Define

$$
B_{ij}=\mathbf 1\!\left[C_{ij}^{DD}\le\tau_c\right]
$$

and

$$
G_{ij}=\mathbf 1\!\left[A_{ij}^{eph}\ge\tau_\alpha\right].
$$

The graph relation is

$$
M_{ij}=
\begin{cases}
2,
&B_{ij}=1\ \text{and}\ G_{ij}=1,
\\
1,
&B_{ij}=1\ \text{and}\ G_{ij}=0,
\\
0,
&B_{ij}=0.
\end{cases}
$$

Interpret the relations as follows:

| Code | Display | Condition | Interpretation |
| --- | --- | --- | --- |
| `2` | Red edge | Fractional DD collapsed and ephemeris AoAs are far apart | Pairwise common-source spoof candidate |
| `1` | Orange dashed edge | Fractional DD collapsed and ephemeris AoAs are close | Collapsed but angularly non-informative pair |
| `0` | No edge | Fractional DD did not collapse | No pairwise common-source relation detected |

The connected-component and corroboration stages can remain unchanged after the
pair-evidence column names and threshold units are updated.

## Intended Claim

The refactored red edge means:

> The pair has nearly identical fractional inter-antenna DD signatures even
> though its two ephemeris-based baseline-relative AoAs are separated by at
> least the configured angular threshold.

Use the term **fractional-DD collapse**, not **absolute DD-AoA collapse**. The
notebook deliberately avoids reconstructing absolute DD-method AoAs because
that would require a reference-angle anchor and carrier-phase branch selection.

A red edge is pairwise evidence compatible with a common received direction; it
is not sufficient by itself to declare both endpoints spoofed. Continue using
the existing pair-alert and corroborated-component distinction.

## Required Notebook Changes

1. Replace `## 3. Reference-independent DD measurement` with a section named
   `## 3. Pairwise fractional-DD collapse`.
2. Remove the `DD_ir - DD_jr` reference-cancellation derivation.
3. State directly that the input is the synchronized pairwise quantity
   `wrap(s_i - s_j)`.
4. Retain ephemeris AoA in degrees after the azimuth/elevation-to-baseline-angle
   calculation.
5. Replace the ephemeris predicted-cycle separation with
   `abs(ephemeris_aoa_i_deg - ephemeris_aoa_j_deg)`.
6. Aggregate ephemeris angular separation only over epochs jointly valid for the
   same satellite pair.
7. Replace the cycle-based ephemeris threshold with a degree threshold.
8. Update graph labels, heatmaps, tables, saved outputs, and tests to show the
   unit of every threshold and metric explicitly.

## Suggested Naming Changes

| Current name | Refactored name |
| --- | --- |
| `EXPECTED_SEPARATION_THRESHOLD_CYCLES` | `EPHEMERIS_AOA_SEPARATION_THRESHOLD_DEG` |
| `median_expected_separation` | `median_ephemeris_aoa_separation_deg` |
| `geometry_informative` | `ephemeris_aoa_separated` |
| `ephemeris_different` | `ephemeris_aoa_far` |
| `common_source_collapse_cycles` | `fractional_dd_collapse_cycles` |
| `median_common_source_collapse` | `median_fractional_dd_collapse_cycles` |

## Pseudocode

```python
for i, j in satellite_pairs:
    common_epochs = jointly_valid_epochs(i, j)

    fractional_dd = wrap(
        single_difference_cycles[i] - single_difference_cycles[j]
    )
    dd_collapse_cycles = median(abs(fractional_dd[common_epochs]))

    ephemeris_separation_deg = median(abs(
        ephemeris_aoa_deg[i][common_epochs]
        - ephemeris_aoa_deg[j][common_epochs]
    ))

    dd_collapsed = dd_collapse_cycles <= collapse_threshold_cycles
    ephemeris_far = (
        ephemeris_separation_deg >= ephemeris_aoa_separation_threshold_deg
    )

    if dd_collapsed and ephemeris_far:
        relation = "informative_common_source"
    elif dd_collapsed:
        relation = "angularly_noninformative_collapse"
    else:
        relation = "noncollapse"
```

## Threshold Calibration

The fractional-DD threshold and ephemeris-angle threshold describe different
statistics and must be calibrated separately.

The paper's reported `10 deg` threshold compares one satellite's reconstructed
DD-method AoA with its ephemeris-based AoA:

$$
\left|\hat\alpha_i^{DD}-\alpha_i^{eph}\right|.
$$

The graph instead thresholds the separation between two ephemeris AoAs:

$$
\left|\alpha_i^{eph}-\alpha_j^{eph}\right|.
$$

Therefore, the paper's `10 deg` value must not be adopted automatically as
`tau_alpha`. Select and freeze `tau_alpha` using independent authentic and
spoofing sessions, then report pair-level and component-level sensitivity.
