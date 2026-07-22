# Mixed-Tracking Placeholder Values Report

## Scope and evidence

This report maps the `XX` placeholders in Section 4.4 of [journal.pdf](journal.pdf) to the latest executed implementation in [AOA_Paper_Refactored.ipynb](AOA_Paper_Refactored.ipynb), specifically the **Minimal Method Extension: Common-Direction Penalty** section.

The numerical evidence comes from these notebook-generated files:

- `outputs/aoa_paper/sp-partial-19-06/18-06-prn23-pairwise/hybrid_best_m_common_direction_summary.csv`
- `outputs/aoa_paper/sp-partial-19-06/18-06-prn23-pairwise/hybrid_best_m_adjusted_classification_scores.csv`
- `outputs/aoa_paper/sp-partial-19-06/18-06-prn23-pairwise/hybrid_best_m_common_direction_group_report.csv`
- `outputs/aoa_paper/sp-partial-19-06/18-06-prn23-pairwise/hybrid_best_m_adjusted_classification_groups.csv`
- `outputs/aoa_paper/sp-partial-19-06/18-06/manuscript_mixed_case/reference_penalty_runs/penalty_reference_summary.csv`
- `outputs/aoa_paper/sp-partial-19-06/18-06/manuscript_mixed_case/reference_penalty_runs/penalty_aggregated_classification_scores.csv`
- `outputs/aoa_paper/sp-partial-19-06/18-06/manuscript_mixed_case/reference_penalty_runs/penalty_pairs_aggregated_across_references.csv`

The PRN 23 notebook and result files were written on 2026-07-20 at 03:30:29-03:30:30 +07. The notebook was subsequently executed with reference PRNs 12, 11, and 9 by `run_penalty_reference_notebooks.py`. The runner preserves the source notebook, writes a separate executed copy and result directory for each reference, and median-aggregates valid pair residuals across references.

## Execution model and manuscript aggregation

Each individual notebook execution is a fixed-reference diagnostic. Its reference has no DD reconstruction and is excluded from that run, leaving 10 classified satellites and 9 partners per score.

The manuscript-style aggregate combines the four reference runs

\[
R=\{12,23,11,9\}
\]

by taking the median of every pair's available per-reference \(q_{ij}\) and \(c_{ij}\) values. Pairs involving no reference candidate have four contributing references, pairs involving one candidate have three, and pairs formed by two reference candidates have two. This produces all 55 unique pairs among the 11 manuscript satellites and restores 10 partners per satellite.

The score is no longer the unpenalized score in draft Eq. (38). It is

\[
C_i^{(\mathrm{adj})}=\max\left(0,C_i^{(q)}-C_i^{(c)}\right),
\]

where the pairwise journal-consistency test uses \(\tau_q=0.20\), the common-direction penalty uses \(\tau_c=0.02\), and the satellite decision threshold remains \(\eta=0.5\). A pair satisfying both tests has net contribution zero.

These thresholds are dimensionless direction-cosine values. The notebook explicitly treats them as scenario demonstration values, not authentic-calibrated operating thresholds.

## Equation (47) and calibration-percentile placeholders

The draft currently says:

> \(\tau_p=XX\), corresponding to the \(XX\)-th percentile of the authentic pairwise-residual distribution.

Do not fill both placeholders with numbers. The current implementation does not derive its threshold from an authentic calibration percentile.

Recommended replacement:

> For this fixed-reference diagnostic, the pairwise geometry threshold was manually fixed at \(\tau_q=0.20\). The common-direction penalty threshold was set to \(\tau_c=0.02\), within the observed gap between the six compact counterfeit-counterfeit residuals (maximum 0.0108) and the next-smallest pairwise common-direction residual (0.0367). These are scenario demonstration values rather than authentic-calibrated operating thresholds.

Therefore:

| Draft placeholder | Current value or action |
| --- | --- |
| Eq. (47), `tau_p = XX` | Use \(\tau_q=0.20\), and add \(\tau_c=0.02\). If the manuscript retains \(\tau_p\) notation, the numeric pairwise value is still 0.20, but the notebook calls it \(\tau_q\). |
| `XX-th percentile` | No value. Delete or rewrite the percentile claim because the threshold is manually fixed. |

## Table 3 replacement values - four-reference aggregate

The existing `Consistent pairs` and `Score C_i` columns should be revised because the new method subtracts a penalty count. A transparent replacement is to report the positive count \(A\), penalty count \(B\), net adjusted contribution, adjusted score, and adjusted decision.

| PRN | Ground truth | Positive pairs `A/10` | Penalty pairs `B/10` | Net adjusted contribution | \(C_i^{(\mathrm{adj})}\) | Current decision | Correct? |
| ---: | --- | ---: | ---: | ---: | ---: | --- | --- |
| 6 | Counterfeit | 6/10 | 3/10 | 3/10 | 0.30 | Counterfeit | Yes |
| 9 | Counterfeit | 8/10 | 3/10 | 5/10 | 0.50 | Authentic | **No** |
| 11 | Counterfeit | 0/10 | 3/10 | -3/10 | 0.00 after clipping | Counterfeit | Yes |
| 20 | Counterfeit | 6/10 | 3/10 | 3/10 | 0.30 | Counterfeit | Yes |
| 10 | Authentic | 5/10 | 0/10 | 5/10 | 0.50 | Authentic | Yes |
| 12 | Authentic | 4/10 | 0/10 | 4/10 | 0.40 | Counterfeit | **No** |
| 15 | Authentic | 9/10 | 0/10 | 9/10 | 0.90 | Authentic | Yes |
| 18 | Authentic | 5/10 | 0/10 | 5/10 | 0.50 | Authentic | Yes |
| 23 | Authentic | 8/10 | 0/10 | 8/10 | 0.80 | Authentic | Yes |
| 24 | Authentic | 9/10 | 0/10 | 9/10 | 0.90 | Authentic | Yes |
| 32 | Authentic | 6/10 | 0/10 | 6/10 | 0.60 | Authentic | Yes |

If the existing two numerical columns must be retained, rename `Consistent pairs` to `Net adjusted contributions` and use the `Net adjusted contribution` and \(C_i^{(\mathrm{adj})}\) columns above. Keeping the label `Consistent pairs` would be misleading because, for example, PRN 6 has six positive pairs but a net score of only 3/10 after three penalties.

The draft currently labels every counterfeit PRN as `Counterfeit` in the decision column. Under the aggregate result, PRN 9 is instead decided **Authentic**. Authentic PRN 12 is decided **Counterfeit**.

## Equation (49) and summary placeholders

The draft's `eta = XX` placeholder should be filled as

\[
\eta=0.5.
\]

Recommended replacement:

> Using the decision threshold \(\eta=0.5\), the adjusted method correctly classified 6 out of 7 authentic signals and 3 out of 4 counterfeit signals. The authentic acceptance rate was therefore 85.7%, while the counterfeit detection rate was 75.0%. The resulting overall satellite-level classification accuracy was 81.8% (9/11).

Placeholder mapping:

| Draft placeholder | Current value |
| --- | ---: |
| `eta = XX` | 0.5 |
| `XX out of seven authentic signals` | 6 out of 7 |
| `XX out of four counterfeit signals` | 3 out of 4 |
| Authentic acceptance `XX.X%` | 85.7% |
| Counterfeit detection `XX.X%` | 75.0% |
| Overall accuracy `XX.X%` | 81.8% |

For the four-reference aggregate, the unpenalized journal score and the adjusted penalty score each classify 9 of 11 satellites correctly, but they do not necessarily misclassify the same satellites. The penalty's purpose is to remove common-direction spoof-pair votes rather than to guarantee a higher accuracy for every reference choice.

## Table 4 placeholders

| Reference | Authentic acceptance | Counterfeit detection | Overall accuracy | Status |
| --- | ---: | ---: | ---: | --- |
| PRN 12 | 83.3% (5/6) | 100.0% (4/4) | 90.0% (9/10) | Executed notebook result; PRN 10 misclassified |
| PRN 23 | 100.0% (6/6) | 75.0% (3/4) | 90.0% (9/10) | Executed notebook result; PRN 9 misclassified |
| PRN 11 | 71.4% (5/7) | 66.7% (2/3) | 70.0% (7/10) | Executed notebook result; PRNs 9, 12, and 18 misclassified |
| PRN 9 | 85.7% (6/7) | 100.0% (3/3) | 90.0% (9/10) | Executed notebook result; PRN 12 misclassified |
| Aggregated references | 85.7% (6/7) | 75.0% (3/4) | 81.8% (9/11) | Median aggregation of valid \(q_{ij}\) and \(c_{ij}\) across the four references; PRNs 9 and 12 misclassified |

These percentages can directly replace all 15 `XX.X%` placeholders in Table 4.

## Other draft corrections exposed by the placeholder review

1. Section 4.1.3 lists authentic PRN 13, while Section 4.4, Table 3, and the current evaluated classifier use authentic PRN 24. The mixed-dataset description should be made consistent with the evaluated set.
2. Individual fixed-reference tables use `/9`; the four-reference aggregate Table 3 uses `/10` and includes all 11 satellites.
3. Figure 14 is visibly unresolved in the current PDF (`figures/mixed_pairwise_residual_heatmap.pdf` is printed as a missing graphic). The new combined figure is `reference_penalty_runs/figure_penalty_reference_heatmaps.pdf` and shows \(q_{ij}\), \(c_{ij}\), and \(A_{ij}-B_{ij}\) for every reference plus the median aggregate.
4. The notebook states that its hybrid best-m/DP reconstruction is label-dependent and remains a scenario diagnostic. The draft should not describe these numbers as a blind deployable detector result without an additional label-independent reconstruction and validation run.

## Copy-ready minimal result block

If the paper is revised to report only what the current penalty implementation actually demonstrates, the compact numerical block is:

> Four fixed-reference diagnostics were executed using PRNs 12, 23, 11, and 9. For every satellite pair, the available per-reference pairwise geometry residuals and common-direction residuals were median-aggregated before applying \(\tau_q=0.20\), \(\tau_c=0.02\), and \(\eta=0.5\). The resulting 11-satellite classifier correctly classified six of seven authentic signals and three of four counterfeit signals, corresponding to an authentic acceptance rate of 85.7%, a counterfeit detection rate of 75.0%, and an overall accuracy of 81.8% (9/11). PRNs 9 and 12 were misclassified.
