"""Deterministic graph classification of pairwise GNSS spatial signatures.

This module classifies already-aggregated pair evidence.  It deliberately does
not parse receiver data, plot results, or assign ground-truth authenticity.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from itertools import combinations
from typing import Sequence

import numpy as np
import pandas as pd


class EvidenceState(str, Enum):
    """Recording-level strength and topology of the observed evidence."""

    UNDETERMINED = "undetermined"
    NO_GROUP_EVIDENCE = "no_group_evidence"
    PAIR_ALERTS_ONLY = "pair_alerts_only"
    SINGLE_SUBSET_COMPONENT = "single_subset_component"
    SINGLE_ALL_NODE_COMPONENT = "single_all_node_component"
    MULTIPLE_COMPONENTS = "multiple_components"


@dataclass(frozen=True)
class ComponentClassifierConfig:
    """Fixed operating thresholds for graph evidence classification."""

    fractional_dd_collapse_threshold_cycles: float = 0.03
    ephemeris_predicted_separation_threshold_cycles: float = 0.05
    min_pair_epochs: int = 10
    min_observed_pair_coverage: float = 1.0
    min_collapse_fraction: float = 1.0
    min_corroborated_size: int = 3


@dataclass(frozen=True)
class GraphClassification:
    """Structured result that keeps pair alerts separate from groups."""

    pair_evidence: pd.DataFrame
    component_summary: pd.DataFrame
    node_summary: pd.DataFrame
    pair_alerts: tuple[tuple[int, int], ...]
    corroborated_groups: tuple[tuple[int, ...], ...]
    evidence_state: EvidenceState


_PAIR_COLUMNS = {
    "left_svid",
    "right_svid",
    "median_fractional_dd_collapse_cycles",
    "median_ephemeris_predicted_pair_separation_cycles",
    "epoch_count",
}


def _validate_inputs(
    eligible_nodes: Sequence[int],
    aggregated_pairs: pd.DataFrame,
    config: ComponentClassifierConfig,
) -> tuple[int, ...]:
    nodes = tuple(int(node) for node in eligible_nodes)
    if len(nodes) != len(set(nodes)):
        raise ValueError("eligible_nodes contains duplicates.")
    if len(nodes) < 2:
        raise ValueError("At least two eligible nodes are required.")
    if config.fractional_dd_collapse_threshold_cycles < 0:
        raise ValueError(
            "fractional_dd_collapse_threshold_cycles must be non-negative."
        )
    if config.ephemeris_predicted_separation_threshold_cycles < 0:
        raise ValueError(
            "ephemeris_predicted_separation_threshold_cycles must be non-negative."
        )
    if config.min_pair_epochs < 1:
        raise ValueError("min_pair_epochs must be at least one.")
    if not 0.0 <= config.min_observed_pair_coverage <= 1.0:
        raise ValueError("min_observed_pair_coverage must be between zero and one.")
    if not 0.0 <= config.min_collapse_fraction <= 1.0:
        raise ValueError("min_collapse_fraction must be between zero and one.")
    if config.min_corroborated_size < 3:
        raise ValueError("min_corroborated_size must be at least three.")

    missing = _PAIR_COLUMNS - set(aggregated_pairs.columns)
    if missing:
        raise ValueError(f"aggregated_pairs is missing required columns: {sorted(missing)}")
    if aggregated_pairs.empty:
        return tuple(sorted(nodes))

    endpoints = set(aggregated_pairs["left_svid"].astype(int)) | set(
        aggregated_pairs["right_svid"].astype(int)
    )
    unknown = endpoints - set(nodes)
    if unknown:
        raise ValueError(f"Pair endpoints are not eligible nodes: {sorted(unknown)}")
    if aggregated_pairs["left_svid"].astype(int).eq(
        aggregated_pairs["right_svid"].astype(int)
    ).any():
        raise ValueError("Self-pairs are not allowed.")
    canonical = aggregated_pairs.apply(
        lambda row: tuple(sorted((int(row.left_svid), int(row.right_svid)))), axis=1
    )
    if canonical.duplicated().any():
        raise ValueError("Each unordered satellite pair must appear at most once.")
    numeric = aggregated_pairs[
        [
            "median_fractional_dd_collapse_cycles",
            "median_ephemeris_predicted_pair_separation_cycles",
            "epoch_count",
        ]
    ].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("Pair metrics and epoch_count must be finite numbers.")
    if numeric["epoch_count"].lt(0).any():
        raise ValueError("epoch_count must be non-negative.")
    return tuple(sorted(nodes))


def _connected_components(
    nodes: Sequence[int], edges: Sequence[tuple[int, int]]
) -> tuple[tuple[int, ...], ...]:
    adjacency = {node: set() for node in nodes}
    for left, right in edges:
        adjacency[left].add(right)
        adjacency[right].add(left)
    components: list[tuple[int, ...]] = []
    visited: set[int] = set()
    for node in sorted(node for node in nodes if adjacency[node]):
        if node in visited:
            continue
        stack = [node]
        component: list[int] = []
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            component.append(current)
            stack.extend(sorted(adjacency[current] - visited, reverse=True))
        components.append(tuple(sorted(component)))
    return tuple(sorted(components, key=lambda group: (-len(group), group)))


def classify_common_source_components(
    eligible_nodes: Sequence[int],
    aggregated_pairs: pd.DataFrame,
    *,
    config: ComponentClassifierConfig,
) -> GraphClassification:
    """Classify geometry-informative collapsed pairs and corroborated groups.

    A two-node informative component is retained as a pair alert.  Components
    with three or more nodes are corroborated only if their observed internal
    pairs satisfy the configured coverage and collapse-fraction requirements.
    """
    nodes = _validate_inputs(eligible_nodes, aggregated_pairs, config)
    supplied: dict[tuple[int, int], pd.Series] = {}
    for _, row in aggregated_pairs.iterrows():
        pair = tuple(sorted((int(row["left_svid"]), int(row["right_svid"]))))
        supplied[pair] = row

    pair_rows: list[dict[str, object]] = []
    for left, right in combinations(nodes, 2):
        source = supplied.get((left, right))
        present = source is not None
        epochs = int(source["epoch_count"]) if present else 0
        observed = present and epochs >= config.min_pair_epochs
        fractional_dd_collapse_cycles = (
            float(source["median_fractional_dd_collapse_cycles"])
            if present
            else float("nan")
        )
        ephemeris_predicted_pair_separation_cycles = (
            float(source["median_ephemeris_predicted_pair_separation_cycles"])
            if present
            else float("nan")
        )
        fractional_dd_collapsed = (
            observed
            and fractional_dd_collapse_cycles
            <= config.fractional_dd_collapse_threshold_cycles
        )
        predicted_phase_distinguishable = (
            observed
            and ephemeris_predicted_pair_separation_cycles
            >= config.ephemeris_predicted_separation_threshold_cycles
        )
        informative_edge = fractional_dd_collapsed and predicted_phase_distinguishable
        if not observed:
            relation = "missing"
        elif informative_edge:
            relation = "informative_common_source"
        elif fractional_dd_collapsed:
            relation = "predicted_phase_aliased_collapse"
        else:
            relation = "noncollapse"
        pair_rows.append(
            {
                "pair": f"PRN {left}–PRN {right}",
                "left_svid": left,
                "right_svid": right,
                "median_fractional_dd_collapse_cycles": fractional_dd_collapse_cycles,
                "median_ephemeris_predicted_pair_separation_cycles": (
                    ephemeris_predicted_pair_separation_cycles
                ),
                "epoch_count": epochs,
                "observed": observed,
                "fractional_dd_collapsed": fractional_dd_collapsed,
                "predicted_phase_distinguishable": predicted_phase_distinguishable,
                "informative_common_source_edge": informative_edge,
                "relation": relation,
            }
        )
    pairs = pd.DataFrame(pair_rows)
    pair_alerts = tuple(
        (int(row.left_svid), int(row.right_svid))
        for row in pairs.loc[pairs["informative_common_source_edge"]].itertuples()
    )
    candidate_components = _connected_components(nodes, pair_alerts)
    component_columns = [
        "component_id",
        "component_svids",
        "component_size",
        "possible_pair_count",
        "observed_pair_count",
        "observed_pair_coverage",
        "collapse_pair_count",
        "collapse_fraction_observed",
        "informative_pair_count",
        "noncollapse_pair_count",
        "corroborated_common_signature_group",
        "rejection_reasons",
    ]
    component_rows: list[dict[str, object]] = []
    corroborated: list[tuple[int, ...]] = []
    for component_id, component in enumerate(candidate_components, start=1):
        internal = pairs.loc[
            pairs["left_svid"].isin(component) & pairs["right_svid"].isin(component)
        ]
        possible_count = len(component) * (len(component) - 1) // 2
        observed_count = int(internal["observed"].sum())
        collapse_count = int(internal["fractional_dd_collapsed"].sum())
        informative_count = int(internal["informative_common_source_edge"].sum())
        noncollapse_count = int(
            (internal["observed"] & ~internal["fractional_dd_collapsed"]).sum()
        )
        coverage = observed_count / possible_count
        collapse_fraction = collapse_count / observed_count if observed_count else 0.0
        reasons: list[str] = []
        if len(component) < config.min_corroborated_size:
            reasons.append("pair_level_only")
        if coverage < config.min_observed_pair_coverage:
            reasons.append("insufficient_pair_coverage")
        if collapse_fraction < config.min_collapse_fraction:
            reasons.append("internal_noncollapse")
        accepted = not reasons
        if accepted:
            corroborated.append(component)
        component_rows.append(
            {
                "component_id": component_id,
                "component_svids": component,
                "component_size": len(component),
                "possible_pair_count": possible_count,
                "observed_pair_count": observed_count,
                "observed_pair_coverage": coverage,
                "collapse_pair_count": collapse_count,
                "collapse_fraction_observed": collapse_fraction,
                "informative_pair_count": informative_count,
                "noncollapse_pair_count": noncollapse_count,
                "corroborated_common_signature_group": accepted,
                "rejection_reasons": tuple(reasons),
            }
        )
    corroborated_groups = tuple(corroborated)

    alert_nodes = {node for pair in pair_alerts for node in pair}
    corroborated_nodes = {node for group in corroborated_groups for node in group}
    possible_node_pairs = len(nodes) - 1
    observed_by_node = {
        node: int(
            pairs.loc[
                (pairs["left_svid"].eq(node) | pairs["right_svid"].eq(node)), "observed"
            ].sum()
        )
        for node in nodes
    }
    node_summary = pd.DataFrame(
        {
            "svid": nodes,
            "status": [
                "corroborated_group_member"
                if node in corroborated_nodes
                else "pair_alert_member"
                if node in alert_nodes
                else "undetermined_due_to_coverage"
                if observed_by_node[node] / possible_node_pairs
                < config.min_observed_pair_coverage
                else "not_assigned_to_detected_group"
                for node in nodes
            ],
            "observed_pair_count": [observed_by_node[node] for node in nodes],
            "possible_pair_count": possible_node_pairs,
        }
    )
    if len(corroborated_groups) > 1:
        state = EvidenceState.MULTIPLE_COMPONENTS
    elif len(corroborated_groups) == 1:
        state = (
            EvidenceState.SINGLE_ALL_NODE_COMPONENT
            if set(corroborated_groups[0]) == set(nodes)
            else EvidenceState.SINGLE_SUBSET_COMPONENT
        )
    elif pair_alerts:
        state = EvidenceState.PAIR_ALERTS_ONLY
    elif not pairs["observed"].all():
        state = EvidenceState.UNDETERMINED
    else:
        state = EvidenceState.NO_GROUP_EVIDENCE
    return GraphClassification(
        pair_evidence=pairs,
        component_summary=pd.DataFrame(component_rows, columns=component_columns),
        node_summary=node_summary,
        pair_alerts=pair_alerts,
        corroborated_groups=corroborated_groups,
        evidence_state=state,
    )
