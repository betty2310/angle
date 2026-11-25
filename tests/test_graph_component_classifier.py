import unittest

import pandas as pd

from graph_component_classifier import (
    ComponentClassifierConfig,
    EvidenceState,
    classify_common_source_components,
)


class GraphComponentClassifierTests(unittest.TestCase):
    @staticmethod
    def pairs(*rows):
        return pd.DataFrame(
            [
                {
                    "left_svid": left,
                    "right_svid": right,
                    "median_fractional_dd_collapse_cycles": (
                        fractional_dd_collapse_cycles
                    ),
                    "median_ephemeris_predicted_pair_separation_cycles": (
                        predicted_phase_separation_cycles
                    ),
                    "epoch_count": epoch_count,
                }
                for (
                    left,
                    right,
                    fractional_dd_collapse_cycles,
                    predicted_phase_separation_cycles,
                    epoch_count,
                ) in rows
            ]
        )

    def test_informative_two_satellite_pair_is_an_alert_not_a_corroborated_group(self):
        pairs = self.pairs((6, 20, 0.02, 0.15, 20))

        result = classify_common_source_components(
            [6, 20], pairs, config=ComponentClassifierConfig()
        )

        self.assertEqual(result.pair_alerts, ((6, 20),))
        self.assertEqual(result.corroborated_groups, ())
        self.assertEqual(result.evidence_state, EvidenceState.PAIR_ALERTS_ONLY)
        self.assertEqual(result.node_summary.set_index("svid")["status"].to_dict(), {
            6: "pair_alert_member",
            20: "pair_alert_member",
        })

    def test_two_satellite_alias_and_noncollapse_do_not_raise_an_alert(self):
        for pairs in (
            self.pairs((6, 20, 0.02, 0.02, 20)),
            self.pairs((6, 20, 0.04, 0.15, 20)),
        ):
            with self.subTest(pairs=pairs.to_dict("records")):
                result = classify_common_source_components(
                    [6, 20], pairs, config=ComponentClassifierConfig()
                )
                self.assertEqual(result.pair_alerts, ())
                self.assertEqual(result.evidence_state, EvidenceState.NO_GROUP_EVIDENCE)

    def test_threshold_boundaries_are_inclusive(self):
        result = classify_common_source_components(
            [6, 20],
            self.pairs((6, 20, 0.03, 0.05, 10)),
            config=ComponentClassifierConfig(),
        )

        self.assertEqual(result.pair_alerts, ((6, 20),))

    def test_pair_below_epoch_requirement_is_explicitly_missing(self):
        result = classify_common_source_components(
            [6, 20],
            self.pairs((6, 20, 0.01, 0.15, 9)),
            config=ComponentClassifierConfig(min_pair_epochs=10),
        )

        self.assertEqual(result.pair_alerts, ())
        self.assertEqual(result.pair_evidence.iloc[0]["relation"], "missing")
        self.assertEqual(result.evidence_state, EvidenceState.UNDETERMINED)

    def test_complete_three_node_component_is_corroborated(self):
        result = classify_common_source_components(
            [6, 9, 20],
            self.pairs(
                (6, 9, 0.01, 0.15, 20),
                (6, 20, 0.02, 0.02, 20),
                (9, 20, 0.01, 0.20, 20),
            ),
            config=ComponentClassifierConfig(),
        )

        self.assertEqual(result.corroborated_groups, ((6, 9, 20),))
        self.assertEqual(result.evidence_state, EvidenceState.SINGLE_ALL_NODE_COMPONENT)
        summary = result.component_summary.iloc[0]
        self.assertEqual(summary["informative_pair_count"], 2)
        self.assertEqual(summary["noncollapse_pair_count"], 0)
        self.assertEqual(summary["observed_pair_coverage"], 1.0)

    def test_informative_chain_with_contradictory_internal_pair_is_not_corroborated(self):
        result = classify_common_source_components(
            [1, 2, 3],
            self.pairs(
                (1, 2, 0.01, 0.15, 20),
                (2, 3, 0.01, 0.15, 20),
                (1, 3, 0.20, 0.15, 20),
            ),
            config=ComponentClassifierConfig(),
        )

        self.assertEqual(result.corroborated_groups, ())
        self.assertEqual(result.evidence_state, EvidenceState.PAIR_ALERTS_ONLY)
        self.assertIn("internal_noncollapse", result.component_summary.iloc[0]["rejection_reasons"])

    def test_missing_internal_pair_is_undetermined_not_negative_evidence(self):
        result = classify_common_source_components(
            [1, 2, 3],
            self.pairs(
                (1, 2, 0.01, 0.15, 20),
                (2, 3, 0.01, 0.15, 20),
            ),
            config=ComponentClassifierConfig(),
        )

        missing = result.pair_evidence.query("left_svid == 1 and right_svid == 3").iloc[0]
        self.assertFalse(missing["observed"])
        self.assertEqual(missing["relation"], "missing")
        self.assertEqual(result.corroborated_groups, ())
        self.assertIn("insufficient_pair_coverage", result.component_summary.iloc[0]["rejection_reasons"])

    def test_explicit_node_without_pairs_is_preserved_and_marked_undetermined(self):
        result = classify_common_source_components(
            [1, 2, 3],
            self.pairs((1, 2, 0.01, 0.15, 20)),
            config=ComponentClassifierConfig(),
        )

        statuses = result.node_summary.set_index("svid")["status"].to_dict()
        self.assertEqual(statuses[3], "undetermined_due_to_coverage")
        self.assertIn(3, tuple(result.node_summary.svid))

    def test_two_disjoint_corroborated_components_are_not_merged(self):
        rows = []
        for group in ((1, 2, 3), (4, 5, 6)):
            rows.extend(
                (left, right, 0.01, 0.15, 20)
                for left, right in ((group[0], group[1]), (group[0], group[2]), (group[1], group[2]))
            )
        rows.extend(
            (left, right, 0.20, 0.15, 20)
            for left in (1, 2, 3)
            for right in (4, 5, 6)
        )

        result = classify_common_source_components(
            [1, 2, 3, 4, 5, 6], self.pairs(*rows), config=ComponentClassifierConfig()
        )

        self.assertEqual(result.corroborated_groups, ((1, 2, 3), (4, 5, 6)))
        self.assertEqual(result.evidence_state, EvidenceState.MULTIPLE_COMPONENTS)

    def test_one_component_on_strict_subset_leaves_other_nodes_unassigned(self):
        result = classify_common_source_components(
            [1, 2, 3, 4],
            self.pairs(
                (1, 2, 0.01, 0.15, 20), (1, 3, 0.01, 0.15, 20),
                (2, 3, 0.01, 0.15, 20), (1, 4, 0.20, 0.15, 20),
                (2, 4, 0.20, 0.15, 20), (3, 4, 0.20, 0.15, 20),
            ),
            config=ComponentClassifierConfig(),
        )

        self.assertEqual(result.evidence_state, EvidenceState.SINGLE_SUBSET_COMPONENT)
        self.assertEqual(
            result.node_summary.set_index("svid").loc[4, "status"],
            "not_assigned_to_detected_group",
        )

    def test_empty_pair_input_returns_undetermined_for_explicit_nodes(self):
        empty = pd.DataFrame(columns=[
            "left_svid", "right_svid", "median_fractional_dd_collapse_cycles",
            "median_ephemeris_predicted_pair_separation_cycles", "epoch_count",
        ])

        result = classify_common_source_components(
            [1, 2], empty, config=ComponentClassifierConfig()
        )

        self.assertEqual(result.evidence_state, EvidenceState.UNDETERMINED)
        self.assertTrue(result.node_summary.status.eq("undetermined_due_to_coverage").all())

    def test_invalid_pair_tables_are_rejected_at_the_boundary(self):
        cases = {
            "duplicate": self.pairs((1, 2, 0.01, 0.15, 20), (2, 1, 0.01, 0.15, 20)),
            "self_pair": self.pairs((1, 1, 0.01, 0.15, 20)),
            "unknown_node": self.pairs((1, 3, 0.01, 0.15, 20)),
            "nan_metric": self.pairs((1, 2, float("nan"), 0.15, 20)),
        }
        for name, pairs in cases.items():
            with self.subTest(name=name), self.assertRaises(ValueError):
                classify_common_source_components(
                    [1, 2], pairs, config=ComponentClassifierConfig()
                )

    def test_invalid_configuration_is_rejected(self):
        invalid_configs = (
            ComponentClassifierConfig(fractional_dd_collapse_threshold_cycles=-0.01),
            ComponentClassifierConfig(ephemeris_predicted_separation_threshold_cycles=-0.01),
            ComponentClassifierConfig(min_pair_epochs=0),
            ComponentClassifierConfig(min_observed_pair_coverage=1.1),
            ComponentClassifierConfig(min_collapse_fraction=-0.1),
            ComponentClassifierConfig(min_corroborated_size=2),
        )
        for config in invalid_configs:
            with self.subTest(config=config), self.assertRaises(ValueError):
                classify_common_source_components([1, 2], self.pairs((1, 2, 0.1, 0.15, 20)), config=config)

    def test_input_order_does_not_change_structured_output(self):
        pairs = self.pairs(
            (1, 2, 0.01, 0.15, 20),
            (1, 3, 0.01, 0.15, 20),
            (2, 3, 0.01, 0.15, 20),
        )
        forward = classify_common_source_components(
            [1, 2, 3], pairs, config=ComponentClassifierConfig()
        )
        reverse = classify_common_source_components(
            [3, 2, 1], pairs.iloc[::-1], config=ComponentClassifierConfig()
        )

        self.assertEqual(forward.pair_alerts, reverse.pair_alerts)
        self.assertEqual(forward.corroborated_groups, reverse.corroborated_groups)
        self.assertEqual(forward.evidence_state, reverse.evidence_state)


if __name__ == "__main__":
    unittest.main()
