import unittest

import pandas as pd

from aoa_pipeline import (
    _pivot_rawx,
    compute_ephemeris_predicted_pair_separation_cycles,
)


class EphemerisPairwisePhaseProjectionTests(unittest.TestCase):
    def test_integer_cycle_separation_is_fractionally_aliased(self):
        separation = compute_ephemeris_predicted_pair_separation_cycles(
            60.0,
            120.0,
            baseline_over_wavelength=1.0,
        )

        self.assertAlmostEqual(float(separation), 0.0, places=12)

    def test_equal_degree_separations_have_geometry_dependent_phase_separation(self):
        near_baseline = compute_ephemeris_predicted_pair_separation_cycles(
            10.0,
            20.0,
            baseline_over_wavelength=1.0,
        )
        broadside = compute_ephemeris_predicted_pair_separation_cycles(
            80.0,
            90.0,
            baseline_over_wavelength=1.0,
        )

        self.assertAlmostEqual(float(near_baseline), 0.0451151322, places=9)
        self.assertAlmostEqual(float(broadside), 0.1736481777, places=9)
        self.assertGreater(float(broadside), float(near_baseline))


class RawxCarrierPhaseQualityTests(unittest.TestCase):
    def test_phase_pivot_rejects_unresolved_half_cycle_measurements(self):
        rawx = pd.DataFrame(
            {
                "epoch_index": [0, 1, 2],
                "svid": [6, 6, 6],
                "carrier_phase_cycles": [10.0, 11.5, 12.0],
                "cp_valid": [1, 1, 0],
                "half_cycle": [1, 0, 1],
            }
        )

        result = _pivot_rawx(
            rawx,
            "carrier_phase_cycles",
            pd.Index([0, 1, 2], name="epoch_index"),
            require_valid_phase=True,
        )

        self.assertEqual(result.loc[0, 6], 10.0)
        self.assertTrue(pd.isna(result.loc[1, 6]))
        self.assertTrue(pd.isna(result.loc[2, 6]))


if __name__ == "__main__":
    unittest.main()
