import unittest

from aoa_pipeline import compute_ephemeris_predicted_pair_separation_cycles


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


if __name__ == "__main__":
    unittest.main()
