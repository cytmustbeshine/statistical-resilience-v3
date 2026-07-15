from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from two_factor_traffic_resilience import DemandEfficiencyModel, two_dimension_event_states


class DemandEfficiencyTests(unittest.TestCase):
    def sample(self, seed=8):
        rng = np.random.default_rng(seed)
        demand = rng.normal(size=(60, 3))
        efficiency = rng.normal(size=(60, 3))
        speed = 0.8 * efficiency + rng.normal(scale=0.3, size=efficiency.shape)
        occupancy = 0.7 * efficiency + rng.normal(scale=0.35, size=efficiency.shape)
        return demand, speed, occupancy, 40

    def test_demand_is_flow_score(self):
        demand, speed, occupancy, train_end = self.sample()
        model = DemandEfficiencyModel().fit(demand, speed, occupancy, train_end)
        result = model.transform(demand, speed, occupancy)
        np.testing.assert_allclose(result["demand_score"], demand)

    def test_flow_does_not_change_efficiency(self):
        demand, speed, occupancy, train_end = self.sample()
        model = DemandEfficiencyModel().fit(demand, speed, occupancy, train_end)
        first = model.transform(demand, speed, occupancy)["efficiency_score"]
        second = model.transform(demand + 999, speed, occupancy)["efficiency_score"]
        np.testing.assert_allclose(first, second, equal_nan=True)

    def test_speed_does_not_change_demand(self):
        demand, speed, occupancy, train_end = self.sample()
        model = DemandEfficiencyModel().fit(demand, speed, occupancy, train_end)
        result = model.transform(demand, speed + 999, occupancy)
        np.testing.assert_allclose(result["demand_score"], demand)

    def test_speed_only_is_direct_efficiency(self):
        demand, speed, _, train_end = self.sample()
        model = DemandEfficiencyModel().fit(demand, speed, None, train_end)
        result = model.transform(demand, speed, None)
        self.assertEqual(model.efficiency_mode_, "speed_observed")
        np.testing.assert_allclose(result["efficiency_score"], speed)

    def test_bridge_efficiency_unavailable(self):
        demand, _, _, train_end = self.sample()
        model = DemandEfficiencyModel().fit(demand, None, None, train_end)
        result = model.transform(demand, None, None)
        self.assertEqual(model.efficiency_mode_, "unavailable")
        self.assertTrue(np.isnan(result["efficiency_score"]).all())

    def test_speed_loading_positive(self):
        demand, speed, occupancy, train_end = self.sample()
        model = DemandEfficiencyModel().fit(demand, speed, occupancy, train_end)
        self.assertGreater(model.efficiency_factor_.loadings_[0], 0)

    def test_factor_train_only(self):
        demand, speed, occupancy, train_end = self.sample()
        changed_speed = speed.copy(); changed_speed[train_end:] += 999
        changed_occupancy = occupancy.copy(); changed_occupancy[train_end:] -= 999
        first = DemandEfficiencyModel().fit(demand, speed, occupancy, train_end)
        second = DemandEfficiencyModel().fit(demand, changed_speed, changed_occupancy, train_end)
        np.testing.assert_allclose(first.efficiency_factor_.loadings_, second.efficiency_factor_.loadings_)
        np.testing.assert_allclose(first.efficiency_factor_.residual_variance_, second.efficiency_factor_.residual_variance_)

    def test_missing_occupancy_uses_speed(self):
        demand, speed, occupancy, train_end = self.sample()
        occupancy[:, 0] = np.nan
        model = DemandEfficiencyModel().fit(demand, speed, occupancy, train_end)
        result = model.transform(demand, speed, occupancy)
        self.assertTrue(np.isfinite(result["efficiency_score"][:, 0]).all())
        self.assertTrue(np.all(result["efficiency_observed_modalities"][:, 0] == 1))

    def test_all_efficiency_modalities_missing_nan(self):
        demand, speed, occupancy, train_end = self.sample()
        model = DemandEfficiencyModel().fit(demand, speed, occupancy, train_end)
        speed[0, 0] = np.nan; occupancy[0, 0] = np.nan
        result = model.transform(demand, speed, occupancy)
        self.assertTrue(np.isnan(result["efficiency_score"][0, 0]))

    def test_information_reduces_efficiency_variance(self):
        demand, speed, occupancy, train_end = self.sample()
        model = DemandEfficiencyModel().fit(demand, speed, occupancy, train_end)
        full = model.transform(demand, speed, occupancy)["efficiency_posterior_variance"]
        occupancy_missing = np.full_like(occupancy, np.nan)
        reduced = model.transform(demand, speed, occupancy_missing)["efficiency_posterior_variance"]
        self.assertLess(np.nanmean(full), np.nanmean(reduced))

    def test_npz_roundtrip(self):
        demand, speed, occupancy, train_end = self.sample()
        model = DemandEfficiencyModel().fit(demand, speed, occupancy, train_end)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "l4.npz"
            model.save(path)
            loaded = DemandEfficiencyModel.load(path)
            np.testing.assert_allclose(
                model.transform(demand, speed, occupancy)["efficiency_score"],
                loaded.transform(demand, speed, occupancy)["efficiency_score"],
                equal_nan=True,
            )

    def test_quadrants_mutually_exclusive(self):
        demand = np.array([0, 2, 0, 2], dtype=float)
        efficiency = np.array([0, 0, 2, 2], dtype=float)
        states = two_dimension_event_states(demand, efficiency, 1, 1)
        masks = np.stack([
            states["both_high"], states["demand_only_high"],
            states["efficiency_only_high"], states["neither_high"],
        ])
        self.assertTrue(np.all(masks.sum(axis=0) == 1))

    def test_quadrant_meaning(self):
        states = two_dimension_event_states(
            np.array([0, 2, 0, 2], float), np.array([0, 0, 2, 2], float), 1, 1
        )
        np.testing.assert_array_equal(states["neither_high"], [True, False, False, False])
        np.testing.assert_array_equal(states["demand_only_high"], [False, True, False, False])
        np.testing.assert_array_equal(states["efficiency_only_high"], [False, False, True, False])
        np.testing.assert_array_equal(states["both_high"], [False, False, False, True])

    def test_missing_efficiency_no_joint_state(self):
        states = two_dimension_event_states(np.array([1.0]), None, 1.0, None)
        self.assertFalse(states["available"])

    def test_no_scalar_joint_score_returned(self):
        demand, speed, occupancy, train_end = self.sample()
        result = DemandEfficiencyModel().fit(demand, speed, occupancy, train_end).transform(
            demand, speed, occupancy
        )
        self.assertNotIn("joint_score", result)
        self.assertNotIn("combined_deficit", result)


if __name__ == "__main__":
    unittest.main()