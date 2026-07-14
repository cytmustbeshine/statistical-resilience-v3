from __future__ import annotations

import copy
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from statistical_resilience_profile import (
    CDF_SOURCE_CELL,
    CDF_SOURCE_GLOBAL,
    CDF_SOURCE_NODE,
    CDF_SOURCE_NODE_DAYTYPE,
    CDF_SOURCE_ROBUST_PARAMETRIC,
    CDF_SOURCE_UNAVAILABLE,
    compute_probabilistic_resilience_state,
    fit_hierarchical_normal_profile,
    fit_shrunk_conditional_ecdf,
    load_conditional_ecdf_npz,
    query_conditional_cdf,
    save_conditional_ecdf_npz,
)


class ConditionalCdfSourceTests(unittest.TestCase):
    def sample(self):
        timestamps = pd.date_range("2024-01-01", periods=12, freq="12h").to_numpy()
        values = np.column_stack(
            [
                np.linspace(-3.0, 4.0, 12),
                np.linspace(5.0, 16.0, 12),
            ]
        )
        return values, timestamps, 8

    def fitted(self):
        values, timestamps, train_end = self.sample()
        profile = fit_hierarchical_normal_profile(
            values,
            train_end,
            timestamps,
            time_of_day_bins=2,
            day_type_mode="none",
            shrinkage_candidates=(1,),
            minimum_cell_samples=1,
            transform_mode="identity_robust",
        )
        ecdf = fit_shrunk_conditional_ecdf(
            values, profile, train_end, timestamps, min_ecdf_samples=2
        )
        return values, timestamps, train_end, profile, ecdf

    def query_same_cell(self, profile, ecdf):
        query = np.array([[-4.0, 4.0], [0.0, 8.0], [5.0, 18.0]])
        timestamps = np.repeat(np.datetime64("2024-01-01T00:00:00"), len(query))
        return query_conditional_cdf(query, profile, ecdf, timestamps)

    def test_cdf_source_cell(self):
        _, _, _, profile, ecdf = self.fitted()
        result = self.query_same_cell(profile, ecdf)
        self.assertTrue(np.all(result["source_level"] == CDF_SOURCE_CELL))

    def test_cdf_source_node_daytype(self):
        _, _, _, profile, ecdf = self.fitted()
        ecdf = copy.deepcopy(ecdf)
        ecdf["cell_sorted_values"] = {}
        result = self.query_same_cell(profile, ecdf)
        self.assertTrue(np.all(result["source_level"] == CDF_SOURCE_NODE_DAYTYPE))

    def test_cdf_source_node(self):
        _, _, _, profile, ecdf = self.fitted()
        ecdf = copy.deepcopy(ecdf)
        ecdf["cell_sorted_values"] = {}
        ecdf["node_daytype_sorted_values"] = {}
        result = self.query_same_cell(profile, ecdf)
        self.assertTrue(np.all(result["source_level"] == CDF_SOURCE_NODE))

    def test_cdf_source_global(self):
        _, _, _, profile, ecdf = self.fitted()
        ecdf = copy.deepcopy(ecdf)
        ecdf["cell_sorted_values"] = {}
        ecdf["node_daytype_sorted_values"] = {}
        ecdf["node_sorted_values"] = {}
        result = self.query_same_cell(profile, ecdf)
        self.assertTrue(np.all(result["source_level"] == CDF_SOURCE_GLOBAL))

    def test_cdf_source_parametric(self):
        _, _, _, profile, ecdf = self.fitted()
        ecdf = copy.deepcopy(ecdf)
        ecdf["cell_sorted_values"] = {}
        ecdf["node_daytype_sorted_values"] = {}
        ecdf["node_sorted_values"] = {}
        ecdf["global_sorted_values"] = np.array([])
        result = self.query_same_cell(profile, ecdf)
        self.assertTrue(np.all(result["source_level"] == CDF_SOURCE_ROBUST_PARAMETRIC))
        self.assertTrue(np.isfinite(result["cdf"]).all())

    def test_cdf_source_unavailable(self):
        _, _, _, profile, ecdf = self.fitted()
        profile = copy.deepcopy(profile)
        profile["global_count"] = 0
        ecdf = copy.deepcopy(ecdf)
        ecdf["cell_sorted_values"] = {}
        ecdf["node_daytype_sorted_values"] = {}
        ecdf["node_sorted_values"] = {}
        ecdf["global_sorted_values"] = np.array([])
        result = self.query_same_cell(profile, ecdf)
        self.assertTrue(np.all(result["source_level"] == CDF_SOURCE_UNAVAILABLE))
        self.assertTrue(np.isnan(result["cdf"]).all())

    def test_cdf_monotonicity_by_source(self):
        _, _, _, profile, base = self.fitted()
        configurations = []
        for cleared in range(5):
            ecdf = copy.deepcopy(base)
            if cleared >= 1:
                ecdf["cell_sorted_values"] = {}
            if cleared >= 2:
                ecdf["node_daytype_sorted_values"] = {}
            if cleared >= 3:
                ecdf["node_sorted_values"] = {}
            if cleared >= 4:
                ecdf["global_sorted_values"] = np.array([])
            configurations.append(ecdf)
        for ecdf in configurations:
            result = self.query_same_cell(profile, ecdf)
            self.assertTrue(np.all(np.diff(result["cdf"][:, 0]) >= 0))
            self.assertTrue(np.all(np.diff(result["cdf"][:, 1]) >= 0))

    def test_cdf_train_only(self):
        values, timestamps, train_end, profile_a, ecdf_a = self.fitted()
        changed = values.copy()
        changed[train_end:] = 99999.0
        profile_b = fit_hierarchical_normal_profile(
            changed,
            train_end,
            timestamps,
            time_of_day_bins=2,
            day_type_mode="none",
            shrinkage_candidates=(1,),
            minimum_cell_samples=1,
            transform_mode="identity_robust",
        )
        ecdf_b = fit_shrunk_conditional_ecdf(
            changed, profile_b, train_end, timestamps, min_ecdf_samples=2
        )
        np.testing.assert_allclose(profile_a["location_shrunk"], profile_b["location_shrunk"])
        np.testing.assert_allclose(ecdf_a["global_sorted_values"], ecdf_b["global_sorted_values"])

    def test_missing_value_returns_nan(self):
        _, timestamps, _, profile, ecdf = self.fitted()
        result = query_conditional_cdf(np.array([[np.nan, 7.0]]), profile, ecdf, timestamps[:1])
        self.assertTrue(np.isnan(result["cdf"][0, 0]))
        self.assertFalse(result["valid_mask"][0, 0])
        state = compute_probabilistic_resilience_state(
            np.array([[np.nan, 7.0]]), profile, ecdf, timestamps[:1]
        )
        self.assertTrue(np.isnan(state["probabilistic_deficit"][0, 0]))

    def test_standardized_negative_not_clipped(self):
        _, _, _, profile, ecdf = self.fitted()
        result = self.query_same_cell(profile, ecdf)
        self.assertTrue(np.isfinite(result["cdf"][0, 0]))
        self.assertEqual(result["transformed_values"][0, 0], -4.0)

    def test_probability_bounds(self):
        _, _, _, profile, ecdf = self.fitted()
        result = self.query_same_cell(profile, ecdf)
        finite = result["cdf"][np.isfinite(result["cdf"])]
        self.assertGreaterEqual(finite.min(), 1e-4)
        self.assertLessEqual(finite.max(), 1 - 1e-4)

    def test_source_level_shape_dtype(self):
        _, _, _, profile, ecdf = self.fitted()
        result = self.query_same_cell(profile, ecdf)
        self.assertEqual(result["source_level"].shape, (3, 2))
        self.assertTrue(np.issubdtype(result["source_level"].dtype, np.integer))

    def test_parent_used_before_parametric(self):
        _, _, _, profile, ecdf = self.fitted()
        ecdf = copy.deepcopy(ecdf)
        ecdf["cell_sorted_values"] = {}
        result = self.query_same_cell(profile, ecdf)
        self.assertFalse(np.any(result["source_level"] == CDF_SOURCE_ROBUST_PARAMETRIC))
        self.assertTrue(np.all(result["source_level"] == CDF_SOURCE_NODE_DAYTYPE))

    def test_cdf_query_reproducible(self):
        _, _, _, profile, ecdf = self.fitted()
        expected = self.query_same_cell(profile, ecdf)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ecdf.npz"
            save_conditional_ecdf_npz(path, ecdf)
            loaded = load_conditional_ecdf_npz(path)
            actual = self.query_same_cell(profile, loaded)
        np.testing.assert_allclose(expected["cdf"], actual["cdf"])
        np.testing.assert_array_equal(expected["source_level"], actual["source_level"])

    def test_legacy_interface_compatible(self):
        values, timestamps, _, profile, ecdf = self.fitted()
        result = compute_probabilistic_resilience_state(values, profile, ecdf, timestamps)
        for key in (
            "p_lower",
            "probabilistic_deficit",
            "system_deficit",
            "ecdf_fallback_mask",
            "train_deficit_quantiles",
        ):
            self.assertIn(key, result)


if __name__ == "__main__":
    unittest.main()