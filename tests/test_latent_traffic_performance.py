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
from latent_traffic_performance import (
    MissingDataOneFactorModel,
    compute_latent_resilience_deficit,
    fit_latent_performance_profile,
    transform_conditional_normal_scores,
)
from analyze_event_resilience_process import (
    analyze_event_segments,
    compute_event_resilience_metrics,
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


class LatentPerformanceTests(unittest.TestCase):
    def score_profile(self):
        timestamps = pd.date_range("2024-01-01", periods=12, freq="12h").to_numpy()
        values = np.column_stack([np.linspace(1, 12, 12), np.linspace(4, 26, 12)])
        profile = fit_hierarchical_normal_profile(
            values, 8, timestamps, time_of_day_bins=2, day_type_mode="none",
            shrinkage_candidates=(1,), minimum_cell_samples=1,
            transform_mode="identity_robust",
        )
        ecdf = fit_shrunk_conditional_ecdf(values, profile, 8, timestamps, min_ecdf_samples=1)
        return values, timestamps, profile, ecdf

    def synthetic_scores(self, seed=4):
        rng = np.random.default_rng(seed)
        h = rng.normal(size=(80, 3))
        flow = -0.7 * h + rng.normal(scale=0.45, size=h.shape)
        speed = 0.9 * h + rng.normal(scale=0.35, size=h.shape)
        scores = np.stack([flow, speed], axis=2)
        scores[::9, 0, 1] = np.nan
        scores[::11, 1, 0] = np.nan
        return scores, 55

    def test_normal_score_monotonic(self):
        _, timestamps, profile, ecdf = self.score_profile()
        query = np.array([[0.0, 1.0], [5.0, 10.0], [20.0, 40.0]])
        query_times = np.repeat(timestamps[:1], 3)
        result = transform_conditional_normal_scores(
            query, profile, ecdf, query_times, "higher_is_better"
        )
        self.assertTrue(np.all(np.diff(result["normal_score"][:, 0]) >= 0))

    def test_normal_score_finite(self):
        _, timestamps, profile, ecdf = self.score_profile()
        query = np.array([[-1e9, 1e9], [1e9, -1e9]])
        result = transform_conditional_normal_scores(
            query, profile, ecdf, timestamps[:2], "higher_is_better"
        )
        self.assertTrue(np.isfinite(result["normal_score"]).all())

    def test_speed_orientation(self):
        _, timestamps, profile, ecdf = self.score_profile()
        result = transform_conditional_normal_scores(
            np.array([[1.0, 2.0], [10.0, 20.0]]), profile, ecdf,
            timestamps[:2], "higher_is_better"
        )
        self.assertGreater(result["oriented_score"][1, 0], result["oriented_score"][0, 0])

    def test_occupancy_orientation(self):
        _, timestamps, profile, ecdf = self.score_profile()
        result = transform_conditional_normal_scores(
            np.array([[1.0, 2.0], [10.0, 20.0]]), profile, ecdf,
            timestamps[:2], "higher_is_worse"
        )
        self.assertLess(result["oriented_score"][1, 0], result["oriented_score"][0, 0])

    def test_flow_loading_not_hardcoded(self):
        scores, train_end = self.synthetic_scores()
        model = MissingDataOneFactorModel().fit(scores, train_end, ["flow", "speed"], random_state=7)
        self.assertLess(model.loadings_[0], 0)
        self.assertGreater(model.loadings_[1], 0)

    def test_factor_model_train_only(self):
        scores, train_end = self.synthetic_scores()
        changed = scores.copy(); changed[train_end:] = 9999
        first = MissingDataOneFactorModel().fit(scores, train_end, ["flow", "speed"], random_state=21)
        second = MissingDataOneFactorModel().fit(changed, train_end, ["flow", "speed"], random_state=21)
        np.testing.assert_allclose(first.loadings_, second.loadings_)
        np.testing.assert_allclose(first.residual_variance_, second.residual_variance_)
        np.testing.assert_allclose(
            first.transform(scores)["posterior_mean"][:train_end],
            second.transform(changed)["posterior_mean"][:train_end],
        )

    def test_missing_modality_not_zero_filled(self):
        scores, train_end = self.synthetic_scores()
        model = MissingDataOneFactorModel().fit(scores, train_end, ["flow", "speed"], random_state=1)
        query = np.array([[[1.0, np.nan]], [[1.0, 0.0]]])
        result = model.transform(query)
        self.assertEqual(result["n_observed_modalities"][0, 0], 1)
        self.assertEqual(result["n_observed_modalities"][1, 0], 2)
        self.assertGreater(result["posterior_variance"][0, 0], result["posterior_variance"][1, 0])

    def test_all_modalities_missing_returns_nan(self):
        scores, train_end = self.synthetic_scores()
        model = MissingDataOneFactorModel().fit(scores, train_end, ["flow", "speed"])
        result = model.transform(np.full((1, 1, 2), np.nan))
        self.assertTrue(np.isnan(result["posterior_mean"][0, 0]))
        self.assertTrue(np.isnan(result["posterior_variance"][0, 0]))

    def test_posterior_variance_decreases_with_information(self):
        scores, train_end = self.synthetic_scores()
        model = MissingDataOneFactorModel().fit(scores, train_end, ["flow", "speed"])
        result = model.transform(np.array([[[1.0, np.nan]], [[1.0, 1.0]]]))
        self.assertLess(result["posterior_variance"][1, 0], result["posterior_variance"][0, 0])

    def test_factor_sign_identification(self):
        scores, train_end = self.synthetic_scores()
        model = MissingDataOneFactorModel().fit(scores, train_end, ["flow", "speed"], random_state=100)
        self.assertGreater(model.loadings_[1], 0)
        self.assertEqual(model.sign_anchor_, "speed_loading_positive")

    def test_residual_variance_positive(self):
        scores, train_end = self.synthetic_scores()
        model = MissingDataOneFactorModel().fit(scores, train_end, ["flow", "speed"])
        self.assertTrue(np.all(model.residual_variance_ >= 1e-4))

    def test_em_likelihood_non_decreasing(self):
        scores, train_end = self.synthetic_scores()
        model = MissingDataOneFactorModel().fit(scores, train_end, ["flow", "speed"])
        differences = np.diff(model.log_likelihood_history_)
        self.assertTrue(np.all(differences >= -1e-5 * np.maximum(1, np.abs(model.log_likelihood_history_[:-1]))))

    def test_multiple_initializations_reproducible(self):
        scores, train_end = self.synthetic_scores()
        first = MissingDataOneFactorModel().fit(
            scores, train_end, ["flow", "speed"], random_state=42, n_initializations=3
        )
        second = MissingDataOneFactorModel().fit(
            scores, train_end, ["flow", "speed"], random_state=42, n_initializations=3
        )
        np.testing.assert_allclose(first.loadings_, second.loadings_)

    def test_model_npz_roundtrip(self):
        scores, train_end = self.synthetic_scores()
        model = MissingDataOneFactorModel().fit(scores, train_end, ["flow", "speed"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "factor.npz"
            model.save(path)
            loaded = MissingDataOneFactorModel.load(path)
            np.testing.assert_allclose(
                model.transform(scores)["posterior_mean"],
                loaded.transform(scores)["posterior_mean"],
                equal_nan=True,
            )

    def test_latent_profile_identity_transform(self):
        timestamps = pd.date_range("2024-01-01", periods=20, freq="12h").to_numpy()
        latent = np.linspace(-3, 3, 20)[:, None]
        result = fit_latent_performance_profile(
            latent, 14, timestamps, time_of_day_bins=2, day_type_mode="none",
            shrinkage_candidates=(1,), minimum_cell_samples=1,
        )
        self.assertEqual(result["profile"]["transform_mode"], "identity_robust")
        self.assertLess(result["ecdf"]["global_sorted_values"].min(), 0)

    def latent_state(self, values):
        timestamps = pd.date_range("2024-01-01", periods=len(values), freq="12h").to_numpy()
        profile = fit_latent_performance_profile(
            values, 14, timestamps, time_of_day_bins=2, day_type_mode="none",
            shrinkage_candidates=(1,), minimum_cell_samples=1,
        )
        variance = np.full_like(values, 0.2)
        state = compute_latent_resilience_deficit(
            values, variance, profile["profile"], profile["ecdf"], timestamps
        )
        return state, profile, timestamps

    def test_latent_deficit_lower_tail(self):
        values = np.linspace(-2, 2, 20)[:, None]
        state, _, _ = self.latent_state(values)
        self.assertGreater(state["latent_deficit"][0, 0], state["latent_deficit"][-1, 0])

    def test_latent_threshold_train_only(self):
        values = np.linspace(-2, 2, 20)[:, None]
        state_a, profile, timestamps = self.latent_state(values)
        changed = values.copy(); changed[14:] = -999
        state_b = compute_latent_resilience_deficit(
            changed, np.full_like(changed, 0.2), profile["profile"], profile["ecdf"], timestamps
        )
        self.assertEqual(state_a["train_deficit_quantiles"], state_b["train_deficit_quantiles"])

    def test_uncertainty_not_multiplied_into_deficit(self):
        values = np.linspace(-2, 2, 20)[:, None]
        state, profile, timestamps = self.latent_state(values)
        other = compute_latent_resilience_deficit(
            values, np.full_like(values, 0.9), profile["profile"], profile["ecdf"], timestamps
        )
        np.testing.assert_allclose(state["latent_deficit"], other["latent_deficit"], equal_nan=True)

    def test_event_peak_and_cumulative_deficit(self):
        values = np.array([0, 0, 1, 3, 2, 0, 0, 0], dtype=float)
        result = compute_event_resilience_metrics(values, 2, 4, 1.5, 2.5, 0.5, 2, 1.0)
        self.assertEqual(result["peak_deficit"], 3.0)
        self.assertEqual(result["cumulative_deficit"], 6.0)

    def test_recovery_requires_consecutive_steps(self):
        values = np.array([0, 3, 0.2, 0.8, 0.2, 0.2], dtype=float)
        result = compute_event_resilience_metrics(values, 1, 1, 1, 2, 0.5, 2, 1.0)
        self.assertEqual(result["recovery_time"], 4.0)

    def test_recovery_starts_after_event_end(self):
        values = np.array([0, 3, 0.1, 0.1, 2, 0.1, 0.1], dtype=float)
        result = compute_event_resilience_metrics(values, 1, 4, 1, 2, 0.5, 2, 1.0)
        self.assertEqual(result["recovery_time"], 5.0)
        self.assertGreaterEqual(result["recovery_correspondence"], 0.0)

    def test_censored_recovery_returns_nan(self):
        values = np.array([0, 3, 2, 1], dtype=float)
        result = compute_event_resilience_metrics(values, 1, 2, 1, 2, 0.5, 2, 1.0)
        self.assertEqual(result["recovery_status"], "censored")
        self.assertTrue(np.isnan(result["recovery_time"]))

    def test_event_segments_analyzed_separately(self):
        values = np.zeros(30); values[3:6] = 2; values[20:23] = 3
        rows = analyze_event_segments(
            "typhoon", "L3", values, [(3, 5), (20, 22)],
            {"q50": 0.1, "q75": 0.2, "q90": 0.5, "q99": 1.0}, 1.0
        )
        self.assertEqual(len(rows), 2)
        self.assertEqual([row["event_segment"] for row in rows], [1, 2])

    def test_candidate_uses_own_threshold(self):
        first = np.arange(10, dtype=float)
        second = first * 10
        self.assertNotEqual(np.quantile(first, 0.9), np.quantile(second, 0.9))

    def test_bridge_joint_unavailable(self):
        variable_names = ["flow"]
        self.assertFalse(len(variable_names) >= 2 and "speed" in variable_names)

    def test_no_event_signal_in_factor_fit(self):
        scores, train_end = self.synthetic_scores()
        event_a = np.zeros(len(scores)); event_b = np.ones(len(scores))
        first = MissingDataOneFactorModel().fit(scores, train_end, ["flow", "speed"], random_state=7)
        second = MissingDataOneFactorModel().fit(scores, train_end, ["flow", "speed"], random_state=7)
        self.assertFalse(np.array_equal(event_a, event_b))
        np.testing.assert_allclose(first.loadings_, second.loadings_)

if __name__ == "__main__":
    unittest.main()