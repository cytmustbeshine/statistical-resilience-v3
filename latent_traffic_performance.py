"""Latent traffic performance and probabilistic resilience definitions."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.special import ndtri

from statistical_resilience_profile import (
    fit_hierarchical_normal_profile,
    fit_shrunk_conditional_ecdf,
    query_conditional_cdf,
)


def transform_conditional_normal_scores(
    values: np.ndarray,
    profile: dict[str, object],
    conditional_ecdf: dict[str, object],
    timestamps: np.ndarray,
    orientation: str,
    probability_floor: float = 1e-4,
) -> dict[str, np.ndarray]:
    """Map one traffic variable to an oriented conditional normal-score scale.

    Mathematically, ``Z = Phi^-1(clip(F_train(x | node, bin, daytype)))``.
    ``higher_is_worse`` reverses the sign before the measurement model, while
    ``data_driven_loading`` leaves flow orientation to its train-only loading.

    Args:
        values: Observations with shape ``[T, N]``.
        profile: Train-only hierarchical robust profile.
        conditional_ecdf: Train-only hierarchical ECDF.
        timestamps: Query timestamps.
        orientation: ``higher_is_better``, ``higher_is_worse`` or
            ``data_driven_loading``.
        probability_floor: Symmetric clipping floor before ``Phi^-1``.

    Returns:
        Raw CDF, finite normal scores, oriented scores, source codes and masks.
    """
    if orientation not in {"higher_is_better", "higher_is_worse", "data_driven_loading"}:
        raise ValueError("invalid orientation")
    query = query_conditional_cdf(
        values,
        profile,
        conditional_ecdf,
        timestamps=timestamps,
        probability_floor=probability_floor,
    )
    raw_cdf = query["cdf"]
    normal = np.where(query["valid_mask"], ndtri(raw_cdf), np.nan)
    oriented = -normal if orientation == "higher_is_worse" else normal.copy()
    return {
        "normal_score": normal,
        "raw_cdf": raw_cdf,
        "oriented_score": oriented,
        "source_level": query["source_level"],
        "valid_mask": query["valid_mask"],
    }


def _as_score_cube(scores: np.ndarray) -> np.ndarray:
    array = np.asarray(scores, dtype=float)
    if array.ndim != 3:
        raise ValueError("scores must have shape [T,N,J]")
    return np.where(np.isfinite(array), array, np.nan)


def _posterior_flat(data: np.ndarray, loadings: np.ndarray, residual: np.ndarray):
    valid = np.isfinite(data)
    precision = np.where(valid, loadings[None, :] ** 2 / residual[None, :], 0.0).sum(axis=1)
    variance = 1.0 / (1.0 + precision)
    weighted = np.where(
        valid,
        data * loadings[None, :] / residual[None, :],
        0.0,
    ).sum(axis=1)
    mean = variance * weighted
    observed = valid.sum(axis=1)
    mean[observed == 0] = np.nan
    variance[observed == 0] = np.nan
    return mean, variance, observed, valid


def _pattern_statistics(data: np.ndarray) -> list[dict[str, np.ndarray | int]]:
    """Compress observations into counts and scatter matrices by missing pattern."""
    valid = np.isfinite(data)
    patterns, inverse = np.unique(valid, axis=0, return_inverse=True)
    statistics = []
    for pattern_index, pattern in enumerate(patterns):
        columns = np.flatnonzero(pattern)
        if columns.size == 0:
            continue
        observed = data[inverse == pattern_index][:, columns]
        statistics.append(
            {
                "columns": columns,
                "count": int(len(observed)),
                "scatter": observed.T @ observed,
            }
        )
    return statistics


def _statistics_log_likelihood(statistics, loadings, residual) -> float:
    """Evaluate observed Gaussian log likelihood from sufficient statistics."""
    total = 0.0
    for pattern in statistics:
        columns = pattern["columns"]
        count = pattern["count"]
        scatter = pattern["scatter"]
        covariance = np.diag(residual[columns]) + np.outer(loadings[columns], loadings[columns])
        sign, logdet = np.linalg.slogdet(covariance)
        if sign <= 0:
            return -np.inf
        inverse = np.linalg.inv(covariance)
        quadratic_sum = float(np.trace(inverse @ scatter))
        total += -0.5 * (
            count * (len(columns) * np.log(2 * np.pi) + logdet) + quadratic_sum
        )
    return float(total)

class MissingDataOneFactorModel:
    """Gaussian one-factor model fitted by EM with arbitrary missing patterns."""

    def __init__(self):
        self.loadings_: np.ndarray | None = None
        self.residual_variance_: np.ndarray | None = None
        self.variable_names_: list[str] = []
        self.log_likelihood_history_: list[float] = []
        self.converged_: bool = False
        self.n_iter_: int = 0
        self.train_end_: int = 0
        self.sign_anchor_: str = "unidentified"
        self.initialization_diagnostics_: list[dict[str, object]] = []
        self.min_residual_variance_: float = 1e-4

    def _initial_parameters(self, data: np.ndarray, rng: np.random.Generator):
        n_variables = data.shape[1]
        variance = np.nanvar(data, axis=0)
        variance = np.where(np.isfinite(variance) & (variance > 1e-3), variance, 1.0)
        loadings = rng.normal(0.55, 0.12, n_variables)
        residual = np.maximum(variance - np.minimum(loadings**2, 0.8 * variance), self.min_residual_variance_)
        return loadings, residual

    def _fit_one(self, data, seed, max_iter, tolerance):
        rng = np.random.default_rng(seed)
        loadings, residual = self._initial_parameters(data, rng)
        statistics = _pattern_statistics(data)
        history: list[float] = []
        converged = False
        n_variables = data.shape[1]
        for iteration in range(1, max_iter + 1):
            numerator = np.zeros(n_variables)
            denominator = np.zeros(n_variables)
            sum_squares = np.zeros(n_variables)
            counts = np.zeros(n_variables, dtype=int)
            for pattern in statistics:
                columns = pattern["columns"]
                count = pattern["count"]
                scatter = pattern["scatter"]
                local_loadings = loadings[columns]
                local_residual = residual[columns]
                posterior_variance = 1.0 / (
                    1.0 + np.sum(local_loadings**2 / local_residual)
                )
                coefficient = posterior_variance * local_loadings / local_residual
                sum_z_h = scatter @ coefficient
                sum_h_second = float(
                    count * posterior_variance + coefficient @ scatter @ coefficient
                )
                numerator[columns] += sum_z_h
                denominator[columns] += sum_h_second
                sum_squares[columns] += np.diag(scatter)
                counts[columns] += count
            new_loadings = loadings.copy()
            estimable = denominator > 0
            new_loadings[estimable] = numerator[estimable] / denominator[estimable]
            new_residual = residual.copy()
            residual_estimable = counts > 0
            residual_sum = (
                sum_squares
                - 2 * new_loadings * numerator
                + new_loadings**2 * denominator
            )
            new_residual[residual_estimable] = np.maximum(
                residual_sum[residual_estimable] / counts[residual_estimable],
                self.min_residual_variance_,
            )
            loadings, residual = new_loadings, new_residual
            likelihood = _statistics_log_likelihood(statistics, loadings, residual)
            history.append(likelihood)
            if len(history) >= 2:
                improvement = history[-1] - history[-2]
                scale = max(abs(history[-2]), 1.0)
                if improvement >= -1e-7 * scale and abs(improvement) <= tolerance * scale:
                    converged = True
                    break
        return {
            "seed": int(seed),
            "loadings": loadings,
            "residual": residual,
            "history": history,
            "converged": converged,
            "n_iter": iteration,
            "final_log_likelihood": history[-1] if history else -np.inf,
        }

    def fit(
        self,
        scores: np.ndarray,
        train_end: int,
        variable_names: list[str],
        max_iter: int = 500,
        tolerance: float = 1e-6,
        min_residual_variance: float = 1e-4,
        random_state: int | list[int] = 42,
        n_initializations: int = 1,
    ) -> "MissingDataOneFactorModel":
        """Fit parameters using only ``scores[:train_end]``.

        Args:
            scores: Oriented score cube with shape ``[T, N, J]``.
            train_end: Exclusive training cutoff.
            variable_names: Names corresponding to the final dimension.
            max_iter: Maximum EM iterations per initialization.
            tolerance: Relative log-likelihood convergence tolerance.
            min_residual_variance: Positive floor for diagonal ``Psi``.
            random_state: Base seed or an explicit list of initialization seeds.
            n_initializations: Number of deterministic consecutive seeds when a
                scalar ``random_state`` is supplied.

        Returns:
            The fitted model using the highest training log likelihood solution.
        """
        cube = _as_score_cube(scores)
        if cube.shape[2] != len(variable_names):
            raise ValueError("variable_names length does not match scores")
        end = int(np.clip(train_end, 0, len(cube)))
        if end == 0:
            raise ValueError("empty training segment")
        data = cube[:end].reshape(-1, cube.shape[2])
        if not np.isfinite(data).any():
            raise ValueError("training scores contain no finite observations")
        self.variable_names_ = list(variable_names)
        self.train_end_ = end
        self.min_residual_variance_ = float(min_residual_variance)
        if isinstance(random_state, (list, tuple, np.ndarray)):
            seeds = [int(seed) for seed in random_state]
        else:
            seeds = [int(random_state) + offset for offset in range(max(1, int(n_initializations)))]
        solutions = [self._fit_one(data, seed, max_iter, tolerance) for seed in seeds]
        best = max(solutions, key=lambda item: item["final_log_likelihood"])
        loadings = np.asarray(best["loadings"], dtype=float)
        residual = np.maximum(np.asarray(best["residual"], dtype=float), self.min_residual_variance_)

        anchor_index = None
        anchor_name = None
        for candidate in ("speed", "occupancy"):
            if candidate in self.variable_names_:
                anchor_index = self.variable_names_.index(candidate)
                anchor_name = candidate
                break
        if anchor_index is not None:
            if loadings[anchor_index] < 0:
                loadings = -loadings
            self.sign_anchor_ = f"{anchor_name}_loading_positive"
        else:
            self.sign_anchor_ = "unidentified_no_speed_or_occupancy"

        self.loadings_ = loadings
        self.residual_variance_ = residual
        self.log_likelihood_history_ = [float(value) for value in best["history"]]
        self.converged_ = bool(best["converged"])
        self.n_iter_ = int(best["n_iter"])
        self.initialization_diagnostics_ = [
            {
                "seed": int(solution["seed"]),
                "final_log_likelihood": float(solution["final_log_likelihood"]),
                "converged": bool(solution["converged"]),
                "n_iter": int(solution["n_iter"]),
                "loadings": np.asarray(solution["loadings"], dtype=float),
                "residual_variance": np.asarray(solution["residual"], dtype=float),
            }
            for solution in solutions
        ]
        return self

    def transform(self, scores: np.ndarray) -> dict[str, np.ndarray]:
        """Compute posterior mean and variance without zero-filling missing modes.

        For observed set ``O``, ``Var(H|Z_O) = (1 + lambda_O' Psi_O^-1
        lambda_O)^-1`` and ``E(H|Z_O) = Var(H|Z_O) lambda_O' Psi_O^-1 Z_O``.
        """
        if self.loadings_ is None or self.residual_variance_ is None:
            raise RuntimeError("model is not fitted")
        cube = _as_score_cube(scores)
        if cube.shape[2] != len(self.loadings_):
            raise ValueError("score variable count differs from fitted model")
        flat = cube.reshape(-1, cube.shape[2])
        mean, variance, observed, valid = _posterior_flat(
            flat, self.loadings_, self.residual_variance_
        )
        bit_weights = (1 << np.arange(cube.shape[2], dtype=np.int64))[None, :]
        pattern = np.sum(valid.astype(np.int64) * bit_weights, axis=1)
        shape = cube.shape[:2]
        return {
            "posterior_mean": mean.reshape(shape),
            "posterior_variance": variance.reshape(shape),
            "n_observed_modalities": observed.reshape(shape),
            "observation_pattern": pattern.reshape(shape),
            "valid_mask": (observed > 0).reshape(shape),
        }

    @property
    def communalities_(self) -> np.ndarray:
        if self.loadings_ is None or self.residual_variance_ is None:
            raise RuntimeError("model is not fitted")
        return self.loadings_**2 / (self.loadings_**2 + self.residual_variance_)

    def save(self, path: str | Path) -> None:
        """Save fitted numeric parameters and compact metadata to NPZ."""
        if self.loadings_ is None or self.residual_variance_ is None:
            raise RuntimeError("model is not fitted")
        np.savez_compressed(
            path,
            loadings=self.loadings_,
            residual_variance=self.residual_variance_,
            variable_names=np.asarray(self.variable_names_, dtype=object),
            log_likelihood_history=np.asarray(self.log_likelihood_history_, dtype=float),
            converged=np.asarray(self.converged_),
            n_iter=np.asarray(self.n_iter_),
            train_end=np.asarray(self.train_end_),
            sign_anchor=np.asarray(self.sign_anchor_),
            min_residual_variance=np.asarray(self.min_residual_variance_),
        )

    @classmethod
    def load(cls, path: str | Path) -> "MissingDataOneFactorModel":
        """Load a model saved by :meth:`save`."""
        model = cls()
        with np.load(path, allow_pickle=True) as archive:
            model.loadings_ = np.asarray(archive["loadings"], dtype=float)
            model.residual_variance_ = np.asarray(archive["residual_variance"], dtype=float)
            model.variable_names_ = [str(value) for value in archive["variable_names"].tolist()]
            model.log_likelihood_history_ = archive["log_likelihood_history"].astype(float).tolist()
            model.converged_ = bool(archive["converged"].item())
            model.n_iter_ = int(archive["n_iter"].item())
            model.train_end_ = int(archive["train_end"].item())
            model.sign_anchor_ = str(archive["sign_anchor"].item())
            model.min_residual_variance_ = float(archive["min_residual_variance"].item())
        return model

def fit_latent_performance_profile(
    posterior_mean: np.ndarray,
    train_end: int,
    timestamps: np.ndarray,
    **profile_kwargs,
) -> dict[str, object]:
    """Fit the shared hierarchical profile to signed latent performance scores.

    ``H`` is never clipped or log-transformed; the wrapper always uses
    ``identity_robust`` and only training observations.
    """
    profile = fit_hierarchical_normal_profile(
        posterior_mean,
        train_end,
        timestamps,
        transform_mode="identity_robust",
        **profile_kwargs,
    )
    ecdf = fit_shrunk_conditional_ecdf(
        posterior_mean,
        profile,
        train_end,
        timestamps,
    )
    profile.update(
        variable_name="latent_performance",
        tail_direction="lower",
        transform_mode="identity_robust",
    )
    return {"profile": profile, "ecdf": ecdf}


def _trimmed_system(values: np.ndarray, trim_fraction: float) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    output = np.full(len(array), np.nan)
    trim = float(np.clip(trim_fraction, 0.0, 0.49))
    for time_index, row in enumerate(array):
        finite = np.sort(row[np.isfinite(row)])
        cut = int(len(finite) * trim)
        if len(finite) - 2 * cut > 0:
            finite = finite[cut : len(finite) - cut]
        if finite.size:
            output[time_index] = np.mean(finite)
    return output


def compute_latent_resilience_deficit(
    posterior_mean: np.ndarray,
    posterior_variance: np.ndarray,
    latent_profile: dict[str, object],
    latent_ecdf: dict[str, object],
    timestamps: np.ndarray,
    probability_floor: float = 1e-4,
    system_trim_fraction: float = 0.10,
    deficit_clip_quantile: float = 0.999,
) -> dict[str, np.ndarray]:
    """Compute instantaneous probabilistic latent-performance loss.

    The node deficit is ``D_H=-log(F_train(H_hat))*I(H_hat<mu_H)``. Posterior
    variance is returned as measurement uncertainty but is not multiplied into
    the first-version deficit. System loss is a NaN-aware symmetric trimmed mean.

    Args:
        posterior_mean: ``E(H|Z_observed)`` with shape ``[T,N]``.
        posterior_variance: ``Var(H|Z_observed)`` with the same shape.
        latent_profile: Train-only hierarchical robust profile for ``H_hat``.
        latent_ecdf: Train-only hierarchical ECDF for ``H_hat``.
        timestamps: Query timestamps.
        probability_floor: Lower probability clip.
        system_trim_fraction: Fraction trimmed from each node tail.
        deficit_clip_quantile: Train-only node-deficit clipping quantile.

    Returns:
        Node/system deficits and resilience states, uncertainty, source codes,
        and train-only q50/q75/q90/q95/q99 thresholds.
    """
    mean = np.asarray(posterior_mean, dtype=float)
    variance = np.asarray(posterior_variance, dtype=float)
    if mean.shape != variance.shape:
        raise ValueError("posterior mean and variance shapes differ")
    query = query_conditional_cdf(
        mean,
        latent_profile,
        latent_ecdf,
        timestamps=timestamps,
        probability_floor=probability_floor,
    )
    probability = query["cdf"]
    location = query["conditional_median"]
    valid = query["valid_mask"] & np.isfinite(variance)
    surprise = -np.log(probability)
    deficit = np.where(valid, np.where(mean < location, surprise, 0.0), np.nan)
    train_end = min(int(latent_profile["train_end_exclusive"]), len(deficit))
    train_values = deficit[:train_end][np.isfinite(deficit[:train_end])]
    clip_value = (
        float(np.quantile(train_values, deficit_clip_quantile))
        if train_values.size
        else -np.log(probability_floor)
    )
    deficit = np.clip(deficit, 0.0, max(clip_value, 1e-6))
    system = _trimmed_system(deficit, system_trim_fraction)
    train_system = system[:train_end][np.isfinite(system[:train_end])]
    quantiles = {
        f"q{int(q * 100):02d}": float(np.quantile(train_system, q)) if train_system.size else np.nan
        for q in (0.5, 0.75, 0.9, 0.95, 0.99)
    }
    return {
        "p_latent_lower": probability,
        "latent_surprisal": surprise,
        "latent_deficit": deficit,
        "node_resilience": np.exp(-deficit),
        "system_deficit": system,
        "system_resilience": np.exp(-system),
        "posterior_variance": variance.copy(),
        "conditional_median": location,
        "source_level": query["source_level"],
        "valid_mask": valid,
        "deficit_clip_value": np.asarray(clip_value),
        "train_deficit_quantiles": quantiles,
        "high_state_threshold": np.asarray(quantiles["q90"]),
        "extreme_state_threshold": np.asarray(quantiles["q99"]),
    }