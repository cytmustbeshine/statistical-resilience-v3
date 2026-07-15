"""Two-dimension demand and operating-efficiency traffic resilience."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from latent_traffic_performance import MissingDataOneFactorModel


class DemandEfficiencyModel:
    """Separate flow demand/service state from speed-occupancy efficiency state.

    Flow is never fitted as an efficiency indicator. With speed only, the
    efficiency score is observed directly. With speed and orientation-reversed
    occupancy, a train-only missing-data one-factor model estimates efficiency.
    """

    def __init__(self):
        self.efficiency_mode_: str = "unavailable"
        self.efficiency_factor_: MissingDataOneFactorModel | None = None
        self.train_end_: int = 0
        self.variable_names_: list[str] = []

    def fit(
        self,
        demand_scores: np.ndarray,
        speed_scores: np.ndarray | None,
        occupancy_scores: np.ndarray | None,
        train_end: int,
        initialization_seeds: list[int] | tuple[int, ...] = (1, 7, 21, 42, 100),
        max_iter: int = 500,
    ) -> "DemandEfficiencyModel":
        """Fit only the efficiency measurement relation using training data.

        Args:
            demand_scores: Flow conditional normal scores, shape ``[T,N]``.
            speed_scores: Higher-is-better speed scores or ``None``.
            occupancy_scores: Higher-is-better reversed occupancy scores or ``None``.
            train_end: Exclusive training cutoff.
            initialization_seeds: Deterministic train-only factor starts.
            max_iter: Maximum EM iterations.

        Returns:
            Fitted two-dimension model. Demand remains the supplied flow score.
        """
        demand = np.asarray(demand_scores, dtype=float)
        if demand.ndim != 2:
            raise ValueError("demand_scores must have shape [T,N]")
        self.train_end_ = int(np.clip(train_end, 0, len(demand)))
        if speed_scores is None:
            self.efficiency_mode_ = "unavailable"
            self.variable_names_ = []
            return self
        speed = np.asarray(speed_scores, dtype=float)
        if speed.shape != demand.shape:
            raise ValueError("speed and demand score shapes differ")
        if occupancy_scores is None:
            self.efficiency_mode_ = "speed_observed"
            self.variable_names_ = ["speed"]
            return self
        occupancy = np.asarray(occupancy_scores, dtype=float)
        if occupancy.shape != demand.shape:
            raise ValueError("occupancy and demand score shapes differ")
        cube = np.stack([speed, occupancy], axis=2)
        self.efficiency_factor_ = MissingDataOneFactorModel().fit(
            cube,
            self.train_end_,
            ["speed", "occupancy"],
            max_iter=max_iter,
            random_state=list(initialization_seeds),
        )
        self.efficiency_mode_ = "speed_occupancy_factor"
        self.variable_names_ = ["speed", "occupancy"]
        return self

    def transform(
        self,
        demand_scores: np.ndarray,
        speed_scores: np.ndarray | None,
        occupancy_scores: np.ndarray | None,
    ) -> dict[str, np.ndarray]:
        """Return separated demand and efficiency states without scalar mixing."""
        demand = np.asarray(demand_scores, dtype=float)
        if self.efficiency_mode_ == "unavailable":
            efficiency = np.full_like(demand, np.nan)
            variance = np.full_like(demand, np.nan)
            observed = np.zeros(demand.shape, dtype=int)
        elif self.efficiency_mode_ == "speed_observed":
            efficiency = np.asarray(speed_scores, dtype=float).copy()
            variance = np.where(np.isfinite(efficiency), 0.0, np.nan)
            observed = np.isfinite(efficiency).astype(int)
        else:
            speed = np.asarray(speed_scores, dtype=float)
            occupancy = np.asarray(occupancy_scores, dtype=float)
            posterior = self.efficiency_factor_.transform(
                np.stack([speed, occupancy], axis=2)
            )
            efficiency = posterior["posterior_mean"]
            variance = posterior["posterior_variance"]
            observed = posterior["n_observed_modalities"]
        return {
            "demand_score": demand.copy(),
            "efficiency_score": efficiency,
            "efficiency_posterior_variance": variance,
            "efficiency_observed_modalities": observed,
            "demand_valid_mask": np.isfinite(demand),
            "efficiency_valid_mask": np.isfinite(efficiency),
        }

    def save(self, path: str | Path) -> None:
        """Save compact model parameters to NPZ."""
        payload = {
            "efficiency_mode": np.asarray(self.efficiency_mode_),
            "train_end": np.asarray(self.train_end_),
            "variable_names": np.asarray(self.variable_names_, dtype=object),
        }
        if self.efficiency_factor_ is not None:
            payload.update(
                loadings=self.efficiency_factor_.loadings_,
                residual_variance=self.efficiency_factor_.residual_variance_,
                sign_anchor=np.asarray(self.efficiency_factor_.sign_anchor_),
            )
        np.savez_compressed(path, **payload)

    @classmethod
    def load(cls, path: str | Path) -> "DemandEfficiencyModel":
        """Load a saved demand-efficiency model."""
        model = cls()
        with np.load(path, allow_pickle=True) as archive:
            model.efficiency_mode_ = str(archive["efficiency_mode"].item())
            model.train_end_ = int(archive["train_end"].item())
            model.variable_names_ = [str(value) for value in archive["variable_names"].tolist()]
            if "loadings" in archive.files:
                factor = MissingDataOneFactorModel()
                factor.loadings_ = np.asarray(archive["loadings"], dtype=float)
                factor.residual_variance_ = np.asarray(archive["residual_variance"], dtype=float)
                factor.variable_names_ = model.variable_names_.copy()
                factor.train_end_ = model.train_end_
                factor.sign_anchor_ = str(archive["sign_anchor"].item())
                model.efficiency_factor_ = factor
        return model


def two_dimension_event_states(
    demand_system_deficit: np.ndarray,
    efficiency_system_deficit: np.ndarray | None,
    demand_threshold: float,
    efficiency_threshold: float | None,
) -> dict[str, np.ndarray]:
    """Classify mutually exclusive demand-efficiency high-state quadrants.

    Returns both-high, demand-only, efficiency-only and neither masks. When the
    efficiency dimension is unavailable, ``available`` is false and no joint
    state is fabricated.
    """
    demand = np.asarray(demand_system_deficit, dtype=float)
    if efficiency_system_deficit is None or efficiency_threshold is None:
        return {"available": False}
    efficiency = np.asarray(efficiency_system_deficit, dtype=float)
    if demand.shape != efficiency.shape:
        raise ValueError("system deficit shapes differ")
    valid = np.isfinite(demand) & np.isfinite(efficiency)
    demand_high = demand > demand_threshold
    efficiency_high = efficiency > efficiency_threshold
    both = valid & demand_high & efficiency_high
    demand_only = valid & demand_high & ~efficiency_high
    efficiency_only = valid & ~demand_high & efficiency_high
    neither = valid & ~demand_high & ~efficiency_high
    return {
        "available": True,
        "both_high": both,
        "demand_only_high": demand_only,
        "efficiency_only_high": efficiency_only,
        "neither_high": neither,
        "valid_mask": valid,
    }