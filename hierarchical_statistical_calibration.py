"""Hierarchical robust residual calibration for physical traffic forecasts."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from l4_prediction_evaluation import regression_metrics


METRICS = ("mae", "rmse", "smape", "wape")
WEIGHT_GRID = (0.85, 0.90, 0.95, 1.00, 1.05)


@dataclass(frozen=True)
class HierarchicalCalibration:
    blend_weight: float
    correction: np.ndarray
    correction_shrinkage: float
    calibration_metric_ratios: dict[str, float]

    def apply(self, prediction: np.ndarray, persistence: np.ndarray) -> np.ndarray:
        prediction = np.asarray(prediction, dtype=float)
        persistence = np.asarray(persistence, dtype=float)
        if prediction.shape != persistence.shape:
            raise ValueError("prediction and persistence shapes differ")
        blended = self.blend_weight * prediction + (1.0 - self.blend_weight) * persistence
        calibrated = blended + self.correction_shrinkage * self.correction
        return np.maximum(calibrated, 0.0)

    def save(self, path: Path) -> None:
        np.savez_compressed(
            path,
            blend_weight=np.asarray(self.blend_weight),
            correction=np.asarray(self.correction, dtype=float),
            correction_shrinkage=np.asarray(self.correction_shrinkage),
            metric_names=np.asarray(list(self.calibration_metric_ratios)),
            metric_ratios=np.asarray(list(self.calibration_metric_ratios.values()), dtype=float),
        )

    @classmethod
    def load(cls, path: Path) -> "HierarchicalCalibration":
        archive = np.load(path, allow_pickle=True)
        names = [str(value) for value in archive["metric_names"].tolist()]
        ratios = [float(value) for value in archive["metric_ratios"].tolist()]
        return cls(
            blend_weight=float(archive["blend_weight"]),
            correction=np.asarray(archive["correction"], dtype=float),
            correction_shrinkage=float(archive["correction_shrinkage"]),
            calibration_metric_ratios=dict(zip(names, ratios)),
        )


def fit_hierarchical_calibration(
    truth: np.ndarray,
    prediction: np.ndarray,
    persistence: np.ndarray,
    correction_shrinkage: float = 0.25,
    weight_grid: tuple[float, ...] = WEIGHT_GRID,
) -> HierarchicalCalibration:
    truth = np.asarray(truth, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    persistence = np.asarray(persistence, dtype=float)
    if truth.shape != prediction.shape or truth.shape != persistence.shape:
        raise ValueError("truth, prediction and persistence shapes must match")
    if truth.ndim != 4:
        raise ValueError("forecast tensors must have shape [windows,horizon,nodes,features]")
    if not 0.0 <= correction_shrinkage <= 1.0:
        raise ValueError("correction_shrinkage must be between 0 and 1")
    baseline_metrics = regression_metrics(truth, prediction)
    best = None
    for blend_weight in weight_grid:
        blended = float(blend_weight) * prediction + (1.0 - float(blend_weight)) * persistence
        residual = np.where(np.isfinite(truth), truth - blended, np.nan)
        correction = np.nanmedian(residual, axis=0)
        correction = np.where(np.isfinite(correction), correction, 0.0)
        calibrated = np.maximum(blended + correction_shrinkage * correction, 0.0)
        calibrated_metrics = regression_metrics(truth, calibrated)
        ratios = {
            metric: float(calibrated_metrics[metric] / baseline_metrics[metric])
            for metric in METRICS
        }
        score = (max(ratios.values()), float(np.mean(list(ratios.values()))), abs(float(blend_weight) - 1.0))
        if best is None or score < best[0]:
            best = (score, float(blend_weight), correction, ratios)
    if best is None:
        raise RuntimeError("no calibration candidate was evaluated")
    _, blend_weight, correction, ratios = best
    return HierarchicalCalibration(
        blend_weight=blend_weight,
        correction=correction,
        correction_shrinkage=float(correction_shrinkage),
        calibration_metric_ratios=ratios,
    )
