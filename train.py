"""Training entry point for D-STSGCN backbone experiments.

DGCN/quality_v2/OOD-scaled paths are retained for legacy reproduction and for
reuse by the upcoming traffic-resilience auxiliary model. They are no longer
treated as active thesis main models by ``run_experiments.py``.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from data import (
    SplitScaler,
    TrafficWindowDataset,
    build_static_adjacency,
    load_wide_traffic_csv,
    resolve_time_index,
    split_traffic_window_indices,
)
from model import DSTSGCN
from resilience_metrics import (
    compute_R_series,
    compute_baseline,
    compute_log_resilience_series,
    compute_robust_shrunk_baseline,
)


def parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_optional_suffix(value: str) -> str | None:
    value = value.strip()
    if value.lower() in {"", "none", "null"}:
        return None
    return value


def parse_optional_name(value: str) -> str | None:
    value = value.strip()
    if value.lower() in {"", "none", "null"}:
        return None
    return value


def resolve_feature_index(feature_name: str | None, feature_names: list[str]) -> int | None:
    if feature_name is None:
        return None
    if feature_name not in feature_names:
        raise ValueError(
            f"Feature column {feature_name!r} is not available. "
            f"Available feature names: {feature_names}"
        )
    return feature_names.index(feature_name)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def extract_event_signal(x: torch.Tensor, event_feature_idx: int | None) -> torch.Tensor | None:
    if event_feature_idx is None:
        return None
    return x[:, -1, :, event_feature_idx].mean(dim=1, keepdim=True)


def event_window_values(event_series: np.ndarray, history: int, num_windows: int) -> np.ndarray:
    return np.asarray(
        [event_series[min(idx + history - 1, len(event_series) - 1)] for idx in range(num_windows)],
        dtype=np.float64,
    )


def split_dataset(
    dataset: TrafficWindowDataset,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    split_mode: str = "chronological",
    event_series: np.ndarray | None = None,
    event_window_threshold: float = 0.0,
    explicit_event_index: int | None = None,
    event_test_prehistory_steps: int | None = None,
) -> tuple[Subset, Subset, Subset, dict[str, object]]:
    """Split windows using the shared, auditable project split implementation."""
    train_indices, val_indices, test_indices, info = split_traffic_window_indices(
        len(dataset),
        history=dataset.history,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        split_mode=split_mode,
        event_series=event_series,
        event_window_threshold=event_window_threshold,
        explicit_event_index=explicit_event_index,
        event_test_prehistory_steps=event_test_prehistory_steps,
    )
    return (
        Subset(dataset, train_indices.tolist()),
        Subset(dataset, val_indices.tolist()),
        Subset(dataset, test_indices.tolist()),
        info,
    )

@torch.no_grad()
def estimate_train_ood_profile(
    dataset: torch.utils.data.Dataset,
    ood_clip: float = 5.0,
    use_volatility: bool = True,
) -> dict[str, float]:
    """Estimate robust train-only OOD profile from scaled traffic windows."""
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    chunks: list[torch.Tensor] = []
    for x, _ in loader:
        chunks.append(x[..., 0].float().reshape(-1).cpu())
    if not chunks:
        return {"enabled": True, "count": 0}

    all_traffic = torch.cat(chunks)
    traffic_median = torch.median(all_traffic)
    traffic_mad = torch.median(torch.abs(all_traffic - traffic_median)).clamp_min(1e-6)
    robust_scale = 1.4826 * traffic_mad

    raw_values: list[torch.Tensor] = []
    vol_values: list[torch.Tensor] = []
    for x, _ in loader:
        traffic = x[..., 0].float()
        robust_z = torch.abs((traffic - traffic_median) / robust_scale)
        raw_ood = torch.clamp(robust_z, 0.0, float(ood_clip)).mean(dim=(1, 2))
        raw_values.append(raw_ood.cpu())
        if traffic.size(1) > 1:
            vol_values.append(torch.abs(torch.diff(traffic, dim=1)).mean(dim=(1, 2)).cpu())
        else:
            vol_values.append(torch.zeros(traffic.size(0)))

    raw_all = torch.cat(raw_values).float()
    vol_all = torch.cat(vol_values).float()
    raw_median = torch.median(raw_all)
    raw_mad = torch.median(torch.abs(raw_all - raw_median)).clamp_min(1e-6)
    vol_median = torch.median(vol_all)
    vol_mad = torch.median(torch.abs(vol_all - vol_median)).clamp_min(1e-6)
    raw_z = (raw_all - raw_median) / (1.4826 * raw_mad)
    vol_z = (vol_all - vol_median) / (1.4826 * vol_mad)
    combined = 0.7 * raw_z + 0.3 * vol_z if use_volatility else raw_z

    def q(values: torch.Tensor, p: float) -> float:
        return float(torch.quantile(values, p).item())

    return {
        "enabled": True,
        "count": int(raw_all.numel()),
        "traffic_median": float(traffic_median.item()),
        "traffic_mad": float(traffic_mad.item()),
        "raw_ood_median": float(raw_median.item()),
        "raw_ood_mad": float(raw_mad.item()),
        "raw_ood_q95": q(raw_all, 0.95),
        "raw_ood_q99": q(raw_all, 0.99),
        "volatility_median": float(vol_median.item()),
        "volatility_mad": float(vol_mad.item()),
        "combined_ood_q95": q(combined, 0.95),
        "combined_ood_q99": q(combined, 0.99),
        "ood_clip": float(ood_clip),
        "use_volatility": bool(use_volatility),
    }


def extract_ood_intensity(
    x: torch.Tensor,
    profile: dict[str, float] | None,
    ood_clip: float = 5.0,
    use_volatility: bool = True,
) -> torch.Tensor | None:
    if not profile:
        return None
    traffic = x[..., 0]
    traffic_median = x.new_tensor(float(profile.get("traffic_median", 0.0)))
    traffic_mad = x.new_tensor(max(float(profile.get("traffic_mad", 1.0)), 1e-6))
    raw_median = x.new_tensor(float(profile.get("raw_ood_median", 0.0)))
    raw_mad = x.new_tensor(max(float(profile.get("raw_ood_mad", 1.0)), 1e-6))
    vol_median = x.new_tensor(float(profile.get("volatility_median", 0.0)))
    vol_mad = x.new_tensor(max(float(profile.get("volatility_mad", 1.0)), 1e-6))
    q95 = x.new_tensor(float(profile.get("combined_ood_q95", 0.0)))
    q99 = x.new_tensor(float(profile.get("combined_ood_q99", 1.0)))

    robust_z = torch.abs((traffic - traffic_median) / (1.4826 * traffic_mad))
    raw_ood = torch.clamp(robust_z, 0.0, float(ood_clip)).mean(dim=(1, 2))
    raw_z = (raw_ood - raw_median) / (1.4826 * raw_mad)
    if use_volatility and traffic.size(1) > 1:
        volatility = torch.abs(torch.diff(traffic, dim=1)).mean(dim=(1, 2))
        vol_z = (volatility - vol_median) / (1.4826 * vol_mad)
        combined = 0.7 * raw_z + 0.3 * vol_z
    else:
        combined = raw_z
    intensity = torch.relu(combined - q95) / torch.clamp(q99 - q95, min=1e-6)
    return torch.clamp(intensity, 0.0, 1.0).unsqueeze(-1)


class ResilienceAuxWindowDataset(TrafficWindowDataset):
    """Traffic window dataset with train-only statistical resilience targets."""

    def __init__(
        self,
        data: np.ndarray,
        resilience_targets: np.ndarray,
        resilience_baselines: np.ndarray,
        history: int = 12,
        horizon: int = 12,
        target_dim: int = 0,
    ) -> None:
        super().__init__(data, history=history, horizon=horizon, target_dim=target_dim)
        expected = len(self)
        if resilience_targets.shape[:3] != (expected, horizon, data.shape[1]):
            raise ValueError(
                "resilience_targets must have shape [num_windows, horizon, num_nodes, 1]."
            )
        if resilience_baselines.shape[:3] != (expected, horizon, data.shape[1]):
            raise ValueError(
                "resilience_baselines must have shape [num_windows, horizon, num_nodes, 1]."
            )
        self.resilience_targets = resilience_targets.astype(np.float32)
        self.resilience_baselines = resilience_baselines.astype(np.float32)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x, y = super().__getitem__(idx)
        r = torch.from_numpy(self.resilience_targets[idx])
        baseline = torch.from_numpy(self.resilience_baselines[idx])
        return x, y, r, baseline


def build_resilience_supervision(
    raw_data: np.ndarray,
    train_time_end: int,
    history: int,
    horizon: int,
    time_of_day_bins: int = 288,
    baseline_mode: str = "mean_tod",
    shrinkage_candidates: tuple[float, ...] = (0, 1, 3, 7, 14, 28),
    log_ratio_mad_clip: float = 6.0,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Build train-only ratio or robust log-ratio resilience supervision."""
    raw_traffic = raw_data[..., 0].astype(np.float64)
    if baseline_mode == "robust_shrunk":
        baseline = compute_robust_shrunk_baseline(
            raw_traffic,
            train_time_end,
            time_of_day_bins=time_of_day_bins,
            shrinkage_candidates=shrinkage_candidates,
        )
        target_series = compute_log_resilience_series(
            raw_traffic,
            baseline,
            mad_clip=log_ratio_mad_clip,
        )
        target_mode = "log_ratio"
    elif baseline_mode == "mean_tod":
        baseline = compute_baseline(raw_traffic, train_time_end, time_of_day_bins=time_of_day_bins)
        target_series = compute_R_series(raw_traffic, baseline, use_tod=True)
        target_mode = "ratio"
    else:
        raise ValueError("resilience baseline mode must be mean_tod or robust_shrunk.")

    tod_mean = np.asarray(baseline["tod_mean"], dtype=np.float64)
    bins = np.arange(raw_traffic.shape[0]) % tod_mean.shape[0]
    baseline_series = np.maximum(tod_mean[bins], 1e-6)
    num_windows = max(0, raw_data.shape[0] - history - horizon + 1)
    targets = np.zeros((num_windows, horizon, raw_data.shape[1], 1), dtype=np.float32)
    denominators = np.zeros_like(targets)
    for idx in range(num_windows):
        start = idx + history
        end = start + horizon
        targets[idx, :, :, 0] = target_series[start:end]
        denominators[idx, :, :, 0] = baseline_series[start:end]
    default_target = 0.0 if target_mode == "log_ratio" else 1.0
    targets = np.nan_to_num(targets, nan=default_target, posinf=default_target, neginf=default_target)
    if target_mode == "ratio":
        targets = np.clip(targets, 0.0, 2.0)
    denominators = np.maximum(
        np.nan_to_num(denominators, nan=1e-6, posinf=1e-6, neginf=1e-6),
        1e-6,
    ).astype(np.float32)
    train_target = target_series[:train_time_end]
    report: dict[str, object] = {
        "enabled": True,
        "baseline_mode": baseline_mode,
        "target_mode": target_mode,
        "time_of_day_bins": int(time_of_day_bins),
        "train_time_end": int(train_time_end),
        "num_windows": int(num_windows),
        "target_mean": float(np.nanmean(targets)) if targets.size else float("nan"),
        "target_std": float(np.nanstd(targets)) if targets.size else float("nan"),
        "train_target_mean": float(np.nanmean(train_target)) if train_target.size else float("nan"),
        "train_target_std": float(np.nanstd(train_target)) if train_target.size else float("nan"),
        "baseline_mean": float(np.nanmean(baseline_series)),
        "baseline_min": float(np.nanmin(baseline_series)),
        "log_ratio_mad_clip": float(log_ratio_mad_clip),
    }
    for key in (
        "selected_shrinkage_m",
        "shrinkage_cv_scores",
        "train_days",
        "fallback_reason",
        "log_ratio_median",
        "log_ratio_mad",
    ):
        if key in baseline:
            report[key] = baseline[key]
    return targets, denominators, report

def unpack_batch(
    batch: tuple[torch.Tensor, ...],
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Move a traffic batch to device and expose optional resilience labels."""
    if len(batch) == 2:
        x, y = batch
        return x.to(device), y.to(device), None, None
    if len(batch) == 4:
        x, y, resilience_target, resilience_baseline = batch
        return (
            x.to(device),
            y.to(device),
            resilience_target.to(device),
            resilience_baseline.to(device),
        )
    raise ValueError(f"Unexpected batch size: {len(batch)}")


def masked_huber_loss(loss_fn: nn.Module, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Huber loss over finite target positions only."""
    mask = torch.isfinite(target)
    if not bool(mask.any()):
        return pred.new_tensor(0.0)
    return loss_fn(pred[mask], target[mask])



def masked_mean(values: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Return a finite masked mean or a differentiable zero scalar."""
    valid = torch.isfinite(values)
    if mask is not None:
        valid = valid & mask.to(dtype=torch.bool, device=values.device)
    if not bool(valid.any()):
        return values.sum() * 0.0
    return values[valid].mean()


def empirical_cvar_loss(
    elementwise_loss: torch.Tensor,
    alpha: float = 0.90,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Estimate empirical conditional value at risk.

    ``CVaR_alpha(L) = eta + E[(L-eta)_+] / (1-alpha)``, where ``eta`` is the
    empirical alpha quantile. The quantile is detached, while tail losses retain
    their gradients.
    """
    if not 0.0 <= alpha < 1.0:
        raise ValueError("CVaR alpha must satisfy 0 <= alpha < 1.")
    valid = torch.isfinite(elementwise_loss)
    if mask is not None:
        valid = valid & mask.to(dtype=torch.bool, device=elementwise_loss.device)
    losses = elementwise_loss[valid]
    if losses.numel() == 0:
        return elementwise_loss.sum() * 0.0
    eta = torch.quantile(losses.detach(), float(alpha))
    return eta + F.relu(losses - eta).mean() / max(1.0 - float(alpha), 1e-6)

def compute_metric_sums(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    mask = torch.isfinite(target)
    pred = torch.where(mask, pred, torch.zeros_like(pred))
    target = torch.where(mask, target, torch.zeros_like(target))
    abs_err = torch.abs(pred - target)
    sq_err = (pred - target) ** 2
    denom = torch.clamp((torch.abs(pred) + torch.abs(target)) / 2.0, min=1e-6)
    return {
        "abs_sum": float(abs_err[mask].sum().item()),
        "sq_sum": float(sq_err[mask].sum().item()),
        "smape_sum": float((abs_err / denom)[mask].sum().item()),
        "target_abs_sum": float(torch.abs(target)[mask].sum().item()),
        "count": float(mask.sum().item()),
    }


def run_epoch(
    model: DSTSGCN,
    loader: DataLoader,
    static_adj: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    temporal_reg_weight: float,
    sparse_reg_weight: float,
    event_feature_idx: int | None,
    ood_profile: dict[str, float] | None,
    ood_enabled: bool,
    ood_clip: float,
    ood_use_volatility: bool,
    resilience_aux_enabled: bool,
    resilience_loss_weight: float,
    resilience_consistency_weight: float,
    resilience_target_mode: str,
    consistency_weight_current: float,
    cvar_enabled: bool,
    traffic_cvar_alpha: float,
    traffic_cvar_weight: float,
    resilience_cvar_alpha: float,
    resilience_cvar_weight: float,
    uncertainty_weighting: bool,
    traffic_raw_mean: float,
    traffic_raw_std: float,
    device: str,
) -> dict[str, float]:
    """Train one epoch with optional statistical resilience objectives."""
    model.train()
    totals = {
        "loss": 0.0,
        "temporal_reg": 0.0,
        "ood_intensity_mean": 0.0,
        "ood_scale_mean": 0.0,
        "traffic_loss": 0.0,
        "traffic_cvar_loss": 0.0,
        "resilience_loss": 0.0,
        "resilience_cvar_loss": 0.0,
        "resilience_consistency_loss": 0.0,
    }
    last_weights = (1.0, 1.0, 1.0)
    for batch_data in loader:
        x, y, resilience_target, resilience_baseline = unpack_batch(batch_data, device)
        event_signal = extract_event_signal(x, event_feature_idx)
        ood_intensity = (
            extract_ood_intensity(x, ood_profile, ood_clip, ood_use_volatility)
            if ood_enabled
            else None
        )
        optimizer.zero_grad()
        pred, aux = model(
            x,
            static_adj,
            event_signal=event_signal,
            ood_intensity=ood_intensity,
            return_aux=True,
        )
        temporal_reg = aux.get("temporal_reg", x.new_tensor(0.0))
        traffic_element = F.smooth_l1_loss(pred, y, reduction="none")
        traffic_mask = torch.isfinite(y)
        traffic_mean_loss = masked_mean(traffic_element, traffic_mask)
        traffic_cvar_loss = (
            empirical_cvar_loss(traffic_element, traffic_cvar_alpha, traffic_mask)
            if cvar_enabled
            else x.new_tensor(0.0)
        )
        traffic_loss = traffic_mean_loss + traffic_cvar_weight * traffic_cvar_loss
        resilience_mean_loss = x.new_tensor(0.0)
        resilience_cvar_loss = x.new_tensor(0.0)
        resilience_loss = x.new_tensor(0.0)
        consistency_loss = x.new_tensor(0.0)
        if resilience_aux_enabled and resilience_target is not None:
            resilience_pred = aux.get("resilience_pred")
            if resilience_pred is None:
                raise RuntimeError("resilience_aux is enabled but resilience_pred is missing.")
            if resilience_target_mode == "log_ratio":
                deficit_pred = F.relu(-resilience_pred)
                deficit_target = F.relu(-resilience_target)
                res_element = F.smooth_l1_loss(deficit_pred, deficit_target, reduction="none")
                res_mask = torch.isfinite(resilience_target)
                resilience_mean_loss = masked_mean(res_element, res_mask)
                resilience_cvar_loss = (
                    empirical_cvar_loss(res_element, resilience_cvar_alpha, res_mask)
                    if cvar_enabled
                    else x.new_tensor(0.0)
                )
                resilience_loss = resilience_mean_loss + resilience_cvar_weight * resilience_cvar_loss
                if consistency_weight_current > 0 and resilience_baseline is not None:
                    pred_raw = torch.clamp(
                        pred * float(traffic_raw_std) + float(traffic_raw_mean), min=0.0
                    )
                    pred_log_ratio = torch.log(
                        (pred_raw + 1e-6) / (torch.clamp(resilience_baseline, min=1e-6) + 1e-6)
                    )
                    consistency_loss = masked_huber_loss(loss_fn, resilience_pred, pred_log_ratio)
            else:
                resilience_mean_loss = masked_huber_loss(loss_fn, resilience_pred, resilience_target)
                resilience_loss = resilience_mean_loss
                if resilience_consistency_weight > 0 and resilience_baseline is not None:
                    pred_raw = pred * float(traffic_raw_std) + float(traffic_raw_mean)
                    pred_ratio = torch.clamp(
                        pred_raw / torch.clamp(resilience_baseline, min=1e-6), 0.0, 2.0
                    )
                    consistency_loss = masked_huber_loss(loss_fn, resilience_pred, pred_ratio.detach())

        if uncertainty_weighting:
            s_y = torch.clamp(model.traffic_log_var, -5.0, 5.0)
            s_r = torch.clamp(model.resilience_log_var, -5.0, 5.0)
            s_c = torch.clamp(model.consistency_log_var, -5.0, 5.0)
            loss = torch.exp(-s_y) * traffic_loss + s_y
            if resilience_aux_enabled:
                loss = loss + torch.exp(-s_r) * resilience_loss + s_r
                if consistency_weight_current > 0:
                    loss = loss + torch.exp(-s_c) * consistency_weight_current * consistency_loss + s_c
            last_weights = (
                float(torch.exp(-s_y).detach().item()),
                float(torch.exp(-s_r).detach().item()),
                float(torch.exp(-s_c).detach().item()),
            )
        else:
            loss = (
                traffic_loss
                + resilience_loss_weight * resilience_loss
                + consistency_weight_current * consistency_loss
            )
        loss = loss + temporal_reg_weight * temporal_reg
        if sparse_reg_weight > 0 and hasattr(model.graph_learner, "global_residual"):
            loss = loss + sparse_reg_weight * torch.abs(model.graph_learner.global_residual).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        batch_size = x.size(0)
        values = {
            "loss": loss,
            "temporal_reg": temporal_reg,
            "ood_intensity_mean": aux.get("ood_intensity_mean", x.new_tensor(0.0)),
            "ood_scale_mean": aux.get("ood_scale_mean", x.new_tensor(1.0)),
            "traffic_loss": traffic_mean_loss,
            "traffic_cvar_loss": traffic_cvar_loss,
            "resilience_loss": resilience_mean_loss,
            "resilience_cvar_loss": resilience_cvar_loss,
            "resilience_consistency_loss": consistency_loss,
        }
        for key, value in values.items():
            totals[key] += float(value.detach().item()) * batch_size
    n = max(len(loader.dataset), 1)
    result = {key: value / n for key, value in totals.items()}
    result.update(
        {
            "traffic_uncertainty_weight": last_weights[0],
            "resilience_uncertainty_weight": last_weights[1],
            "consistency_uncertainty_weight": last_weights[2],
        }
    )
    return result

@torch.no_grad()
def evaluate(
    model: DSTSGCN,
    loader: DataLoader,
    static_adj: torch.Tensor,
    event_feature_idx: int | None,
    ood_profile: dict[str, float] | None,
    ood_enabled: bool,
    ood_clip: float,
    ood_use_volatility: bool,
    resilience_aux_enabled: bool,
    traffic_raw_mean: float,
    traffic_raw_std: float,
    device: str,
) -> dict[str, object]:
    model.eval()
    totals = {"abs_sum": 0.0, "sq_sum": 0.0, "smape_sum": 0.0, "target_abs_sum": 0.0, "count": 0.0}
    resilience_abs_sum = 0.0
    resilience_count = 0.0
    horizon_abs_sum = None
    horizon_count = None
    all_abs_errors: list[torch.Tensor] = []
    for batch_data in loader:
        x, y, resilience_target, _ = unpack_batch(batch_data, device)
        event_signal = extract_event_signal(x, event_feature_idx)
        ood_intensity = (
            extract_ood_intensity(x, ood_profile, ood_clip, ood_use_volatility)
            if ood_enabled
            else None
        )
        if resilience_aux_enabled and resilience_target is not None:
            pred, aux = model(
                x,
                static_adj,
                event_signal=event_signal,
                ood_intensity=ood_intensity,
                return_aux=True,
            )
            resilience_pred = aux.get("resilience_pred")
            if resilience_pred is not None:
                r_mask = torch.isfinite(resilience_target)
                r_abs = torch.where(
                    r_mask,
                    torch.abs(resilience_pred - resilience_target),
                    torch.zeros_like(resilience_target),
                )
                resilience_abs_sum += float(r_abs.sum().item())
                resilience_count += float(r_mask.sum().item())
        else:
            pred = model(x, static_adj, event_signal=event_signal, ood_intensity=ood_intensity)
        pred_metric = pred * traffic_raw_std + traffic_raw_mean
        y_metric = y * traffic_raw_std + traffic_raw_mean
        batch_metrics = compute_metric_sums(pred_metric, y_metric)
        for key in totals:
            totals[key] += batch_metrics[key]
        mask = torch.isfinite(y_metric)
        abs_err = torch.where(mask, torch.abs(pred_metric - y_metric), torch.zeros_like(y_metric))
        if bool(mask.any()):
            all_abs_errors.append(abs_err[mask].detach().cpu())
        batch_horizon_abs = abs_err.sum(dim=(0, 2, 3)).detach().cpu()
        batch_horizon_count = mask.sum(dim=(0, 2, 3)).detach().cpu()
        if horizon_abs_sum is None:
            horizon_abs_sum = batch_horizon_abs
            horizon_count = batch_horizon_count
        else:
            horizon_abs_sum += batch_horizon_abs
            horizon_count += batch_horizon_count
    count = max(totals["count"], 1.0)
    horizon_mae = (
        (horizon_abs_sum / torch.clamp(horizon_count, min=1)).tolist()
        if horizon_abs_sum is not None and horizon_count is not None
        else []
    )
    if all_abs_errors:
        flat_errors = torch.cat(all_abs_errors)
        q90 = torch.quantile(flat_errors, 0.90)
        tail = flat_errors[flat_errors >= q90]
        tail_mae_q90 = float(tail.mean().item()) if tail.numel() else float("nan")
    else:
        tail_mae_q90 = float("nan")
    return {
        "mae": totals["abs_sum"] / count,
        "rmse": (totals["sq_sum"] / count) ** 0.5,
        "smape": totals["smape_sum"] / count * 100,
        "wape": totals["abs_sum"] / max(totals["target_abs_sum"], 1e-6) * 100,
        "horizon_mae": horizon_mae,
        "tail_mae_q90": tail_mae_q90,
        "resilience_mae": resilience_abs_sum / max(resilience_count, 1.0)
        if resilience_count > 0
        else float("nan"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--dataset-name", default="")
    parser.add_argument("--model-name", default="")
    parser.add_argument("--canonical-model-name", default="")
    parser.add_argument("--experiment-stage", default="")
    parser.add_argument("--time-col", default="Time")
    parser.add_argument("--value-suffix", default="_volume")
    parser.add_argument("--exclude-cols", default="ID,id")
    parser.add_argument("--extra-feature-cols", default="")
    parser.add_argument("--node-feature-suffixes", default="")
    parser.add_argument("--add-time-features", action="store_true")
    parser.add_argument("--history", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-blocks", type=int, default=2)
    parser.add_argument("--graph-learner-type", choices=["lmln", "attention"], default="lmln")
    parser.add_argument("--fusion-mode", choices=["fusion", "static", "dynamic"], default="fusion")
    parser.add_argument("--fusion-type", choices=["quality", "quality_ood_scaled"], default="quality")
    parser.add_argument("--dynamic-top-k", type=int, default=3)
    parser.add_argument("--quality-gate-bias", type=float, default=-1.0)
    parser.add_argument("--num-diffusion-steps", type=int, default=2)
    parser.add_argument("--matrix-hidden-dim", type=int, default=256)
    parser.add_argument("--event-col", default="")
    parser.add_argument("--ood-gamma-max", type=float, default=0.35)
    parser.add_argument("--ood-clip", type=float, default=5.0)
    parser.add_argument("--ood-use-volatility", action="store_true")
    parser.add_argument("--resilience-aux", action="store_true")
    parser.add_argument("--resilience-loss-weight", type=float, default=0.3)
    parser.add_argument("--resilience-consistency-weight", type=float, default=0.1)
    parser.add_argument("--resilience-time-of-day-bins", type=int, default=288)
    parser.add_argument("--resilience-target-mode", choices=["ratio", "log_ratio"], default="ratio")
    parser.add_argument("--resilience-baseline-mode", choices=["mean_tod", "robust_shrunk"], default="mean_tod")
    parser.add_argument("--resilience-shrinkage-candidates", default="0,1,3,7,14,28")
    parser.add_argument("--resilience-log-ratio-mad-clip", type=float, default=6.0)
    parser.add_argument("--resilience-consistency-warmup-epochs", type=int, default=5)
    parser.add_argument("--traffic-cvar-alpha", type=float, default=0.90)
    parser.add_argument("--traffic-cvar-weight", type=float, default=0.20)
    parser.add_argument("--resilience-cvar-alpha", type=float, default=0.90)
    parser.add_argument("--resilience-cvar-weight", type=float, default=0.20)
    parser.add_argument("--disable-cvar", action="store_true")
    parser.add_argument("--resilience-uncertainty-weighting", action="store_true")
    parser.add_argument("--disable-resilience-uncertainty-weighting", action="store_true")
    parser.add_argument("--adj-source", choices=["corr", "road", "granger", "mix"], default="corr")
    parser.add_argument("--adj-path", default="")
    parser.add_argument("--adj-from-col", default="from_node")
    parser.add_argument("--adj-to-col", default="to_node")
    parser.add_argument("--adj-weight-col", default="")
    parser.add_argument("--adj-link-id-col", default="link_id")
    parser.add_argument("--adj-undirected", action="store_true")
    parser.add_argument("--adj-geo-wkt-col", default="geo_wkt")
    parser.add_argument("--road-knn", type=int, default=3)
    parser.add_argument("--corr-threshold", type=float, default=0.2)
    parser.add_argument("--corr-method", choices=["pearson", "shrinkage", "partial", "oas_partial"], default="pearson")
    parser.add_argument("--corr-shrinkage-lambda", type=float, default=0.1)
    parser.add_argument("--auto-corr-shrinkage", action="store_true")
    parser.add_argument("--road-weight", type=float, default=0.5)
    parser.add_argument("--granger-lag", type=int, default=3)
    parser.add_argument("--granger-p-threshold", type=float, default=0.05)
    parser.add_argument("--granger-cache-path", default="")
    parser.add_argument("--static-adj-train-only", action="store_true")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument("--lr-patience", type=int, default=5)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--sparse-reg-weight", type=float, default=1e-4)
    parser.add_argument("--temporal-reg-weight", type=float, default=1e-3)
    parser.add_argument("--max-nodes", type=int, default=None)
    parser.add_argument("--output-dir", default=r"D:\TrafficGNN\outputs")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-mode", choices=["chronological", "event_aware", "event_aligned"], default="chronological")
    parser.add_argument("--explicit-event-time", default="")
    parser.add_argument("--event-test-prehistory-steps", type=int, default=None)
    parser.add_argument("--event-window-threshold", type=float, default=0.0)
    args = parser.parse_args()

    if args.fusion_type == "quality_ood_scaled" and args.fusion_mode != "fusion":
        raise ValueError("quality_ood_scaled requires --fusion-mode fusion.")

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data, node_names, feature_names = load_wide_traffic_csv(
        args.csv,
        value_suffix=parse_optional_suffix(args.value_suffix),
        time_col=args.time_col,
        max_nodes=args.max_nodes,
        extra_feature_cols=parse_csv_list(args.extra_feature_cols),
        node_feature_suffixes=parse_csv_list(args.node_feature_suffixes),
        add_time_features=args.add_time_features,
        exclude_cols=parse_csv_list(args.exclude_cols),
        return_feature_names=True,
    )
    raw_data = data.copy()
    event_col = parse_optional_name(args.event_col)
    event_feature_idx = resolve_feature_index(event_col, feature_names)
    raw_event_series = data[:, 0, event_feature_idx].copy() if event_feature_idx is not None else None

    train_time_end = int(len(data) * 0.6)
    scaler = SplitScaler(num_traffic_features=1)
    scaler.fit(raw_data[:train_time_end])
    data = scaler.transform(raw_data)

    adj_data = data[:train_time_end] if args.static_adj_train_only else data
    static_np = build_static_adjacency(
        adj_data,
        node_names,
        source=args.adj_source,
        corr_threshold=args.corr_threshold,
        corr_method=args.corr_method,
        corr_shrinkage_lambda=args.corr_shrinkage_lambda,
        auto_corr_shrinkage=args.auto_corr_shrinkage,
        adj_path=args.adj_path or None,
        road_weight=args.road_weight,
        granger_lag=args.granger_lag,
        granger_p_threshold=args.granger_p_threshold,
        granger_cache_path=args.granger_cache_path or None,
        from_col=args.adj_from_col,
        to_col=args.adj_to_col,
        weight_col=args.adj_weight_col or None,
        link_id_col=args.adj_link_id_col or None,
        directed=not args.adj_undirected,
        road_knn=args.road_knn,
        geo_wkt_col=args.adj_geo_wkt_col,
    )
    static_adj = torch.tensor(static_np, dtype=torch.float32, device=args.device)

    resilience_aux_enabled = bool(args.resilience_aux)
    resilience_report: dict[str, object] = {"enabled": False}
    if resilience_aux_enabled:
        shrinkage_candidates = tuple(
            float(value) for value in parse_csv_list(args.resilience_shrinkage_candidates)
        )
        resilience_targets, resilience_baselines, resilience_report = build_resilience_supervision(
            raw_data,
            train_time_end=train_time_end,
            history=args.history,
            horizon=args.horizon,
            time_of_day_bins=args.resilience_time_of_day_bins,
            baseline_mode=args.resilience_baseline_mode,
            shrinkage_candidates=shrinkage_candidates,
            log_ratio_mad_clip=args.resilience_log_ratio_mad_clip,
        )
        dataset = ResilienceAuxWindowDataset(
            data,
            resilience_targets=resilience_targets,
            resilience_baselines=resilience_baselines,
            history=args.history,
            horizon=args.horizon,
            target_dim=0,
        )
    else:
        dataset = TrafficWindowDataset(data, history=args.history, horizon=args.horizon, target_dim=0)
    explicit_event_index = (
        resolve_time_index(args.csv, args.time_col, args.explicit_event_time)
        if args.explicit_event_time else None
    )
    train_set, val_set, test_set, split_info = split_dataset(
        dataset,
        split_mode=args.split_mode,
        event_series=raw_event_series,
        event_window_threshold=args.event_window_threshold,
        explicit_event_index=explicit_event_index,
        event_test_prehistory_steps=args.event_test_prehistory_steps,
    )
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    ood_enabled = args.fusion_mode == "fusion" and args.fusion_type == "quality_ood_scaled"
    ood_profile = (
        estimate_train_ood_profile(train_set, args.ood_clip, args.ood_use_volatility)
        if ood_enabled
        else {"enabled": False}
    )

    model = DSTSGCN(
        num_nodes=len(node_names),
        input_dim=data.shape[-1],
        output_dim=1,
        horizon=args.horizon,
        hidden_dim=args.hidden_dim,
        num_blocks=args.num_blocks,
        graph_learner_type=args.graph_learner_type,
        fusion_mode=args.fusion_mode,
        fusion_type=args.fusion_type,
        dynamic_top_k=args.dynamic_top_k if args.dynamic_top_k > 0 else None,
        quality_gate_bias=args.quality_gate_bias,
        matrix_hidden_dim=args.matrix_hidden_dim,
        num_diffusion_steps=args.num_diffusion_steps,
        event_dim=1 if event_feature_idx is not None else 0,
        ood_gamma_max=args.ood_gamma_max,
        resilience_aux=resilience_aux_enabled,
        resilience_target_mode=args.resilience_target_mode,
        resilience_uncertainty_weighting=(
            args.resilience_uncertainty_weighting
            and not args.disable_resilience_uncertainty_weighting
        ),
    ).to(args.device)
    trainable_params = count_trainable_parameters(model)
    effective_graph_learner_type = getattr(model, "graph_learner_type", args.graph_learner_type)
    canonical_model_name = (
        args.canonical_model_name
        or args.model_name
        or (
            "quality_resilience_aux_v1"
            if resilience_aux_enabled
            else "quality_ood_scaled_v1"
            if args.fusion_type == "quality_ood_scaled"
            else ("stsgcn" if args.fusion_mode == "static" else "dgcn" if args.fusion_mode == "dynamic" else "quality_v2")
        )
    )
    dataset_name = args.dataset_name or Path(args.csv).stem

    config_payload = vars(args).copy()
    config_payload.update(
        {
            "dataset": dataset_name,
            "model_name": args.model_name or canonical_model_name,
            "canonical_model_name": canonical_model_name,
            "experiment_stage": args.experiment_stage,
            "candidate_main_model": bool(args.experiment_stage.lower() == "a3"),
            "consistency_enabled": bool(resilience_aux_enabled and args.resilience_consistency_weight > 0),
            "requested_graph_learner_type": args.graph_learner_type,
            "graph_learner_type": effective_graph_learner_type,
            "split_info": split_info,
            "feature_names": feature_names,
            "node_count": len(node_names),
            "trainable_parameters": trainable_params,
            "effective_graph_learner_type": effective_graph_learner_type,
            "requested_fusion_type": args.fusion_type,
            "effective_fusion_type": args.fusion_type,
            "ood_enabled": ood_enabled,
            "ood_profile": ood_profile,
            "resilience_aux_enabled": resilience_aux_enabled,
            "resilience_loss_weight": args.resilience_loss_weight if resilience_aux_enabled else 0.0,
            "resilience_consistency_weight": args.resilience_consistency_weight if resilience_aux_enabled else 0.0,
            "resilience_time_of_day_bins": args.resilience_time_of_day_bins,
            "resilience_target_mode": args.resilience_target_mode,
            "resilience_baseline_mode": args.resilience_baseline_mode,
            "resilience_uncertainty_weighting": (
                args.resilience_uncertainty_weighting
                and not args.disable_resilience_uncertainty_weighting
            ),
            "cvar_enabled": not args.disable_cvar,
            "traffic_cvar_alpha": args.traffic_cvar_alpha,
            "traffic_cvar_weight": args.traffic_cvar_weight,
            "resilience_cvar_alpha": args.resilience_cvar_alpha,
            "resilience_cvar_weight": args.resilience_cvar_weight,
            "resilience_consistency_warmup_epochs": args.resilience_consistency_warmup_epochs,
            "resilience_supervision": resilience_report,
        }
    )
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(config_payload, f, ensure_ascii=False, indent=2)

    print(f"Loaded data: time={data.shape[0]}, nodes={data.shape[1]}, features={data.shape[2]}")
    print(
        f"Split mode: {args.split_mode} | train={split_info['train_windows']} "
        f"val={split_info['val_windows']} test={split_info['test_windows']}"
    )
    if split_info.get("warning"):
        print(f"[WARNING] {split_info['warning']}")
    if split_info.get("event_window_counts") is not None:
        print(f"Event window counts: {split_info['event_window_counts']}")
    print(f"Device: {args.device}")
    print(f"Dataset: {dataset_name}")
    print(f"Model: {canonical_model_name}")
    print(f"Graph learner: {effective_graph_learner_type}")
    print(f"Fusion mode: {args.fusion_mode}")
    print(f"Fusion type: {args.fusion_type}")
    print(f"Trainable parameters: {trainable_params}")
    if ood_enabled:
        print("[OODScaled] enabled=yes")
        print(
            "[OODScaled] "
            f"gamma_max={args.ood_gamma_max:g} ood_clip={args.ood_clip:g} "
            f"use_volatility={'yes' if args.ood_use_volatility else 'no'}"
        )
        print(
            "[OODScaled] "
            f"combined_q95={ood_profile.get('combined_ood_q95', 0.0):.6f} "
            f"combined_q99={ood_profile.get('combined_ood_q99', 0.0):.6f}"
        )
    if resilience_aux_enabled:
        print("[ResilienceAux] enabled=yes")
        print(
            "[ResilienceAux] "
            f"loss_weight={args.resilience_loss_weight:g} "
            f"consistency_weight={args.resilience_consistency_weight:g} "
            f"time_of_day_bins={args.resilience_time_of_day_bins}"
        )
        print(
            "[ResilienceAux] "
            f"target_mean={resilience_report.get('target_mean', 0.0):.6f} "
            f"target_std={resilience_report.get('target_std', 0.0):.6f} "
            f"baseline_mean={resilience_report.get('baseline_mean', 0.0):.6f}"
        )
    print(f"Dynamic top-k: {args.dynamic_top_k if args.dynamic_top_k > 0 else 'disabled'}")
    print(f"Diffusion steps: {args.num_diffusion_steps}")
    print(f"Temporal regularization weight: {args.temporal_reg_weight:g}")
    print("Event column: " + (f"{event_col} (feature index {event_feature_idx})" if event_col else "disabled"))
    print(f"Adjacency source: {args.adj_source}")
    print(f"Correlation method: {args.corr_method} (threshold={args.corr_threshold:g})")
    print(f"Static adjacency train only: {'yes' if args.static_adj_train_only else 'no'}")
    print(f"Extra features: {parse_csv_list(args.extra_feature_cols) or 'none'}")
    print(f"Node feature suffixes: {parse_csv_list(args.node_feature_suffixes) or 'none'}")
    print(f"Time features: {'enabled' if args.add_time_features else 'disabled'}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.lr_factor,
        patience=args.lr_patience,
        min_lr=args.min_lr,
    )
    loss_fn = nn.HuberLoss()
    traffic_raw_mean = float(np.asarray(scaler.traffic_scaler.mean).reshape(-1)[0])
    traffic_raw_std = float(np.asarray(scaler.traffic_scaler.std).reshape(-1)[0])
    best_val = float("inf")
    best_path = output_dir / "best_dstsgcn.pt"

    for epoch in range(1, args.epochs + 1):
        train_stats = run_epoch(
            model,
            train_loader,
            static_adj,
            optimizer,
            loss_fn,
            args.temporal_reg_weight,
            args.sparse_reg_weight,
            event_feature_idx,
            ood_profile,
            ood_enabled,
            args.ood_clip,
            args.ood_use_volatility,
            resilience_aux_enabled,
            args.resilience_loss_weight,
            args.resilience_consistency_weight,
            args.resilience_target_mode,
            args.resilience_consistency_weight
            * min(1.0, epoch / max(args.resilience_consistency_warmup_epochs, 1)),
            not args.disable_cvar,
            args.traffic_cvar_alpha,
            args.traffic_cvar_weight,
            args.resilience_cvar_alpha,
            args.resilience_cvar_weight,
            args.resilience_uncertainty_weighting
            and not args.disable_resilience_uncertainty_weighting,
            traffic_raw_mean,
            traffic_raw_std,
            args.device,
        )
        val_metrics = evaluate(
            model,
            val_loader,
            static_adj,
            event_feature_idx,
            ood_profile,
            ood_enabled,
            args.ood_clip,
            args.ood_use_volatility,
            resilience_aux_enabled,
            traffic_raw_mean,
            traffic_raw_std,
            args.device,
        )
        scheduler.step(float(val_metrics["mae"]))
        current_lr = optimizer.param_groups[0]["lr"]
        log_line = (
            f"Epoch {epoch:03d} | train_loss={train_stats['loss']:.4f} | "
            f"temporal_reg={train_stats['temporal_reg']:.6f} | "
            f"val_mae={val_metrics['mae']:.4f} | val_rmse={val_metrics['rmse']:.4f} | "
            f"val_smape={val_metrics['smape']:.2f}% | val_wape={val_metrics['wape']:.2f}% | "
        )
        if ood_enabled:
            log_line += (
                f"ood_intensity_mean={train_stats['ood_intensity_mean']:.4f} | "
                f"ood_scale_mean={train_stats['ood_scale_mean']:.4f} | "
            )
        if resilience_aux_enabled:
            log_line += (
                f"traffic_loss={train_stats['traffic_loss']:.4f} | "
                f"res_loss={train_stats['resilience_loss']:.4f} | "
                f"res_consistency={train_stats['resilience_consistency_loss']:.4f} | "
                f"traffic_cvar={train_stats['traffic_cvar_loss']:.4f} | "
                f"res_cvar={train_stats['resilience_cvar_loss']:.4f} | "
                f"val_res_mae={val_metrics['resilience_mae']:.4f} | "
                f"val_tail_q90={val_metrics['tail_mae_q90']:.4f} | "
                f"uw=({train_stats['traffic_uncertainty_weight']:.3f},"
                f"{train_stats['resilience_uncertainty_weight']:.3f},"
                f"{train_stats['consistency_uncertainty_weight']:.3f}) | "
            )
        log_line += f"lr={current_lr:.2e}"
        print(log_line)
        if float(val_metrics["mae"]) < best_val:
            best_val = float(val_metrics["mae"])
            torch.save(model.state_dict(), best_path)

    model.load_state_dict(torch.load(best_path, map_location=args.device))
    test_metrics = evaluate(
        model,
        test_loader,
        static_adj,
        event_feature_idx,
        ood_profile,
        ood_enabled,
        args.ood_clip,
        args.ood_use_volatility,
        resilience_aux_enabled,
        traffic_raw_mean,
        traffic_raw_std,
        args.device,
    )
    horizon_mae = test_metrics["horizon_mae"]
    display_indices = [idx for idx in (0, 2, 5, 11) if idx < len(horizon_mae)]
    horizon_display = " ".join(f"h{idx + 1}={horizon_mae[idx]:.4f}" for idx in display_indices)
    print(
        f"Test MAE={test_metrics['mae']:.4f}, "
        f"RMSE={test_metrics['rmse']:.4f}, "
        f"SMAPE={test_metrics['smape']:.2f}%, "
        f"WAPE={test_metrics['wape']:.2f}%"
        + (f" | {horizon_display}" if horizon_display else "")
    )


if __name__ == "__main__":
    main()


