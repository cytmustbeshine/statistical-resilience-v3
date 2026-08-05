"""Statistical seasonal-residual spatial-temporal identity forecaster."""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset


def fit_shrunk_seasonal_baseline(
    physical_values: np.ndarray,
    timestamps: np.ndarray,
    train_time_end_exclusive: int,
    shrinkage: float = 7.0,
) -> tuple[np.ndarray, dict[str, object]]:
    """Fit node x time-slot x weekday/weekend medians on training rows only."""
    values = np.asarray(physical_values, dtype=float)
    if values.ndim != 3 or values.shape[-1] != 1:
        raise ValueError("physical_values must have shape [time,nodes,1]")
    end = int(train_time_end_exclusive)
    if end <= 0 or end > len(values):
        raise ValueError("invalid train_time_end_exclusive")
    if shrinkage < 0:
        raise ValueError("shrinkage must be non-negative")
    time_index = pd.DatetimeIndex(pd.to_datetime(np.asarray(timestamps)))
    if len(time_index) != len(values) or time_index.isna().any():
        raise ValueError("timestamps must be finite and aligned")
    slots = np.asarray((time_index.hour * 60 + time_index.minute) // 5, dtype=int)
    day_types = np.asarray(time_index.dayofweek >= 5, dtype=int)
    train = values[:end, :, 0]
    node_median = np.nanmedian(train, axis=0)
    global_median = float(np.nanmedian(train))
    node_median = np.where(np.isfinite(node_median), node_median, global_median)
    slot_median = np.empty((288, train.shape[1]), dtype=float)
    stratum = np.empty((288, 2, train.shape[1]), dtype=float)
    counts = np.zeros((288, 2, train.shape[1]), dtype=float)
    for slot in range(288):
        slot_mask = slots[:end] == slot
        if slot_mask.any():
            current_slot = np.nanmedian(train[slot_mask], axis=0)
        else:
            current_slot = node_median.copy()
        current_slot = np.where(np.isfinite(current_slot), current_slot, node_median)
        slot_median[slot] = current_slot
        for day_type in (0, 1):
            mask = slot_mask & (day_types[:end] == day_type)
            finite_count = np.isfinite(train[mask]).sum(axis=0) if mask.any() else np.zeros(train.shape[1])
            if mask.any():
                location = np.nanmedian(train[mask], axis=0)
            else:
                location = current_slot.copy()
            location = np.where(np.isfinite(location), location, current_slot)
            weight = finite_count / (finite_count + float(shrinkage))
            stratum[slot, day_type] = weight * location + (1.0 - weight) * current_slot
            counts[slot, day_type] = finite_count
    baseline = stratum[slots, day_types][..., None]
    digest = hashlib.sha256(np.ascontiguousarray(baseline).view(np.uint8)).hexdigest()
    metadata = {
        "method": "node_slot_daytype_median_shrinkage",
        "train_time_end_exclusive": end,
        "shrinkage": float(shrinkage),
        "baseline_sha256": digest,
        "minimum_stratum_count": float(counts.min()),
        "maximum_stratum_count": float(counts.max()),
    }
    return baseline, metadata


class StatisticalIdentityWindowDataset(Dataset):
    """Traffic windows with frozen seasonal targets and calendar identities."""

    def __init__(
        self,
        scaled_values: np.ndarray,
        scaled_seasonal_baseline: np.ndarray,
        timestamps: np.ndarray,
        history: int,
        horizon: int,
    ) -> None:
        self.values = np.asarray(scaled_values, dtype=np.float32)
        self.baseline = np.asarray(scaled_seasonal_baseline, dtype=np.float32)
        self.timestamps = pd.DatetimeIndex(pd.to_datetime(np.asarray(timestamps)))
        self.history = int(history)
        self.horizon = int(horizon)
        if self.values.shape != self.baseline.shape:
            raise ValueError("values and seasonal baseline shapes differ")

    def __len__(self) -> int:
        return max(0, len(self.values) - self.history - self.horizon + 1)

    def __getitem__(self, index: int):
        target_start = index + self.history
        x = self.values[index:target_start]
        y = self.values[target_start:target_start + self.horizon]
        baseline = self.baseline[target_start:target_start + self.horizon]
        origin = self.timestamps[target_start]
        slot = int((origin.hour * 60 + origin.minute) // 5)
        day_of_week = int(origin.dayofweek)
        return (
            torch.from_numpy(x),
            torch.from_numpy(y),
            torch.from_numpy(baseline),
            torch.tensor(slot, dtype=torch.long),
            torch.tensor(day_of_week, dtype=torch.long),
        )


class ResidualMLPBlock(nn.Module):
    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(width, width),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.network(x))


class StatisticalIdentityResidualForecaster(nn.Module):
    """STID-style identity model with statistical residual anchoring."""

    def __init__(
        self,
        num_nodes: int,
        history: int = 12,
        horizon: int = 12,
        embedding_dim: int = 32,
        hidden_width: int = 256,
        num_blocks: int = 3,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.history = int(history)
        self.horizon = int(horizon)
        self.history_projection = nn.Linear(self.history, embedding_dim)
        self.node_embedding = nn.Parameter(torch.empty(self.num_nodes, embedding_dim))
        self.time_of_day_embedding = nn.Embedding(288, embedding_dim)
        self.day_of_week_embedding = nn.Embedding(7, embedding_dim)
        self.robust_state_projection = nn.Linear(3, embedding_dim)
        self.input_projection = nn.Linear(embedding_dim * 6, hidden_width)
        self.blocks = nn.ModuleList(
            [ResidualMLPBlock(hidden_width, dropout) for _ in range(num_blocks)]
        )
        self.output_projection = nn.Linear(hidden_width, self.horizon)
        nn.init.xavier_uniform_(self.node_embedding)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def _robust_state(self, x: torch.Tensor) -> torch.Tensor:
        history = x[..., 0].permute(0, 2, 1)
        median = history.median(dim=-1).values
        mad = torch.abs(history - median.unsqueeze(-1)).median(dim=-1).values.clamp_min(1e-3)
        level = (history[..., -1] - median) / mad
        trend = (history[..., -1] - history[..., 0]) / max(self.history - 1, 1)
        volatility = torch.abs(history[..., 1:] - history[..., :-1]).median(dim=-1).values
        return torch.stack((level, trend, volatility), dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        static_adj: torch.Tensor,
        seasonal_baseline: torch.Tensor,
        origin_slot: torch.Tensor,
        origin_day_of_week: torch.Tensor,
        return_aux: bool = False,
    ):
        batch_size, _, num_nodes, _ = x.shape
        if num_nodes != self.num_nodes:
            raise ValueError(f"Expected {self.num_nodes} nodes, received {num_nodes}")
        history = x[..., 0].permute(0, 2, 1)
        history_embedding = self.history_projection(history)
        graph_context = torch.einsum("nm,bmd->bnd", static_adj, history_embedding)
        node_identity = self.node_embedding.unsqueeze(0).expand(batch_size, -1, -1)
        time_identity = self.time_of_day_embedding(origin_slot).unsqueeze(1).expand(-1, num_nodes, -1)
        week_identity = self.day_of_week_embedding(origin_day_of_week).unsqueeze(1).expand(-1, num_nodes, -1)
        robust_state = self.robust_state_projection(self._robust_state(x))
        hidden = torch.cat(
            (
                history_embedding,
                graph_context,
                node_identity,
                time_identity,
                week_identity,
                robust_state,
            ),
            dim=-1,
        )
        hidden = self.input_projection(hidden)
        for block in self.blocks:
            hidden = block(hidden)
        residual = self.output_projection(hidden).permute(0, 2, 1).unsqueeze(-1)
        prediction = seasonal_baseline + residual
        if return_aux:
            return prediction, {
                "residual_prediction": residual,
                "history_embedding": history_embedding,
                "graph_context": graph_context,
            }
        return prediction
