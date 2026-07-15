"""Data utilities for traffic forecasting experiments."""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy import stats
import torch
from torch.utils.data import Dataset


def resolve_time_index(path: str, time_col: str, target_time: str) -> int:
    """Resolve a timestamp to its nearest row index after chronological sorting."""
    frame = read_csv_with_fallback(path)
    if time_col not in frame.columns:
        raise ValueError(f"Time column {time_col!r} is not present in {path}.")
    parsed = pd.to_datetime(frame[time_col], errors="coerce")
    if float(parsed.notna().mean()) < 0.95:
        raise ValueError(f"Cannot reliably parse {time_col!r} in {path} as datetimes.")
    ordered = parsed.sort_values(kind="stable").reset_index(drop=True)
    target = pd.Timestamp(target_time)
    deltas = (ordered - target).abs()
    if deltas.isna().all():
        raise ValueError(f"No valid timestamps are available in {path}.")
    return int(deltas.idxmin())


def split_traffic_window_indices(
    num_windows: int,
    history: int,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    split_mode: str = "chronological",
    event_series: np.ndarray | None = None,
    event_window_threshold: float = 0.0,
    explicit_event_index: int | None = None,
    event_test_prehistory_steps: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    """Build auditable chronological, event-aware, or event-aligned splits.

    ``event_aligned`` keeps the original train prefix unchanged and places the
    validation/test boundary before the explicit event. The first test target
    starts one full history window before the event by default, so both the
    pre-disruption context and the complete disruption are forecast.
    """
    if split_mode not in {"chronological", "event_aware", "event_aligned"}:
        raise ValueError(f"Unknown split mode: {split_mode}")
    indices = np.arange(num_windows, dtype=int)
    train_end = int(num_windows * train_ratio)
    val_end = int(num_windows * (train_ratio + val_ratio))
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]
    warning = ""

    if split_mode == "event_aligned":
        if explicit_event_index is None:
            warning = "event-aligned split requested without an explicit event index; chronological split retained"
        else:
            prehistory = history if event_test_prehistory_steps is None else max(0, int(event_test_prehistory_steps))
            test_start = int(explicit_event_index) - history - prehistory
            test_start = max(train_end + 1, min(test_start, num_windows - 1))
            val_indices = indices[train_end:test_start]
            test_indices = indices[test_start:]
            if val_indices.size == 0:
                raise ValueError("event-aligned split leaves no validation windows")

    elif split_mode == "event_aware" and event_series is not None:
        event_values = np.asarray(
            [event_series[min(i + history - 1, len(event_series) - 1)] for i in range(num_windows)],
            dtype=np.float64,
        )
        post_train = indices[train_end:]
        post_event = post_train[event_values[post_train] > event_window_threshold]
        if post_event.size:
            selected = post_event[-min(post_event.size, len(test_indices)):]
            mask = np.zeros(num_windows, dtype=bool)
            mask[selected] = True
            remaining = len(test_indices) - len(selected)
            if remaining > 0:
                candidates = post_train[~mask[post_train]]
                mask[candidates[-remaining:]] = True
            proposed_test = np.sort(post_train[mask[post_train]])
            remaining_post = post_train[~mask[post_train]]
            if remaining_post.size >= len(val_indices):
                test_indices = proposed_test
                val_indices = remaining_post[:len(val_indices)]
            else:
                warning = "event-aware split could not preserve validation size; chronological split retained"
        else:
            warning = "no post-train event windows; chronological split retained"
    elif split_mode == "event_aware":
        warning = "event-aware split requested without event data; chronological split retained"

    event_counts = None
    if event_series is not None:
        event_values = np.asarray(
            [event_series[min(i + history - 1, len(event_series) - 1)] for i in range(num_windows)],
            dtype=np.float64,
        )
        event_counts = {
            name: int(np.sum(event_values[split] > event_window_threshold))
            for name, split in (("train", train_indices), ("val", val_indices), ("test", test_indices))
        }
    first_test_target = int(test_indices[0] + history) if test_indices.size else None
    info = {
        "split_mode": split_mode,
        "num_windows": int(num_windows),
        "train_windows": int(train_indices.size),
        "val_windows": int(val_indices.size),
        "test_windows": int(test_indices.size),
        "train_start_window": int(train_indices[0]) if train_indices.size else None,
        "train_end_window_exclusive": int(train_indices[-1] + 1) if train_indices.size else None,
        "val_start_window": int(val_indices[0]) if val_indices.size else None,
        "val_end_window_exclusive": int(val_indices[-1] + 1) if val_indices.size else None,
        "test_start_window": int(test_indices[0]) if test_indices.size else None,
        "first_test_target_index": first_test_target,
        "explicit_event_index": int(explicit_event_index) if explicit_event_index is not None else None,
        "event_precoverage_steps": (
            int(explicit_event_index - first_test_target)
            if explicit_event_index is not None and first_test_target is not None else None
        ),
        "event_window_threshold": float(event_window_threshold),
        "event_window_counts": event_counts,
        "warning": warning,
    }
    return train_indices, val_indices, test_indices, info


def split_traffic_window_indices_strict(
    num_timesteps: int,
    history: int,
    horizon: int,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    """Build target-disjoint chronological forecasting-window splits.

    Raw-time boundaries define the split membership of the complete target
    horizon. A validation or test input may use earlier historical context, but
    every target timestamp belongs to exactly one split. Windows whose targets
    cross a raw-time boundary are omitted.

    Args:
        num_timesteps: Number of rows in the chronological raw series.
        history: Number of historical input steps.
        horizon: Number of future target steps.
        train_ratio: Exclusive raw-time training boundary ratio.
        val_ratio: Validation share following the training boundary.

    Returns:
        Train, validation and test window-start indices plus boundary metadata.
    """
    if num_timesteps < 0 or history <= 0 or horizon <= 0:
        raise ValueError("num_timesteps must be nonnegative; history and horizon must be positive")
    if not (0.0 < train_ratio < 1.0 and 0.0 <= val_ratio < 1.0):
        raise ValueError("invalid train/validation ratios")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be less than 1")
    num_windows = max(0, num_timesteps - history - horizon + 1)
    windows = np.arange(num_windows, dtype=int)
    target_start = windows + history
    target_end = target_start + horizon - 1
    train_time_end = int(num_timesteps * train_ratio)
    val_time_end = int(num_timesteps * (train_ratio + val_ratio))
    train = windows[target_end < train_time_end]
    val = windows[(target_start >= train_time_end) & (target_end < val_time_end)]
    test = windows[(target_start >= val_time_end) & (target_end < num_timesteps)]
    info = {
        "split_mode": "chronological_strict",
        "num_timesteps": int(num_timesteps),
        "num_windows": int(num_windows),
        "train_time_end_exclusive": int(train_time_end),
        "val_time_end_exclusive": int(val_time_end),
        "train_windows": int(len(train)),
        "val_windows": int(len(val)),
        "test_windows": int(len(test)),
        "omitted_boundary_windows": int(num_windows - len(train) - len(val) - len(test)),
        "target_disjoint": True,
    }
    return train, val, test, info

class StandardScaler:
    """Simple z-score scaler for traffic tensors."""

    def __init__(self) -> None:
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def fit(self, data: np.ndarray) -> None:
        self.mean = data.mean(axis=(0, 1), keepdims=True)
        self.std = data.std(axis=(0, 1), keepdims=True)
        self.std[self.std < 1e-6] = 1.0

    def transform(self, data: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("Scaler has not been fitted.")
        return (data - self.mean) / self.std

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("Scaler has not been fitted.")
        return data * self.std + self.mean


class SplitScaler:
    """
    Scaler that normalizes traffic features and extra features independently.

    This avoids scale interference when weather columns (e.g. temperature,
    humidity) are concatenated with traffic volume/speed columns.
    """

    def __init__(self, num_traffic_features: int = 1) -> None:
        self.num_traffic = num_traffic_features
        self.traffic_scaler = StandardScaler()
        self.extra_scaler = StandardScaler()

    def fit(self, data: np.ndarray) -> None:
        """data: [time, nodes, features]"""
        self.traffic_scaler.fit(data[..., : self.num_traffic])
        if data.shape[-1] > self.num_traffic:
            self.extra_scaler.fit(data[..., self.num_traffic :])

    def transform(self, data: np.ndarray) -> np.ndarray:
        traffic = self.traffic_scaler.transform(data[..., : self.num_traffic])
        if data.shape[-1] > self.num_traffic:
            extra = self.extra_scaler.transform(data[..., self.num_traffic :])
            return np.concatenate([traffic, extra], axis=-1)
        return traffic

    def inverse_transform_traffic(self, data: np.ndarray) -> np.ndarray:
        """Inverse-transform only the traffic feature slice."""
        return self.traffic_scaler.inverse_transform(data)


def read_csv_with_fallback(path: str) -> pd.DataFrame:
    """Read CSV files that may use UTF-8 or common Chinese encodings."""
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    return pd.read_csv(path)


class TrafficWindowDataset(Dataset):
    """
    Sliding-window traffic forecasting dataset.

    Args:
        data: [time, num_nodes, input_dim]
        history: number of historical steps
        horizon: number of future steps
        target_dim: which feature dimension to predict
    """

    def __init__(
        self,
        data: np.ndarray,
        history: int = 12,
        horizon: int = 12,
        target_dim: int = 0,
    ) -> None:
        if data.ndim != 3:
            raise ValueError("data must have shape [time, num_nodes, input_dim].")
        self.data = data.astype(np.float32)
        self.history = history
        self.horizon = horizon
        self.target_dim = target_dim

    def __len__(self) -> int:
        return max(0, self.data.shape[0] - self.history - self.horizon + 1)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx : idx + self.history]
        y = self.data[
            idx + self.history : idx + self.history + self.horizon,
            :,
            self.target_dim : self.target_dim + 1,
        ]
        return torch.from_numpy(x), torch.from_numpy(y)


def load_wide_traffic_csv(
    path: str,
    value_suffix: str | None = None,
    time_col: str | None = None,
    max_nodes: int | None = None,
    extra_feature_cols: list[str] | None = None,
    node_feature_suffixes: list[str] | None = None,
    add_time_features: bool = False,
    exclude_cols: list[str] | None = None,
    value_columns: list[str] | None = None,
    return_feature_names: bool = False,
) -> tuple[np.ndarray, list[str]] | tuple[np.ndarray, list[str], list[str]]:
    """
    Load a wide traffic table into [time, nodes, features].

    Example supported columns:
        Time, 1111570_volume, 1116139_volume, ...

    If value_suffix is provided, only columns ending with that suffix are used.
    If value_columns is provided, those columns are loaded in exactly the
    supplied order. This is required for matched flow-speed subnetworks and
    takes precedence over suffix scanning.
    For example, value_suffix="_volume" loads traffic volume columns.
    """

    df = read_csv_with_fallback(path)
    if time_col and time_col in df.columns:
        parsed_time = pd.to_datetime(df[time_col], errors="coerce")
        if float(parsed_time.notna().mean()) >= 0.95:
            df = (
                df.assign(__parsed_time_for_sort=parsed_time)
                .sort_values("__parsed_time_for_sort")
                .drop(columns="__parsed_time_for_sort")
                .reset_index(drop=True)
            )
        else:
            df = df.sort_values(time_col).reset_index(drop=True)

    exclude_set = set(exclude_cols or [])
    if value_columns is not None:
        numeric_cols = list(map(str, value_columns))
        if len(numeric_cols) != len(set(numeric_cols)):
            raise ValueError("value_columns contains duplicate columns")
        missing = [column for column in numeric_cols if column not in df.columns]
        if missing:
            raise ValueError(f"Requested value columns are missing: {missing}")
        forbidden = [
            column for column in numeric_cols if column == time_col or column in exclude_set
        ]
        if forbidden:
            raise ValueError(f"Requested value columns are excluded identifiers: {forbidden}")
        nonnumeric = [
            column for column in numeric_cols if not pd.api.types.is_numeric_dtype(df[column])
        ]
        if nonnumeric:
            raise ValueError(f"Requested value columns are not numeric: {nonnumeric}")
        if value_suffix and any(not column.endswith(value_suffix) for column in numeric_cols):
            raise ValueError("Explicit value columns do not all match value_suffix")
    else:
        numeric_cols = []
        for col in df.columns:
            if col == time_col or col in exclude_set:
                continue
            if value_suffix and not str(col).endswith(value_suffix):
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                numeric_cols.append(col)

    if max_nodes is not None:
        numeric_cols = numeric_cols[:max_nodes]
    if not numeric_cols:
        raise ValueError("No numeric traffic columns were found.")

    feature_names = ["traffic"]

    values = df[numeric_cols].astype("float32").to_numpy()
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    data = values[:, :, None]

    if node_feature_suffixes and value_suffix:
        node_bases = [str(col)[: -len(value_suffix)] for col in numeric_cols]
        for suffix in node_feature_suffixes:
            if not suffix or suffix == value_suffix:
                continue
            feature_cols = [f"{base}{suffix}" for base in node_bases]
            if not all(
                col in df.columns and pd.api.types.is_numeric_dtype(df[col])
                for col in feature_cols
            ):
                continue
            feature_values = df[feature_cols].astype("float32").to_numpy()
            feature_values = np.nan_to_num(
                feature_values,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            data = np.concatenate([data, feature_values[:, :, None]], axis=-1)
            feature_names.append(suffix)

    if extra_feature_cols:
        skipped_extra_cols = [
            col
            for col in extra_feature_cols
            if col in df.columns and not pd.api.types.is_numeric_dtype(df[col])
        ]
        if skipped_extra_cols:
            print(
                "[WARNING] Skipping non-numeric extra feature columns: "
                f"{skipped_extra_cols}"
            )
        available_extra_cols = [
            col
            for col in extra_feature_cols
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col])
        ]
        if available_extra_cols:
            extra = df[available_extra_cols].astype("float32").to_numpy()
            extra = np.nan_to_num(extra, nan=0.0, posinf=0.0, neginf=0.0)
            extra = extra[:, None, :]
            extra = np.repeat(extra, repeats=len(numeric_cols), axis=1)
            data = np.concatenate([data, extra], axis=-1)
            feature_names.extend(available_extra_cols)

    if add_time_features and time_col and time_col in df.columns:
        dt = pd.to_datetime(df[time_col], errors="coerce")
        minutes = (dt.dt.hour.fillna(0) * 60 + dt.dt.minute.fillna(0)).to_numpy()
        day = dt.dt.dayofweek.fillna(0).to_numpy()
        time_features = np.stack(
            [
                np.sin(2 * np.pi * minutes / 1440.0),
                np.cos(2 * np.pi * minutes / 1440.0),
                np.sin(2 * np.pi * day / 7.0),
                np.cos(2 * np.pi * day / 7.0),
            ],
            axis=1,
        ).astype("float32")
        time_features = time_features[:, None, :]
        time_features = np.repeat(time_features, repeats=len(numeric_cols), axis=1)
        data = np.concatenate([data, time_features], axis=-1)
        feature_names.extend(
            [
                "time_sin_day",
                "time_cos_day",
                "time_sin_week",
                "time_cos_week",
            ]
        )
    if return_feature_names:
        return data, numeric_cols, feature_names
    return data, numeric_cols


def auto_shrinkage_lambda(data: np.ndarray) -> float:
    """
    Heuristic shrinkage strength based on sample/node ratio and covariance noise.

    This keeps the implementation dependency-light. It is intentionally
    conservative: small samples or many nodes receive stronger shrinkage.
    """
    series = data[:, :, 0]
    num_samples, num_nodes = series.shape
    ratio_term = num_nodes / max(num_samples, 1)
    corr = np.corrcoef(series.T)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    offdiag = corr[~np.eye(num_nodes, dtype=bool)] if num_nodes > 1 else np.array([0.0])
    noise_term = float(np.std(offdiag))
    value = 0.05 + 2.0 * ratio_term + 0.15 * noise_term
    return float(np.clip(value, 0.05, 0.40))


def shrinkage_correlation(data: np.ndarray, shrinkage_lambda: float) -> np.ndarray:
    """Estimate a shrinkage correlation matrix from traffic series."""
    cov_shrink = shrinkage_covariance(data, shrinkage_lambda)
    return covariance_to_correlation(cov_shrink)


def shrinkage_covariance(data: np.ndarray, shrinkage_lambda: float) -> np.ndarray:
    """Estimate a diagonal-target shrinkage covariance matrix."""
    series = data[:, :, 0].astype(np.float64)
    series = np.nan_to_num(series, nan=0.0, posinf=0.0, neginf=0.0)
    cov = np.cov(series, rowvar=False)
    cov = np.atleast_2d(np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0))
    lam = float(np.clip(shrinkage_lambda, 0.0, 1.0))
    target = np.diag(np.diag(cov))
    return (1.0 - lam) * cov + lam * target


def covariance_to_correlation(cov: np.ndarray) -> np.ndarray:
    """Convert a covariance matrix to a correlation matrix."""
    std = np.sqrt(np.maximum(np.diag(cov), 1e-12))
    corr = cov / np.outer(std, std)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    return corr


def partial_correlation(data: np.ndarray, shrinkage_lambda: float) -> np.ndarray:
    """
    Estimate a partial-correlation matrix from a shrinkage covariance.

    The off-diagonal entries measure conditional dependence after controlling
    the remaining nodes, which is often a better statistical graph prior than
    raw synchronous correlation.
    """
    cov = shrinkage_covariance(data, shrinkage_lambda)
    precision = np.linalg.pinv(cov)
    diag = np.sqrt(np.maximum(np.diag(precision), 1e-12))
    partial = -precision / np.outer(diag, diag)
    np.fill_diagonal(partial, 1.0)
    return np.nan_to_num(partial, nan=0.0, posinf=0.0, neginf=0.0)



def oas_partial_correlation(data: np.ndarray) -> tuple[np.ndarray, float]:
    """Estimate an absolute partial-correlation graph using OAS shrinkage."""
    series = np.asarray(data[:, :, 0], dtype=np.float64)
    series = np.nan_to_num(series, nan=0.0, posinf=0.0, neginf=0.0)
    try:
        from sklearn.covariance import OAS

        estimator = OAS(assume_centered=False).fit(series)
        cov = estimator.covariance_
        shrinkage = float(estimator.shrinkage_)
    except Exception:
        centered = series - series.mean(axis=0, keepdims=True)
        empirical = centered.T @ centered / max(series.shape[0], 1)
        p = empirical.shape[0]
        mu = float(np.trace(empirical) / max(p, 1))
        alpha = float(np.mean(empirical**2))
        denominator = max((series.shape[0] + 1.0) * (alpha - mu**2 / max(p, 1)), 1e-12)
        shrinkage = float(np.clip((alpha + mu**2) / denominator, 0.0, 1.0))
        cov = (1.0 - shrinkage) * empirical + shrinkage * mu * np.eye(p)
    try:
        precision = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        precision = np.linalg.pinv(cov)
    scale = np.sqrt(np.maximum(np.diag(precision), 1e-12))
    partial = -precision / np.outer(scale, scale)
    partial = np.abs(np.nan_to_num(partial, nan=0.0, posinf=0.0, neginf=0.0))
    np.fill_diagonal(partial, 1.0)
    return partial, shrinkage
def build_correlation_adj(
    data: np.ndarray,
    threshold: float = 0.2,
    method: str = "pearson",
    shrinkage_lambda: float = 0.1,
    auto_shrinkage: bool = False,
) -> np.ndarray:
    """
    Build a static graph from a statistical dependence estimate.

    Use this when physical topology is unavailable. If you have true road
    topology, replace this with the adjacency built from from_node/to_node.
    """

    series = data[:, :, 0]
    if method == "pearson":
        corr = np.corrcoef(series.T)
    elif method == "shrinkage":
        lam = auto_shrinkage_lambda(data) if auto_shrinkage else shrinkage_lambda
        corr = shrinkage_correlation(data, lam)
        print(f"[INFO] Shrinkage correlation lambda={lam:.4f}")
    elif method == "partial":
        lam = auto_shrinkage_lambda(data) if auto_shrinkage else shrinkage_lambda
        corr = partial_correlation(data, lam)
        print(f"[INFO] Partial correlation shrinkage lambda={lam:.4f}")
    elif method == "oas_partial":
        corr, lam = oas_partial_correlation(data)
        print(f"[INFO] OAS partial correlation shrinkage={lam:.4f}")
    else:
        raise ValueError("corr method must be one of: pearson, shrinkage, partial, oas_partial.")
    corr = np.nan_to_num(corr, nan=0.0)
    adj = np.maximum(corr, 0.0)
    adj[adj < threshold] = 0.0
    np.fill_diagonal(adj, 1.0)
    return adj.astype(np.float32)


def _lagged_design(series: np.ndarray, max_lag: int) -> tuple[np.ndarray, list[np.ndarray]]:
    target = series[max_lag:]
    lagged = [series[max_lag - lag : -lag] for lag in range(1, max_lag + 1)]
    return target, lagged


def granger_p_value(
    cause: np.ndarray,
    target: np.ndarray,
    max_lag: int = 3,
) -> float:
    """
    F-test p-value for whether `cause` Granger-causes `target`.

    Restricted model: target history only.
    Unrestricted model: target history plus cause history.
    """

    cause = np.asarray(cause, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    valid = np.isfinite(cause) & np.isfinite(target)
    cause = cause[valid]
    target = target[valid]
    if cause.size <= max_lag * 3:
        return 1.0

    y, target_lags = _lagged_design(target, max_lag)
    _, cause_lags = _lagged_design(cause, max_lag)
    ones = np.ones_like(y)
    restricted = np.column_stack([ones, *target_lags])
    unrestricted = np.column_stack([ones, *target_lags, *cause_lags])

    try:
        beta_r, *_ = np.linalg.lstsq(restricted, y, rcond=None)
        beta_u, *_ = np.linalg.lstsq(unrestricted, y, rcond=None)
    except np.linalg.LinAlgError:
        return 1.0

    resid_r = y - restricted @ beta_r
    resid_u = y - unrestricted @ beta_u
    rss_r = float(np.sum(resid_r**2))
    rss_u = float(np.sum(resid_u**2))
    df_num = max_lag
    df_den = y.size - unrestricted.shape[1]
    if df_den <= 0 or rss_u <= 1e-12 or rss_r <= rss_u:
        return 1.0

    f_stat = ((rss_r - rss_u) / df_num) / (rss_u / df_den)
    if not np.isfinite(f_stat) or f_stat <= 0:
        return 1.0
    return float(stats.f.sf(f_stat, df_num, df_den))


def build_granger_adjacency(
    data: np.ndarray,
    max_lag: int = 3,
    p_threshold: float = 0.05,
    cache_path: str | None = None,
) -> np.ndarray:
    """
    Build a directed static graph using pairwise Granger causality tests.

    adj[i, j] > 0 means node i's history helps predict node j under an F-test.
    Edge weight is 1 - p_value for significant pairs.
    """

    series = data[:, :, 0]
    num_nodes = series.shape[1]
    if cache_path:
        path = Path(cache_path)
        if path.exists():
            cached = np.load(path)
            if cached.shape == (num_nodes, num_nodes):
                return cached.astype(np.float32)
            print(
                "[WARNING] Ignoring Granger cache with unexpected shape: "
                f"{cached.shape}, expected {(num_nodes, num_nodes)}"
            )

    adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for cause_idx in range(num_nodes):
        for target_idx in range(num_nodes):
            if cause_idx == target_idx:
                continue
            p_value = granger_p_value(
                series[:, cause_idx],
                series[:, target_idx],
                max_lag=max_lag,
            )
            if p_value < p_threshold:
                adj[cause_idx, target_idx] = 1.0 - p_value
    np.fill_diagonal(adj, 1.0)

    if cache_path:
        path = Path(cache_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, adj)
        print(f"[INFO] Granger adjacency saved to {path}")
    return adj.astype(np.float32)


def base_node_id(name: str) -> str:
    """Convert feature column names such as 1111570_volume to node ids."""
    value = str(name)
    for suffix in (
        "_volume",
        "_speed",
        "_flow",
        "_tpi",
        "_occupancy",
        "_density",
    ):
        if value.endswith(suffix):
            return value[: -len(suffix)]
    return value


def parse_wkt_centroid(value: object) -> tuple[float, float] | None:
    """Approximate a WKT geometry centroid from coordinate pairs."""
    numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", str(value))
    if len(numbers) < 2:
        return None
    coords = [float(item) for item in numbers]
    xs = coords[0::2]
    ys = coords[1::2]
    if not xs or not ys:
        return None
    return float(np.mean(xs)), float(np.mean(ys))


def load_road_adjacency_csv(
    path: str,
    node_names: list[str],
    from_col: str = "from_node",
    to_col: str = "to_node",
    weight_col: str | None = None,
    link_id_col: str | None = "link_id",
    directed: bool = True,
    road_knn: int = 3,
    geo_wkt_col: str = "geo_wkt",
) -> np.ndarray:
    """
    Build a road-topology adjacency for the selected traffic nodes.

    Two CSV layouts are supported:
    1. Link table: link_id, from_node, to_node. Selected traffic nodes are
       matched to link_id, then links are connected when upstream.to_node
       equals downstream.from_node.
    2. Edge list: from_col and to_col directly contain traffic node ids.
    """

    df = read_csv_with_fallback(path)
    node_ids = [base_node_id(name) for name in node_names]
    node_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}
    n = len(node_names)
    adj = np.zeros((n, n), dtype=np.float32)

    def add_edge(src: str, dst: str, weight: float = 1.0) -> None:
        src = str(src)
        dst = str(dst)
        if src not in node_to_idx or dst not in node_to_idx:
            return
        i = node_to_idx[src]
        j = node_to_idx[dst]
        adj[i, j] = max(adj[i, j], float(weight))
        if not directed:
            adj[j, i] = max(adj[j, i], float(weight))

    # Case 1: road-link table. Build link-to-link topology through endpoints.
    if link_id_col and link_id_col in df.columns and from_col in df.columns and to_col in df.columns:
        extra_cols = []
        if weight_col and weight_col in df.columns:
            extra_cols.append(weight_col)
        if geo_wkt_col and geo_wkt_col in df.columns:
            extra_cols.append(geo_wkt_col)
        work = df[[link_id_col, from_col, to_col] + extra_cols].copy()
        work[link_id_col] = work[link_id_col].astype(str)
        selected = work[work[link_id_col].isin(node_to_idx)].copy()
        if not selected.empty:
            rows = selected.to_dict("records")
            for upstream in rows:
                for downstream in rows:
                    if str(upstream[link_id_col]) == str(downstream[link_id_col]):
                        continue
                    connects = str(upstream[to_col]) == str(downstream[from_col])
                    if not directed:
                        connects = connects or str(upstream[from_col]) == str(downstream[to_col])
                    if connects:
                        weight = 1.0
                        if weight_col and weight_col in selected.columns:
                            raw = downstream.get(weight_col, 1.0)
                            if pd.notna(raw):
                                weight = float(raw)
                        add_edge(upstream[link_id_col], downstream[link_id_col], weight)

            # If sampled detector links are not directly adjacent, use a
            # geometry-based kNN road graph as a physical topology fallback.
            if float(adj.sum() - np.trace(adj)) <= 0.0 and road_knn > 0 and geo_wkt_col in selected.columns:
                centroids: list[tuple[str, tuple[float, float]]] = []
                for row in selected.to_dict("records"):
                    centroid = parse_wkt_centroid(row.get(geo_wkt_col))
                    if centroid is not None:
                        centroids.append((str(row[link_id_col]), centroid))
                for src_id, (src_x, src_y) in centroids:
                    distances = []
                    for dst_id, (dst_x, dst_y) in centroids:
                        if src_id == dst_id:
                            continue
                        dist = ((src_x - dst_x) ** 2 + (src_y - dst_y) ** 2) ** 0.5
                        distances.append((dist, dst_id))
                    for dist, dst_id in sorted(distances)[:road_knn]:
                        add_edge(src_id, dst_id, 1.0)
                        add_edge(dst_id, src_id, 1.0)

    # Case 2: direct edge list between selected traffic node ids.
    if from_col in df.columns and to_col in df.columns:
        cols = [from_col, to_col]
        if weight_col and weight_col in df.columns:
            cols.append(weight_col)
        for row in df[cols].itertuples(index=False):
            src = getattr(row, from_col)
            dst = getattr(row, to_col)
            weight = 1.0
            if weight_col and weight_col in df.columns:
                raw = getattr(row, weight_col)
                if pd.notna(raw):
                    weight = float(raw)
            add_edge(src, dst, weight)

    np.fill_diagonal(adj, 1.0)
    if float(adj.sum() - np.trace(adj)) <= 0.0:
        raise ValueError(
            "Road adjacency has no matched non-self edges. Check node ids and adjacency columns."
        )
    return adj


def build_static_adjacency(
    data: np.ndarray,
    node_names: list[str],
    source: str = "corr",
    corr_threshold: float = 0.2,
    corr_method: str = "pearson",
    corr_shrinkage_lambda: float = 0.1,
    auto_corr_shrinkage: bool = False,
    adj_path: str | None = None,
    road_weight: float = 0.5,
    granger_lag: int = 3,
    granger_p_threshold: float = 0.05,
    granger_cache_path: str | None = None,
    from_col: str = "from_node",
    to_col: str = "to_node",
    weight_col: str | None = None,
    link_id_col: str | None = "link_id",
    directed: bool = True,
    road_knn: int = 3,
    geo_wkt_col: str = "geo_wkt",
) -> np.ndarray:
    """Build corr, road, granger, or mixed static adjacency."""

    corr_adj = build_correlation_adj(
        data,
        threshold=corr_threshold,
        method=corr_method,
        shrinkage_lambda=corr_shrinkage_lambda,
        auto_shrinkage=auto_corr_shrinkage,
    )
    if source == "corr":
        return corr_adj
    if source == "granger":
        return build_granger_adjacency(
            data,
            max_lag=granger_lag,
            p_threshold=granger_p_threshold,
            cache_path=granger_cache_path,
        )

    if not adj_path:
        raise ValueError("--adj-path is required when --adj-source is road or mix.")

    road_adj = load_road_adjacency_csv(
        adj_path,
        node_names,
        from_col=from_col,
        to_col=to_col,
        weight_col=weight_col,
        link_id_col=link_id_col,
        directed=directed,
        road_knn=road_knn,
        geo_wkt_col=geo_wkt_col,
    )
    if source == "road":
        return road_adj.astype(np.float32)
    if source == "mix":
        road_weight = min(max(float(road_weight), 0.0), 1.0)
        mixed = road_weight * road_adj + (1.0 - road_weight) * corr_adj
        np.fill_diagonal(mixed, 1.0)
        return mixed.astype(np.float32)
    raise ValueError("source must be one of: corr, road, granger, mix.")

