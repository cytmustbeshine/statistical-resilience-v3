"""Convert PEMS .npz traffic tensors to the project's wide CSV format.

Expected input shape is [time, nodes, features]. The first feature is treated
as the target traffic volume by default and is written as *_volume columns.
Additional feature channels can be exported as aligned per-node suffix columns
such as *_occupancy and *_speed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def load_npz_array(path: Path, key: str | None) -> np.ndarray:
    """Load a [T, N, F] array from an NPZ file."""
    with np.load(path) as payload:
        if key:
            if key not in payload:
                raise KeyError(f"Key {key!r} not found. Available keys: {list(payload.keys())}")
            array = payload[key]
        elif "data" in payload:
            array = payload["data"]
        else:
            keys = list(payload.keys())
            if len(keys) != 1:
                raise KeyError(
                    "Could not infer NPZ array key. "
                    f"Use --key. Available keys: {keys}"
                )
            array = payload[keys[0]]
    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 2:
        array = array[..., None]
    if array.ndim != 3:
        raise ValueError(f"Expected [time, nodes, features], got shape {array.shape}")
    return np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)


def convert_npz_to_wide_csv(
    input_path: Path,
    output_path: Path,
    dataset_name: str,
    key: str | None,
    start_time: str,
    freq: str,
    feature_names: list[str],
) -> None:
    """Write PEMS tensor features as a wide CSV."""
    data = load_npz_array(input_path, key)
    time_steps, num_nodes, num_features = data.shape
    if len(feature_names) < num_features:
        feature_names = feature_names + [
            f"feature{i}" for i in range(len(feature_names), num_features)
        ]
    feature_names = feature_names[:num_features]

    times = pd.date_range(start=start_time, periods=time_steps, freq=freq)
    frame: dict[str, object] = {"Time": times}
    node_ids = [f"{dataset_name}_{idx:03d}" for idx in range(num_nodes)]
    for feature_idx, feature_name in enumerate(feature_names):
        suffix = "_volume" if feature_idx == 0 else f"_{feature_name}"
        for node_idx, node_id in enumerate(node_ids):
            frame[f"{node_id}{suffix}"] = data[:, node_idx, feature_idx]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(frame).to_csv(output_path, index=False, encoding="utf-8-sig")
    print(
        f"[OK] Wrote {output_path} | "
        f"time={time_steps}, nodes={num_nodes}, features={num_features}"
    )
    print(f"[INFO] Target suffix: _volume")
    if num_features > 1:
        suffixes = [f"_{name}" for name in feature_names[1:]]
        print(f"[INFO] Optional node feature suffixes: {','.join(suffixes)}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to pemsXX.npz")
    parser.add_argument("--output", required=True, help="Output wide CSV path")
    parser.add_argument("--dataset-name", default="pems", help="Node column prefix")
    parser.add_argument("--key", default="", help="NPZ key. Empty = infer or use 'data'.")
    parser.add_argument("--start-time", default="2018-01-01 00:00:00")
    parser.add_argument("--freq", default="5min")
    parser.add_argument(
        "--feature-names",
        default="volume,occupancy,speed",
        help="Comma-separated names for feature channels. First becomes _volume.",
    )
    args = parser.parse_args()
    feature_names = [
        item.strip() for item in args.feature_names.split(",") if item.strip()
    ]
    if not feature_names:
        raise ValueError("--feature-names must contain at least one name.")
    convert_npz_to_wide_csv(
        input_path=Path(args.input),
        output_path=Path(args.output),
        dataset_name=args.dataset_name,
        key=args.key.strip() or None,
        start_time=args.start_time,
        freq=args.freq,
        feature_names=feature_names,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
