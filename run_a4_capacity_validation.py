"""Run the A4 validation-only DSTSGCN capacity screen."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from data import TrafficWindowDataset, build_static_adjacency
from l4_prediction_evaluation import regression_metrics
from l4_prediction_pipeline import DUAL_SPACE_MISSING_PROTOCOL, PROFILE_COMPATIBLE_SPLIT_PROTOCOL, target_tensor_from_indices
from run_l4_prediction_baseline import collect_predictions, limited_indices, model_for_baseline, prepare_variable
from train import run_epoch, set_seed
from temporal_decoder import AutoregressiveForecastWrapper

TASKS = (("bridge", "flow"), ("rainstorm", "flow"), ("rainstorm", "speed"), ("typhoon", "flow"), ("typhoon", "speed"))

def zero_initialize_residual_head(model: nn.Module) -> nn.Module:
    """Initialize the final traffic projection to an exact zero residual."""
    output = getattr(model, "output", None)
    if output is None or len(output) == 0 or not isinstance(output[-1], nn.Linear):
        raise TypeError("DSTSGCN output must end with nn.Linear")
    nn.init.zeros_(output[-1].weight)
    if output[-1].bias is not None:
        nn.init.zeros_(output[-1].bias)
    return model
class ResidualForecastWrapper(nn.Module):
    """Add a persistence anchor while the graph model learns a scaled residual."""

    def __init__(self, base_model: nn.Module, horizon: int, residual_scale: float = 1.0):
        super().__init__()
        if residual_scale <= 0:
            raise ValueError("residual_scale must be positive")
        self.base_model = base_model
        self.graph_learner = base_model.graph_learner
        self.horizon = int(horizon)
        self.residual_scale = float(residual_scale)

    def forward(self, x: torch.Tensor, static_adj: torch.Tensor, *args, **kwargs):
        base_output = self.base_model(x, static_adj, *args, **kwargs)
        if isinstance(base_output, tuple):
            residual, aux = base_output
            output = self._add_anchor(x, residual)
            return output, aux
        return self._add_anchor(x, base_output)

    def _add_anchor(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        anchor = x[:, -1:, :, :].expand(-1, self.horizon, -1, -1)
        return anchor + self.residual_scale * residual


def parse_int_list(text: str) -> list[int]:
    values = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not values or any(value <= 0 for value in values):
        raise ValueError("integer candidate list must contain positive values")
    return list(dict.fromkeys(values))


def parse_float_list(text: str) -> list[float]:
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not values or any(value <= 0 for value in values):
        raise ValueError("float candidate list must contain positive values")
    return list(dict.fromkeys(values))


def candidate_summary(rows: pd.DataFrame, control_hidden_dim: int = 64) -> pd.DataFrame:
    task_columns = ["dataset", "variable", "seed"]
    control = rows.loc[rows["hidden_dim"] == control_hidden_dim, task_columns + ["best_val_mae"]].rename(columns={"best_val_mae": "control_val_mae"})
    compared = rows.merge(control, on=task_columns, how="left", validate="many_to_one")
    compared["val_mae_ratio"] = compared["best_val_mae"] / compared["control_val_mae"]
    grouped = compared.groupby(["hidden_dim", "lr"], as_index=False).agg(
        tasks=("val_mae_ratio", "size"),
        finite_tasks=("finite", "sum"),
        improved_tasks=("val_mae_ratio", lambda values: int(np.sum(np.asarray(values) < 1.0))),
        mean_val_mae_ratio=("val_mae_ratio", "mean"),
        worst_val_mae_ratio=("val_mae_ratio", "max"),
        parameter_count=("parameter_count", "max"),
    )
    grouped["advances"] = ((grouped["tasks"] == len(TASKS)) & (grouped["finite_tasks"] == len(TASKS)) & (grouped["improved_tasks"] >= 4) & (grouped["mean_val_mae_ratio"] <= 0.97) & (grouped["worst_val_mae_ratio"] <= 1.05))
    return grouped.sort_values(["mean_val_mae_ratio", "worst_val_mae_ratio"]).reset_index(drop=True)


def train_validation_only(dataset: str, variable: str, hidden_dim: int, lr: float, args, residual_scale: float = 0.0, zero_init_residual: bool = False, autoregressive_decoder: bool = False) -> dict[str, object]:
    prepared = prepare_variable(dataset, variable, args)
    bundle = prepared["bundle"]
    train_indices = limited_indices(bundle["train_indices"], args.max_train_windows)
    val_indices = limited_indices(bundle["val_indices"], args.max_val_windows)
    base_dataset = TrafficWindowDataset(bundle["scaled_values"], args.history, args.horizon)
    train_loader = DataLoader(Subset(base_dataset, train_indices.tolist()), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(Subset(base_dataset, val_indices.tolist()), batch_size=args.batch_size, shuffle=False)
    train_end = int(bundle["split_info"]["train_time_end_exclusive"])
    adjacency = build_static_adjacency(bundle["scaled_values"][:train_end], prepared["model_names"], source="corr", corr_threshold=0.2)
    device = torch.device(args.device)
    static_adj = torch.tensor(adjacency, dtype=torch.float32, device=device)
    set_seed(args.seed)
    model = model_for_baseline(len(prepared["model_names"]), args.horizon, hidden_dim, args.num_blocks)
    if residual_scale > 0:
        if zero_init_residual:
            zero_initialize_residual_head(model)
        model = ResidualForecastWrapper(model, args.horizon, residual_scale)
    if autoregressive_decoder:
        if residual_scale > 0:
            raise ValueError("autoregressive decoder and residual wrapper are mutually exclusive")
        model = AutoregressiveForecastWrapper(model, hidden_dim, args.horizon)
    model = model.to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.HuberLoss()
    raw_mean = float(np.asarray(bundle["scaler"].mean).reshape(-1)[0])
    raw_std = float(np.asarray(bundle["scaler"].std).reshape(-1)[0])
    best_val_mae = float("inf")
    best_epoch = 0
    best_state = None
    history = []
    for epoch in range(1, args.epochs + 1):
        train_stats = run_epoch(model, train_loader, static_adj, optimizer, loss_fn, 1e-3, 1e-4, None, None, False, 5.0, False, False, 0.0, 0.0, "ratio", 0.0, False, 0.90, 0.0, 0.90, 0.0, False, raw_mean, raw_std, device)
        val_true_scaled, val_pred_scaled = collect_predictions(model, val_loader, static_adj, device)
        if prepared["missing_space_protocol"] == DUAL_SPACE_MISSING_PROTOCOL:
            val_true = target_tensor_from_indices(prepared["selected_physical_values"], val_indices, args.history, args.horizon)
        else:
            val_true = bundle["scaler"].inverse_transform(val_true_scaled)
        val_pred = bundle["scaler"].inverse_transform(val_pred_scaled)
        val_mae = float(regression_metrics(val_true, val_pred)["mae"])
        history.append({"epoch": epoch, "train_loss": float(train_stats["loss"]), "val_mae": val_mae})
        if np.isfinite(val_mae) and val_mae < best_val_mae:
            best_val_mae = val_mae
            best_epoch = epoch
            best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
    if best_state is None:
        raise RuntimeError(f"no finite validation checkpoint for {dataset}/{variable}")
    run_dir = Path(args.output_dir) / f"seed_{args.seed}" / f"h{hidden_dim}_lr{lr:g}_rs{residual_scale:g}_zi{int(zero_init_residual)}_ar{int(autoregressive_decoder)}" / dataset / variable
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": best_state, "metadata": {"stage": "A4-0", "validation_only": True, "dataset": dataset, "variable": variable, "hidden_dim": hidden_dim, "lr": lr, "residual_scale": residual_scale, "zero_init_residual": zero_init_residual, "autoregressive_decoder": autoregressive_decoder, "seed": args.seed, "best_epoch": best_epoch, "best_val_mae": best_val_mae}}, run_dir / "best_validation_only.pt")
    result = {"stage": "A4-0", "validation_only": True, "dataset": dataset, "variable": variable, "seed": args.seed, "hidden_dim": hidden_dim, "lr": lr, "residual_scale": residual_scale, "zero_init_residual": zero_init_residual, "autoregressive_decoder": autoregressive_decoder, "parameter_count": parameter_count, "train_windows": len(train_indices), "validation_windows": len(val_indices), "epochs": args.epochs, "best_epoch": best_epoch, "best_val_mae": best_val_mae, "finite": bool(np.isfinite(best_val_mae)), "history": history}
    (run_dir / "validation_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=r"D:\TrafficGNN\outputs\a4_capacity_validation")
    parser.add_argument("--l4-dir", default=r"D:\TrafficGNN\outputs\two_factor_traffic_resilience_l4")
    parser.add_argument("--source-profile-dir", default=r"D:\TrafficGNN\outputs\latent_traffic_performance_l3")
    parser.add_argument("--hidden-dims", default="64,128,160")
    parser.add_argument("--learning-rates", default="0.001")
    parser.add_argument("--max-nodes", type=int, default=41)
    parser.add_argument("--history", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-blocks", type=int, default=2)
    parser.add_argument("--max-train-windows", type=int, default=0)
    parser.add_argument("--max-val-windows", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--split-protocol", default=PROFILE_COMPATIBLE_SPLIT_PROTOCOL)
    parser.add_argument("--missing-space-protocol", default=DUAL_SPACE_MISSING_PROTOCOL)
    args = parser.parse_args()
    hidden_dims = parse_int_list(args.hidden_dims)
    learning_rates = parse_float_list(args.learning_rates)
    if 64 not in hidden_dims:
        raise ValueError("A4-0 requires hidden_dim=64 as the validation control")
    rows = []
    for hidden_dim in hidden_dims:
        for lr in learning_rates:
            for dataset, variable in TASKS:
                print(f"[A4-0] h={hidden_dim} lr={lr:g} {dataset}/{variable}", flush=True)
                rows.append(train_validation_only(dataset, variable, hidden_dim, lr, args))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([{key: value for key, value in row.items() if key != "history"} for row in rows])
    frame.to_csv(output_dir / "a4_validation_tasks.csv", index=False, encoding="utf-8-sig")
    summary = candidate_summary(frame)
    summary.to_csv(output_dir / "a4_validation_candidates.csv", index=False, encoding="utf-8-sig")
    advancing = summary.loc[summary["advances"]]
    selected = None if advancing.empty else advancing.iloc[0].to_dict()
    decision = {"stage": "A4-0", "validation_only": True, "test_loader_constructed": False, "candidate_selected": selected is not None, "selected_candidate": selected, "formal_test_authorized": selected is not None}
    (output_dir / "a4_validation_decision.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(decision, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())