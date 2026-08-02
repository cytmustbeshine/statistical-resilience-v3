"""Orchestrate the preregistered three-seed E-L4-2 formal experiment."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from audit_l4_prediction_pipeline import DATASETS, ordered_frame
from l4_prediction_pipeline import paired_column_plan


def run_command(command: list[str], manifest: list[dict[str, object]], task: str) -> None:
    print(f"[formal] {task}", flush=True)
    completed = subprocess.run(command, check=False)
    row = {"task": task, "returncode": int(completed.returncode), "command": command}
    manifest.append(row)
    if completed.returncode != 0:
        raise RuntimeError(f"formal task failed: {task}")


def dcrnn_specs(dataset: str) -> list[tuple[str, list[str], str]]:
    config = DATASETS[dataset]
    frame = ordered_frame(config)
    plan = paired_column_plan(list(frame.columns), config["flow_suffix"], config["speed_suffix"], 41)
    if dataset == "bridge":
        return [("flow", plan["selected_flow_columns"], str(config["flow_suffix"] or "none"))]
    return [
        ("flow", plan["paired_flow_columns"], str(config["flow_suffix"])),
        ("speed", plan["paired_speed_columns"], str(config["speed_suffix"])),
    ]


def run(args) -> None:
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    manifest: list[dict[str, object]] = []
    python = sys.executable

    for seed in seeds:
        m1_tag = f"seed_{seed}"
        m1_decision = root / "m1_dstsgcn" / f"e_l4_1_{m1_tag}_decision.json"
        if not m1_decision.exists():
            run_command([
                python, str(Path(__file__).with_name("run_l4_prediction_baseline.py")),
                "--run-tag", m1_tag,
                "--datasets", "bridge,rainstorm,typhoon",
                "--output-dir", str(root / "m1_dstsgcn"),
                "--max-nodes", "41", "--history", "12", "--horizon", "12",
                "--epochs", str(args.epochs), "--batch-size", "64",
                "--hidden-dim", "64", "--num-blocks", "2",
                "--max-train-windows", "0", "--max-eval-windows", "0",
                "--seed", str(seed), "--device", args.device,
                "--split-protocol", "profile_compatible_event_external",
                "--missing-space-protocol", "dual_space_train_only_median",
            ], manifest, f"M1 seed={seed}")
        else:
            manifest.append({"task": f"M1 seed={seed}", "returncode": 0, "skipped_complete": True})

        m23_decision = root / "m2_m3_dstsgcn" / f"seed_{seed}_decision.json"
        if not m23_decision.exists():
            run_command([
                python, str(Path(__file__).with_name("run_final_l4_a3.py")),
                "--datasets", "bridge,rainstorm,typhoon", "--variants", "m2,m3",
                "--output-dir", str(root / "m2_m3_dstsgcn"), "--run-tag", f"seed_{seed}",
                "--max-nodes", "41", "--history", "12", "--horizon", "12",
                "--epochs", str(args.epochs), "--batch-size", "64",
                "--hidden-dim", "64", "--num-blocks", "2",
                "--max-train-windows", "0", "--max-eval-windows", "0",
                "--seed", str(seed), "--device", args.device,
            ], manifest, f"M2/M3 seed={seed}")
        else:
            manifest.append({"task": f"M2/M3 seed={seed}", "returncode": 0, "skipped_complete": True})

        for dataset in ("bridge", "rainstorm", "typhoon"):
            config = DATASETS[dataset]
            for variable, columns, suffix in dcrnn_specs(dataset):
                task_dir = root / "m0_dcrnn" / dataset / variable / f"seed_{seed}"
                complete = all((task_dir / name).exists() for name in (
                    "best_dcrnn_resilience_official_adapted.pt", "metrics.json", "traffic_predictions.npz"
                ))
                task = f"M0 {dataset}/{variable} seed={seed}"
                if complete:
                    manifest.append({"task": task, "returncode": 0, "skipped_complete": True})
                    continue
                run_command([
                    python,
                    str(Path(__file__).parent / "baselines" / "dcrnn_resilience_official_adapted" / "train.py"),
                    "--csv", str(config["csv"]), "--dataset-name", dataset,
                    "--value-suffix", suffix, "--value-columns", ",".join(columns),
                    "--time-col", str(config["time_col"]), "--max-nodes", str(len(columns)),
                    "--history-steps", "12", "--horizon", "12",
                    "--rnn-units", "256", "--num-rnn-layers", "2", "--max-diffusion-step", "1",
                    "--output-dir", str(task_dir), "--seed", str(seed),
                    "--epochs", str(args.epochs), "--batch-size", "64",
                    "--max-train-windows", "0", "--max-eval-windows", "0", "--device", args.device,
                    "--split-protocol", "profile_compatible_event_external",
                    "--missing-space-protocol", "dual_space_train_only_median",
                ], manifest, task)

        (root / "formal_run_manifest.json").write_text(
            json.dumps({"seeds": seeds, "epochs": args.epochs, "tasks": manifest}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=r"D:\TrafficGNN\outputs\e_l4_2_final_aligned_a3\formal_final")
    parser.add_argument("--seeds", default="42,2024,3407")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
