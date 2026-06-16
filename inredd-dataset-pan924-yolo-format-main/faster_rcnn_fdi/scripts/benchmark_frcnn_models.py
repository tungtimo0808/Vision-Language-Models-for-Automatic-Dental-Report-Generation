"""Benchmark 3 Faster R-CNN backbones with shallow/deep fine-tuning.

Runs 6 experiments:
- 3 model architectures
- 2 trainable_backbone_layers settings (shallow, deep)

Outputs:
- faster_rcnn_fdi/outputs/benchmark/benchmark_results.json
- faster_rcnn_fdi/outputs/benchmark/benchmark_results.csv
- faster_rcnn_fdi/outputs/benchmark/benchmark_table.md
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser(description="Benchmark Faster R-CNN architectures")
    parser.add_argument("--root", type=Path, default=Path("."), help="Repository root")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("faster_rcnn_fdi/configs/train_config.yaml"),
        help="Base config path",
    )
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--dry-run", action="store_true", help="Run only one batch per run")
    parser.add_argument(
        "--max-runs",
        type=int,
        default=0,
        help="Run only first N experiments (0 = all)",
    )
    parser.add_argument(
        "--recommend-by",
        type=str,
        default="f1_then_runtime",
        choices=["f1_only", "f1_then_runtime"],
        help="Recommendation rule for RTX3050",
    )
    return parser.parse_args()


def run_train(
    root: Path,
    config_path: Path,
    architecture: str,
    trainable_layers: int,
    output_subdir: str,
    device: str,
    epochs: int | None,
    dry_run: bool,
    resume_path: Path | None,
) -> tuple[bool, str]:
    """Run one training experiment as subprocess."""
    cmd = [
        sys.executable,
        str(root / "faster_rcnn_fdi/scripts/train_frcnn.py"),
        "--root",
        str(root),
        "--config",
        str(config_path),
        "--architecture",
        architecture,
        "--trainable-backbone-layers",
        str(trainable_layers),
        "--output-subdir",
        output_subdir,
        "--device",
        device,
    ]

    if epochs is not None:
        cmd.extend(["--epochs", str(epochs)])
    if resume_path is not None and resume_path.exists():
        cmd.extend(["--resume", str(resume_path)])
    if dry_run:
        cmd.append("--dry-run")

    try:
        subprocess.run(cmd, check=True)
        return True, "ok"
    except subprocess.CalledProcessError as exc:
        return False, f"failed(code={exc.returncode})"


def load_shared_resolution(config_path: Path) -> tuple[int, int]:
    """Load shared min/max size from base config."""
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    data_cfg = cfg.get("data", {})
    return int(data_cfg.get("min_size", 512)), int(data_cfg.get("max_size", 768))


def pick_recommendation(rows: list[dict[str, Any]], rule: str) -> dict[str, Any] | None:
    """Pick recommended run for RTX3050 from completed experiments."""
    completed = [r for r in rows if r["status"] == "ok"]
    if not completed:
        return None

    if rule == "f1_only":
        return max(completed, key=lambda r: float(r["test_f1"]))

    return sorted(completed, key=lambda r: (-float(r["test_f1"]), float(r["runtime_sec"])))[0]


def main() -> None:
    """Benchmark entrypoint."""
    args = parse_args()

    root = args.root.resolve()
    config_path = (root / args.config).resolve()
    shared_min_size, shared_max_size = load_shared_resolution(config_path)

    benchmark_dir = root / "faster_rcnn_fdi/outputs/benchmark"
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    experiments = [
        {
            "architecture": "fasterrcnn_mobilenet_v3_large_320_fpn",
            "label": "mbv3_320",
        },
        {
            "architecture": "fasterrcnn_mobilenet_v3_large_fpn",
            "label": "mbv3_large",
        },
        {
            "architecture": "fasterrcnn_resnet50_fpn",
            "label": "resnet50",
        },
    ]

    depth_settings = [
        {"trainable_layers": 1, "depth_name": "shallow"},
        {"trainable_layers": 3, "depth_name": "deep"},
        {"trainable_layers": 5, "depth_name": "full"},
    ]

    plan: list[dict[str, Any]] = []
    for exp in experiments:
        for depth in depth_settings:
            plan.append(
                {
                    **exp,
                    **depth,
                    "run_name": f"{exp['label']}_{depth['depth_name']}",
                }
            )

    if args.max_runs > 0:
        plan = plan[: args.max_runs]

    rows: list[dict[str, Any]] = []

    for index, item in enumerate(plan, start=1):
        run_name = str(item["run_name"])
        output_subdir = f"faster_rcnn_fdi/outputs/bench/{run_name}"
        resume_path = root / output_subdir / "last.pth"

        print(f"[{index}/{len(plan)}] Running {run_name} ...")
        t0 = time.time()
        ok, status = run_train(
            root=root,
            config_path=config_path,
            architecture=str(item["architecture"]),
            trainable_layers=int(item["trainable_layers"]),
            output_subdir=output_subdir,
            device=str(args.device),
            epochs=args.epochs,
            dry_run=bool(args.dry_run),
            resume_path=resume_path,
        )
        runtime_sec = round(time.time() - t0, 2)

        result_path = root / output_subdir / "test_results.json"
        test_f1 = 0.0
        test_precision = 0.0
        test_recall = 0.0
        test_map_50 = 0.0
        test_map_50_95 = 0.0
        best_val_loss = None
        best_epoch = None

        if ok and result_path.exists():
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            det = payload.get("test_detection", {})
            test_f1 = float(det.get("f1", 0.0))
            test_precision = float(det.get("precision", 0.0))
            test_recall = float(det.get("recall", 0.0))
            test_map = payload.get("test_map", {})
            test_map_50 = float(test_map.get("map_50", 0.0))
            test_map_50_95 = float(test_map.get("map_50_95", 0.0))
            best_val_loss = payload.get("best_val_loss")
            best_epoch = payload.get("best_epoch")

        row = {
            "run_name": run_name,
            "architecture": item["architecture"],
            "depth_name": item["depth_name"],
            "trainable_backbone_layers": int(item["trainable_layers"]),
            "min_size": int(shared_min_size),
            "max_size": int(shared_max_size),
            "status": status,
            "runtime_sec": runtime_sec,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "test_precision": round(test_precision, 6),
            "test_recall": round(test_recall, 6),
            "test_f1": round(test_f1, 6),
            "test_map_50": round(test_map_50, 6),
            "test_map_50_95": round(test_map_50_95, 6),
            "output_dir": output_subdir,
        }
        rows.append(row)

    completed = [r for r in rows if r["status"] == "ok"]
    if completed:
        best_by_f1 = max(completed, key=lambda r: float(r["test_f1"]))
        best_for_3050 = pick_recommendation(rows, args.recommend_by)
    else:
        best_by_f1 = None
        best_for_3050 = None

    summary = {
        "device": args.device,
        "dry_run": args.dry_run,
        "fairness_policy": {
            "shared_config": str(config_path.relative_to(root)),
            "shared_min_size": int(shared_min_size),
            "shared_max_size": int(shared_max_size),
            "same_train_eval_params_for_all_runs": True,
            "only_changes": ["architecture", "trainable_backbone_layers", "output_subdir"],
        },
        "runs": rows,
        "best_by_f1": best_by_f1,
        "recommendation_rule": args.recommend_by,
        "recommended_for_rtx3050_4gb": best_for_3050,
    }

    json_path = benchmark_dir / "benchmark_results.json"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_path = benchmark_dir / "benchmark_results.csv"
    if rows:
        fieldnames = list(rows[0].keys())
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    md_path = benchmark_dir / "benchmark_table.md"
    lines = [
        "| run_name | architecture | depth | layers | min_size | max_size | test_precision | test_recall | test_f1 | test_map_50 | test_map_50_95 | runtime_sec | status | output_dir |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for r in rows:
        lines.append(
            "| {run_name} | {architecture} | {depth_name} | {trainable_backbone_layers} | {min_size} | {max_size} | "
            "{test_precision:.6f} | {test_recall:.6f} | {test_f1:.6f} | {test_map_50:.6f} | {test_map_50_95:.6f} | "
            "{runtime_sec:.2f} | {status} | {output_dir} |".format(**r)
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Saved: {json_path}")
    print(f"Saved: {csv_path}")
    print(f"Saved: {md_path}")

    if best_for_3050 is not None:
        print("Recommended for RTX 3050 4GB:")
        print(best_for_3050)


if __name__ == "__main__":
    main()
