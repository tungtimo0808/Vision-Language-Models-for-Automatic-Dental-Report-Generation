"""Plot the key benchmark figures from saved JSON results.

Reads:
- <benchmark-dir>/benchmark_results.json  (the per-run summary)
- <run>/test_results.json                 (per-class metrics + confusion matrix)
- <run>/metrics_history.json              (per-epoch training curves)

Produces (into <benchmark-dir>/plots/):
1. model_comparison.png      - F1 / macro-F1 / mAP@0.5 across all runs
2. per_class_f1_<run>.png    - per-class F1 (+support) for the best run
3. confusion_<run>.png       - row-normalized confusion matrix for the best run
4. training_curves_<run>.png - train/val loss and val F1 over epochs for the best run

Plots 2-4 are skipped automatically when the data is absent (e.g. the FDI workspace,
whose runs do not store per-class metrics).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - depends on user env
    plt = None


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser(description="Plot benchmark figures")
    parser.add_argument("--root", type=Path, default=Path("."), help="Repository root")
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        default=Path("faster_rcnn_disease/outputs/benchmark"),
        help="Directory containing benchmark_results.json",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    """Load a JSON file, returning None if missing."""
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def plot_model_comparison(runs: list[dict[str, Any]], out_path: Path) -> None:
    """Grouped bar chart comparing F1 / macro-F1 / mAP@0.5 across runs."""
    done = [r for r in runs if r.get("status") in ("ok", "cached")]
    if not done:
        print("[skip] model_comparison: no completed runs")
        return

    names = [r["run_name"] for r in done]
    f1 = [float(r.get("test_f1", 0.0)) for r in done]
    macro = [float(r.get("test_macro_f1", 0.0)) for r in done]
    map50 = [float(r.get("test_map_50", 0.0)) for r in done]

    x = range(len(names))
    width = 0.27
    fig, ax = plt.subplots(figsize=(max(8, len(names) * 1.1), 5))
    ax.bar([i - width for i in x], f1, width, label="F1 (global)")
    ax.bar(list(x), macro, width, label="macro-F1")
    ax.bar([i + width for i in x], map50, width, label="mAP@0.5")
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1)
    ax.set_title("Disease detection - model comparison")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[ok] {out_path}")


def plot_per_class_f1(test_results: dict[str, Any], run_name: str, out_path: Path) -> None:
    """Bar chart of per-class F1 for one run, sorted ascending, support annotated."""
    per_class = test_results.get("test_per_class")
    if not per_class:
        print("[skip] per_class_f1: no per-class metrics in test_results")
        return

    items = sorted(per_class.items(), key=lambda kv: kv[1]["f1"])
    labels = [k for k, _ in items]
    f1 = [v["f1"] for _, v in items]
    support = [v["support"] for _, v in items]

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.8), 5))
    bars = ax.bar(labels, f1, color="#4C72B0")
    for bar, sup in zip(bars, support):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"n={sup}",
                ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("F1")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Per-class F1 - {run_name}")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[ok] {out_path}")


def plot_confusion(test_results: dict[str, Any], run_name: str, out_path: Path) -> None:
    """Row-normalized confusion matrix heatmap for one run."""
    matrix = test_results.get("confusion_matrix")
    labels = test_results.get("confusion_labels")
    if not matrix or not labels:
        print("[skip] confusion: no confusion matrix in test_results")
        return

    norm = []
    for row in matrix:
        total = sum(row)
        norm.append([(v / total) if total > 0 else 0.0 for v in row])

    n = len(labels)
    fig, ax = plt.subplots(figsize=(max(7, n * 0.6), max(6, n * 0.6)))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("ground truth")
    ax.set_title(f"Confusion (row-normalized) - {run_name}")
    for i in range(n):
        for j in range(n):
            if norm[i][j] > 0.01:
                ax.text(j, i, f"{norm[i][j]:.2f}", ha="center", va="center",
                        color="white" if norm[i][j] > 0.5 else "black", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[ok] {out_path}")


def plot_training_curves(history: dict[str, Any], run_name: str, out_path: Path) -> None:
    """Train/val loss and val F1 over epochs for one run."""
    hist = history.get("history") if isinstance(history, dict) else None
    if not hist:
        print("[skip] training_curves: no metrics history")
        return

    epochs = [h["epoch"] for h in hist]
    train_loss = [h.get("train_loss") for h in hist]
    val_loss = [h.get("val_loss") for h in hist]
    val_f1 = [h.get("val_f1") for h in hist]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(epochs, train_loss, label="train_loss", color="#C44E52")
    ax1.plot(epochs, val_loss, label="val_loss", color="#DD8452")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(epochs, val_f1, label="val_f1", color="#55A868")
    ax2.set_ylabel("val F1")
    ax2.set_ylim(0, 1)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper center")
    ax1.set_title(f"Training curves - {run_name}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[ok] {out_path}")


def main() -> None:
    """Plotting entrypoint."""
    args = parse_args()

    if plt is None:
        raise SystemExit("matplotlib is not installed. Install it: pip install matplotlib")

    root = args.root.resolve()
    benchmark_dir = (root / args.benchmark_dir).resolve()
    plots_dir = benchmark_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    summary = load_json(benchmark_dir / "benchmark_results.json")
    if summary is None:
        raise SystemExit(f"No benchmark_results.json in {benchmark_dir}. Run the benchmark first.")

    runs = summary.get("runs", [])
    plot_model_comparison(runs, plots_dir / "model_comparison.png")

    # Best run: prefer macro-F1 (fairer under class imbalance), fall back to F1.
    best = summary.get("best_by_macro_f1") or summary.get("best_by_f1")
    if not best:
        print("[skip] per-run plots: no best run found")
        return

    run_name = best["run_name"]
    run_dir = root / best["output_dir"]
    test_results = load_json(run_dir / "test_results.json")
    history = load_json(run_dir / "metrics_history.json")

    if test_results:
        plot_per_class_f1(test_results, run_name, plots_dir / f"per_class_f1_{run_name}.png")
        plot_confusion(test_results, run_name, plots_dir / f"confusion_{run_name}.png")
    if history:
        plot_training_curves(history, run_name, plots_dir / f"training_curves_{run_name}.png")

    print(f"Plots saved to: {plots_dir}")


if __name__ == "__main__":
    main()
