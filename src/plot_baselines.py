import argparse
import os
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


MODEL_DISPLAY = {
    "signature_only": "Signature-only",
    "rf_novelty_only": "RF Novelty-only",
    "hybrid_signature_rf_novelty": "Hybrid Signature + RF Novelty",
}

MODEL_ORDER = [
    "signature_only",
    "rf_novelty_only",
    "hybrid_signature_rf_novelty",
]


def find_latest_summary(result_base_dir: str) -> Path:
    base = Path(result_base_dir)
    if not base.exists():
        raise FileNotFoundError(f"Result base directory not found: {base}")

    run_dirs = [p for p in base.iterdir() if p.is_dir()]
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found under: {base}")

    latest_run = max(run_dirs, key=lambda p: p.stat().st_mtime)
    summary_path = latest_run / "summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"summary.csv not found in latest run dir: {latest_run}")
    return summary_path


def load_baseline_rows(summary_path: str | Path, quantile: float) -> pd.DataFrame:
    df = pd.read_csv(summary_path)
    df = df[df["quantile"] == quantile].copy()
    df = df[df["model"].isin(MODEL_ORDER)].copy()
    if df.empty:
        raise ValueError(f"No baseline rows found for quantile={quantile} in {summary_path}")

    df["model_order"] = df["model"].map({m: i for i, m in enumerate(MODEL_ORDER)})
    df = df.sort_values("model_order")
    df["Model"] = df["model"].map(MODEL_DISPLAY).fillna(df["model"])
    return df


def save_bar_chart(df: pd.DataFrame, metric: str, ylabel: str, out_path: str | Path):
    plt.figure(figsize=(8, 5))
    plt.bar(df["Model"], df[metric])
    plt.title(f"Baseline Comparison: {ylabel}")
    plt.ylabel(ylabel)
    plt.xlabel("Model")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def save_grouped_chart(df: pd.DataFrame, out_path: str | Path):
    plot_df = df.set_index("Model")[["precision", "recall", "f1"]].copy()
    plot_df.columns = ["Precision", "Recall", "F1-score"]

    plt.figure(figsize=(9, 5.5))
    plot_df.plot(kind="bar", ax=plt.gca())
    plt.title("Baseline Comparison")
    plt.ylabel("Score")
    plt.xlabel("Model")
    plt.xticks(rotation=15, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot baseline comparison charts from summary.csv")
    parser.add_argument("--summary", type=str, default=None, help="Path to summary.csv")
    parser.add_argument(
        "--result-base-dir",
        type=str,
        default="result_baselines",
        help="Base result directory to auto-find latest summary.csv when --summary is omitted",
    )
    parser.add_argument("--quantile", type=float, default=0.95)
    parser.add_argument("--out-dir", type=str, default="result_plots")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    summary_path = Path(args.summary) if args.summary else find_latest_summary(args.result_base_dir)
    df = load_baseline_rows(summary_path, args.quantile)

    save_bar_chart(df, "precision", "Precision", Path(args.out_dir) / "precision_comparison.png")
    save_bar_chart(df, "recall", "Recall", Path(args.out_dir) / "recall_comparison.png")
    save_bar_chart(df, "f1", "F1-score", Path(args.out_dir) / "f1_comparison.png")
    save_bar_chart(df, "far", "FAR", Path(args.out_dir) / "far_comparison.png")
    save_grouped_chart(df, Path(args.out_dir) / "baseline_grouped.png")

    print("Using summary:", summary_path)
    print("Saved plots to:", Path(args.out_dir).resolve())


if __name__ == "__main__":
    main()
