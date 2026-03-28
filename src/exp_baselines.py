import argparse
import json
import os
import sys
from argparse import Namespace
from pathlib import Path

import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from best_config import get_best_experiment_config
from exp_rf_novelty import run_rf_novelty_experiment


MODEL_ORDER = [
    "signature_only",
    "rf_novelty_only",
    "hybrid_signature_rf_novelty",
]

MODEL_DISPLAY = {
    "signature_only": "Signature-only",
    "rf_novelty_only": "RF Novelty-only",
    "hybrid_signature_rf_novelty": "Hybrid Signature + RF Novelty",
}


def get_project_root() -> Path:
    return Path(CURRENT_DIR).resolve().parent


def get_latest_run_dir(result_base_dir: str) -> Path:
    """
    Locate the most recently created run directory under the given base result dir.
    Example:
        result_baselines/
            rf_novelty_20260328_120001/
            rf_novelty_20260328_120315/
    """
    base = get_project_root() / result_base_dir
    if not base.exists():
        raise FileNotFoundError(f"Result base directory not found: {base}")

    run_dirs = [p for p in base.iterdir() if p.is_dir()]
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found under: {base}")

    latest = max(run_dirs, key=lambda p: p.stat().st_mtime)
    return latest


def build_args_from_best(
    input_dir: str,
    result_dir: str,
    max_files: int | None,
    max_rows_per_file: int | None,
    random_state: int | None,
) -> Namespace:
    best_cfg = get_best_experiment_config()
    rf_cfg = best_cfg["rf_config"]

    return Namespace(
        input_dir=input_dir,
        max_rows_per_file=max_rows_per_file,
        max_files=max_files,
        use_signature=best_cfg.get("use_signature", True),
        n_estimators=rf_cfg["n_estimators"],
        max_depth=rf_cfg["max_depth"],
        min_samples_leaf=rf_cfg["min_samples_leaf"],
        alpha=rf_cfg["alpha"],
        quantiles=rf_cfg["quantiles"],
        random_state=random_state if random_state is not None else rf_cfg.get("random_state", 42),
        result_dir=result_dir,
    )


def load_summary(latest_run_dir: Path) -> pd.DataFrame:
    summary_path = latest_run_dir / "summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"summary.csv not found in: {latest_run_dir}")
    return pd.read_csv(summary_path)


def build_baseline_table(summary_df: pd.DataFrame, quantile: float) -> pd.DataFrame:
    df = summary_df.copy()

    df = df[df["quantile"] == quantile].copy()
    if df.empty:
        raise ValueError(f"No rows found for quantile={quantile}")

    df["model_order"] = df["model"].map({m: i for i, m in enumerate(MODEL_ORDER)})
    df = df.sort_values("model_order")

    out = df[["model", "precision", "recall", "f1", "far"]].copy()
    out["Model"] = out["model"].map(MODEL_DISPLAY).fillna(out["model"])

    out = out[["Model", "precision", "recall", "f1", "far"]].rename(
        columns={
            "precision": "Precision",
            "recall": "Recall",
            "f1": "F1-score",
            "far": "FAR",
        }
    )

    for col in ["Precision", "Recall", "F1-score", "FAR"]:
        out[col] = out[col].astype(float).round(4)

    return out.reset_index(drop=True)


def save_baseline_outputs(
    latest_run_dir: Path,
    baseline_df: pd.DataFrame,
    quantile: float,
):
    csv_path = latest_run_dir / "baseline_table.csv"
    md_path = latest_run_dir / "baseline_table.md"
    tex_path = latest_run_dir / "baseline_table.tex"
    json_path = latest_run_dir / "baseline_report.json"

    baseline_df.to_csv(csv_path, index=False)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(baseline_df.to_markdown(index=False))

    latex_df = baseline_df.copy()
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\\begin{table}[ht]\n")
        f.write("\\centering\n")
        f.write(f"\\caption{{Baseline comparison at quantile $q={quantile}$}}\n")
        f.write("\\label{tab:baseline_comparison}\n")
        f.write(latex_df.to_latex(index=False, escape=False))
        f.write("\\end{table}\n")

    report = {
        "quantile": quantile,
        "baseline_table": baseline_df.to_dict(orient="records"),
        "output_files": {
            "csv": str(csv_path),
            "markdown": str(md_path),
            "latex": str(tex_path),
            "json": str(json_path),
        },
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return {
        "csv": csv_path,
        "markdown": md_path,
        "latex": tex_path,
        "json": json_path,
    }


def main():
    parser = argparse.ArgumentParser(description="Run saved best config and generate baseline comparison table.")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory or CSV path for CIC-IDS2017.")
    parser.add_argument("--result-dir", type=str, default="result_baselines", help="Base result directory.")
    parser.add_argument("--max-files", type=int, default=8, help="Optional cap on number of source CSV files.")
    parser.add_argument("--max-rows-per-file", type=int, default=None, help="Optional row cap per CSV after loading.")
    parser.add_argument("--random-state", type=int, default=None, help="Optional override for random state.")
    parser.add_argument("--quantile", type=float, default=0.95, help="Quantile to extract for baseline table.")
    args = parser.parse_args()

    run_args = build_args_from_best(
        input_dir=args.input_dir,
        result_dir=args.result_dir,
        max_files=args.max_files,
        max_rows_per_file=args.max_rows_per_file,
        random_state=args.random_state,
    )

    print("\nRunning baseline experiment with saved best config...")
    run_rf_novelty_experiment(run_args)

    latest_run_dir = get_latest_run_dir(args.result_dir)
    print(f"\nLatest run directory: {latest_run_dir}")

    summary_df = load_summary(latest_run_dir)
    baseline_df = build_baseline_table(summary_df, quantile=args.quantile)

    output_paths = save_baseline_outputs(
        latest_run_dir=latest_run_dir,
        baseline_df=baseline_df,
        quantile=args.quantile,
    )

    print("\nBaseline comparison table")
    print(baseline_df.to_string(index=False))

    print("\nSaved files:")
    for k, v in output_paths.items():
        print(f" - {k}: {v}")


if __name__ == "__main__":
    main()