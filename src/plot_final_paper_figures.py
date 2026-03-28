import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt


def save_bar_chart(df: pd.DataFrame, x_col: str, y_col: str, title: str, ylabel: str, out_path: str):
    plt.figure(figsize=(8, 5))
    plt.bar(df[x_col], df[y_col])
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel("Model")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate paper-style result figures.")
    parser.add_argument("--input-csv", type=str, required=True, help="Path to final_model_comparison.csv")
    parser.add_argument("--out-dir", type=str, default="paper_figures", help="Output directory for figures")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.input_csv)

    save_bar_chart(
        df, "model", "precision",
        "Precision Comparison Across Models",
        "Precision",
        os.path.join(args.out_dir, "precision_comparison.png"),
    )

    save_bar_chart(
        df, "model", "recall",
        "Recall Comparison Across Models",
        "Recall",
        os.path.join(args.out_dir, "recall_comparison.png"),
    )

    save_bar_chart(
        df, "model", "f1",
        "F1-score Comparison Across Models",
        "F1-score",
        os.path.join(args.out_dir, "f1_comparison.png"),
    )

    save_bar_chart(
        df, "model", "far",
        "False Alarm Rate Comparison Across Models",
        "FAR",
        os.path.join(args.out_dir, "far_comparison.png"),
    )

    plot_df = df.set_index("model")[["precision", "recall", "f1"]]
    plot_df.columns = ["Precision", "Recall", "F1-score"]

    plt.figure(figsize=(10, 5.5))
    plot_df.plot(kind="bar", ax=plt.gca())
    plt.title("Metric Comparison Across Models")
    plt.ylabel("Score")
    plt.xlabel("Model")
    plt.xticks(rotation=20, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "grouped_metrics_comparison.png"), dpi=200, bbox_inches="tight")
    plt.close()

    print("Saved figures to:", os.path.abspath(args.out_dir))


if __name__ == "__main__":
    main()