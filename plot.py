from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError as exc:
    raise SystemExit(
        "matplotlib and seaborn are required for plot.py. Install them with: pip install matplotlib seaborn"
    ) from exc


ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

DATA_SOURCES = {
    "Human": (ROOT / "Human" / "human_classified.csv", "latin-1"),
    "Gemini": (ROOT / "Gemini" / "Gemini_parse.csv", "utf-8"),
    "OpenAI": (ROOT / "Openai" / "openai_parse.csv", "utf-8"),
    "Claude": (ROOT / "Claude" / "claude_parse.csv", "utf-8"),
}

SCORE_COLUMNS = ["semantic", "logical", "decision", "final_score"]
ATTACK_ORDER = ["None", "Semantic", "Logic", "Decision"]
MODEL_ORDER = ["Human", "Gemini", "OpenAI", "Claude"]
MODEL_COLORS = {
    "Human": "#4E79A7",
    "Gemini": "#F28E2B",
    "OpenAI": "#59A14F",
    "Claude": "#E15759",
}

sns.set_theme(style="whitegrid", context="talk")


def load_csv(path: Path, encoding: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding=encoding)
    df["video_id"] = df["video_id"].astype(str).str.strip()
    df["is_poisoned"] = (
        df["is_poisoned"].astype(str).str.strip().str.upper().map({"TRUE": True, "FALSE": False})
    )
    df["attack_level"] = df["attack_level"].astype(str).str.strip()
    for col in SCORE_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[
        ["video_id", "is_poisoned", "attack_level", *SCORE_COLUMNS]
    ].sort_values("video_id")


def rmse(series_a: pd.Series, series_b: pd.Series) -> float:
    diff = series_a - series_b
    return float(np.sqrt(np.mean(np.square(diff))))


def compute_metrics(datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    human = datasets["Human"].copy()
    records = []

    for model_name in MODEL_ORDER:
        df = datasets[model_name].copy()

        true_rate = float(df["is_poisoned"].mean() * 100.0)

        if model_name == "Human":
            poisoned_accuracy = 100.0
            attack_accuracy = 100.0
            overall_rmse = 0.0
            semantic_rmse = 0.0
            logical_rmse = 0.0
            decision_rmse = 0.0
            final_score_rmse = 0.0
        else:
            merged = human.merge(
                df,
                on="video_id",
                suffixes=("_human", "_model"),
                how="inner",
            )

            poisoned_accuracy = float(
                (merged["is_poisoned_human"] == merged["is_poisoned_model"]).mean() * 100.0
            )
            attack_accuracy = float(
                (merged["attack_level_human"] == merged["attack_level_model"]).mean() * 100.0
            )

            component_rmses = {
                col: rmse(merged[f"{col}_human"], merged[f"{col}_model"])
                for col in SCORE_COLUMNS
            }
            all_human_scores = merged[[f"{col}_human" for col in SCORE_COLUMNS]].to_numpy().reshape(-1)
            all_model_scores = merged[[f"{col}_model" for col in SCORE_COLUMNS]].to_numpy().reshape(-1)
            overall_rmse = float(np.sqrt(np.mean(np.square(all_human_scores - all_model_scores))))

            semantic_rmse = component_rmses["semantic"]
            logical_rmse = component_rmses["logical"]
            decision_rmse = component_rmses["decision"]
            final_score_rmse = component_rmses["final_score"]

        records.append(
            {
                "model": model_name,
                "true_rate_pct": true_rate,
                "poisoned_accuracy_pct": poisoned_accuracy,
                "attack_level_accuracy_pct": attack_accuracy,
                "overall_rmse": overall_rmse,
                "semantic_rmse": semantic_rmse,
                "logical_rmse": logical_rmse,
                "decision_rmse": decision_rmse,
                "final_score_rmse": final_score_rmse,
            }
        )

    return pd.DataFrame(records)


def plot_true_rate(metrics: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(
        data=metrics,
        x="model",
        y="true_rate_pct",
        hue="model",
        palette=MODEL_COLORS,
        dodge=False,
        ax=ax,
    )
    if ax.legend_ is not None:
        ax.legend_.remove()
    ax.set_ylim(0, 100)
    ax.set_xlabel("")
    ax.set_ylabel("True Rate (%)")
    ax.set_title("Poisoned Rate by Annotator / Model")

    for patch, value in zip(ax.patches, metrics["true_rate_pct"]):
        ax.text(
            patch.get_x() + patch.get_width() / 2,
            value + 1.2,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    fig.tight_layout()
    fig.savefig(FIG_DIR / "true_rate_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_accuracy(metrics: pd.DataFrame) -> None:
    compare = metrics[metrics["model"] != "Human"].copy()
    plot_df = compare.melt(
        id_vars="model",
        value_vars=["poisoned_accuracy_pct", "attack_level_accuracy_pct"],
        var_name="metric",
        value_name="accuracy",
    )
    plot_df["metric"] = plot_df["metric"].map(
        {
            "poisoned_accuracy_pct": "Poisoned Label Accuracy",
            "attack_level_accuracy_pct": "Attack-Level Accuracy",
        }
    )

    fig, ax = plt.subplots(figsize=(9, 5.6))
    sns.barplot(
        data=plot_df,
        x="model",
        y="accuracy",
        hue="metric",
        palette=["#4E79A7", "#9C755F"],
        ax=ax,
    )
    ax.set_ylim(0, 100)
    ax.set_xlabel("")
    ax.set_ylabel("Agreement with Human (%)")
    ax.set_title("Agreement Rate with Human Labels")
    ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.03),
        ncol=2,
    )

    for patch in ax.patches:
        value = patch.get_height()
        if not np.isfinite(value) or value <= 0:
            continue
        ax.text(
            patch.get_x() + patch.get_width() / 2,
            value + 1.2,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIG_DIR / "accuracy_vs_human.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_rmse(metrics: pd.DataFrame) -> None:
    compare = metrics[metrics["model"] != "Human"].copy()
    plot_df = compare.melt(
        id_vars="model",
        value_vars=[
            "overall_rmse",
            "semantic_rmse",
            "logical_rmse",
            "decision_rmse",
            "final_score_rmse",
        ],
        var_name="rmse_type",
        value_name="rmse",
    )
    plot_df["rmse_type"] = plot_df["rmse_type"].map(
        {
            "overall_rmse": "Overall",
            "semantic_rmse": "Semantic",
            "logical_rmse": "Logical",
            "decision_rmse": "Decision",
            # "final_score_rmse": "Final",
        }
    )

    fig, ax = plt.subplots(figsize=(11.8, 5.8))
    sns.barplot(
        data=plot_df,
        x="rmse_type",
        y="rmse",
        hue="model",
        palette={k: v for k, v in MODEL_COLORS.items() if k != "Human"},
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("RMSE vs Human")
    ax.set_title("Score RMSE Compared with Human Labels")
    ax.legend(
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0.0,
    )

    for patch in ax.patches:
        value = patch.get_height()
        if not np.isfinite(value) or value <= 0:
            continue
        ax.text(
            patch.get_x() + patch.get_width() / 2,
            value + 0.004,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=0,
        )

    fig.tight_layout(rect=(0, 0, 0.86, 1))
    fig.savefig(FIG_DIR / "rmse_vs_human.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_summary(metrics: pd.DataFrame, datasets: dict[str, pd.DataFrame]) -> None:
    summary = metrics.copy()
    summary["samples"] = summary["model"].map({name: len(df) for name, df in datasets.items()})
    summary = summary[
        [
            "model",
            "samples",
            "true_rate_pct",
            "poisoned_accuracy_pct",
            "attack_level_accuracy_pct",
            "overall_rmse",
            "semantic_rmse",
            "logical_rmse",
            "decision_rmse",
            "final_score_rmse",
        ]
    ]
    summary.to_csv(FIG_DIR / "metrics_summary.csv", index=False)


def main() -> None:
    datasets = {
        name: load_csv(path, encoding)
        for name, (path, encoding) in DATA_SOURCES.items()
    }

    metrics = compute_metrics(datasets)
    plot_true_rate(metrics)
    plot_accuracy(metrics)
    plot_rmse(metrics)
    save_summary(metrics, datasets)

    print(f"Saved plots to: {FIG_DIR}")
    for filename in [
        "true_rate_comparison.png",
        "accuracy_vs_human.png",
        "rmse_vs_human.png",
        "metrics_summary.csv",
    ]:
        print(f" - {FIG_DIR / filename}")


if __name__ == "__main__":
    main()
