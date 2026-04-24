from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError as exc:
    raise SystemExit(
        "matplotlib and seaborn are required for plot_improvement.py. Install them with: pip install matplotlib seaborn"
    ) from exc


ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

HUMAN_SOURCE = (ROOT / "Human" / "human_classified.csv", "latin-1")
MODEL_SOURCES = {
    "Gemini": {
        "Un-Improved": (ROOT / "Gemini" / "Gemini_parse.csv", "utf-8"),
        "Improved": (ROOT / "Improvement" / "Gemini" / "gemini_improved_parse.csv", "utf-8"),
    },
    "OpenAI": {
        "Un-Improved": (ROOT / "Openai" / "openai_parse.csv", "utf-8"),
        "Improved": (ROOT / "Improvement" / "Openai" / "openai_improved_parse.csv", "utf-8"),
    },
    "Claude": {
        "Un-Improved": (ROOT / "Claude" / "claude_parse.csv", "utf-8"),
        "Improved": (ROOT / "Improvement" / "Claude" / "claude_improved_parse.csv", "utf-8"),
    },
}

SCORE_COLUMNS = ["semantic", "logical", "decision", "final_score"]
RMSE_SCORE_COLUMNS = ["semantic", "logical", "decision"]
MODEL_ORDER = ["Gemini", "OpenAI", "Claude"]
VARIANT_ORDER = ["Un-Improved", "Improved"]
VARIANT_COLORS = {
    "Un-Improved": "#F28E2B",
    "Improved": "#4E79A7",
}
RMSE_ORDER = ["Overall", "Semantic", "Logical", "Decision"]

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
    return df[["video_id", "is_poisoned", "attack_level", *SCORE_COLUMNS]].sort_values("video_id")


def rmse(series_a: pd.Series, series_b: pd.Series) -> float:
    diff = series_a - series_b
    return float(np.sqrt(np.mean(np.square(diff))))


def compute_provider_metrics(human: pd.DataFrame, provider: str, variant: str, df: pd.DataFrame) -> dict:
    true_rate = float(df["is_poisoned"].mean() * 100.0)

    merged = human.merge(df, on="video_id", suffixes=("_human", "_model"), how="inner")
    samples = len(merged)
    poisoned_accuracy = float(
        (merged["is_poisoned_human"] == merged["is_poisoned_model"]).mean() * 100.0
    )
    attack_accuracy = float(
        (merged["attack_level_human"] == merged["attack_level_model"]).mean() * 100.0
    )

    component_rmses = {
        col: rmse(merged[f"{col}_human"], merged[f"{col}_model"])
        for col in RMSE_SCORE_COLUMNS
    }
    all_human_scores = merged[[f"{col}_human" for col in RMSE_SCORE_COLUMNS]].to_numpy().reshape(-1)
    all_model_scores = merged[[f"{col}_model" for col in RMSE_SCORE_COLUMNS]].to_numpy().reshape(-1)
    overall_rmse = float(np.sqrt(np.mean(np.square(all_human_scores - all_model_scores))))
    semantic_rmse = component_rmses["semantic"]
    logical_rmse = component_rmses["logical"]
    decision_rmse = component_rmses["decision"]

    return {
        "provider": provider,
        "variant": variant,
        "samples": samples,
        "true_rate_pct": true_rate,
        "poisoned_accuracy_pct": poisoned_accuracy,
        "attack_level_accuracy_pct": attack_accuracy,
        "overall_rmse": overall_rmse,
        "semantic_rmse": semantic_rmse,
        "logical_rmse": logical_rmse,
        "decision_rmse": decision_rmse,
    }


def build_metrics() -> pd.DataFrame:
    human = load_csv(*HUMAN_SOURCE)
    records = []

    for provider in MODEL_ORDER:
        for variant in ["Un-Improved", "Improved"]:
            records.append(
                compute_provider_metrics(
                    human,
                    provider,
                    variant,
                    load_csv(*MODEL_SOURCES[provider][variant]),
                )
            )

    return pd.DataFrame(records)


def annotate_bars(ax, fmt: str, offset: float) -> None:
    for patch in ax.patches:
        value = patch.get_height()
        if not np.isfinite(value) or value <= 0:
            continue
        ax.text(
            patch.get_x() + patch.get_width() / 2,
            value + offset,
            format(value, fmt),
            ha="center",
            va="bottom",
            fontsize=8,
        )


def plot_true_rate(metrics: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10.8, 5.6))
    sns.barplot(
        data=metrics,
        x="provider",
        y="true_rate_pct",
        hue="variant",
        order=MODEL_ORDER,
        hue_order=VARIANT_ORDER,
        palette=VARIANT_COLORS,
        ax=ax,
    )
    ax.set_ylim(0, 100)
    ax.set_xlabel("")
    ax.set_ylabel("True Rate (%)")
    ax.set_title("True Rate: Un-Improved vs Improved")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)
    annotate_bars(ax, ".1f", 1.2)
    fig.tight_layout(rect=(0, 0, 0.86, 1))
    fig.savefig(FIG_DIR / "improvement_true_rate_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_accuracy(metrics: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15.8, 5.8), sharey=True)
    plot_specs = [
        ("poisoned_accuracy_pct", "Label Accuracy"),
        ("attack_level_accuracy_pct", "Attack-Level Accuracy"),
    ]

    for ax, (col, title) in zip(axes, plot_specs):
        sns.barplot(
            data=metrics,
            x="provider",
            y=col,
            hue="variant",
            order=MODEL_ORDER,
            hue_order=VARIANT_ORDER,
            palette=VARIANT_COLORS,
            ax=ax,
        )
        ax.set_ylim(0, 105)
        ax.set_xlabel("")
        ax.set_ylabel("Agreement with Human (%)")
        ax.set_title(title)
        if ax.legend_ is not None:
            ax.legend_.remove()
        annotate_bars(ax, ".1f", 1.0)

    fig.suptitle("Agreement with Human Labels: Un-Improved vs Improved", y=1.03)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="center right", bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout(rect=(0, 0, 0.88, 0.98))
    fig.savefig(FIG_DIR / "improvement_accuracy_vs_human.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_rmse(metrics: pd.DataFrame) -> None:
    rmse_cols = {
        "overall_rmse": "Overall",
        "semantic_rmse": "Semantic",
        "logical_rmse": "Logical",
        "decision_rmse": "Decision",
    }
    plot_df = metrics.melt(
        id_vars=["provider", "variant"],
        value_vars=list(rmse_cols.keys()),
        var_name="rmse_type",
        value_name="rmse",
    )
    plot_df["rmse_type"] = plot_df["rmse_type"].map(rmse_cols)

    fig, axes = plt.subplots(2, 2, figsize=(15.8, 9.5), sharey=True)
    axes = axes.flatten()

    for ax, rmse_type in zip(axes, RMSE_ORDER):
        subset = plot_df[plot_df["rmse_type"] == rmse_type].copy()
        sns.barplot(
            data=subset,
            x="provider",
            y="rmse",
            hue="variant",
            order=MODEL_ORDER,
            hue_order=VARIANT_ORDER,
            palette=VARIANT_COLORS,
            ax=ax,
        )
        ax.set_xlabel("")
        ax.set_ylabel("RMSE vs Human")
        ax.set_title(rmse_type)
        if ax.legend_ is not None:
            ax.legend_.remove()
        annotate_bars(ax, ".3f", 0.004)

    fig.suptitle("Score RMSE: Un-Improved vs Improved", y=1.02)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="center right", bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout(rect=(0, 0, 0.88, 0.98))
    fig.savefig(FIG_DIR / "improvement_rmse_vs_human.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_summary(metrics: pd.DataFrame) -> None:
    summary = metrics[
        [
            "provider",
            "variant",
            "samples",
            "true_rate_pct",
            "poisoned_accuracy_pct",
            "attack_level_accuracy_pct",
            "overall_rmse",
            "semantic_rmse",
            "logical_rmse",
            "decision_rmse",
        ]
    ].copy()
    summary.to_csv(FIG_DIR / "improvement_metrics_summary.csv", index=False)


def main() -> None:
    metrics = build_metrics()
    plot_true_rate(metrics)
    plot_accuracy(metrics)
    plot_rmse(metrics)
    save_summary(metrics)

    print(f"Saved plots to: {FIG_DIR}")
    for filename in [
        "improvement_true_rate_comparison.png",
        "improvement_accuracy_vs_human.png",
        "improvement_rmse_vs_human.png",
        "improvement_metrics_summary.csv",
    ]:
        print(f" - {FIG_DIR / filename}")


if __name__ == "__main__":
    main()
