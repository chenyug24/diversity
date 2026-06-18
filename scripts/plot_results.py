#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRIC_LABELS = {
    "mean_score": "Final score",
    "mean_value": "Value",
    "mean_diversity": "Diversity",
    "mean_origin": "Origin distance",
    "peak_coverage": "Peak coverage",
    "max_peak_occupancy": "Max peak occupancy",
}

STRATEGY_LABELS = {
    "random": "Random",
    "random_corner": "Random corner",
    "origin_maximizer": "Origin max",
    "independent_search": "Independent",
    "score_following": "Score following",
    "diversity_only": "Diversity only",
    "full_collaboration": "Full collab",
    "random_collaboration": "Random collab",
    "strategic_collaboration": "Strategic collab",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Peak-Divergence experiment results.")
    parser.add_argument("--results", type=Path, required=True, help="Directory with metrics CSVs.")
    parser.add_argument("--out", type=Path, default=None, help="Directory for PNG figures.")
    parser.add_argument("--top", type=int, default=9, help="Number of strategies to show.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = args.results
    out_dir = args.out or results_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    final_path = results_dir / "final_metrics.csv"
    round_path = results_dir / "round_metrics.csv"
    if not final_path.exists():
        raise FileNotFoundError(f"Missing {final_path}")
    if not round_path.exists():
        raise FileNotFoundError(f"Missing {round_path}")

    final = pd.read_csv(final_path)
    rounds = pd.read_csv(round_path)
    order = (
        final.groupby("strategy")["mean_score"]
        .mean()
        .sort_values(ascending=False)
        .head(args.top)
        .index.tolist()
    )
    final = final[final["strategy"].isin(order)].copy()
    rounds = rounds[rounds["strategy"].isin(order)].copy()

    _set_style()
    paths = [
        plot_final_scores(final, order, out_dir),
        plot_components(final, order, out_dir),
        plot_round_curves(rounds, order, out_dir),
        plot_quality_diversity(final, order, out_dir),
    ]
    print("Wrote figures:")
    for path in paths:
        print(path)


def _set_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 180,
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.8,
            "legend.frameon": False,
        }
    )


def _labels(order: list[str]) -> list[str]:
    return [STRATEGY_LABELS.get(strategy, strategy) for strategy in order]


def _palette(count: int) -> list[tuple[float, float, float, float]]:
    cmap = plt.get_cmap("tab10")
    return [cmap(i % 10) for i in range(count)]


def plot_final_scores(final: pd.DataFrame, order: list[str], out_dir: Path) -> Path:
    grouped = final.groupby("strategy")["mean_score"].agg(["mean", "std"]).reindex(order)
    errors = grouped["std"].fillna(0.0).to_numpy()
    y = np.arange(len(order))
    colors = _palette(len(order))

    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    ax.barh(y, grouped["mean"], xerr=errors, color=colors, alpha=0.88, capsize=3)
    ax.set_yticks(y, _labels(order))
    ax.invert_yaxis()
    ax.set_xlabel("Average final score")
    ax.set_title("Peak-Divergence Final Score by Strategy")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    for idx, value in enumerate(grouped["mean"]):
        ax.text(value, idx, f" {value:.1f}", va="center", ha="left", fontsize=9)
    fig.tight_layout()
    path = out_dir / "final_score_by_strategy.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_components(final: pd.DataFrame, order: list[str], out_dir: Path) -> Path:
    metrics = ["mean_value", "mean_diversity", "peak_coverage"]
    grouped = final.groupby("strategy")[metrics].mean().reindex(order)

    normalized = grouped.copy()
    for metric in metrics:
        max_value = max(float(normalized[metric].max()), 1e-9)
        normalized[metric] = normalized[metric] / max_value

    x = np.arange(len(order))
    width = 0.24
    colors = ["#4C78A8", "#F58518", "#54A24B"]

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    for idx, metric in enumerate(metrics):
        ax.bar(
            x + (idx - 1) * width,
            normalized[metric],
            width=width,
            label=METRIC_LABELS[metric],
            color=colors[idx],
            alpha=0.88,
        )

    ax.set_xticks(x, _labels(order), rotation=25, ha="right")
    ax.set_ylabel("Normalized to best strategy")
    ax.set_ylim(0, 1.12)
    ax.set_title("Quality-Diversity Components")
    ax.legend(ncols=3, loc="upper right")
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    path = out_dir / "component_tradeoffs.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_round_curves(rounds: pd.DataFrame, order: list[str], out_dir: Path) -> Path:
    colors = _palette(len(order))
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6), sharex=True)
    metrics = ["mean_score", "mean_value", "peak_coverage"]

    for ax, metric in zip(axes, metrics):
        for strategy, color in zip(order, colors):
            subset = rounds[rounds["strategy"] == strategy]
            grouped = subset.groupby("round")[metric].mean()
            ax.plot(
                grouped.index,
                grouped.values,
                marker="o",
                linewidth=2,
                markersize=3.5,
                color=color,
                label=STRATEGY_LABELS.get(strategy, strategy),
            )
        ax.set_title(METRIC_LABELS.get(metric, metric))
        ax.set_xlabel("Round")
        ax.grid(True)

    axes[0].set_ylabel("Mean across seeds")
    axes[-1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.suptitle("Round-by-Round Dynamics", y=1.04, fontsize=14)
    fig.tight_layout()
    path = out_dir / "round_dynamics.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_quality_diversity(final: pd.DataFrame, order: list[str], out_dir: Path) -> Path:
    grouped = (
        final.groupby("strategy")
        .agg(
            mean_value=("mean_value", "mean"),
            mean_diversity=("mean_diversity", "mean"),
            mean_score=("mean_score", "mean"),
            peak_coverage=("peak_coverage", "mean"),
        )
        .reindex(order)
    )
    colors = _palette(len(order))
    sizes = 50 + 28 * grouped["peak_coverage"].fillna(0.0).to_numpy()
    offsets = [
        (7, 8),
        (7, -10),
        (7, 7),
        (7, -7),
        (7, 7),
        (7, -7),
        (7, 7),
        (7, -7),
        (7, 7),
    ]

    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    for idx, strategy in enumerate(order):
        row = grouped.loc[strategy]
        ax.scatter(
            row["mean_diversity"],
            row["mean_value"],
            s=sizes[idx],
            color=colors[idx],
            alpha=0.82,
            edgecolor="white",
            linewidth=1.0,
            label=STRATEGY_LABELS.get(strategy, strategy),
        )
        ax.annotate(
            STRATEGY_LABELS.get(strategy, strategy),
            xy=(row["mean_diversity"], row["mean_value"]),
            xytext=offsets[idx % len(offsets)],
            textcoords="offset points",
            va="center",
            fontsize=8.5,
        )

    ax.set_xlabel("Average diversity")
    ax.set_ylabel("Average value")
    ax.set_title("Quality-Diversity Tradeoff")
    ax.grid(True)
    fig.tight_layout()
    path = out_dir / "quality_diversity_scatter.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


if __name__ == "__main__":
    main()
