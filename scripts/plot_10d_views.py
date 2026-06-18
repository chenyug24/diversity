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


STRATEGY_LABELS = {
    "random": "Random",
    "independent_search": "Independent",
    "score_following": "Score following",
    "full_collaboration": "Full collab",
    "strategic_collaboration": "Strategic collab",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot 10D coordinate views.")
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--sample-per-strategy", type=int, default=45)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out or args.results / "figures_10d"
    out_dir.mkdir(parents=True, exist_ok=True)

    agent_path = args.results / "agent_scores.csv"
    peak_path = args.results / "peaks.csv"
    if not agent_path.exists():
        raise FileNotFoundError(
            f"Missing {agent_path}. Re-run the experiment with --write-agent-scores."
        )

    agents = pd.read_csv(agent_path)
    peaks = pd.read_csv(peak_path) if peak_path.exists() else None
    z_cols = [col for col in agents.columns if col.startswith("z")]
    z_cols = sorted(z_cols, key=lambda col: int(col[1:]))

    sampled = _sample_agents(agents, args.sample_per_strategy)
    _set_style()
    paths = [
        plot_parallel_coordinates(sampled, z_cols, out_dir),
        plot_coordinate_heatmap(sampled, z_cols, out_dir),
        plot_pca_projection(agents, peaks, z_cols, out_dir),
    ]

    print("Wrote 10D figures:")
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


def _sample_agents(agents: pd.DataFrame, sample_per_strategy: int) -> pd.DataFrame:
    pieces = []
    for _, group in agents.groupby("strategy_run"):
        count = min(sample_per_strategy, len(group))
        top = group.nlargest(max(1, count // 3), "score")
        remaining_count = count - len(top)
        rest = group.drop(top.index)
        if remaining_count > 0 and len(rest) > 0:
            random_part = rest.sample(
                n=min(remaining_count, len(rest)),
                random_state=7,
            )
            pieces.append(pd.concat([top, random_part], ignore_index=True))
        else:
            pieces.append(top)
    return pd.concat(pieces, ignore_index=True)


def _strategy_order(frame: pd.DataFrame) -> list[str]:
    return (
        frame.groupby("strategy_run")["score"]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )


def _colors(order: list[str]) -> dict[str, tuple[float, float, float, float]]:
    cmap = plt.get_cmap("tab10")
    return {strategy: cmap(idx % 10) for idx, strategy in enumerate(order)}


def plot_parallel_coordinates(frame: pd.DataFrame, z_cols: list[str], out_dir: Path) -> Path:
    order = _strategy_order(frame)
    colors = _colors(order)
    x = np.arange(len(z_cols))

    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    for strategy in order:
        subset = frame[frame["strategy_run"] == strategy]
        label = STRATEGY_LABELS.get(strategy, strategy)
        for row_idx, row in enumerate(subset[z_cols].to_numpy()):
            ax.plot(
                x,
                row,
                color=colors[strategy],
                alpha=0.16,
                linewidth=1.2,
                label=label if row_idx == 0 else None,
            )
        median = subset[z_cols].median().to_numpy()
        ax.plot(x, median, color=colors[strategy], linewidth=2.5)

    ax.set_xticks(x, [f"z{i}" for i in range(len(z_cols))])
    ax.set_ylim(-2, 102)
    ax.set_ylabel("Coordinate value")
    ax.set_title("10D Positions as Parallel Coordinates")
    ax.legend(ncols=3, loc="upper center", bbox_to_anchor=(0.5, -0.10))
    fig.tight_layout()
    path = out_dir / "parallel_coordinates_10d.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_coordinate_heatmap(frame: pd.DataFrame, z_cols: list[str], out_dir: Path) -> Path:
    order = _strategy_order(frame)
    sorted_frame = (
        frame.assign(_strategy_rank=frame["strategy_run"].map({s: i for i, s in enumerate(order)}))
        .sort_values(["_strategy_rank", "peak_id", "score"], ascending=[True, True, False])
        .reset_index(drop=True)
    )
    matrix = sorted_frame[z_cols].to_numpy()

    fig, ax = plt.subplots(figsize=(8.8, 8.2))
    image = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0, vmax=100)
    ax.set_xticks(np.arange(len(z_cols)), [f"z{i}" for i in range(len(z_cols))])
    ax.set_ylabel("Agents sorted by strategy, peak, score")
    ax.set_title("10D Coordinate Heatmap")

    boundaries = []
    labels = []
    cursor = 0
    for strategy in order:
        count = int((sorted_frame["strategy_run"] == strategy).sum())
        if count:
            boundaries.append(cursor + count)
            labels.append((cursor + count / 2, STRATEGY_LABELS.get(strategy, strategy)))
            cursor += count
    for boundary in boundaries[:-1]:
        ax.axhline(boundary - 0.5, color="white", linewidth=1.2)
    ax.set_yticks([position for position, _ in labels], [label for _, label in labels])
    cbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.025)
    cbar.set_label("Coordinate value")
    fig.tight_layout()
    path = out_dir / "coordinate_heatmap_10d.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_pca_projection(
    agents: pd.DataFrame,
    peaks: pd.DataFrame | None,
    z_cols: list[str],
    out_dir: Path,
) -> Path:
    agent_coords = agents[z_cols].to_numpy(dtype=float)
    peak_coords = np.empty((0, len(z_cols)))
    if peaks is not None and not peaks.empty:
        first_strategy = str(peaks["strategy"].iloc[0])
        first_seed = int(peaks["seed"].iloc[0])
        peak_subset = peaks[
            (peaks["strategy"] == first_strategy) & (peaks["seed"] == first_seed)
        ]
        mu_cols = [f"mu{i}" for i in range(len(z_cols))]
        peak_coords = peak_subset[mu_cols].to_numpy(dtype=float)

    combined = np.vstack([agent_coords, peak_coords]) if len(peak_coords) else agent_coords
    projected = _pca_2d(combined)
    agent_projected = projected[: len(agent_coords)]
    peak_projected = projected[len(agent_coords) :]

    order = _strategy_order(agents)
    colors = _colors(order)
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    for strategy in order:
        subset = agents["strategy_run"] == strategy
        ax.scatter(
            agent_projected[subset, 0],
            agent_projected[subset, 1],
            s=20,
            alpha=0.58,
            color=colors[strategy],
            label=STRATEGY_LABELS.get(strategy, strategy),
        )

    if len(peak_projected):
        ax.scatter(
            peak_projected[:, 0],
            peak_projected[:, 1],
            s=150,
            marker="*",
            color="black",
            edgecolor="white",
            linewidth=0.8,
            label="Hidden peaks",
            zorder=5,
        )

    ax.set_xlabel("PC1 projection")
    ax.set_ylabel("PC2 projection")
    ax.set_title("2D PCA Projection of the 10D Space")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()
    path = out_dir / "pca_projection_10d.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _pca_2d(matrix: np.ndarray) -> np.ndarray:
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:2].T
    return centered @ components


if __name__ == "__main__":
    main()
