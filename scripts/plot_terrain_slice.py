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


LABELS = {
    "score_following": "Score following",
    "full_collaboration": "Full collab",
    "strategic_collaboration": "Strategic collab",
    "independent_search": "Independent",
    "random": "Random",
}

COLORS = {
    "score_following": "#1f77b4",
    "full_collaboration": "#ff7f0e",
    "strategic_collaboration": "#2ca02c",
    "independent_search": "#d62728",
    "random": "#9467bd",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot a 2D terrain slice of a 10D landscape.")
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--resolution", type=int, default=170)
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["score_following", "full_collaboration", "strategic_collaboration", "independent_search"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out or args.results / "figures_10d"
    out_dir.mkdir(parents=True, exist_ok=True)

    peaks = pd.read_csv(args.results / "peaks.csv")
    agents_path = args.results / "agent_scores.csv"
    agents = pd.read_csv(agents_path) if agents_path.exists() else None

    first_strategy = str(peaks["strategy"].iloc[0])
    first_seed = int(peaks["seed"].iloc[0])
    peaks = peaks[(peaks["strategy"] == first_strategy) & (peaks["seed"] == first_seed)]

    mu_cols = sorted(
        [col for col in peaks.columns if col.startswith("mu")],
        key=lambda col: int(col[2:]),
    )
    centers = peaks[mu_cols].to_numpy(dtype=float)
    heights = peaks["height"].to_numpy(dtype=float)
    widths = peaks["width"].to_numpy(dtype=float)

    d0, d1 = choose_display_dimensions(centers)
    anchor = centers[int(np.argmax(heights))].copy()
    x, y, z = landscape_slice(centers, heights, widths, anchor, d0, d1, args.resolution)

    path = plot_slice(
        x=x,
        y=y,
        z=z,
        centers=centers,
        agents=agents,
        d0=d0,
        d1=d1,
        strategies=args.strategies,
        out_dir=out_dir,
    )
    print(path)


def choose_display_dimensions(centers: np.ndarray) -> tuple[int, int]:
    ranges = centers.max(axis=0) - centers.min(axis=0)
    dims = np.argsort(ranges)[-2:]
    return int(dims[0]), int(dims[1])


def landscape_slice(
    centers: np.ndarray,
    heights: np.ndarray,
    widths: np.ndarray,
    anchor: np.ndarray,
    d0: int,
    d1: int,
    resolution: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = np.linspace(0, 100, resolution)
    ys = np.linspace(0, 100, resolution)
    x, y = np.meshgrid(xs, ys)
    points = np.tile(anchor, (x.size, 1))
    points[:, d0] = x.ravel()
    points[:, d1] = y.ravel()
    deltas = points[:, None, :] - centers[None, :, :]
    squared_l2 = np.sum(deltas * deltas, axis=2)
    values = heights[None, :] * np.exp(-squared_l2 / (2.0 * widths[None, :] ** 2))
    z = values.max(axis=1).reshape(x.shape)
    return x, y, z


def plot_slice(
    *,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    centers: np.ndarray,
    agents: pd.DataFrame | None,
    d0: int,
    d1: int,
    strategies: list[str],
    out_dir: Path,
) -> Path:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 180,
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
        }
    )

    levels = np.linspace(float(z.min()), float(z.max()), 18)
    fig = plt.figure(figsize=(14, 5.8))

    ax1 = fig.add_subplot(1, 2, 1)
    contour = ax1.contourf(x, y, z, levels=levels, cmap="terrain")
    ax1.contour(x, y, z, levels=levels[::2], colors="black", alpha=0.28, linewidths=0.8)
    ax1.scatter(
        centers[:, d0],
        centers[:, d1],
        marker="*",
        s=150,
        c="white",
        edgecolor="black",
        linewidth=0.9,
        label="Hidden peaks projected",
    )

    if agents is not None:
        for strategy in strategies:
            group = agents[agents["strategy_run"] == strategy]
            if group.empty:
                continue
            sample = group.sample(n=min(45, len(group)), random_state=3)
            ax1.scatter(
                sample[f"z{d0}"],
                sample[f"z{d1}"],
                s=16,
                alpha=0.58,
                c=COLORS.get(strategy, "#444444"),
                label=LABELS.get(strategy, strategy),
            )

    ax1.set_xlabel(f"z{d0}")
    ax1.set_ylabel(f"z{d1}")
    ax1.set_title("Topographic Slice of the 10D Value Landscape")
    ax1.legend(loc="upper right", fontsize=8)
    colorbar = fig.colorbar(contour, ax=ax1, fraction=0.045, pad=0.04)
    colorbar.set_label("Hidden value V(z)")

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    stride = 3
    ax2.plot_surface(
        x[::stride, ::stride],
        y[::stride, ::stride],
        z[::stride, ::stride],
        cmap="terrain",
        linewidth=0,
        antialiased=True,
        alpha=0.94,
    )
    ax2.contour(
        x,
        y,
        z,
        zdir="z",
        offset=float(z.min()) - 6,
        levels=levels[::2],
        cmap="terrain",
        alpha=0.75,
    )
    ax2.set_xlabel(f"z{d0}")
    ax2.set_ylabel(f"z{d1}")
    ax2.set_zlabel("Value")
    ax2.set_title("Same Slice as a Surface")
    ax2.view_init(elev=28, azim=-58)
    ax2.set_zlim(float(z.min()) - 6, float(z.max()) + 5)

    fig.suptitle(
        "A 2D Terrain Slice Through the 10D Peak-Divergence Landscape",
        fontsize=15,
        y=1.02,
    )
    fig.tight_layout()
    path = out_dir / "ten_dim_terrain_slice.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


if __name__ == "__main__":
    main()
