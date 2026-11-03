# pyright: reportUnknownMemberType=false
from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

if TYPE_CHECKING:
    import polars as pl
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


def plot_fitness_tracks(summary_df: pl.DataFrame) -> tuple[Figure, Axes]:
    generations = summary_df.get_column("generation").to_numpy()
    max_fitnesses = summary_df.get_column("max_fitness").to_numpy()
    mean_fitnesses = summary_df.get_column("mean_fitness").to_numpy()
    min_fitnesses = summary_df.get_column("min_fitness").to_numpy()

    fig, ax = plt.subplots(figsize=(10, 6))

    _ = ax.plot(
        generations,
        max_fitnesses,
        label="Max Fitness",
        color="#10b981",
    )
    _ = ax.plot(
        generations,
        mean_fitnesses,
        label="Mean Fitness",
        color="#3b82f6",
    )
    _ = ax.plot(
        generations,
        min_fitnesses,
        label="Min Fitness",
        color="#ef4444",
    )
    _ = ax.fill_between(
        generations,
        min_fitnesses,
        max_fitnesses,
        color="#3b82f6",
        alpha=0.1,
        label="Spread",
    )

    _ = ax.set_title("Evolutionary Fitness Trajectory Across Generations")
    _ = ax.set_xlabel("Generation")
    _ = ax.set_ylabel("Fitness Value")
    _ = ax.legend(loc="best")
    ax.grid(True, linestyle="--", alpha=0.6)

    return fig, ax


def plot_param_volatility(volatility_df: pl.DataFrame) -> tuple[Figure, Axes]:
    fig, ax = plt.subplots(figsize=(10, 6))

    generations = volatility_df.get_column("generation").to_numpy()
    param_stds = volatility_df.get_column("param_std").to_numpy()

    _ = ax.plot(
        generations,
        param_stds,
        color="#8b5cf6",
        linestyle="-.",
        label="HoF Vector Volatility (Std Dev)",
    )

    _ = ax.set_title("Consequent Coefficient & Parameter Variance Trajectory")
    _ = ax.set_xlabel("Generation")
    _ = ax.set_ylabel("Standard Deviation Spread")
    _ = ax.legend(loc="best")
    ax.grid(True, linestyle="--", alpha=0.5)

    return fig, ax


def plot_hall_of_fame_turnover(
    turnover_df: pl.DataFrame,
) -> tuple[Figure, Axes]:
    fig, ax = plt.subplots(figsize=(10, 6))

    generations = turnover_df.get_column("generation").to_numpy()
    rates = turnover_df.get_column("turnover_rate").to_numpy()

    _ = ax.plot(
        generations,
        rates,
        color="#f59e0b",
        label="HoF Strategy Replacement Turnover Rate",
    )

    _ = ax.set_title("Hall of Fame Strategy Turnover Tracking")
    _ = ax.set_xlabel("Generation")
    _ = ax.set_ylabel("Turnover Ratio (New / Capacity)")
    _ = ax.legend(loc="best")
    ax.grid(True, linestyle="--", alpha=0.5)

    return fig, ax


def plot_geometry_drift(
    drift_df: pl.DataFrame,
    gene_indices: list[int],
) -> tuple[Figure, Axes]:
    """Plots linguistic coordinate migration trajectories over generations."""
    fig, ax = plt.subplots(figsize=(10, 6))

    generations = drift_df.get_column("generation").to_numpy()
    palette = ["#ec4899", "#3b82f6", "#10b981", "#8b5cf6", "#f59e0b"]

    for idx, gene_idx in enumerate(gene_indices):
        column_name = f"gene_{gene_idx}"
        coordinates = drift_df.get_column(column_name).to_numpy()

        _ = ax.plot(
            generations,
            coordinates,
            color=palette[idx % len(palette)],
            label=f"Gene Position Offset: {gene_idx}",
        )

    _ = ax.set_title("Linguistic Term Geometry Drift Trajectory")
    _ = ax.set_xlabel("Generation")
    _ = ax.set_ylabel("Chromosome Coordinate Value")
    _ = ax.legend(loc="best")
    ax.grid(True, linestyle="--", alpha=0.5)

    return fig, ax
