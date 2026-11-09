# pyright: reportUnknownMemberType=false
from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

if TYPE_CHECKING:
    import polars as pl
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


def plot_fitness_tracks(summary_df: pl.DataFrame) -> tuple[Figure, Axes]:
    generations = summary_df.get_column("generation").to_numpy()  # type: ignore[arg-type]
    max_fitnesses = summary_df.get_column("max_fitness").to_numpy()  # type: ignore[arg-type]
    mean_fitnesses = summary_df.get_column("mean_fitness").to_numpy()  # type: ignore[arg-type]
    min_fitnesses = summary_df.get_column("min_fitness").to_numpy()  # type: ignore[arg-type]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    _ = ax1.plot(
        generations,
        max_fitnesses,
        label="Max Fitness",
        color="#10b981",
        lw=2,
    )
    _ = ax1.plot(
        generations,
        mean_fitnesses,
        label="Mean Fitness",
        color="#3b82f6",
        lw=1.5,
    )
    _ = ax1.plot(
        generations,
        min_fitnesses,
        label="Min Fitness",
        color="#ef4444",
        lw=1,
    )
    _ = ax1.fill_between(
        generations,
        min_fitnesses,
        max_fitnesses,
        color="#3b82f6",
        alpha=0.1,
        label="Spread",
    )

    _ = ax1.set_title(
        "Evolutionary Fitness Trajectory & Pareto Frontier Scaling",
    )
    _ = ax1.set_xlabel("Generation")
    _ = ax1.set_ylabel("Scalar Aggregated Fitness")
    _ = ax1.legend(loc="upper left")
    ax1.grid(True, linestyle="--", alpha=0.6)

    if "front_size" in summary_df.columns:
        ax2 = ax1.twinx()
        front_sizes = summary_df.get_column("front_size").to_numpy()  # type: ignore[arg-type]
        _ = ax2.plot(
            generations,
            front_sizes,
            color="#6b7280",
            linestyle=":",
            label="Pareto Front Size",
            alpha=0.8,
        )
        _ = ax2.set_ylabel(
            "Non-Dominated Individual Count (Front Size)",
            color="#6b7280",
        )
        _ = ax2.tick_params(axis="y", labelcolor="#6b7280")
        _ = ax2.legend(loc="upper right")

    return fig, ax1


def plot_param_volatility(volatility_df: pl.DataFrame) -> tuple[Figure, Axes]:
    fig, ax = plt.subplots(figsize=(10, 6))

    generations = volatility_df.get_column("generation").to_numpy()  # type: ignore[arg-type]
    param_stds = volatility_df.get_column("param_std").to_numpy()  # type: ignore[arg-type]

    _ = ax.plot(
        generations,
        param_stds,
        color="#8b5cf6",
        linestyle="-.",
        label="Front Variant Volatility (Std Dev)",
    )

    _ = ax.set_title("Consequent Coefficient & Parameter Variance Trajectory")
    _ = ax.set_xlabel("Generation")
    _ = ax.set_ylabel("Standard Deviation Spread Across Front")
    _ = ax.legend(loc="best")
    ax.grid(True, linestyle="--", alpha=0.5)

    return fig, ax


def plot_hall_of_fame_turnover(
    turnover_df: pl.DataFrame,
) -> tuple[Figure, Axes]:
    fig, ax = plt.subplots(figsize=(10, 6))

    generations = turnover_df.get_column("generation").to_numpy()  # type: ignore[arg-type]
    rates = turnover_df.get_column("turnover_rate").to_numpy()  # type: ignore[arg-type]

    _ = ax.plot(
        generations,
        rates,
        color="#f59e0b",
        label="Pareto Set Frontier Replacement Rate",
    )

    _ = ax.set_title("Pareto Front Strategy Turnover Tracking")
    _ = ax.set_xlabel("Generation")
    _ = ax.set_ylabel("Turnover Ratio (New Frontiersmen / Total Front)")
    _ = ax.legend(loc="best")
    ax.grid(True, linestyle="--", alpha=0.5)

    return fig, ax


def plot_geometry_drift(
    drift_df: pl.DataFrame,
    gene_indices: list[int],
) -> tuple[Figure, Axes]:
    fig, ax = plt.subplots(figsize=(10, 6))

    generations = drift_df.get_column("generation").to_numpy()  # type: ignore[arg-type]
    palette = ["#ec4899", "#3b82f6", "#10b981", "#8b5cf6", "#f59e0b"]

    for idx, gene_idx in enumerate(gene_indices):
        column_name = f"gene_{gene_idx}"
        coordinates = drift_df.get_column(column_name).to_numpy()  # type: ignore[arg-type]

        _ = ax.plot(
            generations,
            coordinates,
            color=palette[idx % len(palette)],
            label=f"Centroid Gene Offset: {gene_idx}",
        )

    _ = ax.set_title(
        "Linguistic Term Geometry Drift Trajectory (Front Average)",
    )
    _ = ax.set_xlabel("Generation")
    _ = ax.set_ylabel("Chromosome Coordinate Value")
    _ = ax.legend(loc="best")
    ax.grid(True, linestyle="--", alpha=0.5)

    return fig, ax
