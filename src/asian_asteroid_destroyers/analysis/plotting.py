from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.text import Text

if TYPE_CHECKING:
    import polars as pl
    from matplotlib.axes import Axes


def _create_canvas() -> tuple[Figure, Axes]:
    fig = Figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1)
    return fig, ax


def _enable_strict_grid(ax: Axes) -> None:
    for line in ax.xaxis.get_gridlines():
        line.set_visible(True)
    for line in ax.yaxis.get_gridlines():
        line.set_visible(True)


def plot_fitness_tracks(
    summary_dataframe: pl.DataFrame,
) -> tuple[Figure, Axes]:
    generations = summary_dataframe.get_column("generation").to_numpy()
    max_fitnesses = summary_dataframe.get_column("max_fitness").to_numpy()
    mean_fitnesses = summary_dataframe.get_column("mean_fitness").to_numpy()
    min_fitnesses = summary_dataframe.get_column("min_fitness").to_numpy()

    fig, ax = _create_canvas()

    max_line = Line2D(
        generations,
        max_fitnesses,
        label="Max Fitness",
        color="#10b981",
    )
    mean_line = Line2D(
        generations,
        mean_fitnesses,
        label="Mean Fitness",
        color="#3b82f6",
    )
    min_line = Line2D(
        generations,
        min_fitnesses,
        label="Min Fitness",
        color="#ef4444",
    )

    _ = ax.add_line(max_line)
    _ = ax.add_line(mean_line)
    _ = ax.add_line(min_line)

    x_vertices = np.concatenate([generations, generations[::-1]])
    y_vertices = np.concatenate([min_fitnesses, max_fitnesses[::-1]])
    vertices = [list(zip(x_vertices, y_vertices))]

    poly = PolyCollection(vertices)
    poly.set_facecolor("#3b82f6")
    poly.set_alpha(0.1)
    poly.set_label("Spread")
    _ = ax.add_collection(poly)

    ax.relim()
    ax.autoscale_view()

    title_text = Text(
        x=0.5,
        y=0.95,
        text="Evolutionary Fitness Trajectory Across Generations",
    )
    title_text.set_horizontalalignment("center")
    _ = fig.add_artist(title_text)

    ax.xaxis.label.set_text("Generation")
    ax.yaxis.label.set_text("Fitness Value")

    _enable_strict_grid(ax)

    return fig, ax


def plot_param_volatility(
    volatility_dataframe: pl.DataFrame,
) -> tuple[Figure, Axes]:
    fig, ax = _create_canvas()

    generations = volatility_dataframe.get_column("generation").to_numpy()
    param_stds = volatility_dataframe.get_column("param_std").to_numpy()

    vol_line = Line2D(
        generations,
        param_stds,
        color="#8b5cf6",
        linestyle="-.",
        label="HoF Vector Volatility (Std Dev)",
    )

    _ = ax.add_line(vol_line)
    ax.relim()
    ax.autoscale_view()

    title_text = Text(
        x=0.5,
        y=0.95,
        text="Consequent Coefficient & Parameter Variance Trajectory",
    )
    title_text.set_horizontalalignment("center")
    _ = fig.add_artist(title_text)

    ax.xaxis.label.set_text("Generation")
    ax.yaxis.label.set_text("Standard Deviation Spread")

    _enable_strict_grid(ax)

    return fig, ax


def plot_hall_of_fame_turnover(
    turnover_dataframe: pl.DataFrame,
) -> tuple[Figure, Axes]:
    fig, ax = _create_canvas()

    generations = turnover_dataframe.get_column("generation").to_numpy()
    rates = turnover_dataframe.get_column("turnover_rate").to_numpy()

    turnover_line = Line2D(
        generations,
        rates,
        color="#f59e0b",  # Amber
        label="HoF Strategy Replacement Turnover Rate",
    )

    _ = ax.add_line(turnover_line)
    ax.relim()
    ax.autoscale_view()

    title_text = Text(
        x=0.5,
        y=0.95,
        text="Hall of Fame Strategy Turnover Tracking",
    )
    title_text.set_horizontalalignment("center")
    _ = fig.add_artist(title_text)

    ax.xaxis.label.set_text("Generation")
    ax.yaxis.label.set_text("Turnover Ratio (New / Capacity)")

    _enable_strict_grid(ax)

    return fig, ax


def plot_geometry_drift(
    drift_dataframe: pl.DataFrame,
    gene_indices: list[int],
) -> tuple[Figure, Axes]:
    fig, ax = _create_canvas()

    generations = drift_dataframe.get_column("generation").to_numpy()

    palette = ["#ec4899", "#3b82f6", "#10b981", "#8b5cf6", "#f59e0b"]

    for index, gene_index in enumerate(gene_indices):
        column_name = f"gene_{gene_index}"
        coordinates = drift_dataframe.get_column(column_name).to_numpy()

        line = Line2D(
            generations,
            coordinates,
            color=palette[index % len(palette)],
            label=f"Gene Position Offset: {gene_index}",
        )
        _ = ax.add_line(line)

    ax.relim()
    ax.autoscale_view()

    title_text = Text(
        x=0.5,
        y=0.95,
        text="Linguistic Term Geometry Drift Trajectory",
    )
    title_text.set_horizontalalignment("center")
    _ = fig.add_artist(title_text)

    ax.xaxis.label.set_text("Generation")
    ax.yaxis.label.set_text("Chromosome Coordinate Value")

    _enable_strict_grid(ax)

    return fig, ax
