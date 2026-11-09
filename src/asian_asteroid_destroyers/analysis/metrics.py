from __future__ import annotations

from typing import TYPE_CHECKING, Final

import numpy as np
import polars as pl

if TYPE_CHECKING:
    from pathlib import Path

    import numpy.typing as npt

    from asian_asteroid_destroyers.chromosomes import FlatGenome


class EvolutionMetrics:
    _logbook_lazyframe: Final[pl.LazyFrame]
    _history_lazyframe: Final[pl.LazyFrame]

    def __init__(
        self,
        logbook_path: str | Path,
        history_path: str | Path,
    ) -> None:
        self._logbook_lazyframe = pl.scan_parquet(logbook_path)  # type: ignore[arg-type]
        self._history_lazyframe = pl.scan_parquet(history_path)  # type: ignore[arg-type]

    def extract_pareto_front_genomes(
        self,
        generation: int | None = None,
    ) -> list[FlatGenome]:
        query = self._history_lazyframe

        if generation is not None:
            query = query.filter(pl.col("generation") == generation)
        else:
            query = query.sort("generation", descending=True).limit(1)

        processed_query = query.explode("hall_of_fame").select(
            pl.col("hall_of_fame").struct.field("genome").alias("genome"),
        )

        result_dataframe = processed_query.collect()  # type: ignore[arg-type]

        if result_dataframe.is_empty():
            raise ValueError(
                "Could not resolve any front configurations for "
                + f"generation filter: {generation}"
            )

        return result_dataframe.get_column("genome").to_list()

    def collect_training_summary(self) -> pl.DataFrame:
        summary_lf = self._logbook_lazyframe.select(
            "generation",
            "max_fitness",
            "mean_fitness",
            "min_fitness",
        ).sort("generation")

        front_size_lf = self._history_lazyframe.select(
            "generation",
            pl.col("hall_of_fame").list.len().alias("front_size"),
        )

        return summary_lf.join(  # type: ignore[arg-type]
            front_size_lf,
            on="generation",
            how="left",
        ).collect()

    def compute_hall_of_fame_turnover(self) -> pl.DataFrame:
        sorted_history = self._history_lazyframe.sort("generation").collect()  # type: ignore[arg-type]

        if sorted_history.is_empty():
            return pl.DataFrame(
                schema={
                    "generation": pl.Int64(),
                    "turnover_rate": pl.Float64(),
                },
            )

        generations: list[int] = sorted_history.get_column(
            "generation"
        ).to_list()
        exploded = sorted_history.explode("hall_of_fame").select(
            pl.col("generation"),
            pl.col("hall_of_fame").struct.field("genome").alias("genome"),
        )

        turnover_records: list[dict[str, int | float]] = []
        previous_set: set[tuple[float, ...]] = set()

        for index, generation in enumerate(generations):
            current_raw: list[FlatGenome] = (
                exploded.filter(pl.col("generation") == generation)
                .get_column("genome")
                .to_list()
            )
            current_genomes = {tuple(genome) for genome in current_raw}

            if index == 0:
                turnover_rate = 0.0
            else:
                new_entrants = current_genomes - previous_set
                turnover_rate = len(new_entrants) / max(
                    1, len(current_genomes)
                )

            turnover_records.append(
                {"generation": generation, "turnover_rate": turnover_rate},
            )
            previous_set = current_genomes

        return pl.DataFrame(
            turnover_records,
            schema={"generation": pl.Int64(), "turnover_rate": pl.Float64()},
        )

    def collect_geometry_drift(self, gene_indices: list[int]) -> pl.DataFrame:
        flattened_query = self._history_lazyframe.explode(
            "hall_of_fame"
        ).select(
            pl.col("generation"),
            pl.col("hall_of_fame").struct.field("genome").alias("genome"),
        )

        projections = [pl.col("generation")]
        for index in gene_indices:
            projections.append(
                pl.col("genome").list.get(index).alias(f"gene_{index}"),
            )

        return (
            flattened_query.select(projections)  # type: ignore[arg-type]
            .group_by("generation")
            .agg(
                [
                    pl.col(f"gene_{index}").mean().alias(f"gene_{index}")
                    for index in gene_indices
                ]
            )
            .sort("generation")
            .collect()
        )

    def collect_param_volatility(self) -> pl.DataFrame:
        return (
            self._history_lazyframe.explode("hall_of_fame")  # type: ignore[arg-type]
            .select(
                pl.col("generation"),
                pl.col("hall_of_fame").struct.field("genome").alias("genome"),
            )
            .explode("genome")
            .group_by("generation")
            .agg(pl.col("genome").std().alias("param_std"))
            .sort("generation")
            .collect()
        )

    def extract_weighted_best_genome(
        self,
        weights: list[float] | npt.NDArray[np.float64] | None = None,
        generation: int | None = None,
    ) -> FlatGenome:
        query = self._history_lazyframe

        if generation is not None:
            query = query.filter(pl.col("generation") == generation)
        else:
            query = query.sort("generation", descending=True).limit(1)

        generation_df = (
            query.explode("hall_of_fame")  # type: ignore[arg-type]
            .select(
                pl.col("hall_of_fame").struct.field("genome").alias("genome"),
                pl.col("hall_of_fame")
                .struct.field("fitness")
                .alias("fitness"),
            )
            .collect()
        )

        if generation_df.is_empty():
            raise ValueError(
                "Could not resolve history snapshots to compute "
                + f"weighted best coordinates for generation: {generation}"
            )

        front_genomes: list[FlatGenome] = generation_df.get_column(
            "genome"
        ).to_list()

        raw_fitness: list[list[float]] = generation_df.get_column(
            "fitness"
        ).to_list()
        fitness_matrix = np.array(raw_fitness, dtype=np.float64)

        ideal_target = np.array(
            [
                np.max(fitness_matrix[:, 0]),
                np.max(fitness_matrix[:, 1]),
                np.max(fitness_matrix[:, 2]),
                np.max(fitness_matrix[:, 3]),
                np.max(fitness_matrix[:, 4]),
            ],
            dtype=np.float64,
        )

        min_bounds: npt.NDArray[np.float64] = np.min(fitness_matrix, axis=0)  # pyright: ignore[reportAny]
        ranges = ideal_target - min_bounds
        ranges[ranges == 0.0] = 1.0

        normalized_deltas = (fitness_matrix - ideal_target) / ranges

        if weights is not None:
            weight_array = np.array(weights, dtype=np.float64)
            if weight_array.shape[0] != fitness_matrix.shape[1]:
                raise ValueError(
                    f"Length of weights ({weight_array.shape[0]}) must match "
                    + f"number of fitness objectives ({fitness_matrix.shape[1]})."
                )
            weighted_deltas = normalized_deltas * weight_array
        else:
            weighted_deltas = normalized_deltas

        distances: npt.NDArray[np.float64] = np.linalg.norm(  # pyright: ignore[reportAny]
            weighted_deltas,
            axis=1,
        )
        best_index = int(np.argmin(distances))

        return front_genomes[best_index]

    def extract_knee_point_genome(self) -> FlatGenome:
        return self.extract_weighted_best_genome(weights=None, generation=None)
