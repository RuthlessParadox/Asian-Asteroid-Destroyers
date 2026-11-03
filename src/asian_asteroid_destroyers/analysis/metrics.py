from __future__ import annotations

from typing import TYPE_CHECKING, Final, cast

import polars as pl

from asian_asteroid_destroyers.chromosomes import FlatGenome

if TYPE_CHECKING:
    from pathlib import Path


class EvolutionMetrics:
    _logbook_lazyframe: Final[pl.LazyFrame]
    _history_lazyframe: Final[pl.LazyFrame]

    def __init__(
        self,
        logbook_path: str | Path,
        history_path: str | Path,
    ) -> None:
        self._logbook_lazyframe = pl.scan_parquet(logbook_path)
        self._history_lazyframe = pl.scan_parquet(history_path)

    def extract_best_genome(self, generation: int | None = None) -> FlatGenome:
        query = self._history_lazyframe

        if generation is not None:
            query = query.filter(pl.col("generation") == generation)
        else:
            query = query.sort("generation", descending=True).limit(1)

        processed_query = (
            query.explode("hall_of_fame")
            .select(
                pl.col("generation"),
                pl.col("hall_of_fame").struct.field("rank").alias("rank"),
                pl.col("hall_of_fame").struct.field("genome").alias("genome"),
            )
            .filter(pl.col("rank") == 1)
            .select("genome")
        )

        result_dataframe = processed_query.collect()

        if result_dataframe.is_empty():
            raise ValueError(
                "Could not resolve an elite genome configuration. "
                + "Query returned empty dataframe for generation filter: "
                + f"{generation}"
            )

        return cast(
            FlatGenome,
            result_dataframe.get_column("genome").to_list()[0],
        )

    def collect_training_summary(self) -> pl.DataFrame:
        return (
            self._logbook_lazyframe.select(
                "generation",
                "max_fitness",
                "mean_fitness",
                "min_fitness",
            )
            .sort("generation")
            .collect()
        )

    def compute_hall_of_fame_turnover(self) -> pl.DataFrame:
        sorted_history = self._history_lazyframe.sort("generation").collect()

        if sorted_history.is_empty():
            return pl.DataFrame(
                schema={
                    "generation": pl.Int64(),
                    "turnover_rate": pl.Float64(),
                }
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

        for index, gen in enumerate(generations):
            current_raw: list[list[float]] = (
                exploded.filter(pl.col("generation") == gen)
                .get_column("genome")
                .to_list()
            )
            current_genomes = {tuple(gene) for gene in current_raw}

            if index == 0:
                turnover_rate = 0.0
            else:
                new_entrants = current_genomes - previous_set
                turnover_rate = len(new_entrants) / max(
                    1,
                    len(current_genomes),
                )

            turnover_records.append(
                {"generation": gen, "turnover_rate": turnover_rate},
            )
            previous_set = current_genomes

        return pl.DataFrame(
            turnover_records,
            schema={"generation": pl.Int64(), "turnover_rate": pl.Float64()},
        )

    def collect_geometry_drift(self, gene_indices: list[int]) -> pl.DataFrame:
        flattened_query = (
            self._history_lazyframe.explode("hall_of_fame")
            .select(
                pl.col("generation"),
                pl.col("hall_of_fame").struct.field("rank").alias("rank"),
                pl.col("hall_of_fame").struct.field("genome").alias("genome"),
            )
            .filter(pl.col("rank") == 1)
        )

        projections = [pl.col("generation")]
        for index in gene_indices:
            projections.append(
                pl.col("genome").list.get(index).alias(f"gene_{index}"),
            )

        return flattened_query.select(projections).sort("generation").collect()

    def collect_param_volatility(self) -> pl.DataFrame:
        return (
            self._history_lazyframe.explode("hall_of_fame")
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
