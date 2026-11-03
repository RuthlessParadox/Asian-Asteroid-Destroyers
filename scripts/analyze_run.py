# pyright: reportUnknownMemberType=false
from __future__ import annotations

import pathlib
import sys

from asian_asteroid_destroyers.analysis import (
    EvolutionMetrics,
    plot_fitness_tracks,
    plot_geometry_drift,
    plot_hall_of_fame_turnover,
    plot_param_volatility,
)
from asian_asteroid_destroyers.infrastructure import (
    EMERGENCY_HISTORY_FILENAME,
    EMERGENCY_LOGBOOK_FILENAME,
    get_analyzer_config,
)


def main() -> int:
    args = get_analyzer_config()
    run_path = pathlib.Path(args.run_dir)

    if not run_path.exists():
        print(f"Error: Target workspace path does not exist: {run_path}")
        return 1

    logbook_file = run_path / "final_logbook.parquet"
    history_file = run_path / "final_history.parquet"

    if not logbook_file.exists():
        logbook_file = run_path / EMERGENCY_LOGBOOK_FILENAME
    if not history_file.exists():
        history_file = run_path / EMERGENCY_HISTORY_FILENAME

    if not logbook_file.exists() or not history_file.exists():
        print(
            "Error: Required analytical assets missing inside: "
            + f"{run_path.name}"
        )
        return 1

    print(f"Commencing extraction sweep over workspace: {run_path.name}...")
    metrics_engine = EvolutionMetrics(
        logbook_path=logbook_file,
        history_path=history_file,
    )

    print(" -> Compiling historical convergence tracks...")
    summary_df = metrics_engine.collect_training_summary()
    fit_fig, _ = plot_fitness_tracks(summary_df)
    fit_fig.savefig(str(run_path / "analytics_fitness_tracks.png"), dpi=200)

    print(" -> Evaluating internal parameter spread indices...")
    vol_df = metrics_engine.collect_param_volatility()
    vol_fig, _ = plot_param_volatility(vol_df)
    vol_fig.savefig(str(run_path / "analytics_param_volatility.png"), dpi=200)

    print(" -> Computing elite structural replacement rates...")
    turnover_df = metrics_engine.compute_hall_of_fame_turnover()
    turnover_fig, _ = plot_hall_of_fame_turnover(turnover_df)
    turnover_fig.savefig(str(run_path / "analytics_hof_turnover.png"), dpi=200)

    if args.track_genes:
        print(
            f" -> Mapping spatial drift paths for indices: {args.track_genes}..."
        )
        drift_df = metrics_engine.collect_geometry_drift(args.track_genes)
        drift_fig, _ = plot_geometry_drift(drift_df, args.track_genes)
        drift_fig.savefig(
            str(run_path / "analytics_geometry_drift.png"),
            dpi=200,
        )

    print(
        f"\nSuccess! Full dashboard assets secured to disk inside: {run_path}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
