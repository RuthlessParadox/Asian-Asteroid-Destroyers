import argparse
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EvaluationArgs:
    history_path: Path
    logbook_path: Path
    scenarios_path: Path


def get_evaluation_config() -> EvaluationArgs:
    parser = argparse.ArgumentParser(
        description="Evaluate the performance of the elite individual from "
        + "training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    _ = parser.add_argument(
        "--history-path",
        type=str,
        required=True,
        help="Path to the history .parquet file",
    )
    _ = parser.add_argument(
        "--logbook-path",
        type=str,
        required=True,
        help="Path to the logbook .parquet file",
    )
    _ = parser.add_argument(
        "--scenarios-path",
        type=str,
        required=True,
        help="Path to the scenarios .json file",
    )

    namespace = parser.parse_args()

    history_path = Path(
        namespace.history_path  # pyright: ignore[reportAny]
    )
    logbook_path = Path(
        namespace.logbook_path  # pyright: ignore[reportAny]
    )
    scenarios_path = Path(
        namespace.scenarios_path  # pyright: ignore[reportAny]
    )

    if not history_path.is_file():
        raise FileNotFoundError(f"History file not found at: {history_path}")
    if not logbook_path.is_file():
        raise FileNotFoundError(f"Logbook file not found at: {logbook_path}")
    if not scenarios_path.is_file():
        raise FileNotFoundError(
            f"Scenarios file not found at: {scenarios_path}"
        )

    return EvaluationArgs(
        history_path=history_path,
        logbook_path=logbook_path,
        scenarios_path=scenarios_path,
    )
