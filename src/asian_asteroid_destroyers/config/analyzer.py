import argparse
from dataclasses import dataclass
from pathlib import Path

from .parsers import parse_non_negative_int


@dataclass(frozen=True)
class AnalyzerArgs:
    run_dir: Path
    track_genes: list[int]


def get_analyzer_config() -> AnalyzerArgs:
    parser = argparse.ArgumentParser(
        description="Extract metrics and render analytics for a training run.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    _ = parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Path to targeted experiment workspace directory",
    )
    _ = parser.add_argument(
        "--track-genes",
        type=parse_non_negative_int,
        nargs="+",
        default=[],
        help="Space-separated list of flat gene indices to trace for drift",
    )

    namespace = parser.parse_args()

    run_dir = Path(namespace.run_dir)  # pyright: ignore[reportAny]
    track_genes: list[int] = (  # pyright: ignore[reportAny]
        namespace.track_genes
    )

    if not run_dir.exists():
        raise FileNotFoundError(
            f"Target workspace path does not exist: {run_dir}"
        )

    if len(track_genes) != len(set(track_genes)):
        raise ValueError("Duplicate gene indices are not allowed.")

    return AnalyzerArgs(
        run_dir=run_dir,
        track_genes=track_genes,
    )
