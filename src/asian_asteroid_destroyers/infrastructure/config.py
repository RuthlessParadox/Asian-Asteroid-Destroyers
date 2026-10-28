import datetime
from pathlib import Path


def create_experiment_directory(base_directory: str | Path = "./data") -> Path:
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_directory = Path(base_directory) / f"run_{timestamp}"
    run_directory.mkdir(parents=True, exist_ok=True)
    return run_directory
