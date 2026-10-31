import datetime
from pathlib import Path

POINTER_FILENAME = "active_checkpoint.txt"
EMERGENCY_RUNTIME_FILENAME = "emergency_runtime.pkl.gz"
PERIODIC_RUNTIME_PATTERN = "runtime_gen_*.pkl.gz"


def create_experiment_directory(base_directory: str | Path = "./data") -> Path:
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_directory = Path(base_directory) / f"run_{timestamp}"
    run_directory.mkdir(parents=True, exist_ok=True)
    return run_directory


def update_active_pointer(
    directory: Path,
    current_checkpoint_name: str,
) -> None:
    pointer_path = directory / POINTER_FILENAME
    try:
        temp_pointer = pointer_path.with_suffix(".tmp")
        _ = temp_pointer.write_text(current_checkpoint_name, encoding="utf-8")
        _ = temp_pointer.replace(pointer_path)
    except OSError as error:
        print(f"Warning: Could not rotate active checkpoint pointer: {error}")


def get_periodic_runtime_filename(generation: int) -> str:
    return f"runtime_gen_{generation}.pkl.gz"
