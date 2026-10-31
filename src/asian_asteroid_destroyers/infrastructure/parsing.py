from pathlib import Path


def extract_generation_number(path: Path) -> int:
    try:
        raw_generation = path.stem.split("_gen_")[1].split(".")[0]
        return int(raw_generation)
    except (IndexError, ValueError) as error:
        raise ValueError(
            "Malformed or corrupted checkpoint filename encountered: "
            + f"'{path.name}'. Ensure file follows the "
            + "'runtime_gen_<integer>.pkl.gz' convention."
        ) from error
