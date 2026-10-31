from pathlib import Path

import pytest

from asian_asteroid_destroyers.infrastructure import extract_generation_number


def test_extract_generation_number_successfully_parses_valid_filename() -> (
    None
):
    valid_path = Path("/some/dir/runtime_gen_42.pkl.gz")
    generation = extract_generation_number(valid_path)

    assert generation == 42


def test_extract_generation_number_handles_large_and_zero_values() -> None:
    zero_path = Path("runtime_gen_0.pkl.gz")
    large_path = Path("runtime_gen_10500.pkl.gz")

    assert extract_generation_number(zero_path) == 0
    assert extract_generation_number(large_path) == 10500


def test_extract_generation_number_raises_value_error_on_missing_token() -> (
    None
):
    malformed_path = Path("runtime_checkpoint_42.pkl.gz")

    with pytest.raises(
        ValueError,
        match="Malformed or corrupted checkpoint filename encountered",
    ):
        _ = extract_generation_number(malformed_path)


def test_extract_generation_number_raises_value_error_on_non_integer_value() -> (
    None
):
    corrupted_path = Path("runtime_gen_final_run.pkl.gz")

    with pytest.raises(
        ValueError,
        match="Malformed or corrupted checkpoint filename encountered",
    ):
        _ = extract_generation_number(corrupted_path)


def test_extract_generation_number_handles_complex_or_relative_paths() -> None:
    complex_path = Path("./data/run_20261030/../runtime_gen_7.pkl.gz")

    assert extract_generation_number(complex_path) == 7
