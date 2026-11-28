import numpy as np
import pytest

from asian_asteroid_destroyers.evokit import derive_generation_seed


def test_deterministic_for_same_inputs() -> None:
    seed_a = derive_generation_seed(master_seed=42, generation=3)
    seed_b = derive_generation_seed(master_seed=42, generation=3)

    assert isinstance(seed_a, int)
    assert seed_a == seed_b


@pytest.mark.parametrize(
    ("master_seed", "generation"),
    [
        (42, 0),
        (42, 1),
        (42, 2),
        (43, 0),
        (0, 1),
    ],
)
def test_distinct_seeds_for_different_inputs(
    master_seed: int,
    generation: int,
) -> None:
    seeds: list[int] = [
        derive_generation_seed(master_seed=master_seed, generation=generation),
        derive_generation_seed(
            master_seed=master_seed,
            generation=generation + 1,
        ),
        derive_generation_seed(
            master_seed=master_seed + 1,
            generation=generation,
        ),
    ]
    unique_seeds: set[int] = set(seeds)

    assert len(unique_seeds) == len(seeds)


@pytest.mark.parametrize(
    ("master_seed", "generation"),
    [
        (0, 0),
        (1, 0),
        (123456, 999),
        (2**31 - 1, 2**16),
    ],
)
def test_output_is_positive_uint32_range(
    master_seed: int,
    generation: int,
) -> None:
    seed = derive_generation_seed(
        master_seed=master_seed,
        generation=generation,
    )

    assert 1 <= seed <= int(np.iinfo(np.uint32).max)


@pytest.mark.parametrize(
    ("master_seed", "generation"),
    [
        (2**32, 0),
        (2**64 + 12345, 2**32 + 6789),
        (2**128 + 9876543210, 1),
    ],
)
def test_large_integer_inputs(master_seed: int, generation: int) -> None:
    seed = derive_generation_seed(
        master_seed=master_seed,
        generation=generation,
    )

    assert isinstance(seed, int)
    assert 1 <= seed <= int(np.iinfo(np.uint32).max)


@pytest.mark.parametrize(
    ("first", "second"),
    [
        (0, 1),
        (1, 42),
        (100, 999),
    ],
)
def test_swapped_master_seed_and_generation_differ(
    first: int,
    second: int,
) -> None:
    seed_forward = derive_generation_seed(master_seed=first, generation=second)
    seed_swapped = derive_generation_seed(master_seed=second, generation=first)

    assert seed_forward != seed_swapped


@pytest.mark.parametrize(
    ("master_seed", "generation", "expected_match"),
    [
        (-1, 0, "master_seed"),
        (-42, 5, "master_seed"),
        (0, -1, "generation"),
        (5, -7, "generation"),
        (-1, -1, "master_seed"),
    ],
)
def test_negative_inputs_raise_value_error(
    master_seed: int,
    generation: int,
    expected_match: str,
) -> None:
    with pytest.raises(ValueError, match=expected_match):
        _ = derive_generation_seed(
            master_seed=master_seed, generation=generation
        )


def test_zero_draw_clamps_to_one(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeSeedSequence:
        def __init__(self, entropy: object) -> None:
            _ = entropy

        # noinspection PyMethodMayBeStatic
        def generate_state(
            self,
            n_words: int,
            dtype: object = np.uint32,
        ) -> np.ndarray[tuple[int], np.dtype[np.uint32]]:
            _ = dtype
            return np.zeros(n_words, dtype=np.uint32)

    monkeypatch.setattr(
        np.random,
        "SeedSequence",  # noqa: F821
        FakeSeedSequence,
    )

    seed = derive_generation_seed(master_seed=123, generation=0)

    assert seed == 1
