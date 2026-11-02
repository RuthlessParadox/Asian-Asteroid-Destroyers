import gzip
import pickle
import sys
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from asian_asteroid_destroyers.evokit import (
    TrainingResult,
    clear_fitness_cache,
    clear_individual_cache,
    get_fitness_class,
    get_individual_class,
)
from asian_asteroid_destroyers.infrastructure import (
    CheckpointPayload,
    MainHydratingUnpickler,
)


@pytest.fixture(autouse=True)
def clean_runtime_namespace(request: pytest.FixtureRequest) -> None:
    clear_individual_cache()
    clear_fitness_cache()

    def finalizer() -> None:
        clear_individual_cache()
        clear_fitness_cache()

    request.addfinalizer(finalizer)


def test_unpickler_restores_dynamic_individual_class_on_cold_start(
    tmp_path: Path,
) -> None:
    weights = (1.0, -2.5)
    individual_class = get_individual_class(weights)
    instance = individual_class([5.5, 6.6])

    state = TrainingResult(
        final_population=[instance],
        logbook=None,
        history_logger=None,
        hall_of_fame=None,
    )
    payload: CheckpointPayload = {
        "algorithm_path": "mock.loop",
        "generation": 1,
        "state": state,
        "rng_state": np.random.default_rng(1).bit_generator.state,
    }

    test_file = tmp_path / "individual_test.pkl.gz"
    with gzip.open(test_file, "wb") as file:
        pickle.dump(payload, file)

    clear_individual_cache()
    clear_fitness_cache()
    main_module = sys.modules["__main__"]
    assert not hasattr(main_module, "DynamicIndividual_1_0_neg2_5")

    with gzip.open(test_file, "rb") as file:
        unpickler = MainHydratingUnpickler(file)
        loaded_payload = cast(CheckpointPayload, unpickler.load())

    loaded_population = loaded_payload["state"].final_population
    assert len(loaded_population) == 1

    loaded_individual = loaded_population[0]
    assert loaded_individual == [5.5, 6.6]
    assert (
        loaded_individual.__class__.__name__ == "DynamicIndividual_1_0_neg2_5"
    )
    assert loaded_individual.fitness.weights == weights


def test_unpickler_restores_dynamic_fitness_class_on_cold_start(
    tmp_path: Path,
) -> None:
    weights = (-0.05, 10.0)
    fitness_class = get_fitness_class(weights)
    instance = fitness_class()
    instance.values = (-0.05, 10.0)

    test_file = tmp_path / "fitness_test.pkl.gz"
    with gzip.open(test_file, "wb") as file:
        pickle.dump(instance, file)

    clear_individual_cache()
    clear_fitness_cache()
    main_module = sys.modules["__main__"]
    assert not hasattr(main_module, "DynamicFitness_neg0_05_10_0")

    with gzip.open(test_file, "rb") as file:
        unpickler = MainHydratingUnpickler(file)
        loaded_fitness = cast(object, unpickler.load())

    assert loaded_fitness.__class__.__name__ == "DynamicFitness_neg0_05_10_0"
    assert getattr(loaded_fitness, "weights") == weights


def test_unpickler_passes_through_standard_objects_normally(
    tmp_path: Path,
) -> None:
    standard_data = {"key": "value", "status": True}

    test_file = tmp_path / "standard_test.pkl.gz"
    with gzip.open(test_file, "wb") as file:
        pickle.dump(standard_data, file)

    with gzip.open(test_file, "rb") as file:
        unpickler = MainHydratingUnpickler(file)
        loaded_data = cast(dict[str, object], unpickler.load())

    assert loaded_data == standard_data
