import functools
from collections.abc import Callable
from typing import cast
from unittest.mock import MagicMock

import pytest

from asian_asteroid_destroyers.genetic_algorithm import (
    Individual,
    Population,
    Toolbox,
    build_toolbox,
)


@pytest.fixture
def mock_individual() -> MagicMock:
    return MagicMock(spec=Individual)


@pytest.fixture
def mock_individual_class() -> MagicMock:
    return MagicMock(spec=type[Individual])


@pytest.fixture
def mock_evaluate() -> MagicMock:
    return MagicMock(return_value=(1.0,))


def _assert_partial_mappings(
    assertions: list[tuple[Callable[..., object], MagicMock]],
) -> None:
    for operator, expected_mock in assertions:
        partial_wrapper = cast(functools.partial[object], operator)
        assert partial_wrapper.func == expected_mock


def test_toolbox_initialization_defaults(
    mock_individual: MagicMock,
    mock_evaluate: MagicMock,
) -> None:
    toolbox = Toolbox()

    assert toolbox.clone(mock_individual) is not toolbox.clone(mock_individual)

    items: list[Individual] = [mock_individual]
    assert list(toolbox.map(mock_evaluate, items)) == [(1.0,)]

    with pytest.raises(
        NotImplementedError,
        match="The DEAP operation 'individual' has not been registered yet.",
    ):
        _ = toolbox.individual()

    with pytest.raises(NotImplementedError, match="'population'"):
        _ = toolbox.population(10)

    with pytest.raises(NotImplementedError, match="'evaluate'"):
        _ = toolbox.evaluate(mock_individual)

    with pytest.raises(NotImplementedError, match="'mate'"):
        _ = toolbox.mate(mock_individual, mock_individual)

    with pytest.raises(NotImplementedError, match="'mutate'"):
        _ = toolbox.mutate(mock_individual)

    with pytest.raises(NotImplementedError, match="'select'"):
        _ = toolbox.select([mock_individual], 1)


def test_toolbox_register_and_unregister(mock_individual: MagicMock) -> None:
    toolbox = Toolbox()
    mock_function = MagicMock(return_value=(42.0,))

    toolbox.register("evaluate", mock_function)
    assert toolbox.evaluate(mock_individual) == (42.0,)
    mock_function.assert_called_once_with(mock_individual)

    toolbox.unregister("evaluate")
    assert not hasattr(toolbox, "evaluate")
    assert "evaluate" not in toolbox.__dict__


def test_build_toolbox_with_defaults(
    monkeypatch: pytest.MonkeyPatch,
    mock_individual_class: MagicMock,
    mock_evaluate: MagicMock,
) -> None:
    mock_get_individual_class = MagicMock(return_value=mock_individual_class)
    monkeypatch.setattr(
        "asian_asteroid_destroyers.genetic_algorithm.toolbox"
        + ".get_individual_class",
        mock_get_individual_class,
    )
    mock_mate = MagicMock()
    mock_mutate = MagicMock()
    mock_select = MagicMock()
    monkeypatch.setattr(
        "asian_asteroid_destroyers.genetic_algorithm.toolbox.cx_two_point",
        MagicMock(return_value=mock_mate),
    )
    monkeypatch.setattr(
        "asian_asteroid_destroyers.genetic_algorithm.toolbox.mut_gaussian",
        MagicMock(return_value=mock_mutate),
    )
    monkeypatch.setattr(
        "asian_asteroid_destroyers.genetic_algorithm.toolbox.sel_tournament",
        MagicMock(return_value=mock_select),
    )

    toolbox = build_toolbox(
        evaluation_function=mock_evaluate,
        gene_size=5,
        weights=(1.0, -1.0),
    )

    mock_get_individual_class.assert_called_once_with((1.0, -1.0))

    _assert_partial_mappings(
        [
            (toolbox.evaluate, mock_evaluate),
            (toolbox.mate, mock_mate),
            (toolbox.mutate, mock_mutate),
            (toolbox.select, mock_select),
        ]
    )


def test_build_toolbox_with_custom_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    mock_individual_class: MagicMock,
    mock_evaluate: MagicMock,
) -> None:
    mock_get_individual_class = MagicMock(return_value=mock_individual_class)
    monkeypatch.setattr(
        "asian_asteroid_destroyers.genetic_algorithm.toolbox"
        + ".get_individual_class",
        mock_get_individual_class,
    )
    mock_individual_factory = MagicMock()
    mock_population_factory = MagicMock()
    mock_mate = MagicMock()
    mock_mutate = MagicMock()
    mock_select = MagicMock()

    toolbox = build_toolbox(
        evaluation_function=mock_evaluate,
        gene_size=10,
        individual_factory=mock_individual_factory,
        population_factory=mock_population_factory,
        mate_operator=mock_mate,
        mutate_operator=mock_mutate,
        select_operator=mock_select,
    )

    _assert_partial_mappings(
        [
            (toolbox.individual, mock_individual_factory),
            (toolbox.population, mock_population_factory),
            (toolbox.mate, mock_mate),
            (toolbox.mutate, mock_mutate),
            (toolbox.select, mock_select),
        ]
    )


def test_default_individual_factory_via_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mock_random = MagicMock(side_effect=[0.1, 0.2, 0.3])
    monkeypatch.setattr(
        "asian_asteroid_destroyers.genetic_algorithm.toolbox.random.random",
        mock_random,
    )

    toolbox = build_toolbox(evaluation_function=MagicMock(), gene_size=3)

    result = toolbox.individual()
    passed_args: list[float] = [float(item) for item in result]
    assert passed_args == [0.1, 0.2, 0.3]


def test_default_population_factory_via_build(
    monkeypatch: pytest.MonkeyPatch,
    mock_individual_class: MagicMock,
) -> None:
    monkeypatch.setattr(
        "asian_asteroid_destroyers.genetic_algorithm.toolbox"
        + ".get_individual_class",
        MagicMock(return_value=mock_individual_class),
    )
    mock_population = MagicMock(spec=Population)
    mock_init_repeat = MagicMock(return_value=mock_population)
    monkeypatch.setattr(
        "asian_asteroid_destroyers.genetic_algorithm.toolbox.tools.initRepeat",
        mock_init_repeat,
    )

    toolbox = build_toolbox(evaluation_function=MagicMock(), gene_size=5)
    result = toolbox.population(5)

    assert result == mock_population
    mock_init_repeat.assert_called_once_with(list, toolbox.individual, 5)
