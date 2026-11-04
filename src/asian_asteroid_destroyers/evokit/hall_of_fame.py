from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, cast

from deap import tools
from typing_extensions import override

if TYPE_CHECKING:
    from collections.abc import Iterator

    from .individual import Individual
    from .types import Population


class _DEAPHallOfFame(Protocol):
    maxsize: int
    items: Population

    def update(self, population: Population, /) -> None: ...
    def insert(self, individual: Individual, /) -> None: ...
    def remove(self, index: int, /) -> None: ...
    def clear(self, /) -> None: ...


_UnannotatedParetoFront: type = tools.ParetoFront


class HallOfFame:
    __slots__: tuple[str, ...] = ("_deap_hall_of_fame",)
    _deap_hall_of_fame: _DEAPHallOfFame

    def __init__(self, similar: object = None) -> None:
        self._deap_hall_of_fame = cast(
            _DEAPHallOfFame,
            _UnannotatedParetoFront(similar),
        )

    @property
    def max_size(self) -> int:
        return self._deap_hall_of_fame.maxsize

    @property
    def items(self) -> Population:
        return self._deap_hall_of_fame.items

    @property
    def best(self) -> Individual:
        if not self.items:
            raise IndexError(
                "Cannot retrieve an individual from an empty HallOfFame."
            )
        return self.items[0]

    def update(self, population: Population) -> None:
        self._deap_hall_of_fame.update(population)

    def insert(self, individual: Individual) -> None:
        self._deap_hall_of_fame.insert(individual)

    def remove(self, index: int) -> None:
        self._deap_hall_of_fame.remove(index)

    def clear(self) -> None:
        self._deap_hall_of_fame.clear()

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> Individual:
        return self.items[index]

    def __iter__(self) -> Iterator[Individual]:
        return iter(self.items)

    @override
    def __str__(self) -> str:
        return str(self._deap_hall_of_fame)
