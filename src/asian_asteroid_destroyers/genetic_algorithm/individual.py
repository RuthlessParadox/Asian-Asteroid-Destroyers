from collections.abc import Iterable

from .fitness import Fitness


class Individual(list[float]):
    fitness: Fitness

    def __init__(self, iterable: Iterable[float] | None = None) -> None:
        if iterable is not None:
            super().__init__(iterable)
        else:
            super().__init__()
        if not hasattr(self, "fitness"):
            self.fitness = Fitness()
