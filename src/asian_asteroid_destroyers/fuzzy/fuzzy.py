from typing import Final, NamedTuple, Protocol

from typing_extensions import override


class RuleEvaluationResult(NamedTuple):
    firing_strength: float
    consequent_value: float


class MembershipFunction(Protocol):
    def evaluate(self, crisp_value: float) -> float:
        """Computes the membership degree μ ∈ [0.0, 1.0]."""
        ...


class TriangularMF:
    __slots__: tuple[str, ...] = ("_left", "_center", "_right")
    _left: Final[float]
    _center: Final[float]
    _right: Final[float]

    def __init__(self, left: float, center: float, right: float) -> None:
        if not left <= center <= right:
            raise ValueError("Must satisfy condition: left <= center <= right")
        self._left = left
        self._center = center
        self._right = right

    def evaluate(self, crisp_value: float) -> float:
        if crisp_value <= self._left or crisp_value >= self._right:
            return 0.0
        if crisp_value == self._center:
            return 1.0
        if crisp_value < self._center:
            return (crisp_value - self._left) / (self._center - self._left)
        return (self._right - crisp_value) / (self._right - self._center)

    @override
    def __repr__(self) -> str:
        return (
            f"TriangularMF(left={self._left:.2f}, "
            + f"center={self._center:.2f}, right={self._right:.2f})"
        )


class TrapezoidalMF:
    __slots__: tuple[str, ...] = (
        "_left_base",
        "_left_peak",
        "_right_peak",
        "_right_base",
    )
    _left_base: Final[float]
    _left_peak: Final[float]
    _right_peak: Final[float]
    _right_base: Final[float]

    def __init__(
        self,
        left_base: float,
        left_peak: float,
        right_peak: float,
        right_base: float,
    ) -> None:
        if not left_base <= left_peak <= right_peak <= right_base:
            raise ValueError(
                "Invalid trapezoidal point constraints configuration."
            )
        self._left_base = left_base
        self._left_peak = left_peak
        self._right_peak = right_peak
        self._right_base = right_base

    def evaluate(self, crisp_value: float) -> float:
        if crisp_value <= self._left_base or crisp_value >= self._right_base:
            return 0.0
        if self._left_peak <= crisp_value <= self._right_peak:
            return 1.0
        if crisp_value < self._left_peak:
            return (crisp_value - self._left_base) / (
                self._left_peak - self._left_base
            )
        return (self._right_base - crisp_value) / (
            self._right_base - self._right_peak
        )

    @override
    def __repr__(self) -> str:
        return (
            f"TrapezoidalMF(left_base={self._left_base:.2f}, "
            + f"left_peak={self._left_peak:.2f}, "
            + f"right_peak={self._right_peak:.2f}, "
            + f"right_base={self._right_base:.2f})"
        )


class FuzzyClause:
    __slots__: tuple[str, ...] = ("input_index", "membership_function")
    input_index: Final[int]
    membership_function: Final[MembershipFunction]

    def __init__(
        self,
        input_index: int,
        membership_function: MembershipFunction,
    ) -> None:
        self.input_index = input_index
        self.membership_function = membership_function

    @override
    def __repr__(self) -> str:
        return (
            f"FuzzyClause(input_index={self.input_index}, "
            + f"mf={self.membership_function})"
        )


class TSKRule:
    __slots__: tuple[str, ...] = (
        "_antecedents",
        "_consequent_coefficients",
        "_consequent_offset",
    )
    _antecedents: Final[list[FuzzyClause]]
    _consequent_coefficients: Final[list[float]]
    _consequent_offset: Final[float]

    def __init__(
        self,
        antecedents: list[FuzzyClause],
        consequent_coefficients: list[float],
        consequent_offset: float,
    ) -> None:
        self._antecedents = antecedents
        self._consequent_coefficients = consequent_coefficients
        self._consequent_offset = consequent_offset

    def evaluate(self, inputs: list[float]) -> RuleEvaluationResult:
        firing_strength = 1.0
        for clause in self._antecedents:
            firing_strength *= clause.membership_function.evaluate(
                inputs[clause.input_index],
            )
            if firing_strength == 0.0:
                return RuleEvaluationResult(0.0, 0.0)

        consequent_value = self._consequent_offset
        for index, coefficient in enumerate(self._consequent_coefficients):
            consequent_value += coefficient * inputs[index]

        return RuleEvaluationResult(firing_strength, consequent_value)

    @override
    def __repr__(self) -> str:
        return (
            f"TSKRule(antecedents={self._antecedents}, "
            + f"coefficients={self._consequent_coefficients}, "
            + f"offset={self._consequent_offset:.2f})"
        )


class TSKInferenceEngine:
    __slots__: tuple[str, ...] = ("_rules",)
    _rules: Final[list[TSKRule]]

    def __init__(self, rules: list[TSKRule]) -> None:
        self._rules = rules

    def infer(
        self,
        inputs: list[float],
        fallback_default: float = 0.0,
    ) -> float:
        total_weight = 0.0
        weighted_sum = 0.0

        for rule in self._rules:
            result = rule.evaluate(inputs)
            if result.firing_strength > 0.0:
                total_weight += result.firing_strength
                weighted_sum += (
                    result.firing_strength * result.consequent_value
                )

        if total_weight == 0.0:
            return fallback_default

        return weighted_sum / total_weight

    @override
    def __repr__(self) -> str:
        return f"TSKInferenceEngine(rules_count={len(self._rules)})"
