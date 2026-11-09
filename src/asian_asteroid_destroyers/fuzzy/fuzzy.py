from __future__ import annotations

import math
from typing import TYPE_CHECKING, Final, NamedTuple, Protocol

import numpy as np
from typing_extensions import override

if TYPE_CHECKING:
    import numpy.typing as npt


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
    __slots__: tuple[str, ...] = (
        "_rules_count",
        "_inputs_count",
        "_antecedent_map",
        "_membership_function_types",
        "_membership_function_params",
        "_coefficients",
        "_offsets",
        "_activation_masks",
    )

    _rules_count: int
    _inputs_count: int

    _antecedent_map: npt.NDArray[np.intp]

    _membership_function_types: npt.NDArray[np.int8]
    _membership_function_params: npt.NDArray[np.float64]

    _coefficients: npt.NDArray[np.float64]
    _offsets: npt.NDArray[np.float64]
    _activation_masks: npt.NDArray[np.float64]

    def __init__(
        self,
        rules_count: int,
        inputs_count: int,
        antecedent_map: npt.NDArray[np.intp],
        mf_types: npt.NDArray[np.int8],
        mf_params: npt.NDArray[np.float64],
        coefficients: npt.NDArray[np.float64],
        offsets: npt.NDArray[np.float64],
        activation_masks: npt.NDArray[np.float64],
    ) -> None:
        self._rules_count = rules_count
        self._inputs_count = inputs_count

        # Maps shape: (rules_count, inputs_count) -> stores MF pool index
        self._antecedent_map = antecedent_map

        # Parallel arrays for MF pool lookups
        self._membership_function_types = (
            mf_types  # 0 = Triangular, 1 = Trapezoidal
        )
        self._membership_function_params = mf_params  # Shape: (pool_size, 4)

        # Consequent parameters
        self._coefficients = coefficients  # Shape: (rules_count, inputs_count)
        self._offsets = offsets  # Shape: (rules_count,)
        self._activation_masks = activation_masks

    def infer(
        self,
        inputs: list[float],
        fallback_default: float = 0.0,
    ) -> float:
        input_array = np.array(inputs, dtype=np.float64)

        mapped_membership_function_indices = self._antecedent_map
        params = self._membership_function_params[
            mapped_membership_function_indices
        ]
        types = self._membership_function_types[
            mapped_membership_function_indices
        ]

        input_array_broadcasting = input_array[np.newaxis, :]

        t_left = params[:, :, 0]
        t_center = params[:, :, 1]
        t_right = params[:, :, 2]

        tri_mu = np.where(
            (input_array_broadcasting <= t_left)
            | (input_array_broadcasting >= t_right),
            0.0,
            np.where(
                input_array_broadcasting < t_center,
                (input_array_broadcasting - t_left)
                / np.maximum(t_center - t_left, 1e-10),
                (t_right - input_array_broadcasting)
                / np.maximum(t_right - t_center, 1e-10),
            ),
        )

        z_left_b = params[:, :, 0]
        z_left_p = params[:, :, 1]
        z_right_p = params[:, :, 2]
        z_right_b = params[:, :, 3]

        trap_mu = np.where(
            (input_array_broadcasting <= z_left_b)
            | (input_array_broadcasting >= z_right_b),
            0.0,
            np.where(
                (input_array_broadcasting >= z_left_p)
                & (input_array_broadcasting <= z_right_p),
                1.0,
                np.where(
                    input_array_broadcasting < z_left_p,
                    (input_array_broadcasting - z_left_b)
                    / np.maximum(z_left_p - z_left_b, 1e-10),
                    (z_right_b - input_array_broadcasting)
                    / np.maximum(z_right_b - z_right_p, 1e-10),
                ),
            ),
        )

        clause_mu = np.where(np.equal(types, 0), tri_mu, trap_mu)

        raw_strengths: npt.NDArray[np.float64] = np.prod(clause_mu, axis=1)  # pyright: ignore[reportAny]
        firing_strengths = raw_strengths * self._activation_masks
        firing_strengths: npt.NDArray[np.float64] = np.where(
            np.isnan(firing_strengths) | np.isinf(firing_strengths),
            0.0,
            firing_strengths,
        )
        total_weight = float(np.sum(firing_strengths))

        if total_weight == 0.0:
            return fallback_default

        consequent_values: npt.NDArray[np.float64] = (  # pyright: ignore[reportAny]
            np.sum(self._coefficients * input_array, axis=1) + self._offsets
        )
        consequent_values = np.where(
            np.isnan(consequent_values) | np.isinf(consequent_values),
            0.0,
            consequent_values,
        )
        weighted_sum = float(np.sum(firing_strengths * consequent_values))

        if math.isnan(weighted_sum) or math.isinf(weighted_sum):
            return fallback_default

        return weighted_sum / total_weight

    @override
    def __repr__(self) -> str:
        return f"TSKInferenceEngine(rules_count={self._rules_count})"
