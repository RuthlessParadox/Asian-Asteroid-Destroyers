from __future__ import annotations

from typing import TYPE_CHECKING, Final, NamedTuple

from asian_asteroid_destroyers.fuzzy import (
    FuzzyClause,
    TrapezoidalMF,
    TriangularMF,
    TSKInferenceEngine,
    TSKRule,
)

if TYPE_CHECKING:
    from asian_asteroid_destroyers.fuzzy import MembershipFunction


class MFParamTriangular(NamedTuple):
    """Human-readable parameter gene configuration for a Triangular MF."""

    left: float
    center: float
    right: float


class MFParamTrapezoidal(NamedTuple):
    """Human-readable parameter gene configuration for a Trapezoidal MF."""

    left_base: float
    left_peak: float
    right_peak: float
    right_base: float


class RuleGene(NamedTuple):
    """Maps linguistic intent directly to TSK linear equations.

    Example:
        RuleGene(
            antecedent_mf_indices=[0, 1],  # e.g., Close distance AND Fast speed
            consequent_coefficients=[-0.5, 0.2],
            consequent_offset=10.0
        )
    """

    antecedent_membership_function_indices: list[int]
    consequent_coefficients: list[float]
    consequent_offset: float


class ControllerChromosome(NamedTuple):
    """The full evolutionary genome block for a specific action subsystem.

    This structure is highly human-readable when logged, saved to disk,
    or inspected in a debugger, yet acts as a static blueprint.
    """

    distance_membership_functions: list[MFParamTriangular | MFParamTrapezoidal]
    bearing_membership_functions: list[MFParamTriangular | MFParamTrapezoidal]
    rules: list[RuleGene]


def _compile_membership_function(
    param: MFParamTriangular | MFParamTrapezoidal,
) -> MembershipFunction:
    """Factory that compiles human-readable parameters into an MF instance."""
    if isinstance(param, MFParamTriangular):
        return TriangularMF(param.left, param.center, param.right)
    return TrapezoidalMF(
        param.left_base,
        param.left_peak,
        param.right_peak,
        param.right_base,
    )


def compile_chromosome_to_engine(
    chromosome: ControllerChromosome,
) -> TSKInferenceEngine:
    """Transforms a human-readable chromosome into a fast TSKInferenceEngine.

    This function unpacks the linguistic rules and parameters into an explicit
    index-mapped execution graph. Run this once inside your controller's
    __init__ method during dependency injection.
    """
    compiled_distance_membership_functions: Final[list[MembershipFunction]] = [
        _compile_membership_function(param)
        for param in chromosome.distance_membership_functions
    ]
    compiled_bearing_membership_functions: Final[list[MembershipFunction]] = [
        _compile_membership_function(param)
        for param in chromosome.bearing_membership_functions
    ]

    compiled_rules: Final[list[TSKRule]] = []

    for rule_gene in chromosome.rules:
        antecedents: list[FuzzyClause] = []

        distance_membership_function_index = (
            rule_gene.antecedent_membership_function_indices[0]
        )
        bearing_membership_function_index = (
            rule_gene.antecedent_membership_function_indices[1]
        )

        antecedents.append(
            FuzzyClause(
                0,
                compiled_distance_membership_functions[
                    distance_membership_function_index
                ],
            )
        )

        antecedents.append(
            FuzzyClause(
                1,
                compiled_bearing_membership_functions[
                    bearing_membership_function_index
                ],
            )
        )

        compiled_rules.append(
            TSKRule(
                antecedents=antecedents,
                consequent_coefficients=rule_gene.consequent_coefficients,
                consequent_offset=rule_gene.consequent_offset,
            )
        )

    return TSKInferenceEngine(compiled_rules)
