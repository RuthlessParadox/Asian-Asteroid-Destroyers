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
    """Encapsulates a variable-arity TSK rule structure.

    Attributes:
        antecedents: List of tuples mapping (input_matrix_index, mf_index).
            Example: [(0, 1), (2, 0)] -> (Input 0 is MF 1) AND (Input 2 is MF 0)
        consequent_coefficients: Linear scale weights matching input dimensionality.
        consequent_offset: Intercept/bias value for the TSK equation.
    """

    antecedents: list[tuple[int, int]]
    consequent_coefficients: list[float]
    consequent_offset: float


class SubsystemChromosome(NamedTuple):
    """Evolvable chromosome blueprint block for a singular output system."""

    membership_function_pool: list[MFParamTriangular | MFParamTrapezoidal]
    rules: list[RuleGene]


class ControllerChromosome(NamedTuple):
    """The master chromosome bundle injected directly into the controller."""

    thrust_blueprint: SubsystemChromosome
    turn_blueprint: SubsystemChromosome
    fire_blueprint: SubsystemChromosome


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


def compile_subsystem(blueprint: SubsystemChromosome) -> TSKInferenceEngine:
    """Compiles a subsystem genome into a fast operational engine."""
    # Pre-compile the shared pool of membership function shapes
    compiled_pool: Final[list[MembershipFunction]] = [
        _compile_membership_function(param)
        for param in blueprint.membership_function_pool
    ]

    compiled_rules: list[TSKRule] = []

    for rule_gene in blueprint.rules:
        clauses: list[FuzzyClause] = []

        for input_index, membership_function_index in rule_gene.antecedents:
            clauses.append(
                FuzzyClause(
                    input_index,
                    compiled_pool[membership_function_index],
                ),
            )

        compiled_rules.append(
            TSKRule(
                antecedents=clauses,
                consequent_coefficients=rule_gene.consequent_coefficients,
                consequent_offset=rule_gene.consequent_offset,
            )
        )

    return TSKInferenceEngine(compiled_rules)
