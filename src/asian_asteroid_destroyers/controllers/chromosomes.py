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


def flatten_chromosome(chromosome: ControllerChromosome) -> list[float]:
    flat_genes: list[float] = []

    for blueprint in (
        chromosome.thrust_blueprint,
        chromosome.turn_blueprint,
        chromosome.fire_blueprint,
    ):
        for param in blueprint.membership_function_pool:
            if isinstance(param, MFParamTriangular):
                flat_genes.extend([param.left, param.center, param.right])
            else:
                flat_genes.extend(
                    [
                        param.left_base,
                        param.left_peak,
                        param.right_peak,
                        param.right_base,
                    ],
                )

        for rule in blueprint.rules:
            flat_genes.extend(rule.consequent_coefficients)
            flat_genes.append(rule.consequent_offset)

    return flat_genes


def _hydrate_subsystem(
    flat_genes: list[float],
    start_index: int,
    template: SubsystemChromosome,
) -> tuple[SubsystemChromosome, int]:
    """Decodes a single subsystem chunk from the sequential float array.

    Marches an explicit index pointer through the array based on the
    geometric types and rules present in the provided structural template.
    """
    index = start_index
    hydrated_membership_functions: list[
        MFParamTriangular | MFParamTrapezoidal
    ] = []

    for membership_function_template in template.membership_function_pool:
        if isinstance(membership_function_template, MFParamTriangular):
            hydrated_membership_functions.append(
                MFParamTriangular(
                    left=flat_genes[index],
                    center=flat_genes[index + 1],
                    right=flat_genes[index + 2],
                )
            )
            index += 3
        else:
            hydrated_membership_functions.append(
                MFParamTrapezoidal(
                    left_base=flat_genes[index],
                    left_peak=flat_genes[index + 1],
                    right_peak=flat_genes[index + 2],
                    right_base=flat_genes[index + 3],
                )
            )
            index += 4

    hydrated_rules: list[RuleGene] = []

    for rule_template in template.rules:
        coefficient_count = len(rule_template.consequent_coefficients)

        coefficients: list[float] = []
        for coefficient_i in range(coefficient_count):
            coefficients.append(flat_genes[index + coefficient_i])
        index += coefficient_count

        offset = flat_genes[index]
        index += 1

        hydrated_rules.append(
            RuleGene(
                antecedents=rule_template.antecedents,
                consequent_coefficients=coefficients,
                consequent_offset=offset,
            )
        )

    return (
        SubsystemChromosome(
            membership_function_pool=hydrated_membership_functions,
            rules=hydrated_rules,
        ),
        index,
    )


def hydrate_chromosome_blueprint(
    flat_genes: list[float],
    template: ControllerChromosome,
) -> ControllerChromosome:
    """Hydrates a flat sequence of float genes into a structured chromosome object.

    Uses a template structural architecture to reconstruct separate sub-system
    inference models with full compile-time static type safety.
    """
    thrust_blueprint, next_index = _hydrate_subsystem(
        flat_genes,
        0,
        template.thrust_blueprint,
    )
    turn_blueprint, next_index = _hydrate_subsystem(
        flat_genes,
        next_index,
        template.turn_blueprint,
    )
    fire_blueprint, last_idx = _hydrate_subsystem(
        flat_genes,
        next_index,
        template.fire_blueprint,
    )

    if last_idx != len(flat_genes):
        raise ValueError(
            "Chromosome gene count mismatch. Expected index pointer to finish "
            + f"at array boundary {len(flat_genes)}, "
            + f"but terminated at {last_idx}."
        )

    return ControllerChromosome(
        thrust_blueprint=thrust_blueprint,
        turn_blueprint=turn_blueprint,
        fire_blueprint=fire_blueprint,
    )
