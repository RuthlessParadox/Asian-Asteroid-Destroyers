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

__all__ = [
    "FlatGenome",
    "MFParamTriangular",
    "MFParamTrapezoidal",
    "RuleGene",
    "SubsystemChromosome",
    "ControllerChromosome",
    "compile_subsystem",
    "flatten_chromosome",
    "hydrate_chromosome_blueprint",
    "repair_geometry_invariants",
    "get_default_controller_template",
]

FlatGenome = list[float]


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


def flatten_chromosome(chromosome: ControllerChromosome) -> FlatGenome:
    flat_genes: FlatGenome = []

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
    flat_genes: FlatGenome,
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
    flat_genes: FlatGenome,
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


def repair_geometry_invariants(
    flat_genes: FlatGenome,
    template: ControllerChromosome,
    min_width: float = 0.05,
) -> FlatGenome:
    chromosome = hydrate_chromosome_blueprint(flat_genes, template)

    for subsystem in (
        chromosome.thrust_blueprint,
        chromosome.turn_blueprint,
        chromosome.fire_blueprint,
    ):
        pool = subsystem.membership_function_pool
        for index, param in enumerate(pool):
            if isinstance(param, MFParamTriangular):
                left = param[0]
                center = max(param[1], left + min_width)
                right = max(param[2], center + min_width)

                pool[index] = MFParamTriangular(left, center, right)
            else:
                left_base = param[0]
                left_peak = max(param[1], left_base + min_width)
                right_peak = max(param[2], left_peak + min_width)
                right_base = max(param[3], right_peak + min_width)

                pool[index] = MFParamTrapezoidal(
                    left_base,
                    left_peak,
                    right_peak,
                    right_base,
                )

    return flatten_chromosome(chromosome)


def get_default_controller_template() -> ControllerChromosome:
    def make_isolated_pool() -> list[MFParamTriangular | MFParamTrapezoidal]:
        return [
            # --- Inputs 0: Normalized Angular Bearings / Headings [-1.0, 1.0] ---
            MFParamTrapezoidal(
                -1.0, -1.0, -0.25, 0.0
            ),  # MF 0: Negative / Left
            MFParamTriangular(-0.15, 0.0, 0.15),  # MF 1: Aligned / Center
            MFParamTrapezoidal(0.0, 0.25, 1.0, 1.0),  # MF 2: Positive / Right
            # --- Inputs 1: Normalized Time-To-Impact / Distances [0.0, 1.0] ---
            MFParamTrapezoidal(0.0, 0.0, 0.15, 0.3),  # MF 3: Critical / Close
            MFParamTriangular(0.2, 0.4, 0.6),  # MF 4: Intermediate / Medium
            MFParamTrapezoidal(0.5, 0.8, 1.0, 1.0),  # MF 5: Far / Clear
            # --- Inputs 2: Normalized Engine Velocities [-1.0, 1.0] ---
            MFParamTrapezoidal(
                -1.0, -1.0, -0.1, 0.0
            ),  # MF 6: High Negative Rate
            MFParamTriangular(-0.06, 0.0, 0.06),  # MF 7: Stationary / Linear
            MFParamTrapezoidal(0.0, 0.1, 1.0, 1.0),  # MF 8: High Positive Rate
        ]

    thrust_rules = [
        RuleGene(
            antecedents=[(0, 0), (1, 3), (2, 8)],
            consequent_coefficients=[-0.1, 0.05, -0.2],
            consequent_offset=-0.4,
        ),
        RuleGene(
            antecedents=[(0, 2), (1, 5), (2, 7)],
            consequent_coefficients=[0.05, 0.02, -0.1],
            consequent_offset=0.3,
        ),
    ]

    turn_rules = [
        RuleGene(
            antecedents=[(0, 2), (1, 4), (2, 7)],
            consequent_coefficients=[0.3, 0.0, -0.15],
            consequent_offset=0.25,
        ),
        RuleGene(
            antecedents=[(0, 0), (1, 4), (2, 7)],
            consequent_coefficients=[0.3, 0.0, -0.15],
            consequent_offset=-0.25,
        ),
    ]

    fire_rules = [
        RuleGene(
            antecedents=[(0, 1), (1, 0), (2, 8)],
            consequent_coefficients=[-0.02, -0.05, 0.1],
            consequent_offset=0.1,
        ),
        RuleGene(
            antecedents=[(0, 2), (1, 2), (2, 6)],
            consequent_coefficients=[0.0, 0.0, 0.0],
            consequent_offset=-0.1,
        ),
    ]

    return ControllerChromosome(
        thrust_blueprint=SubsystemChromosome(
            membership_function_pool=make_isolated_pool(),
            rules=thrust_rules,
        ),
        turn_blueprint=SubsystemChromosome(
            membership_function_pool=make_isolated_pool(),
            rules=turn_rules,
        ),
        fire_blueprint=SubsystemChromosome(
            membership_function_pool=make_isolated_pool(),
            rules=fire_rules,
        ),
    )
