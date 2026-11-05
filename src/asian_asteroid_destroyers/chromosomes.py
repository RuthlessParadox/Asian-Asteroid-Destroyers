from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple, cast

import numpy as np

from asian_asteroid_destroyers.fuzzy import TSKInferenceEngine

if TYPE_CHECKING:
    import numpy.typing as npt

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
    "create_geometry_repair_maps",
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
    """Encapsulates a variable-arity TSK rule structure with an activation gate.

    Attributes:
        antecedents: List of tuples mapping (input_matrix_index, mf_index).
        consequent_coefficients: Linear scale weights matching inputs.
        consequent_offset: Intercept/bias value for the TSK equation.
        activation_gate: Continuous value ∈ [0.0, 1.0]. If <= 0.5, rule is dead.
    """

    antecedents: list[tuple[int, int]]
    consequent_coefficients: list[float]
    consequent_offset: float
    activation_gate: float


class SubsystemChromosome(NamedTuple):
    """Evolvable chromosome blueprint block for a singular output system."""

    membership_function_pool: list[MFParamTriangular | MFParamTrapezoidal]
    rules: list[RuleGene]


class ControllerChromosome(NamedTuple):
    """The master chromosome bundle injected directly into the controller."""

    thrust_offensive: SubsystemChromosome
    thrust_defensive: SubsystemChromosome
    turn_offensive: SubsystemChromosome
    turn_defensive: SubsystemChromosome
    threat_gate: SubsystemChromosome


def compile_subsystem(blueprint: SubsystemChromosome) -> TSKInferenceEngine:
    pool = blueprint.membership_function_pool
    pool_size = len(pool)

    membership_function_types = np.zeros(pool_size, dtype=np.int8)
    membership_function_params = np.zeros((pool_size, 4), dtype=np.float64)

    for index, param in enumerate(pool):
        if isinstance(param, MFParamTriangular):
            membership_function_types[index] = 0
            membership_function_params[index, 0:3] = [
                param.left,
                param.center,
                param.right,
            ]
        else:
            membership_function_types[index] = 1
            membership_function_params[index, :] = [
                param.left_base,
                param.left_peak,
                param.right_peak,
                param.right_base,
            ]

    rules_count = len(blueprint.rules)
    inputs_count = len(blueprint.rules[0].consequent_coefficients)

    antecedent_map = np.zeros((rules_count, inputs_count), dtype=np.intp)
    coefficients = np.zeros((rules_count, inputs_count), dtype=np.float64)
    offsets = np.zeros(rules_count, dtype=np.float64)
    activation_masks = np.zeros(rules_count, dtype=np.float64)

    for rule_index, rule in enumerate(blueprint.rules):
        for input_index, membership_function_index in rule.antecedents:
            antecedent_map[rule_index, input_index] = membership_function_index

        for input_index, coefficient in enumerate(
            rule.consequent_coefficients
        ):
            coefficients[rule_index, input_index] = coefficient
        offsets[rule_index] = rule.consequent_offset
        activation_masks[rule_index] = (
            1.0 if rule.activation_gate > 0.5 else 0.0
        )

    return TSKInferenceEngine(
        rules_count=rules_count,
        inputs_count=inputs_count,
        antecedent_map=antecedent_map,
        mf_types=membership_function_types,
        mf_params=membership_function_params,
        coefficients=coefficients,
        offsets=offsets,
        activation_masks=activation_masks,
    )


def flatten_chromosome(chromosome: ControllerChromosome) -> FlatGenome:
    flat_genes: FlatGenome = []

    for blueprint in (
        chromosome.thrust_offensive,
        chromosome.thrust_defensive,
        chromosome.turn_offensive,
        chromosome.turn_defensive,
        chromosome.threat_gate,
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
            flat_genes.append(rule.activation_gate)

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

        gate = flat_genes[index]
        index += 1

        hydrated_rules.append(
            RuleGene(
                antecedents=rule_template.antecedents,
                consequent_coefficients=coefficients,
                consequent_offset=offset,
                activation_gate=gate,
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
    thrust_offensive, next_index = _hydrate_subsystem(
        flat_genes,
        0,
        template.thrust_offensive,
    )
    thrust_defensive, next_index = _hydrate_subsystem(
        flat_genes,
        next_index,
        template.thrust_defensive,
    )
    turn_offensive, next_index = _hydrate_subsystem(
        flat_genes,
        next_index,
        template.turn_offensive,
    )
    turn_defensive, next_index = _hydrate_subsystem(
        flat_genes,
        next_index,
        template.turn_defensive,
    )
    threat_gate, last_idx = _hydrate_subsystem(
        flat_genes,
        next_index,
        template.threat_gate,
    )

    if last_idx != len(flat_genes):
        raise ValueError(
            "Chromosome gene count mismatch. Expected index pointer to finish "
            + f"at array boundary {len(flat_genes)}, "
            + f"but terminated at {last_idx}."
        )

    return ControllerChromosome(
        thrust_offensive=thrust_offensive,
        thrust_defensive=thrust_defensive,
        turn_offensive=turn_offensive,
        turn_defensive=turn_defensive,
        threat_gate=threat_gate,
    )


def create_geometry_repair_maps(
    template: ControllerChromosome,
) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]:
    subsystems = (
        template.thrust_offensive,
        template.thrust_defensive,
        template.turn_offensive,
        template.turn_defensive,
        template.threat_gate,
    )

    triangular_indices: list[list[int]] = []
    trapezoidal_indices: list[list[int]] = []

    current_index = 0
    for blueprint in subsystems:
        for param in blueprint.membership_function_pool:
            param_len = len(param)
            index_range = list(range(current_index, current_index + param_len))
            if param_len == 3:
                triangular_indices.append(index_range)
            elif param_len == 4:
                trapezoidal_indices.append(index_range)
            current_index += param_len

        for rule in blueprint.rules:
            current_index += len(rule.consequent_coefficients) + 2

    triangular_array = (
        np.array(triangular_indices, dtype=np.intp)
        if triangular_indices
        else np.empty((0, 3), dtype=np.intp)
    )
    trapezoidal_array = (
        np.array(trapezoidal_indices, dtype=np.intp)
        if trapezoidal_indices
        else np.empty((0, 4), dtype=np.intp)
    )

    return triangular_array, trapezoidal_array


def repair_geometry_invariants(
    flat_genes: FlatGenome,
    triangular_array: npt.NDArray[np.intp],
    trapezoidal_array: npt.NDArray[np.intp],
    min_width: float = 0.05,
) -> FlatGenome:
    genes = np.array(flat_genes, dtype=np.float64)

    if triangular_array.size > 0:
        genes[triangular_array[:, 1]] = np.maximum(
            genes[triangular_array[:, 1]],
            genes[triangular_array[:, 0]] + min_width,
        )
        genes[triangular_array[:, 2]] = np.maximum(
            genes[triangular_array[:, 2]],
            genes[triangular_array[:, 1]] + min_width,
        )

    if trapezoidal_array.size > 0:
        genes[trapezoidal_array[:, 1]] = np.maximum(
            genes[trapezoidal_array[:, 1]],
            genes[trapezoidal_array[:, 0]] + min_width,
        )
        genes[trapezoidal_array[:, 2]] = np.maximum(
            genes[trapezoidal_array[:, 2]],
            genes[trapezoidal_array[:, 1]] + min_width,
        )
        genes[trapezoidal_array[:, 3]] = np.maximum(
            genes[trapezoidal_array[:, 3]],
            genes[trapezoidal_array[:, 2]] + min_width,
        )

    return cast(FlatGenome, genes.tolist())


def get_default_controller_template() -> ControllerChromosome:
    def make_isolated_pool() -> list[MFParamTriangular | MFParamTrapezoidal]:
        return [
            # --- Inputs 0: Normalized Angular Bearings / Headings ---
            MFParamTrapezoidal(-1.0, -1.0, -0.25, 0.0),  # MF 0: Left
            MFParamTriangular(-0.15, 0.0, 0.15),  # MF 1: Center
            MFParamTrapezoidal(0.0, 0.25, 1.0, 1.0),  # MF 2: Right
            # --- Inputs 1: Normalized Time-To-Impact / Distances ---
            MFParamTrapezoidal(0.0, 0.0, 0.15, 0.3),  # MF 3: Close
            MFParamTriangular(0.2, 0.4, 0.6),  # MF 4: Medium
            MFParamTrapezoidal(0.5, 0.8, 1.0, 1.0),  # MF 5: Far
            # --- Inputs 2: Normalized Velocities / Densities ---
            MFParamTrapezoidal(-1.0, -1.0, -0.1, 0.0),  # MF 6: Neg/Low
            MFParamTriangular(-0.06, 0.0, 0.06),  # MF 7: Zero/Mid
            MFParamTrapezoidal(0.0, 0.1, 1.0, 1.0),  # MF 8: Pos/High
        ]

    # --- Offensive Thrust Subsystem ---
    # Inputs: [normalized_error, norm_dist, norm_turn_rate_ship]
    t_off_rules = [
        RuleGene(
            antecedents=[(0, 1), (1, 5), (2, 7)],
            consequent_coefficients=[0.0, -0.2, 0.0],
            consequent_offset=0.8,
            activation_gate=1.0,
        ),
        RuleGene(
            antecedents=[(0, 0), (1, 4), (2, 8)],
            consequent_coefficients=[-0.1, 0.0, -0.05],
            consequent_offset=0.3,
            activation_gate=1.0,
        ),
    ]

    # --- Defensive Thrust Subsystem ---
    # Inputs: [norm_lateral_threat, normalized_tti, norm_speed_ship]
    t_def_rules = [
        RuleGene(
            antecedents=[(0, 1), (1, 3), (2, 8)],
            consequent_coefficients=[0.0, 0.4, -0.3],
            consequent_offset=-0.6,
            activation_gate=1.0,
        ),
    ]

    # --- Offensive Turn Subsystem ---
    # Inputs: [normalized_error, norm_dist, norm_turn_rate_ship]
    r_off_rules = [
        RuleGene(
            antecedents=[(0, 2), (1, 4), (2, 7)],
            consequent_coefficients=[0.4, 0.0, -0.1],
            consequent_offset=0.3,
            activation_gate=1.0,
        ),
        RuleGene(
            antecedents=[(0, 0), (1, 4), (2, 7)],
            consequent_coefficients=[0.4, 0.0, -0.1],
            consequent_offset=-0.3,
            activation_gate=1.0,
        ),
    ]

    # --- Defensive Turn Subsystem ---
    # Inputs: [norm_lateral_threat, normalized_tti, norm_speed_ship]
    r_def_rules = [
        RuleGene(
            antecedents=[(0, 0), (1, 3), (2, 7)],
            consequent_coefficients=[-0.5, 0.0, 0.0],
            consequent_offset=0.5,
            activation_gate=1.0,
        ),
    ]

    # --- Threat Gate Subsystem ---
    # Inputs: [normalized_tti, norm_density]
    gate_rules = [
        RuleGene(
            antecedents=[(0, 3), (1, 8)],
            consequent_coefficients=[-0.6, 0.4],
            consequent_offset=0.7,
            activation_gate=1.0,
        ),
        RuleGene(
            antecedents=[(0, 5), (1, 6)],
            consequent_coefficients=[0.0, 0.0],
            consequent_offset=0.0,
            activation_gate=1.0,
        ),
    ]

    return ControllerChromosome(
        thrust_offensive=SubsystemChromosome(
            membership_function_pool=make_isolated_pool(),
            rules=t_off_rules,
        ),
        thrust_defensive=SubsystemChromosome(
            membership_function_pool=make_isolated_pool(),
            rules=t_def_rules,
        ),
        turn_offensive=SubsystemChromosome(
            membership_function_pool=make_isolated_pool(),
            rules=r_off_rules,
        ),
        turn_defensive=SubsystemChromosome(
            membership_function_pool=make_isolated_pool(),
            rules=r_def_rules,
        ),
        threat_gate=SubsystemChromosome(
            membership_function_pool=make_isolated_pool(),
            rules=gate_rules,
        ),
    )
