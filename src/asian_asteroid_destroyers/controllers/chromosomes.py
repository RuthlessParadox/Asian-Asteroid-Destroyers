from typing import NamedTuple


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
