import pytest

from asian_asteroid_destroyers.chromosomes import (
    ControllerChromosome,
    MFParamTrapezoidal,
    MFParamTriangular,
    RuleGene,
    SubsystemChromosome,
    compile_subsystem,
    create_geometry_repair_maps,
    flatten_chromosome,
    get_default_controller_template,
    hydrate_chromosome_blueprint,
    repair_geometry_invariants,
)


def test_default_controller_template() -> None:
    template = get_default_controller_template()
    assert isinstance(template, ControllerChromosome)
    assert isinstance(template.thrust_offensive, SubsystemChromosome)
    assert isinstance(template.thrust_defensive, SubsystemChromosome)
    assert isinstance(template.turn_offensive, SubsystemChromosome)
    assert isinstance(template.turn_defensive, SubsystemChromosome)
    assert isinstance(template.threat_gate, SubsystemChromosome)
    assert len(template.thrust_offensive.membership_function_pool) > 0
    assert len(template.thrust_offensive.rules) > 0


def test_flatten_and_hydrate_roundtrip() -> None:
    template = get_default_controller_template()
    flat_genes = flatten_chromosome(template)

    assert isinstance(flat_genes, list)
    assert all(isinstance(g, float) for g in flat_genes)
    assert len(flat_genes) > 0

    hydrated = hydrate_chromosome_blueprint(flat_genes, template)
    assert isinstance(hydrated, ControllerChromosome)
    assert hydrated == template


def test_hydrate_chromosome_blueprint_mismatch_raises() -> None:
    """Test that hydrating with an incorrect gene count raises a ValueError."""
    template = get_default_controller_template()
    flat_genes = flatten_chromosome(template)

    mismatched_genes = [*flat_genes, 99.9]

    with pytest.raises(ValueError, match="Chromosome gene count mismatch"):
        _ = hydrate_chromosome_blueprint(mismatched_genes, template)


def test_compile_subsystem() -> None:
    template = get_default_controller_template()
    engine = compile_subsystem(template.thrust_offensive)

    result = engine.infer([0.1, 0.2, 0.0])
    assert isinstance(result, float)


def test_geometry_repair_maps_and_invariants() -> None:
    subsystem = SubsystemChromosome(
        membership_function_pool=[
            MFParamTriangular(left=0.2, center=0.1, right=0.3),
            MFParamTrapezoidal(
                left_base=0.0, left_peak=0.1, right_peak=0.05, right_base=0.2
            ),
        ],
        rules=[
            RuleGene(
                antecedents=[(0, 0)],
                consequent_coefficients=[1.0],
                consequent_offset=0.0,
                activation_gate=1.0,
            )
        ],
    )

    controller = ControllerChromosome(
        thrust_offensive=subsystem,
        thrust_defensive=subsystem,
        turn_offensive=subsystem,
        turn_defensive=subsystem,
        threat_gate=subsystem,
    )

    triangular_map, trapezoidal_map = create_geometry_repair_maps(controller)
    assert triangular_map.ndim == 2
    assert trapezoidal_map.ndim == 2

    flat_genes = flatten_chromosome(controller)
    repaired_genes = repair_geometry_invariants(
        flat_genes,
        triangular_map,
        trapezoidal_map,
        min_width=0.05,
    )

    assert isinstance(repaired_genes, list)
    assert len(repaired_genes) == len(flat_genes)
