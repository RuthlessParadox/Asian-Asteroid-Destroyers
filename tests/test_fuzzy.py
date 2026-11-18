import numpy as np
import pytest

from asian_asteroid_destroyers.fuzzy import (
    FuzzyClause,
    RuleEvaluationResult,
    TrapezoidalMF,
    TriangularMF,
    TSKInferenceEngine,
    TSKRule,
)


class TestTriangularMF:
    def test_valid_initialization_and_evaluation(self) -> None:
        mf = TriangularMF(0.0, 5.0, 10.0)

        assert mf.evaluate(-1.0) == 0.0
        assert mf.evaluate(0.0) == 0.0
        assert mf.evaluate(10.0) == 0.0
        assert mf.evaluate(11.0) == 0.0

        assert mf.evaluate(5.0) == 1.0
        assert mf.evaluate(2.5) == 0.5
        assert mf.evaluate(7.5) == 0.5

    def test_invalid_initialization(self) -> None:
        with pytest.raises(ValueError, match="Must satisfy condition"):
            _ = TriangularMF(5.0, 0.0, 10.0)

        with pytest.raises(ValueError, match="Must satisfy condition"):
            _ = TriangularMF(0.0, 10.0, 5.0)

    def test_repr(self) -> None:
        mf = TriangularMF(1.0, 2.0, 3.0)
        assert repr(mf) == "TriangularMF(left=1.00, center=2.00, right=3.00)"


class TestTrapezoidalMF:
    def test_valid_initialization_and_evaluation(self) -> None:
        mf = TrapezoidalMF(0.0, 2.0, 4.0, 6.0)

        assert mf.evaluate(-1.0) == 0.0
        assert mf.evaluate(0.0) == 0.0
        assert mf.evaluate(6.0) == 0.0
        assert mf.evaluate(7.0) == 0.0

        assert mf.evaluate(2.0) == 1.0
        assert mf.evaluate(3.0) == 1.0
        assert mf.evaluate(4.0) == 1.0

        assert mf.evaluate(1.0) == 0.5
        assert mf.evaluate(5.0) == 0.5

    def test_invalid_initialization(self) -> None:
        with pytest.raises(
            ValueError,
            match="Invalid trapezoidal point constraints",
        ):
            _ = TrapezoidalMF(0.0, 5.0, 3.0, 6.0)

    def test_repr(self) -> None:
        mf = TrapezoidalMF(1.0, 2.0, 3.0, 4.0)
        assert (
            repr(mf)
            == "TrapezoidalMF(left_base=1.00, left_peak=2.00, "
            + "right_peak=3.00, right_base=4.00)"
        )


class TestFuzzyClause:
    def test_clause_properties_and_repr(self) -> None:
        mf = TriangularMF(0.0, 1.0, 2.0)
        clause = FuzzyClause(input_index=1, membership_function=mf)

        assert clause.input_index == 1
        assert clause.membership_function == mf
        assert "FuzzyClause(input_index=1" in repr(clause)


class TestTSKRule:
    def test_rule_evaluation_success(self) -> None:
        mf_1 = TriangularMF(0.0, 5.0, 10.0)
        mf_2 = TrapezoidalMF(0.0, 2.0, 4.0, 6.0)

        clauses = [
            FuzzyClause(input_index=0, membership_function=mf_1),
            FuzzyClause(input_index=1, membership_function=mf_2),
        ]

        rule = TSKRule(
            antecedents=clauses,
            consequent_coefficients=[1.0, 2.0],
            consequent_offset=3.0,
        )

        inputs = [5.0, 3.0]
        result = rule.evaluate(inputs)

        assert isinstance(result, RuleEvaluationResult)
        assert result.firing_strength == 1.0
        assert result.consequent_value == 14.0

    def test_rule_evaluation_short_circuit(self) -> None:
        mf = TriangularMF(0.0, 5.0, 10.0)
        clauses = [FuzzyClause(input_index=0, membership_function=mf)]

        rule = TSKRule(
            antecedents=clauses,
            consequent_coefficients=[1.0],
            consequent_offset=0.0,
        )

        inputs = [15.0]
        result = rule.evaluate(inputs)

        assert result.firing_strength == 0.0
        assert result.consequent_value == 0.0

    def test_repr(self) -> None:
        rule = TSKRule([], [1.0], 5.5)
        assert "TSKRule" in repr(rule)


class TestTSKInferenceEngine:
    def test_inference_engine_execution(self) -> None:
        rules_count = 1
        inputs_count = 2

        mf_types = np.array([0], dtype=np.int8)
        mf_params = np.array([[0.0, 5.0, 10.0, 0.0]], dtype=np.float64)

        antecedent_map = np.array([[0, 0]], dtype=np.intp)

        coefficients = np.array([[1.0, 1.0]], dtype=np.float64)
        offsets = np.array([0.0], dtype=np.float64)
        activation_masks = np.array([1.0], dtype=np.float64)

        engine = TSKInferenceEngine(
            rules_count=rules_count,
            inputs_count=inputs_count,
            antecedent_map=antecedent_map,
            mf_types=mf_types,
            mf_params=mf_params,
            coefficients=coefficients,
            offsets=offsets,
            activation_masks=activation_masks,
        )

        inputs = [5.0, 5.0]
        result = engine.infer(inputs)

        assert result == pytest.approx(10.0)

    def test_inference_engine_fallback(self) -> None:
        rules_count = 1
        inputs_count = 1

        mf_types = np.array([0], dtype=np.int8)
        mf_params = np.array([[0.0, 5.0, 10.0, 0.0]], dtype=np.float64)
        antecedent_map = np.array([[0]], dtype=np.intp)
        coefficients = np.array([[1.0]], dtype=np.float64)
        offsets = np.array([0.0], dtype=np.float64)
        activation_masks = np.array([1.0], dtype=np.float64)

        engine = TSKInferenceEngine(
            rules_count=rules_count,
            inputs_count=inputs_count,
            antecedent_map=antecedent_map,
            mf_types=mf_types,
            mf_params=mf_params,
            coefficients=coefficients,
            offsets=offsets,
            activation_masks=activation_masks,
        )

        result = engine.infer([20.0], fallback_default=42.0)
        assert result == 42.0

    def test_repr(self) -> None:
        engine = TSKInferenceEngine(
            rules_count=3,
            inputs_count=1,
            antecedent_map=np.zeros((3, 1), dtype=np.intp),
            mf_types=np.zeros(1, dtype=np.int8),
            mf_params=np.zeros((1, 4), dtype=np.float64),
            coefficients=np.zeros((3, 1), dtype=np.float64),
            offsets=np.zeros(3, dtype=np.float64),
            activation_masks=np.ones(3, dtype=np.float64),
        )
        assert repr(engine) == "TSKInferenceEngine(rules_count=3)"
