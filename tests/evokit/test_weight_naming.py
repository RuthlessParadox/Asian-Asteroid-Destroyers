from asian_asteroid_destroyers.evokit import get_clean_weight_string


def test_weight_naming_handles_integers() -> None:
    weights = (1.0, -2.0, 0.0)
    result = get_clean_weight_string(weights)
    assert result == "1_0_neg2_0_0_0"


def test_weight_naming_handles_complex_decimals() -> None:
    weights = (0.005, -123.456)
    result = get_clean_weight_string(weights)
    assert result == "0_005_neg123_456"


def test_weight_naming_handles_empty_tuple() -> None:
    weights: tuple[float, ...] = ()
    result = get_clean_weight_string(weights)
    assert result == ""
