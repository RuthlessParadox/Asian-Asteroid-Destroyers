import argparse

import pytest
from hypothesis import given, strategies as st

from asian_asteroid_destroyers.config import (
    parse_non_negative_float,
    parse_positive_int,
)


@pytest.mark.parametrize(
    "raw_value,expected",
    [
        ("0", 0),
        ("5", 5),
        ("100", 100),
    ],
)
def test_parse_positive_int_valid(raw_value: str, expected: int) -> None:
    assert parse_positive_int(raw_value) == expected


@pytest.mark.parametrize(
    "raw_value,expected_message",
    [
        ("abc", "invalid int value: 'abc'"),
        ("3.14", "invalid int value: '3.14'"),
    ],
)
def test_parse_positive_int_invalid_syntax(
    raw_value: str,
    expected_message: str,
) -> None:
    with pytest.raises(argparse.ArgumentTypeError) as exc_info:
        _ = parse_positive_int(raw_value)
    assert str(exc_info.value) == expected_message


@pytest.mark.parametrize(
    "raw_value,expected_message",
    [
        ("-1", "Value must be non-negative integer (got -1)."),
        ("-100", "Value must be non-negative integer (got -100)."),
    ],
)
def test_parse_positive_int_negative(
    raw_value: str,
    expected_message: str,
) -> None:
    with pytest.raises(argparse.ArgumentTypeError) as exc_info:
        _ = parse_positive_int(raw_value)
    assert str(exc_info.value) == expected_message


@given(value=st.integers(min_value=0))
def test_parse_positive_int_valid_property(value: int) -> None:
    raw = str(value)
    assert parse_positive_int(raw) == value


@given(value=st.integers(max_value=-1))
def test_parse_positive_int_negative_property(value: int) -> None:
    raw = str(value)
    with pytest.raises(argparse.ArgumentTypeError):
        _ = parse_positive_int(raw)


@given(
    raw=st.text(
        alphabet=st.characters(blacklist_categories=("Nd",)),
        min_size=1,
    )
)
def test_parse_positive_int_invalid_syntax_property(raw: str) -> None:
    try:
        _ = int(raw)
        return
    except ValueError:
        pass

    with pytest.raises(argparse.ArgumentTypeError):
        _ = parse_positive_int(raw)


@pytest.mark.parametrize(
    "raw_value,expected",
    [
        ("0.0", 0.0),
        ("0", 0.0),
        ("3.14", 3.14),
        ("42.5", 42.5),
    ],
)
def test_parse_non_negative_float_valid(
    raw_value: str,
    expected: float,
) -> None:
    assert parse_non_negative_float(raw_value) == expected


@pytest.mark.parametrize(
    "raw_value,expected_message",
    [
        ("abc", "invalid float value: 'abc'"),
    ],
)
def test_parse_non_negative_float_invalid_syntax(
    raw_value: str,
    expected_message: str,
) -> None:
    with pytest.raises(argparse.ArgumentTypeError) as exc_info:
        _ = parse_non_negative_float(raw_value)
    assert str(exc_info.value) == expected_message


@pytest.mark.parametrize(
    "raw_value,expected_message",
    [
        ("-0.1", "Value must be a non-negative float (got -0.1)."),
        ("-5.5", "Value must be a non-negative float (got -5.5)."),
    ],
)
def test_parse_non_negative_float_negative(
    raw_value: str,
    expected_message: str,
) -> None:
    with pytest.raises(argparse.ArgumentTypeError) as exc_info:
        _ = parse_non_negative_float(raw_value)
    assert str(exc_info.value) == expected_message


@given(value=st.floats(min_value=0.0, allow_nan=False, allow_infinity=False))
def test_parse_non_negative_float_valid_property(value: float) -> None:
    raw = str(value)
    assert parse_non_negative_float(raw) == value


@given(value=st.floats(max_value=-1e-7, allow_nan=False, allow_infinity=False))
def test_parse_non_negative_float_negative_property(value: float) -> None:
    raw = str(value)
    with pytest.raises(argparse.ArgumentTypeError):
        _ = parse_non_negative_float(raw)


@given(
    raw=st.text(
        alphabet=st.characters(blacklist_categories=("Nd",)),
        min_size=1,
    )
)
def test_parse_non_negative_float_invalid_syntax_property(raw: str) -> None:
    try:
        _ = float(raw)
        return
    except ValueError:
        pass

    with pytest.raises(argparse.ArgumentTypeError):
        _ = parse_non_negative_float(raw)
