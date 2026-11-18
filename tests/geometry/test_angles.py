import pytest

from asian_asteroid_destroyers.geometry import trim_angle


def test_trim_angle_zero() -> None:
    assert trim_angle(0.0) == pytest.approx(0.0)


def test_trim_angle_within_range() -> None:
    assert trim_angle(45.5) == pytest.approx(45.5)
    assert trim_angle(-45.5) == pytest.approx(-45.5)
    assert trim_angle(180.0) == pytest.approx(-180.0)


def test_trim_angle_positive_overflow() -> None:
    assert trim_angle(360.0) == pytest.approx(0.0)
    assert trim_angle(400.0) == pytest.approx(40.0)
    assert trim_angle(540.0) == pytest.approx(-180.0)


def test_trim_angle_negative_overflow() -> None:
    assert trim_angle(-360.0) == pytest.approx(0.0)
    assert trim_angle(-400.0) == pytest.approx(-40.0)


def test_trim_angle_boundaries() -> None:
    assert trim_angle(180.1) == pytest.approx(-179.9)
    assert trim_angle(-180.1) == pytest.approx(179.9)
