import pytest

from asian_asteroid_destroyers.geometry import (
    denorm_thrust,
    denorm_turn_rate,
    norm_ammo_ratio,
    norm_angle,
    norm_asteroid_count,
    norm_distance,
    norm_size,
    norm_speed_asteroid,
    norm_speed_ship,
    norm_tti,
    norm_turn_rate_ship,
    norm_wrapped_distance,
)


def test_norm_angle() -> None:
    assert norm_angle(180.0) == pytest.approx(1.0)
    assert norm_angle(90.0) == pytest.approx(0.5)
    assert norm_angle(0.0) == pytest.approx(0.0)
    assert norm_angle(-180.0) == pytest.approx(-1.0)


def test_norm_distance() -> None:
    assert norm_distance(50.0, 100.0) == pytest.approx(0.5)
    assert norm_distance(100.0, 100.0) == pytest.approx(1.0)
    assert norm_distance(0.0, 100.0) == pytest.approx(0.0)


def test_norm_wrapped_distance() -> None:
    assert norm_wrapped_distance(1.25, 3.0, 4.0) == pytest.approx(0.5)
    assert norm_wrapped_distance(0.0, 3.0, 4.0) == pytest.approx(0.0)
    assert norm_wrapped_distance(5.0, 3.0, 4.0) == pytest.approx(1.0)


def test_norm_speed_asteroid() -> None:
    assert norm_speed_asteroid(240.0) == pytest.approx(1.0)
    assert norm_speed_asteroid(120.0) == pytest.approx(0.5)
    assert norm_speed_asteroid(0.0) == pytest.approx(0.0)


def test_norm_speed_ship() -> None:
    assert norm_speed_ship(240.0) == pytest.approx(1.0)
    assert norm_speed_ship(60.0) == pytest.approx(0.25)
    assert norm_speed_ship(0.0) == pytest.approx(0.0)


def test_norm_turn_rate_ship() -> None:
    assert norm_turn_rate_ship(180.0) == pytest.approx(1.0)
    assert norm_turn_rate_ship(90.0) == pytest.approx(0.5)
    assert norm_turn_rate_ship(0.0) == pytest.approx(0.0)


def test_norm_tti() -> None:
    assert norm_tti(10.0) == pytest.approx(1.0)
    assert norm_tti(5.0) == pytest.approx(0.5)
    assert norm_tti(0.0) == pytest.approx(0.0)


def test_norm_size() -> None:
    assert norm_size(1.0) == pytest.approx(-0.5)
    assert norm_size(2.0) == pytest.approx(0.0)
    assert norm_size(4.0) == pytest.approx(1.0)


def test_norm_asteroid_count() -> None:
    assert norm_asteroid_count(100.0) == pytest.approx(1.0)
    assert norm_asteroid_count(50.0) == pytest.approx(0.5)
    assert norm_asteroid_count(0.0) == pytest.approx(0.0)


def test_norm_ammo_ratio() -> None:
    assert norm_ammo_ratio(1.0) == pytest.approx(0.5)
    assert norm_ammo_ratio(2.0) == pytest.approx(1.0)
    assert norm_ammo_ratio(0.0) == pytest.approx(0.0)
    assert norm_ammo_ratio(3.5) == pytest.approx(1.0)


def test_denorm_thrust() -> None:
    assert denorm_thrust(1.0) == pytest.approx(480.0)
    assert denorm_thrust(0.5) == pytest.approx(240.0)
    assert denorm_thrust(0.0) == pytest.approx(0.0)


def test_denorm_turn_rate() -> None:
    assert denorm_turn_rate(1.0) == pytest.approx(180.0)
    assert denorm_turn_rate(0.5) == pytest.approx(90.0)
    assert denorm_turn_rate(0.0) == pytest.approx(0.0)
