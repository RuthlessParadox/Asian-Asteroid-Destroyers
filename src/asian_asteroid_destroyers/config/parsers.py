import argparse


def parse_positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid int value: '{value}'")

    if parsed <= 0:
        raise argparse.ArgumentTypeError(
            f"Value must be positive integer (got {parsed})."
        )
    return parsed


def parse_non_negative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid int value: '{value}'")

    if parsed < 0:
        raise argparse.ArgumentTypeError(
            f"Value must be non-negative integer (got {parsed})."
        )
    return parsed


def parse_non_negative_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid float value: '{value}'")

    if parsed < 0.0:
        raise argparse.ArgumentTypeError(
            f"Value must be a non-negative float (got {parsed})."
        )
    return parsed
