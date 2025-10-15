import random as _random


def seed(value: int) -> None:
    # Set the seed so runs are reproducible (same random sequence)
    _random.seed(value)


def randint_inclusive(low: int, high: int) -> int:
    # Return an integer in [low, high], inclusive on both ends
    return _random.randint(low, high)


