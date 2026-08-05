import numpy as np


def create_palette(
    number_of_classes: int,
    seed: int = 42,
) -> np.ndarray:
    generator = np.random.default_rng(seed)

    return generator.integers(
        low=0,
        high=256,
        size=(number_of_classes, 3),
        dtype=np.uint8,
    )