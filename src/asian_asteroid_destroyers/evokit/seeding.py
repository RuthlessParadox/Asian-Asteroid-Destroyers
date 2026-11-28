import numpy as np


def derive_generation_seed(master_seed: int, generation: int) -> int:
    sequence = np.random.SeedSequence([master_seed, generation])
    draws = sequence.generate_state(1, dtype=np.uint32)
    seed_value = int(draws[0])  # pyright: ignore[reportAny]
    return seed_value if seed_value > 0 else 1
