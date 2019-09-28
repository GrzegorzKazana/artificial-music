import numpy as np


def create_noise_adder(noise_gen, noise_scale=0.5, **nkwargs):
    def inner(seed_gen, **kwargs):
        return lambda length, batch_size: np.clip((
            noise_scale * noise_gen(length, batch_size, **nkwargs)
            + seed_gen(length, batch_size, **kwargs)), 0, 1
        )
    return inner
