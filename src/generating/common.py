import numpy as np


def create_noise_adder(noise_gen, noise_scale=0.5, **nkwargs):
    def inner(seed_gen):
        return lambda length, input_size, **kwargs: np.clip((
            noise_scale * noise_gen(length, input_size, **nkwargs)
            + seed_gen(length, input_size, **kwargs)), 0, 1
        )
    return inner
