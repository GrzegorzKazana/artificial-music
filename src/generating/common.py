import numpy as np
from math import inf


def create_noise_adder(noise_gen, noise_scale=0.5, clip=(-inf, inf), **nkwargs):
    def inner(seed_gen):
        return lambda length, input_size, **kwargs: np.clip((
            noise_scale * noise_gen(length, input_size, **nkwargs, **
                                    {k: v for k, v in kwargs.items() if k == 'batch_size'})
            + seed_gen(length, input_size, **kwargs)), clip[0], clip[1]
        )
    return inner
