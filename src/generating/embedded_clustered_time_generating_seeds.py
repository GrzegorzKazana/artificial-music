import numpy as np

from src.generating.embedded_generating_seeds import seed_generators as embedded_seed_generators

DEFAULT_DURATION_CLUSTER_SIZE = 24


def get_random_duration(length, batch_size):
    res = np.zeros((batch_size * length, DEFAULT_DURATION_CLUSTER_SIZE))

    for i in range(batch_size * length):
        res[i, np.random.randint(DEFAULT_DURATION_CLUSTER_SIZE)] = 1

    return res.reshape((batch_size, length, DEFAULT_DURATION_CLUSTER_SIZE))


def wrap_seed_generator_w_duration(seed_gen):
    def inner(length, input_size, **kwargs):
        batch_size = kwargs['batch_size'] or 16
        seed = seed_gen(length, input_size, **kwargs)
        duration = get_random_duration(length, batch_size)

        return np.concatenate((seed, duration), axis=2)

    return inner


seed_generators = {
    k: wrap_seed_generator_w_duration(v) for k, v in embedded_seed_generators.items()
}
