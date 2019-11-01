import numpy as np


def identity(x): return x


def recurrent_generate(model, seed, seq_length, window_size, is_binary=False, transform_output=identity):
    """
    Predicts only next step each iteration,
    and then appends generated step to input,
    iterations lasts until desired length is achived.
    Assumes model is returning sequences of dim=3
    """
    x = seed
    accum = [seed]
    for _ in range(seq_length - window_size):
        res = transform_output(model.predict(x))
        next_timestep = res[:, -1:, :].round() if is_binary else res[:, -1:, :]
        x = np.concatenate([x, next_timestep], axis=1)[:, -window_size:, :]
        accum.append(next_timestep)

    return np.concatenate(accum, axis=1)


def linear_generate(model, seed, is_binary=False):
    """
    Transforms given sequence in latent space.
    Assumes model is returning sequences of dim=3
    """
    res = model.predict(seed)
    return res.round() if is_binary else res
