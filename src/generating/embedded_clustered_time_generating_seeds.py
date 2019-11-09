import numpy as np
import random

from .common import create_noise_adder
from .sparse_clustered_time_generating_seeds import wrap_seed_generator_w_duration


# seeds


def random_noise_seed(length, input_size, scaler=0.5, batch_size=16):
    return np.random.normal(size=(batch_size, length, input_size)) * scaler


def zero_seed(length, input_size, word_vectors, batch_size=16):
    zero_vector = word_vectors['']
    return np.tile(zero_vector, (batch_size, length, 1))


def const_frame_seed(length, input_size, word_vectors, batch_size=16):
    """
    assumed length is 1
    """
    res = zero_seed(1, input_size, word_vectors, batch_size)
    for i in range(batch_size):
        note = word_vectors[random.choice(
            list(word_vectors.vocab.keys()))]
        res[i, :, :] = note

    return res


def multi_note_seed(length, input_size, word_vectors, batch_size=16):
    res = np.zeros((batch_size, length, input_size))
    for j in range(batch_size):
        notes_keys = random.sample(list(word_vectors.vocab.keys()), length)
        notes = [word_vectors[n] for n in notes_keys]
        res[j, :, :] = notes

    return res


def multi_note_harmonic_seed(length, input_size, word_vectors, batch_size=16):
    res = np.zeros((batch_size, length, input_size))
    for j in range(batch_size):
        base_note_key = random.choice(
            list(word_vectors.vocab.keys()))
        similar_notes_keys = [name for name, _ in word_vectors.similar_by_word(
            base_note_key, topn=length - 1)]
        notes_keys = [base_note_key, *similar_notes_keys]
        notes = [word_vectors[k] for k in notes_keys]
        random.shuffle(notes)
        res[j, :, :] = notes

    return res


random_noise_adder = create_noise_adder(random_noise_seed)

# refer to artifacts/embedded_seed_generators.json


def get_seed_generators(duration_dict, ignore_shortest=True):
    wrapper = wrap_seed_generator_w_duration(duration_dict, ignore_shortest)

    return {
        "random_noise_seed": wrapper(random_noise_seed),
        "zero_seed": wrapper(zero_seed),
        "const_frame_seed": wrapper(const_frame_seed),
        "multi_note_seed": wrapper(multi_note_seed),
        "multi_note_harmonic_seed": wrapper(multi_note_harmonic_seed),
        "const_frame_seed_noise": wrapper(random_noise_adder(const_frame_seed)),
        "multi_note_seed_noise": wrapper(random_noise_adder(multi_note_seed)),
        "multi_note_harmonic_seed_noise": wrapper(random_noise_adder(multi_note_harmonic_seed)),
    }
