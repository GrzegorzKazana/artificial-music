import numpy as np
import random
from math import inf

# helpers


common_note_times = [
    4, 2, 1.5, 1, 0.75, 0.5,
]


def get_common_note_time():
    return random.choice(common_note_times)


def generate_random_rest_vector(length_seq, batch_size=1):
    """
    [[velocity, delta_time, duration], ...]
    """
    def generate_random():
        vel = 1 if np.random.rand() < 0.75 else 1 - np.random.rand() ** 2
        d_time = 0 if np.random.rand() < 0.25 else get_common_note_time()
        duration = 0 if np.random.rand() < 0.5 else get_common_note_time()
        return [vel, d_time, duration]

    return np.array([
        [generate_random() for __ in range(length_seq)] for _ in range(batch_size)
    ])


def zip_notes_with_rest(batch_note_seqs, batch_rest_seqs):
    return np.concatenate((batch_note_seqs, batch_rest_seqs), axis=2)


def create_random_noise_adder(noise_scale=0.5, clip=(-inf, inf)):
    def inner(seed_gen):
        def inner2(length, input_size, **kwargs):
            noise = np.random.normal(
                size=(kwargs['batch_size'], length, input_size)) * noise_scale
            seed = seed_gen(length, input_size, **kwargs)
            seed[:, :, :input_size] += noise

            return np.clip(seed, clip[0], clip[1])

        return inner2

    return inner


# seeds


def random_noise_seed(length, input_size, scaler=0.5, batch_size=16):
    notes = np.random.normal(size=(batch_size, length, input_size)) * scaler
    rest = generate_random_rest_vector(length, batch_size)

    return zip_notes_with_rest(notes, rest)


def const_frame_seed(length, input_size, word_vectors, batch_size=16):
    """
    assumend length is 1, param is ignored
    """
    note_vecs = [word_vectors[random.choice(
        list(word_vectors.vocab.keys()))] for _ in range(batch_size)]

    notes = np.array(note_vecs).reshape(batch_size, 1, input_size)
    rest = generate_random_rest_vector(1, batch_size)

    return zip_notes_with_rest(notes, rest)


def multi_note_seed(length, input_size, word_vectors, batch_size=16):
    """
    assumes length == num_notes
    """
    notes = np.zeros((batch_size, length, input_size))

    for j in range(batch_size):
        notes_keys = random.sample(list(word_vectors.vocab.keys()), length)
        note_vecs = [word_vectors[n] for n in notes_keys]
        notes[j] = note_vecs

    rest = generate_random_rest_vector(length, batch_size)

    return zip_notes_with_rest(notes, rest)


def multi_note_harmonic_seed(length, input_size, word_vectors, batch_size=16):
    """
    assumes length == num_notes
    """
    notes = np.zeros((batch_size, length, input_size))

    for j in range(batch_size):
        base_note_key = random.choice(
            list(word_vectors.vocab.keys()))

        similar_notes_keys = [name for name, _ in word_vectors.similar_by_word(
            base_note_key, topn=length - 1)]

        notes_keys = [base_note_key, *similar_notes_keys]
        notes_vecs = [word_vectors[k] for k in notes_keys]
        random.shuffle(notes_vecs)

        notes[j] = notes_vecs

    rest = generate_random_rest_vector(length, batch_size)

    return zip_notes_with_rest(notes, rest)


random_noise_adder = create_random_noise_adder()

seed_generators = {
    "random_noise_seed": random_noise_seed,
    "const_frame_seed": const_frame_seed,
    "multi_note_seed": multi_note_seed,
    "multi_note_harmonic_seed": multi_note_harmonic_seed,
    "const_frame_seed_noise": random_noise_adder(const_frame_seed),
    "multi_note_seed_noise": random_noise_adder(multi_note_seed),
    "multi_note_harmonic_seed_noise": random_noise_adder(multi_note_harmonic_seed),
}
