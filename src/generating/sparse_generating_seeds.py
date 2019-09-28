import numpy as np
import random

# helpers
scales = {
    "natural_minor": [0, 2, 3, 5, 7, 8, 10, 12],
    "harmonic_minor": [0, 2, 3, 5, 7, 8, 11, 12],
    "melodic_minor": [0, 2, 3, 5, 7, 9, 11, 12],
    "major": [0, 2, 4, 5, 7, 9, 11, 12],
    "dorian": [0, 2, 4, 5, 7, 9, 10, 12],
    "mixolydian": [0, 2, 4, 5, 7, 9, 10, 12],
    "pentatonic": [0, 3, 5, 7, 10, 12],
}


def get_random_scale():
    return random.choice(list(scales.values()))


def get_random_notes_from_random_scale(base_note, n_notes=1):
    scale = get_random_scale()
    two_octave_scale = scale + [-x for x in scale]
    return [base_note + random.choice(two_octave_scale) for _ in range(n_notes)]


def create_noise_adder(noise_gen, noise_scale=0.5, **nkwargs):
    def inner(seed_gen, **kwargs):
        return lambda length, batch_size: np.clip((
            noise_scale * noise_gen(length, batch_size, **nkwargs)
            + seed_gen(length, batch_size, **kwargs)), 0, 1
        )
    return inner


# seeds
def random_noise_seed(length, input_size, batch_size=16):
    return np.random.random((batch_size, length, input_size)) * 0.5


def zero_seed(length, input_size, batch_size=16):
    return np.zeros((batch_size, length, input_size))


def band_noise_seed(length, input_size, batch_size=16):
    """
    notes in middle range have highier values
    taken from normal distribution for each "hand"
    mu1 = 76, mu2 = 48, std = 12
    """
    mu1 = 76
    mu2 = 48
    std = 12
    hand1 = np.exp(-(np.arange(128) - mu1) ** 2 / (2 * std ** 2))
    hand2 = np.exp(-(np.arange(128) - mu2) ** 2 / (2 * std ** 2))
    hands = hand1 + hand2
    randomness = np.random.random(((batch_size, length, input_size))) * 0.5

    return randomness * hands


def single_note_seed(length, input_size, batch_size=16):
    res = np.zeros((batch_size, length, input_size))
    for i in range(batch_size):
        note = int(np.random.normal(72, 16))
        note = np.clip(note, 0, 127)
        res[i, :, note] = 1

    return res


def single_note_short_seed(length, input_size, batch_size=16, max_note_length=25):
    res = np.zeros((batch_size, length, input_size))
    for i in range(batch_size):
        note = int(np.random.normal(72, 16))
        note = np.clip(note, 0, 127)
        note_on = np.random.randint(length)
        note_off = np.random.randint(
            note_on, min(note_on + max_note_length, length))
        res[i, note_on:note_off, note] = 1

    return res


def multi_note_seed(length, input_size, batch_size=16, num_notes=5):
    res = np.zeros((batch_size, length, input_size))
    for j in range(batch_size):
        notes = np.random.normal(72, 16, size=num_notes).round().astype(int)
        notes = np.clip(notes, 0, 127)
        note_change_times = np.sort(np.random.choice(
            np.arange(length), size=num_notes + 1))
        for i, note in enumerate(notes):
            res[j, note_change_times[i]:note_change_times[i + 1], note] = 1
    return res


def multi_note_harmonic_seed(length, input_size, batch_size=16, num_notes=5):
    res = np.zeros((batch_size, length, input_size))
    base_note = 72
    for j in range(batch_size):
        notes = get_random_notes_from_random_scale(base_note, num_notes)
        note_change_times = np.sort(np.random.choice(
            np.arange(length), size=num_notes + 1))
        for i, note in enumerate(notes):
            res[j, note_change_times[i]:note_change_times[i + 1], note] = 1
    return res


def multi_note_simult_seed(length, input_size, batch_size=16, num_notes=15, max_note_length=25):
    res = np.zeros((batch_size, length, input_size))
    for j in range(batch_size):
        notes = np.random.normal(72, 16, size=num_notes).round().astype(int)
        notes = np.clip(notes, 0, 127)
        for i, note in enumerate(notes):
            note_on = np.random.randint(length)
            note_off = np.random.randint(
                note_on, min(note_on + max_note_length, length))
            res[j, note_on:note_off, note] = 1
    return res


def multi_note_simult_harmonic_seed(length, input_size, batch_size=16, num_notes=15, max_note_length=25):
    res = np.zeros((batch_size, length, input_size))
    base_note = 72
    for j in range(batch_size):
        notes = get_random_notes_from_random_scale(base_note, num_notes)
        for i, note in enumerate(notes):
            note_on = np.random.randint(length)
            note_off = np.random.randint(
                note_on, min(note_on + max_note_length, length))
            res[j, note_on:note_off, note] = 1
    return res


band_noise_adder = create_noise_adder(band_noise_seed)
random_noise_adder = create_noise_adder(random_noise_seed)

# refer to artifacts/seed_generators.jpg
seed_generators = {
    "random_noise_seed": random_noise_seed,
    "zero_seed": zero_seed,
    "band_noise_seed": band_noise_seed,
    "single_note_seed": single_note_seed,
    "single_note_short_seed": single_note_short_seed,
    "multi_note_seed": multi_note_seed,
    "multi_note_harmonic_seed": multi_note_harmonic_seed,
    "multi_note_simult_seed": multi_note_simult_seed,
    "multi_note_simult_harmonic_seed": multi_note_simult_harmonic_seed,
    "single_note_seed_noise": random_noise_adder(single_note_seed),
    "single_note_seed_band noise": band_noise_adder(single_note_seed),
    "single_note_short_seed_noise": random_noise_adder(single_note_short_seed),
    "single_note_short_seed_band noise": band_noise_adder(single_note_short_seed),
    "multi_note_harmonic_seed_noise": random_noise_adder(multi_note_harmonic_seed),
    "multi_note_harmonic_seed_band noise": band_noise_adder(multi_note_harmonic_seed),
    "multi_note_simult_harmonic_seed_noise": random_noise_adder(multi_note_simult_harmonic_seed),
    "multi_note_simult_harmonic_seed_band noise": band_noise_adder(multi_note_simult_harmonic_seed),
}
