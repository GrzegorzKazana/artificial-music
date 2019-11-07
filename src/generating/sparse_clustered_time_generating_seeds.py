import numpy as np

from src.generating.sparse_generating_seeds import random_noise_seed, zero_seed, band_noise_seed, get_random_notes_from_random_scale, random_noise_adder, band_noise_adder

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
        dur_length = seed.shape[1]
        duration = get_random_duration(dur_length, batch_size)

        return np.concatenate((seed, duration), axis=2)

    return inner

# seeds


def single_note_seed(length, input_size, batch_size=16):
    """
    ignores length, assumes 1
    """
    res = np.zeros((batch_size, 1, input_size))
    for i in range(batch_size):
        note = int(np.random.normal(72, 16))
        note = np.clip(note, 0, 127)
        res[i, :, note] = 1

    return res


def multi_note_seed(length, input_size, batch_size=16):
    """
    length == num_notes
    """
    res = np.zeros((batch_size, length, input_size))
    for j in range(batch_size):
        for i in range(length):
            note = np.random.normal(72, 16)
            note = int(np.clip(note, 0, 127))
            res[j, i, note] = 1

    return res


def multi_note_harmonic_seed(length, input_size, batch_size=16):
    res = np.zeros((batch_size, length, input_size))
    base_note = 72
    for j in range(batch_size):
        notes = get_random_notes_from_random_scale(base_note, length)
        for i, note in enumerate(notes):
            res[j, i, note] = 1

    return res


def multi_note_simult_seed(length, input_size, batch_size=16):
    res = np.zeros((batch_size, length, input_size))
    for j in range(batch_size):
        for i in range(length):
            num_notes = np.random.randint(0, 4)
            notes = np.random.normal(
                72, 16, size=num_notes).round().astype(int)
            notes = np.clip(notes, 0, 127)
            res[j, i, notes] = 1

    return res


def multi_note_simult_harmonic_seed(length, input_size, batch_size=16):
    res = np.zeros((batch_size, length, input_size))
    base_note = 72
    for j in range(batch_size):
        notes = get_random_notes_from_random_scale(
            base_note, np.random.randint(1, 3, size=length).sum())
        for i in range(length):
            notes_in_step = np.random.choice(
                notes, size=np.random.randint(1, min(4, len(notes))), replace=False)
            res[j, i, notes_in_step] = 1

    return res


seed_generators = {
    "single_note_seed": wrap_seed_generator_w_duration(single_note_seed),
    "multi_note_seed": wrap_seed_generator_w_duration(multi_note_seed),
    "multi_note_harmonic_seed": wrap_seed_generator_w_duration(multi_note_harmonic_seed),
    "multi_note_simult_seed": wrap_seed_generator_w_duration(multi_note_simult_seed),
    "multi_note_simult_harmonic_seed": wrap_seed_generator_w_duration(multi_note_simult_harmonic_seed),

    "single_note_seed_noise": wrap_seed_generator_w_duration(random_noise_adder(single_note_seed)),
    "multi_note_seed_noise": wrap_seed_generator_w_duration(random_noise_adder(multi_note_seed)),
    "multi_note_harmonic_seed_noise": wrap_seed_generator_w_duration(random_noise_adder(multi_note_harmonic_seed)),
    "multi_note_simult_seed_noise": wrap_seed_generator_w_duration(random_noise_adder(multi_note_simult_seed)),
    "multi_note_simult_harmonic_seed_noise": wrap_seed_generator_w_duration(random_noise_adder(multi_note_simult_harmonic_seed)),

    "single_note_seed_band_noise": wrap_seed_generator_w_duration(band_noise_adder(single_note_seed)),
    "multi_note_seed_band_noise": wrap_seed_generator_w_duration(band_noise_adder(multi_note_seed)),
    "multi_note_harmonic_seed_band_noise": wrap_seed_generator_w_duration(band_noise_adder(multi_note_harmonic_seed)),
    "multi_note_simult_seed_band_noise": wrap_seed_generator_w_duration(band_noise_adder(multi_note_simult_seed)),
    "multi_note_simult_harmonic_seed_band_noise": wrap_seed_generator_w_duration(band_noise_adder(multi_note_simult_harmonic_seed)),
}

# _, axs = plt.subplots(nrows=4, ncols=4, figsize=(20, 20), subplot_kw={'xticks':[], 'yticks':[]})
# for ax, (name, gen) in zip(axs.flat, sg.items()):
#     x = gen(5, 128, batch_size=1)[0]
#     x = np2sparse(x[:, :128], x[:, 128:], duration_dict)
#     ax.imshow(x.T[::-1, :128])
#     ax.set_title(name)
