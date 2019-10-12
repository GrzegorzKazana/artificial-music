import os
import sys
import click
from mido import MidiFile
from tensorflow import keras as K
from gensim.models import KeyedVectors
import matplotlib.pyplot as plt
sys.path.append(os.path.join(os.path.dirname(
    __file__), os.pardir))

from helpers import parse_file_paths

from src.data_processing.common.rw_np_mid import read_numpy_midi, save_numpy_midi
from src.generating.embedded_generating_seeds import seed_generators
from src.generating.generating import recurrent_generate
from src.data_processing.embedding_sparse_notes.reverse_transform import np2sparse
from src.data_processing.sparse_notes_quantized_time.np2mid import np2mid

seed_gens = [
    ('random_noise_seed', lambda length, input_size, wv, batch_size: seed_generators['random_noise_seed'](
        length, input_size, scaler=2.0, batch_size=batch_size)),
    ('zero_seed', lambda length, input_size, wv, batch_size: seed_generators['zero_seed'](
        length, input_size, wv, batch_size=batch_size)),
    ('const_frame_seed', lambda length, input_size, wv, batch_size: seed_generators['const_frame_seed'](
        length, input_size, wv, batch_size=batch_size)),
    ('short_frame_seed', lambda length, input_size, wv, batch_size: seed_generators['short_frame_seed'](
        length, input_size, wv, batch_size=batch_size)),
    ('multi_note_seed', lambda length, input_size, wv, batch_size: seed_generators['multi_note_seed'](
        length, input_size, wv, batch_size=batch_size)),
    ('multi_note_harmonic_seed', lambda length, input_size, wv, batch_size: seed_generators['multi_note_harmonic_seed'](
        length, input_size, wv, batch_size=batch_size)),
    ('const_frame_seed_noise', lambda length, input_size, wv, batch_size: seed_generators['const_frame_seed_noise'](
        length, input_size, word_vectors=wv, batch_size=batch_size)),
    ('short_frame_seed_noise', lambda length, input_size, wv, batch_size: seed_generators['short_frame_seed_noise'](
        length, input_size, word_vectors=wv, batch_size=batch_size)),
    ('multi_note_seed_noise', lambda length, input_size, wv, batch_size: seed_generators['multi_note_seed_noise'](
        length, input_size, word_vectors=wv, batch_size=batch_size)),
    ('multi_note_harmonic_seed_noise', lambda length, input_size, wv, batch_size: seed_generators['multi_note_harmonic_seed_noise'](
        length, input_size, word_vectors=wv, batch_size=batch_size)),
]


@click.command()
@click.option('-m', '--srcmodel', required=True)
@click.option('-w', '--srcwords', required=True)
@click.option('-l', '--seed_length', default=50)
@click.option('-l', '--seq_length', default=400)
@click.option('-i', '--input_size', default=128)
@click.option('-s', '--window_size', default=100)
def main(srcmodel, srcwords, seed_length, seq_length, input_size, window_size, **kwargs):
    model_name = os.path.basename(srcmodel)

    model = K.models.load_model(srcmodel)
    wv = KeyedVectors.load(srcwords)

    output_dir = os.path.join(os.path.dirname(srcmodel), 'samples')
    os.makedirs(output_dir, exist_ok=True)

    for name, gen in seed_gens:
        seed = gen(seed_length, input_size, wv, 1)
        samples = recurrent_generate(
            model, seed, seq_length, window_size)
        sparse_samples = [np2sparse(sample, wv)[0]
                          for sample in samples]
        for i, sample in enumerate(sparse_samples):
            name = f'{model_name}_{name}_{i}'
            mid = np2mid(sample)
            mid.save(name + '.mid')
            fig = plt.figure()
            plt.imshow(sample.toarray().T[::-1, :])
            fig.savefig(name + '.png', dpi=fig.dpi)


if __name__ == '__main__':
    # pylint: disable=no-value-for-parameter
    main()
