import os
import sys
import click
import json
from mido import MidiFile
from gensim.models import KeyedVectors
sys.path.append(os.path.join(os.path.dirname(
    __file__), os.pardir))

from helpers import parse_file_paths

from src.data_processing.common.transpose_midi import transpose_midi
from src.data_processing.common.rw_np_mid import read_numpy_midi, save_numpy_midi
from src.data_processing.embedded_with_time.embed_in_time import mid2np
from src.data_processing.embedded_with_time.unembed_in_time import np2mid

AVAILABLE_DIRECTIONS = [
    'mid2np',
    'np2mid'
]


@click.command()
@click.option('-m', '--mode', required=True, type=click.Choice(AVAILABLE_DIRECTIONS))
@click.option('-s', '--src', required=True)
@click.option('-d', '--dst', required=True)
@click.option('-w', '--word_vectors_path', required=True)
@click.option('-e', '--embedding_dict_path', required=True)
@click.option('-t', '--transpose', is_flag=True, default=False)
def main(mode, src, dst, word_vectors_path, embedding_dict_path, transpose):
    to_numpy = mode == AVAILABLE_DIRECTIONS[0]
    input_paths, output_paths = parse_file_paths(src, dst, to_numpy, '.npy')
    print(input_paths, output_paths)

    wv = KeyedVectors.load(word_vectors_path, mmap='r')
    with open(embedding_dict_path) as fp:
        embedding_dict = json.load(fp)

    for input_p, output_p in zip(input_paths, output_paths):
        if to_numpy:
            mid = MidiFile(input_p)
            if transpose:
                for step in range(-12, 13):
                    transposed_mid = transpose_midi(step, mid)
                    numpy_track = mid2np(transposed_mid, wv, embedding_dict)
                    save_numpy_midi(output_p.replace(
                        '.npy', f'_{step}.npy'), numpy_track)
            else:
                numpy_track = mid2np(mid, wv, embedding_dict)
                save_numpy_midi(output_p, numpy_track)
        else:
            numpy_track = read_numpy_midi(input_p)
            mid = np2mid(numpy_track, wv)
            mid.save(output_p)


if __name__ == '__main__':
    # pylint: disable=no-value-for-parameter
    main()
