import os
import sys
import click
from mido import MidiFile
sys.path.append(os.path.join(os.path.dirname(
    __file__), os.pardir))

from helpers import parse_file_paths

from src.data_processing.common.rw_np_mid import read_numpy_midi, save_numpy_midi
from src.data_processing.just_sparsify_sequence.just_sparsify import just_sparsify
from src.data_processing.transpose_np_track.transpose import transpose


@click.command()
@click.option('-s', '--src', required=True)
@click.option('-d', '--dst', required=True)
@click.option('-t', '--transposition', is_flag=True)
def main(src, dst, transposition, **kwargs):
    input_paths, output_paths = parse_file_paths(src, dst, True)

    for input_p, output_p in zip(input_paths, output_paths):
        mid = MidiFile(input_p)
        numpy_track = just_sparsify(mid)
        if not transposition:
            save_numpy_midi(output_p, numpy_track)
        else:
            for step in range(-12, 13):
                output_path = output_p.replace('.', f'_tr_{str(step)}.')
                transposed = transpose(numpy_track, step)
                save_numpy_midi(output_path, transposed)


if __name__ == '__main__':
    # pylint: disable=no-value-for-parameter
    main()
