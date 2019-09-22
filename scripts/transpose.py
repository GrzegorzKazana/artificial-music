import os
import sys
import click
from mido import MidiFile
sys.path.append(os.path.join(os.path.dirname(
    __file__), os.pardir))

from helpers import parse_file_paths

from src.data_processing.common.rw_np_mid import read_numpy_midi, save_numpy_midi
from src.data_processing.transpose_np_track.transpose import transpose


@click.command()
@click.option('-s', '--src', required=True)
@click.option('-d', '--dst', required=True)
@click.option('-l', '--low_step', required=True, type=int)
@click.option('-h', '--high_step', required=True, type=int)
def main(src, dst, low_step, high_step):
    to_numpy = True
    input_paths, output_paths = parse_file_paths(src, dst, to_numpy)

    for input_p, output_p in zip(input_paths, output_paths):
        track = read_numpy_midi(input_p)
        for step in range(low_step, high_step):
            output_path = output_p.replace('.', f'_tr_{str(step)}.')
            transposed = transpose(track, step)
            save_numpy_midi(output_path, transposed)


if __name__ == '__main__':
    # pylint: disable=no-value-for-parameter
    main()
