import os
import sys
import click
from mido import MidiFile
sys.path.append(os.path.join(os.path.dirname(
    __file__), os.pardir))

from helpers import parse_file_paths

from src.data_processing.common.rw_np_mid import read_numpy_midi, save_numpy_midi
from src.data_processing.sparse_notes_quantized_time.mid2np import mid2np
from src.data_processing.sparse_notes_quantized_time.np2mid import np2mid

AVAILABLE_DIRECTIONS = [
    'mid2np',
    'np2mid'
]


@click.command()
@click.option('-m', '--mode', required=True, type=click.Choice(AVAILABLE_DIRECTIONS))
@click.option('-s', '--src', required=True)
@click.option('-d', '--dst', required=True)
@click.option('--resolution', type=int)
@click.option('--skip_velocities', is_flag=True)
def main(mode, src, dst, **kwargs):
    clean_kwargs = {k: v for k, v in kwargs.items() if v is not None}

    to_numpy = mode == AVAILABLE_DIRECTIONS[0]
    input_paths, output_paths = parse_file_paths(src, dst, to_numpy)

    for input_p, output_p in zip(input_paths, output_paths):
        if to_numpy:
            mid = MidiFile(input_p)
            messages = [m for m in mid]
            numpy_track = mid2np(messages, **clean_kwargs)
            save_numpy_midi(output_p, numpy_track)
        else:
            numpy_track = read_numpy_midi(input_p)
            mid = np2mid(numpy_track, **clean_kwargs)
            mid.save(output_p)


if __name__ == '__main__':
    # pylint: disable=no-value-for-parameter
    main()
