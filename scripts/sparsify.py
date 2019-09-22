import os
import sys
import click
from mido import MidiFile
sys.path.append(os.path.join(os.path.dirname(
    __file__), os.pardir))

from src.data_processing.common.rw_np_mid import read_numpy_midi, save_numpy_midi
from src.data_processing.sparse_notes_quantized_time.mid2np import mid2np
from src.data_processing.sparse_notes_quantized_time.np2mid import np2mid

AVAILABLE_DIRECTIONS = [
    'mid2np',
    'np2mid'
]


def parse_file_paths(src_path, dst_path, to_numpy):
    default_extension = '.npz' if to_numpy else '.mid'
    input_paths = []
    output_paths = []
    if os.path.isfile(src_path):
        input_paths = [src_path]
        output_paths = [dst_path]
    elif os.path.isdir(src_path):
        files_in_dir = os.listdir(src_path)
        output_files = [os.path.splitext(
            f)[0] + default_extension for f in files_in_dir]
        input_file_paths = [os.path.join(src_path, f) for f in files_in_dir]
        output_file_paths = [os.path.join(dst_path, f) for f in output_files]
        input_paths = input_file_paths
        output_paths = output_file_paths

    return input_paths, output_paths


@click.command()
@click.option('-m', '--mode', required=True, type=click.Choice(AVAILABLE_DIRECTIONS))
@click.option('-s', '--src', required=True)
@click.option('-d', '--dst', required=True)
@click.option('--resolution', type=int)
@click.option('--skip_velocities', is_flag=True)
def main(mode, src, dst, **kwargs):
    to_numpy = mode == AVAILABLE_DIRECTIONS[0]
    input_paths, output_paths = parse_file_paths(src, dst, to_numpy)

    for input_p, output_p in zip(input_paths, output_paths):
        if to_numpy:
            mid = MidiFile(input_p)
            messages = [m for m in mid]
            numpy_track = mid2np(messages, **kwargs)
            save_numpy_midi(output_p, numpy_track)
        else:
            numpy_track = read_numpy_midi(input_p)
            mid = np2mid(numpy_track, **kwargs)
            mid.save(output_p)


if __name__ == '__main__':
    # pylint: disable=no-value-for-parameter
    main()
