import os
import sys
import click
from mido import MidiFile
sys.path.append(os.path.join(os.path.dirname(
    __file__), os.pardir))

from helpers import parse_file_paths

from src.data_processing.common.rw_np_mid import read_numpy_midi, save_numpy_midi
from src.data_processing.just_sparsify_sequence.just_sparsify import just_sparsify


@click.command()
@click.option('-s', '--src', required=True)
@click.option('-d', '--dst', required=True)
def main(src, dst, **kwargs):
    input_paths, output_paths = parse_file_paths(src, dst, True)

    for input_p, output_p in zip(input_paths, output_paths):
        mid = MidiFile(input_p)
        numpy_track = just_sparsify(mid)
        save_numpy_midi(output_p, numpy_track)


if __name__ == '__main__':
    # pylint: disable=no-value-for-parameter
    main()
