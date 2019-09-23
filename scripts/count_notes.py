import os
import sys
import click
import json
sys.path.append(os.path.join(os.path.dirname(
    __file__), os.pardir))

from scipy import sparse

from src.data_processing.embedding_sparse_notes.count_notes import create_counters


@click.command()
@click.option('-s', '--src', required=True)
def main(src, **kwargs):
    files_in_folder = os.listdir(src)
    file_paths = [os.path.join(src, f) for f in files_in_folder]

    tracks = [sparse.load_npz(fp).tocsr() for fp in file_paths]
    counter, counter_notes = create_counters(tracks)

    output_dir = os.path.join(src, 'meta')
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, '_counter_nums.json'), 'w') as f:
        json.dump(counter, f, indent=4)

    with open(os.path.join(output_dir, '_counter_notes.json'), 'w') as f:
        json.dump(counter_notes, f, indent=4)


if __name__ == '__main__':
    # pylint: disable=no-value-for-parameter
    main()
