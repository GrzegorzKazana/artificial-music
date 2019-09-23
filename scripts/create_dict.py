import os
import sys
import click
import json
sys.path.append(os.path.join(os.path.dirname(
    __file__), os.pardir))

from scipy import sparse

from helpers import get_valid_files_in_dir, option_kwargs_to_string
from src.data_processing.embedding_sparse_notes.count_notes import create_counters
from src.data_processing.embedding_sparse_notes.create_dict import create_dict


@click.command()
@click.option('-s', '--src', required=True)
@click.option('-i', '--ignore_ratio', type=float)
def main(src, **kwargs):
    clean_kwargs = {k: v for k, v in kwargs.items() if v is not None}

    files_in_folder = get_valid_files_in_dir(src)
    file_paths = [os.path.join(src, f) for f in files_in_folder]

    tracks = [sparse.load_npz(fp) for fp in file_paths]
    counter, counter_notes = create_counters(tracks)
    embedding_dict = create_dict(counter_notes, **clean_kwargs)

    output_dir = os.path.join(src, 'meta')
    os.makedirs(output_dir, exist_ok=True)

    options_str = option_kwargs_to_string(clean_kwargs)

    with open(os.path.join(output_dir, f'_counter_nums_{options_str}.json'), 'w') as f:
        json.dump(counter, f, indent=4)

    with open(os.path.join(output_dir, f'_counter_notes_{options_str}.json'), 'w') as f:
        json.dump(counter_notes, f, indent=4)

    with open(os.path.join(output_dir, f'_embedding_dict_{options_str}.json'), 'w') as f:
        json.dump(embedding_dict, f, indent=4)


if __name__ == '__main__':
    # pylint: disable=no-value-for-parameter
    main()
