import os
import sys
import click
import json
import numpy as np
from mido import MidiFile
sys.path.append(os.path.join(os.path.dirname(
    __file__), os.pardir))

from helpers import parse_file_paths

from src.data_processing.common.rw_np_mid import read_numpy_midi, save_numpy_midi
from src.data_processing.sparse_notes_classified_time.mid2np import mid2np, cluster_and_ohe_durations
from src.data_processing.sparse_notes_classified_time.np2mid import np2mid
from src.data_processing.transpose_np_track.transpose import transpose

AVAILABLE_DIRECTIONS = [
    'mid2np',
    'np2mid'
]


@click.command()
@click.option('-m', '--mode', required=True, type=click.Choice(AVAILABLE_DIRECTIONS))
@click.option('-s', '--src', required=True)
@click.option('-d', '--dst', required=True)
@click.option('-c', '--dur_dict', default='')
@click.option('-t', '--do_transpose', is_flag=True, default=False)
def main(mode, src, dst, dur_dict, do_transpose, **kwargs):
    clean_kwargs = {k: v for k, v in kwargs.items() if v is not None}

    to_numpy = mode == AVAILABLE_DIRECTIONS[0]
    input_paths, output_paths = parse_file_paths(src, dst, to_numpy)

    if to_numpy:
        tracks, durations = zip(
            *[mid2np(MidiFile(input_p), **clean_kwargs) for input_p in input_paths])
        tracks_blob = np.concatenate(tracks, axis=0)
        durations = np.concatenate(durations)
        durations_ohe, _, durations_dict = cluster_and_ohe_durations(
            durations.reshape(-1, 1))

        meta_dir = os.path.join(dst, 'meta')
        os.makedirs(meta_dir, exist_ok=True)

        with open(os.path.join(meta_dir, 'durations_dict.json'), 'w+') as fp:
            json.dump(durations_dict, fp, indent=4, default=lambda x: float(
                x) if isinstance(x, np.float32) else x)

        track_w_duration_blob = np.concatenate(
            (tracks_blob, durations_ohe), axis=1)
        save_numpy_midi(os.path.join(
            meta_dir, 'tracks_blob_w_duration.npz'), track_w_duration_blob)

        track_lens = [t.shape[0] for t in tracks]
        durations_ohe_split = np.split(
            durations_ohe, np.cumsum(track_lens[:-1]))

        for output_p, t, d in zip(output_paths, tracks, durations_ohe_split):
            if do_transpose:
                for step in range(-12, 13):
                    transposed_t = transpose(t, step)
                    track_w_dur = np.concatenate((transposed_t, d), axis=-1)
                    save_numpy_midi(os.path.join(output_p.replace(
                        '.npz', f'_{step}.npz')), track_w_dur)
            else:
                track_w_dur = np.concatenate((t, d), axis=-1)
                save_numpy_midi(os.path.join(output_p), track_w_dur)
    else:
        meta_dir = os.path.join(dst, 'meta')

        assert dur_dict != '', 'Please specify duration dict'

        with open(dur_dict) as fp:
            durations_dict = json.load(fp)

        for input_p, output_p in zip(input_paths, output_paths):
            numpy_track = read_numpy_midi(input_p)
            track, durations = np.split(numpy_track, [128], axis=1)
            mid = np2mid(track, durations, durations_dict, **clean_kwargs)
            mid.save(output_p)


if __name__ == '__main__':
    # pylint: disable=no-value-for-parameter
    main()
