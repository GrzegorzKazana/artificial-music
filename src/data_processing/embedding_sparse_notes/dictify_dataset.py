from scipy import sparse
import numpy as np

from .common import remove_subsequent_notes, concat_tracks, hash_frame, map_hashed_frame_to_names, UNKNOWN_FRAME, TRACK_END, remove_subsequent_dict_values


def dictify_dataset(npz_tracks, embedding_dict):
    """
    takes in list of tracks in sparse format, and an embedding dict
    returns concatenated tracks with notes in the form of dict keys (1d np array)
    """
    def safe_dict_lookup(hashed_frame_note_names):
        return hashed_frame_note_names if hashed_frame_note_names in embedding_dict else UNKNOWN_FRAME

    tracks_hashed = [[map_hashed_frame_to_names(
        hash_frame(frame)) for frame in track.tocsr()] for track in npz_tracks]

    tracks_dicted = [[safe_dict_lookup(frame) for frame in track]
                     for track in tracks_hashed]

    tracks_and_endings = [v for t in tracks_dicted for v in (
        t, [TRACK_END])]

    track_blob = np.concatenate(tracks_and_endings)

    return track_blob, remove_subsequent_dict_values(track_blob)
