import numpy as np
from src.data_processing.embedding_sparse_notes.common import hash_frame, unhash_named_frame, UNKNOWN_FRAME, TRACK_END


def decode_note_vector_track(note_vecs, wv):
    res = []
    for note_vec in note_vecs:
        token = wv.similar_by_vector(note_vec, topn=1)[0][0]
        if token == UNKNOWN_FRAME and len(res) == 0:
            res.append(unhash_named_frame(''))
        elif token == UNKNOWN_FRAME:
            res.append(res[-1])
        elif token == TRACK_END:
            break
        else:
            res.append(unhash_named_frame(token))

    return np.array(res)


def encode_note_frame(frame, wv):
    token = hash_frame(frame)
    return wv[token] if token in wv else wv[UNKNOWN_FRAME]


def encode_frames(frames, wv):
    return np.array([encode_note_frame(f, wv) for f in frames])


def append_track_end(note_vecs, wv):
    return np.concatenate((note_vecs, [wv[TRACK_END]]), axis=0)