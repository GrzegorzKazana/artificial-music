import numpy as np


def just_sparsify(track):
    def one_hot_encode(note_num):
        res = np.zeros(128)
        res[note_num] = 1
        return res

    sparsified = np.stack([one_hot_encode(msg.note)
                           for msg in track if msg.type == 'note_on' and msg.velocity != 0], axis=0)

    return sparsified
