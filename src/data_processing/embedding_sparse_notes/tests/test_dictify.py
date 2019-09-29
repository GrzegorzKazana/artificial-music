from scipy import sparse
import numpy as np
import unittest

from ..dictify_dataset import dictify_dataset
from ..common import UNKNOWN_FRAME, TRACK_END


class DictifyTest(unittest.TestCase):
    def test_dictify_dataset(self):
        tracks = [
            sparse.coo_matrix(np.array([
                [0, 0, 1, 0, 0, 1],
                [0, 0, 1, 1, 0, 0],
                [0, 0, 1, 1, 0, 0],
            ])),
            sparse.coo_matrix(np.array([
                [0, 0, 1, 0, 0, 1],
                [0, 0, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 0, 0],
            ])),
        ]
        embedding_dict = {
            UNKNOWN_FRAME: 0,
            TRACK_END: 1,
            'D-1,D#-1': 3,
            'D-1,F-1': 2,
            '': 1
        }
        result = dictify_dataset(tracks, embedding_dict)
        expected_result = (
            np.array(['D-1,F-1', 'D-1,D#-1', 'D-1,D#-1', TRACK_END,
                      'D-1,F-1', 'D-1,D#-1', '', 'D-1,D#-1', TRACK_END]),
            np.array(['D-1,F-1', 'D-1,D#-1', TRACK_END, 'D-1,F-1',
                      'D-1,D#-1', '', 'D-1,D#-1', TRACK_END])
        )

        self.assertTrue(np.all(result[0] == expected_result[0]))
        self.assertTrue(np.all(result[1] == expected_result[1]))

    def test_dictify_dataset_unknown(self):
        tracks = [
            sparse.coo_matrix(np.array([
                [0, 0, 1, 0, 0, 1],
                [0, 0, 1, 1, 0, 0],
                [0, 0, 1, 1, 0, 0],
            ])),
            sparse.coo_matrix(np.array([
                [0, 0, 1, 0, 0, 1],
                [0, 0, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 0, 0],
            ])),
        ]
        embedding_dict = {
            UNKNOWN_FRAME: 0,
            TRACK_END: 1,
            'D-1,D#-1': 3,
            'D-1,F-1': 2,
        }
        result = dictify_dataset(tracks, embedding_dict)
        expected_result = (
            np.array(['D-1,F-1', 'D-1,D#-1', 'D-1,D#-1', TRACK_END,
                      'D-1,F-1', 'D-1,D#-1', UNKNOWN_FRAME, 'D-1,D#-1', TRACK_END]),
            np.array(['D-1,F-1', 'D-1,D#-1', TRACK_END, 'D-1,F-1',
                      'D-1,D#-1', UNKNOWN_FRAME, 'D-1,D#-1', TRACK_END])
        )

        self.assertTrue(np.all(result[0] == expected_result[0]))
        self.assertTrue(np.all(result[1] == expected_result[1]))
