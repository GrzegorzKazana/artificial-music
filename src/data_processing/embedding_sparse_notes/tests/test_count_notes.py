from scipy import sparse
import numpy as np
import unittest

from ..count_notes import count_note_occurences, transform_counter_num_to_notes, create_counters


class CounterTest(unittest.TestCase):
    def test_count_note_occurences(self):
        track = sparse.csr_matrix(np.array([
            [0, 0, 1, 0, 0, 1],
            [0, 0, 1, 1, 0, 0],
            [0, 0, 1, 1, 0, 0],
            [0, 0, 1, 0, 0, 1],
            [0, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ]))
        result = count_note_occurences(track)
        expected_result = {
            '2,3': 3,
            '2,5': 2,
            '': 1,
        }

        self.assertDictEqual(result, expected_result)

    def test_transform_counter_num_to_notes(self):
        x = {
            '2,3': 3,
            '2,5': 2,
            '': 1,
        }
        result = transform_counter_num_to_notes(x)
        expected_result = {
            'D-1,D#-1': 3,
            'D-1,F-1': 2,
            '': 1
        }

        self.assertDictEqual(result, expected_result)

    def test_create_counters(self):
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
        result = (create_counters(tracks))
        expected_result = (
            {
                '2,3': 3,
                '2,5': 2,
                '': 1,
            },
            {
                'D-1,D#-1': 3,
                'D-1,F-1': 2,
                '': 1
            }
        )

        self.assertSequenceEqual(result, expected_result)
