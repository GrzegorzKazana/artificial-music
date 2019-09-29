from scipy import sparse
import numpy as np
import unittest

from ..reverse_transform import np2dicted, handle_special_tokens, dicted2sparse, np2sparse
from ..common import UNKNOWN_FRAME, TRACK_END


class MockKeyedVectors:
    """
    subsequently returns next vectors, regardless of method arguments
    """

    def __init__(self, words_vectors):
        self.words_vectors = words_vectors
        self.counter = 0

    def similar_by_vector(self, vector, topn=1):
        res = self.words_vectors[self.counter % len(self.words_vectors)][0]
        similarity = self.counter / 10
        self.counter += 1
        print(f'{res}, {similarity}')
        return [(res, similarity), ]


wv = MockKeyedVectors([
    ('a', [1, 2, 3]),
    ('b', [1, 2, 3]),
    ('c', [1, 2, 3]),
    ('d', [1, 2, 3]),
    ('e', [1, 2, 3]),
    ('f', [1, 2, 3]),
    ('g', [1, 2, 3]),
])


class ReverseTransformTest(unittest.TestCase):
    def test_np2dicted(self):
        track = np.array([
            [1, 2, 3],
            [1, 2, 3],
            [1, 2, 3],
        ])
        result = np2dicted(track, wv)
        expected_result = (
            ('a', 'b', 'c'),
            (0.0, 0.1, 0.2),
        )

        self.assertSequenceEqual(result[0], expected_result[0])
        self.assertSequenceEqual(result[1], expected_result[1])

    def test_handle_special_tokens(self):
        track_a = ['a', 'b', UNKNOWN_FRAME, 'c', TRACK_END]
        track_b = ['a', 'b', TRACK_END, 'c', 'd']
        result_a = handle_special_tokens(track_a)
        result_b = handle_special_tokens(track_b)
        expected_result_a = ['a', 'b', 'b', 'c', '']
        expected_result_b = ['a', 'b', '', '', '']

        self.assertSequenceEqual(result_a, expected_result_a)
        self.assertSequenceEqual(result_b, expected_result_b)
