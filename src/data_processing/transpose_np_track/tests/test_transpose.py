import numpy as np
import unittest

from ..transpose import transpose


class TransposeTest(unittest.TestCase):
    def test_transpose(self):
        track = np.array([
            [0, 0, 0, 1, 0, 1, 1, 0],
            [1, 0, 0, 0, 0, 1, 1, 1],
            [0, 0, 0, 1, 0, 0, 0, 0],
        ])
        result_a = transpose(track, -1)
        result_b = transpose(track, 2)
        expected_result_a = np.array([
            [0, 0, 1, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
        ])
        expected_result_b = np.array([
            [0, 0, 0, 0, 0, 1, 0, 1],
            [0, 0, 1, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 1, 0, 0],
        ])

        self.assertTrue(np.allclose(result_a, expected_result_a))
        self.assertTrue(np.allclose(result_b, expected_result_b))
