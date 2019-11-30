import numpy as np
import unittest

from ..tonal_measurements import tonal_range, tonal_hist

track = np.array([
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
])

track_hist = np.array([
    [0, 0, 1, 0, 1, 0, 1, 0, 0, 0],
    [0, 1, 0, 0, 0, 1, 0, 1, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
])


class EvaluationTest(unittest.TestCase):
    def test_tonal_range(self):
        self.assertTrue(tonal_range(track) == 9 - 1)
        self.assertTrue(tonal_range(track[:2]) == 6 - 1)
        self.assertTrue(tonal_range(track[2:]) == 9 - 2)
        self.assertTrue(tonal_range(track[2:4]) == 7 - 7)

    def test_tonal_hist(self):
        self.assertTrue(np.allclose(
            tonal_hist(track_hist),
            np.array([0, 1, 2, 0, 2, 1, 1, 3, 0, 2]) / 12)
        )
