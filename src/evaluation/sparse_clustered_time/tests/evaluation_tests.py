import numpy as np
import unittest

from ..tonal_measurements import tonal_range, tonal_hist
from ..rythm_measurements import rythm_range, rythm_hist


class EvaluationTest(unittest.TestCase):
    def test_tonal_range(self):
        track = np.array([
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
        ])

        self.assertTrue(tonal_range(track) == 9 - 1)
        self.assertTrue(tonal_range(track[:2]) == 6 - 1)
        self.assertTrue(tonal_range(track[2:]) == 9 - 2)
        self.assertTrue(tonal_range(track[2:4]) == 7 - 7)

    def test_tonal_hist(self):
        track_hist = np.array([
            [0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0,
                0, 1, 0, 1, 0, 1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0,
                1, 0, 0, 0, 1, 0, 1, 0, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0,
                0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0,
                0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        ])

        self.assertTrue(np.allclose(
            tonal_hist(track_hist),
            np.array([0, 2, 3, 0, 3, 2, 2, 7, 0, 3, 3, 2]) / 27)
        )

    def test_rythm_range(self):
        duration = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        ])

        self.assertTrue(rythm_range(duration) == 8 - 0)
        self.assertTrue(rythm_range(duration[:2]) == 1 - 0)
        self.assertTrue(rythm_range(duration[2:]) == 8 - 2)
        self.assertTrue(rythm_range(duration[2:4]) == 5 - 2)

    def test_rythm_hist(self):
        duration = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        ])

        self.assertTrue(np.allclose(
            rythm_hist(duration),
            np.array([1, 3, 1, 0, 0, 3, 0, 0, 1, 0]) / 9,
        ))
