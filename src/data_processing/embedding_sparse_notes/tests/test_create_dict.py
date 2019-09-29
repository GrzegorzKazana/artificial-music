from scipy import sparse
import numpy as np
import unittest

from ..create_dict import ignore_rarest_in_counter, create_dict
from ..common import UNKNOWN_FRAME, TRACK_END


class CreateDictTest(unittest.TestCase):
    def test_ignore_rarest_in_counter(self):
        x = {
            'C5': 5,
            'D-1,D#-1': 3,
            'D-1,F-1': 2,
            '': 1
        }
        result = ignore_rarest_in_counter(x, 0.2)
        expected_result = {
            'C5': 5,
            'D-1,D#-1': 3,
        }

        self.assertDictEqual(result, expected_result)

    def test_create_dict(self):
        x = {
            'C5': 5,
            'D-1,D#-1': 3,
            'D-1,F-1': 2,
            '': 1
        }
        result = create_dict(x, ignore_ratio=0.2)
        expected_result = {
            UNKNOWN_FRAME: 0,
            TRACK_END: 1,
            'C5': 2,
            'D-1,D#-1': 3,
        }

        self.assertDictEqual(result, expected_result)
