import unittest
import numpy as np

from ..generating import recurrent_generate, linear_generate


class MockModel:
    """
    mocks seq2seq model
    """

    def predict(self, x):
        assert x.ndim == 3
        return np.arange(0, x.size).reshape(x.shape)


model = MockModel()


class GeneratingTest(unittest.TestCase):
    def test_linear_generate(self):
        x = np.zeros((2, 3, 4))
        result = linear_generate(model, x)
        expected_result = np.arange(0, 24).reshape((2, 3, 4))

        self.assertTrue(np.allclose(result, expected_result))

    def test_recurrent_generate(self):
        x = np.zeros((1, 3, 4))
        result = recurrent_generate(model, x, 10, 2)
        expected_result = np.concatenate([
            np.zeros((1, 3, 4)),
            np.array([8, 9, 10, 11]).reshape(1, 1, 4),
            np.repeat([[4, 5, 6, 7]], 7, axis=0).reshape(1, 7, 4)
        ], axis=1)

        self.assertTrue(np.allclose(result, expected_result))
