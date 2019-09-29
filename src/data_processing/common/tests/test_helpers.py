import unittest
from ..helpers import flow, pipe, unzip


class HelpersTest(unittest.TestCase):
    def test_flow(self):
        x = 3
        f = flow(
            lambda x: x + 2,
            lambda x: x * 3,
        )
        expected_result = 15

        self.assertEqual(f(x), expected_result)

    def test_pipe(self):
        y = pipe(
            3,
            lambda x: x + 2,
            lambda x: x * 3,
        )
        expected_result = 15

        self.assertEqual(y, expected_result)

    def test_unzip(self):
        x = [
            (1, 'a'),
            (2, 'b'),
            (3, 'c'),
        ]
        expected_result = (
            (1, 2, 3),
            ('a', 'b', 'c'),
        )

        self.assertSequenceEqual(unzip(x), expected_result)


if __name__ == '__main__':
    unittest.main()
