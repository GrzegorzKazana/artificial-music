import numpy as np
from scipy import sparse
import unittest

from ..common import remove_subsequent_notes, remove_subsequent_dict_values, concat_tracks, map_note_num_to_name, map_hashed_frame_to_names, hash_frame, unhash_named_frame


class EmbeddingTest(unittest.TestCase):
    def test_remove_subsequent_notes(self):
        x = sparse.csr_matrix(np.array([
            [0, 0, 1],
            [0, 0, 1],
            [0, 1, 1],
            [0, 1, 0],
            [0, 1, 0],
            [0, 0, 0],
        ]))
        result = remove_subsequent_notes(x).toarray()
        expected_result = np.array([
            [0, 0, 1],
            [0, 1, 1],
            [0, 1, 0],
            [0, 0, 0],
        ])

        self.assertTrue(np.allclose(result, expected_result))

    def test_remove_subsequent_dict_values(self):
        x = np.array([1, 2, 2, 3, 4, 4])
        result = remove_subsequent_dict_values(x)
        expected_result = np.array([1, 2, 3, 4])

        self.assertTrue(np.allclose(result, expected_result))

    def test_concat_tracks(self):
        tracks = [
            sparse.coo_matrix(np.array([[1, 2, 3], [4, 5, 6]])),
            sparse.coo_matrix(np.array([[1, 2, 3]])),
            sparse.coo_matrix(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])),
        ]
        result = concat_tracks(tracks).toarray()
        expected_result = np.array([
            [1, 2, 3], [4, 5, 6],
            [1, 2, 3],
            [1, 2, 3], [4, 5, 6], [7, 8, 9]
        ])

        self.assertTrue(np.allclose(result, expected_result))

    def test_map_note_num_to_name(self):
        self.assertEqual(map_note_num_to_name(50), 'D3')
        self.assertEqual(map_note_num_to_name(72), 'C5')

    def test_map_hashed_frame_to_names(self):
        self.assertEqual(map_hashed_frame_to_names(''), '')
        self.assertEqual(map_hashed_frame_to_names('50,72'), 'D3,C5')

    def test_hash_frame(self):
        x = sparse.coo_matrix(np.array([[0, 0, 1, 1, 0, 1]]))
        result = hash_frame(x)
        expected_result = '2,3,5'

        self.assertEqual(result, expected_result)

    def test_unhash_named_frame(self):
        x = 'D3,C5'
        result = unhash_named_frame(x)
        expected_result = np.zeros((1, 128))
        expected_result[0, 50] = 1
        expected_result[0, 72] = 1

        self.assertTrue(np.allclose(result, expected_result))
