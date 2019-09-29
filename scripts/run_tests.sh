#!/bin/bash
python -m unittest src/data_processing/common/tests/test_helpers.py
python -m unittest src/data_processing/embedding_sparse_notes/tests/test_common.py
python -m unittest src/data_processing/embedding_sparse_notes/tests/test_count_notes.py
python -m unittest src/data_processing/embedding_sparse_notes/tests/test_create_dict.py
python -m unittest src/data_processing/embedding_sparse_notes/tests/test_dictify.py
python -m unittest src/data_processing/embedding_sparse_notes/tests/test_reverse_transform.py
python -m unittest src/data_processing/transpose_np_track/tests/test_transpose.py
