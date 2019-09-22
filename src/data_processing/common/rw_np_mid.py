def read_numpy_midi(input_path):
    import numpy as np
    from scipy import sparse
    if 'csv' in input_path:
        return np.loadtxt(input_path, delimiter=",", dtype=np.int32)
    elif 'npy' in input_path:
        return np.load(input_path).astype(np.float32)
    elif 'npz' in input_path:
        sparse_numpy = sparse.load_npz(input_path)
        return sparse_numpy.toarray().astype(np.float32)


def save_numpy_midi(output_path, res):
    import numpy as np
    from scipy import sparse
    if '.csv' in output_path:
        np.savetxt(output_path, res, delimiter=",")
        print(f'Saved to: {output_path}')
    elif '.npy' in output_path:
        np.save(output_path, res)
        print(f'Saved to: {output_path}')
    elif '.npz' in output_path:
        res_sparse = sparse.coo_matrix(res)
        sparsity_factor = 1 - (res_sparse.getnnz() / res.size)
        sparse.save_npz(output_path, res_sparse)
        print(
            f'Saved to: {output_path}, {int(100 * sparsity_factor)}% sparsity')
