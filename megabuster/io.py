from scipy.sparse import csr_matrix
import numpy as np

def load_obsmat(path_obsmat, mask_stacked, size_obsmat, return_diagonal_term=False):
    print("Loading obsmat from path", path_obsmat)

    mm_data = np.load(path_obsmat + "data.npy")
    mm_row_indices = np.load(path_obsmat + "row_indices.npy", )
    mm_column_indices = np.load(path_obsmat + "column_indices.npy")

    sparse_matrix = csr_matrix(
        (mm_data, (mm_row_indices, mm_column_indices)),
        (size_obsmat,size_obsmat),
    )[:,mask_stacked][mask_stacked,:]

    if return_diagonal_term:
        return sparse_matrix, sparse_matrix.diagonal()

    return sparse_matrix

def load_obsmat_no_mask(path_obsmat, size_obsmat, return_diagonal_term=False):
    print("Loading obsmat from path", path_obsmat)

    mm_data = np.load(path_obsmat + "data.npy")
    mm_row_indices = np.load(path_obsmat + "row_indices.npy", )
    mm_column_indices = np.load(path_obsmat + "column_indices.npy")

    sparse_matrix = csr_matrix(
        (mm_data, (mm_row_indices, mm_column_indices)),
        (size_obsmat,size_obsmat),
    )

    if return_diagonal_term:
        return sparse_matrix, sparse_matrix.diagonal()

    return sparse_matrix
