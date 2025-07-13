import os
import numpy as np
import scipy
from scipy.sparse import csr_matrix
from tqdm import tqdm

__all__ = [
    'save_data_ind_j',
    'compute_threshold_matrices',
]

def save_data_ind_j(matrix: scipy.sparse.sparray, path_save, disable_tqdm=True):
    """ Retrieve the data and indices of the non-zero elements of a sparse matrix
    and return them in a format that can be used to create a furax operator.
    """

    if os.path.exists(path_save):
        print(f"## File {path_save} already exists, skipping computation.", flush=True)

    ind_i, ind_j, data = scipy.sparse.find(matrix)
    
    values_i, first_indices_i, counts_i = np.unique(ind_i, return_counts=True, return_index=True)
    values_j, first_indices_j, counts_j = np.unique(ind_j, return_counts=True, return_index=True)
    
    n_data = len(values_i)
    n_pix = n_data//2
    if counts_i.size != 0:
        max_counts_i = np.max(counts_i)
        max_counts_j = np.max(counts_j)
    else:
        max_counts_i = 0
        max_counts_j = 0
        
    new_ind_i = np.zeros((len(values_i), max_counts_i), dtype=np.int32)
    new_data_i = np.zeros((len(values_i), max_counts_i), dtype=np.float64) # Filling the diagonal data with 0
    new_data_j = np.zeros((len(values_j), max_counts_j), dtype=np.float64) # Filling the diagonal data with 0
    new_ind_j = np.zeros((len(values_j), max_counts_j), dtype=np.int64)

    first_indices_i = np.append(first_indices_i, ind_j.size + 1)
    first_indices_j = np.append(first_indices_j, ind_i.size + 1)

    for j in tqdm(range(n_data), disable=disable_tqdm): # Loop on the indices in the mask
        new_ind_i[j,:counts_i[j]] = ind_j[first_indices_i[j]:first_indices_i[j+1]]
        new_data_i[j,:counts_i[j]] = data[first_indices_i[j]:first_indices_i[j+1]]
        
        indices_ = np.where(ind_j == values_j[j])[0]
        new_ind_j[j,:counts_j[j]] = ind_i[indices_]
        new_data_j[j,:counts_j[j]] = data[indices_]
        


    new_data_i = np.array(new_data_i, dtype=np.float64)
    new_ind_i = np.array(new_ind_i, dtype=np.int32)

    new_data_j = np.array(new_data_j, dtype=np.float64)
    new_ind_j = np.array(new_ind_j, dtype=np.int32)

    print(f"## Saving to {path_save}", flush=True)
    np.savez(path_save, new_data_i=new_data_i, new_ind_i=new_ind_i, new_data_j=new_data_j, new_ind_j=new_ind_j, values_i=values_i, values_j=values_j, counts_i=counts_i, counts_j=counts_j)

def compute_threshold_matrices(list_threshold, list_sparse_matrix, list_path_save, nstokes=2, disable_tqdm=True):
    """ Compute the threshold matrices for a list of matrices and save them to a file.
    
    Parameters
    ----------
    list_threshold : list of float
        The list of thresholds to apply to the matrices so that every element is below the corresponding threshold in absolute value
    list_sparse_matrix : list of scipy.sparse.sparray
        The list of sparse matrices to apply the thresholds to.
    path_save : str
        The path to save the resulting matrices.
    """
    
    assert len(list_threshold) == len(list_sparse_matrix), "The list of thresholds and the list of matrices must have the same length."

    n_freq = len(list_sparse_matrix)
    
    npix = list_sparse_matrix[0].shape[0] // nstokes

    for i in range(n_freq):
        big_matrix = list_sparse_matrix[i].copy()

        cond = np.abs(big_matrix.data) < list_threshold[i]
        big_matrix.data[cond] = 0

        label_stokes = ['q', 'u'] if nstokes == 2 else ['i', 'q', 'u']

        for j in range(nstokes):
            for k in range(nstokes):
                save_data_ind_j(big_matrix[j*npix:(j+1)*npix, k*npix:(k+1)*npix], f"{list_path_save[i]}_threshold_{list_threshold[i]}_stokes_{label_stokes[j]}_stokes_{label_stokes[k]}.npz", disable_tqdm=disable_tqdm)
    