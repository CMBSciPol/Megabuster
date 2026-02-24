import os, time
import warnings
import numpy as np
import jax
import jax.numpy as jnp
import scipy
import healpy as hp
from scipy.sparse import csr_matrix
from opt_einsum import contract
from tqdm import tqdm

__all__ = [
    'save_obsmat_precomputation',
    'save_data_ind_j',
    'compute_threshold_matrices',
    'compute_eigenspectrum_from_matrices',
]


def save_obsmat_precomputation(list_path_obsmat_input, list_path_obsmat_output):
    for i, path_obsmat in enumerate(list_path_obsmat_input):
        
        add_ = ''
        if not str(path_obsmat).endswith('.npz'):
            add_ = '.npz'
        
        path_output = str(list_path_obsmat_output[i])
        print(f"Loading obsmat from path {path_obsmat}", flush=True)

        if os.path.exists(path_output + "_data.npy") and os.path.exists(path_output + "_row_indices.npy") and os.path.exists(path_output + "_column_indices.npy"):
            print(f"## Files {path_output}_data.npy, {path_output}_row_indices.npy, {path_output}_column_indices.npy already exist, skipping computation.", flush=True)
            continue
        
        sparse_matrix = scipy.sparse.load_npz(str(path_obsmat)+add_)

        row_indices, column_indices, data = scipy.sparse.find(sparse_matrix)

        print(f"Saving to {path_output}_data.npy, {path_output}_row_indices.npy, {path_output}_column_indices.npy", flush=True)

    
        
        np.save(path_output + "_data.npy", data)
        np.save(path_output + "_row_indices.npy", row_indices)
        np.save(path_output + "_column_indices.npy", column_indices)


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

def compute_eigenspectrum_from_matrices(
    list_scipy_obsmat_masked, 
    inverse_noisecov_QU_ring, 
    mask_B_nest, 
    path_output,
    list_name_output,
    threshold=1e-20
):
    """ Compute the eigenspectrum of the matrices O^T N^{-1} O for each frequency,
    where O is the observation matrix and N is the noise covariance matrix.
    The matrices are computed only on the pixels where the mask is non-zero.
    """

    n_freq = len(list_scipy_obsmat_masked)
    assert len(list_name_output) == n_freq, "The list of output names must have the same length as the list of observation matrices."

    eigh_jitted = jax.jit(jax.numpy.linalg.eigh, static_argnames=['UPLO', 'symmetrize_input'])
    for idx_freq in range(n_freq):
        path_save = path_output+str(list_name_output[idx_freq])
        if not path_save.endswith('.npz'):
            path_save += '.npz'
        if os.path.exists(path_save):
            print(f"## File {path_save} already exists, skipping computation.", flush=True)
            continue
        
        obsmat_array = jnp.array(list_scipy_obsmat_masked[idx_freq].todense())

        time_start = time.time()
        array_to_eigen = contract(
            'ab,b,bc->ac', 
            obsmat_array.T, 
            hp.reorder(inverse_noisecov_QU_ring[idx_freq], r2n=True)[...,mask_B_nest!=0].ravel(), 
            obsmat_array
        )
        array_to_eigen.block_until_ready()
        print("Finish computation in", time.time()-time_start, flush=True)
        
        print("Starting eigenvalue decomp", flush=True)
        eigvals, eigvecs = eigh_jitted(array_to_eigen)
        eigvals.block_until_ready()
        print("Finishing eigenvalue decomp", flush=True)
        print('---', jnp.min(eigvals), jnp.max(eigvals), jnp.mean(eigvals), jnp.std(eigvals), flush=True)
        assert jnp.all(eigvals >= threshold), f"Negative eigenvalues found in the eigenspectrum of frequency {idx_freq}. The matrices are not positive semi-definitewith minimum eigenvalue: {jnp.min(eigvals)}, which can lead to numerical instabilities."

        print("Saving to: ", path_save, flush=True)
        jnp.savez(path_save, eigvals=eigvals, eigvecs=eigvecs)
