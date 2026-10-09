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
