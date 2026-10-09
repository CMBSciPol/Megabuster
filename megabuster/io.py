import jax.numpy as jnp
import numpy as np
import scipy as sp
from opt_einsum import contract

from megabuster.tools import get_dense_furax_operator_from_freq_array

__all__ = [
    'load_obsmat_mask',
    'load_all_obsmat',
    'build_obsmat_operator_from_flattened_matrices'
]


def load_obsmat_mask(path_obsmat, size_obsmat, mask_stacked=None):
    print("Loading obsmat from path", path_obsmat)

    new_path_obsmat = str(path_obsmat).replace('.npz','')
    mm_data = np.load(new_path_obsmat + "_data.npy")
    mm_row_indices = np.load(new_path_obsmat + "_row_indices.npy", )
    mm_column_indices = np.load(new_path_obsmat + "_column_indices.npy")

    if mask_stacked is None:
        mask_stacked = np.ones(size_obsmat, dtype=bool)

    sparse_matrix = sp.sparse.csr_matrix(
        (mm_data, (mm_row_indices, mm_column_indices)),
        (size_obsmat,size_obsmat),
    )[:,mask_stacked][mask_stacked,:]

    return sparse_matrix

def load_obsmat_not_precomputed(path_obsmat, size_obsmat, mask_stacked=None):
    print("Loading obsmat not precomputed from path", path_obsmat)

    if mask_stacked is None:
        mask_stacked = np.ones(size_obsmat, dtype=bool)

    return sp.sparse.load_npz(path_obsmat)[:,mask_stacked][mask_stacked,:]

def load_all_obsmat(list_path_obsmat_prefix, size_obsmat, kind='precomputations_scipy', mask_stacked=None):
    """
    Load all obsmat from a list of paths.
    """

    assert kind in ['precomputations_scipy', 'not_precomputed'], "The parameter 'kind' must be one of 'precomputations_scipy', 'not_precomputed', corresponding respectively to the two different expected formats of the obsmat."

    if kind == 'not_precomputed':
        return [load_obsmat_not_precomputed(
            path_obsmat=str(path_obsmat),
            size_obsmat=size_obsmat,
            mask_stacked=mask_stacked
        ) for path_obsmat in list_path_obsmat_prefix]
    elif kind == 'precomputations_scipy':
        return [load_obsmat_mask(
            path_obsmat=str(path_obsmat),
            size_obsmat=size_obsmat,
            mask_stacked=mask_stacked
        ) for path_obsmat in list_path_obsmat_prefix]

def load_matrix_precond(
        path_matrix_eigen_decomp, 
        nstokes=2,
        power_diagonal=-1,
    ):
    """
    Load the preconditioner matrix from a file.

    Parameters
    ----------
    path_matrix_eigen_decomp : str
        The path to the preconditioner matrix.
    nstokes : int
        The number of Stokes parameters.
    power_diagonal : int
        The power to which the diagonal elements are raised. Default is -1 (for the inverse)

    Returns
    -------
    preconditioner : dict
        The preconditioner as a dictionary with keys 'diag' and 'pinv'.
    """
    if type(path_matrix_eigen_decomp) is str:
        print(f"Loading diagonal preconditioner from {path_matrix_eigen_decomp}")
        add_ = ''
        if not str(path_matrix_eigen_decomp).endswith('.npy'):
            add_ = '.npy'
        diag_data = np.load(f"{path_matrix_eigen_decomp}{add_}")
        return diag_data
    elif type(path_matrix_eigen_decomp) is list:
        print(f"Building and loading matrix eigen decomposition from the list of paths {path_matrix_eigen_decomp}")
        
        n_freq = len(path_matrix_eigen_decomp)
        
        array_eigvals = []
        array_eigvecs = []

        for idx_freq in range(n_freq):
            print(f'Loading matrix eigen decomposition from {path_matrix_eigen_decomp[idx_freq]}', flush=True)
            add_ = ''
            if not str(path_matrix_eigen_decomp[idx_freq]).endswith('.npz'):
                add_ = '.npz'
            dictionary_eig = jnp.load(str(path_matrix_eigen_decomp[idx_freq]) + add_)

            eigvals = dictionary_eig['eigvals']
            eigvecs = dictionary_eig['eigvecs']

            n_pix = eigvals.shape[-1] // nstokes

            array_eigvecs.append(eigvecs)
            array_eigvals.append(eigvals)

        array_eigvals = jnp.array(array_eigvals)
        array_eigvecs = jnp.array(array_eigvecs)
        
        result_unwrapped = contract('fqp,fp,frp->fqr', array_eigvecs, array_eigvals**(power_diagonal), array_eigvecs)


        return jnp.array(
        [
            [
                [result_unwrapped[f,i*n_pix:(i+1)*n_pix,j*n_pix:(j+1)*n_pix] 
                    for j in range(nstokes)
                ] 
                for i in range(nstokes)
            ]
        for f in range(n_freq)]
    )


def build_obsmat_operator_from_flattened_matrices(
        list_scipy_obsmat_masked, 
        nstokes=2,
        return_transpose=False,
    ):
    
    n_freq = len(list_scipy_obsmat_masked)
    n_pix = list_scipy_obsmat_masked[0].shape[0] // nstokes
    matrix_O = jnp.zeros((n_freq, nstokes, nstokes, n_pix, n_pix))
    
    get_matrix = lambda x: x.T if return_transpose else x

    for i in range(n_freq):
        for j in range(nstokes):
            for k in range(nstokes):
                matrix_O = matrix_O.at[i,j,k,...].set(get_matrix(list_scipy_obsmat_masked[i])[j*n_pix:(j+1)*n_pix, k*n_pix:(k+1)*n_pix].todense())

    return get_dense_furax_operator_from_freq_array(matrix_O)
