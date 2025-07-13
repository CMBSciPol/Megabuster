import scipy as sp
import numpy as np

from megabuster.obsmat import to_QU_operators_from_files

__all__ = [
    'load_obsmat_mask',
    'load_all_obsmat',
]

def load_obsmat_mask(path_obsmat, size_obsmat, mask_stacked=None):
    print("Loading obsmat from path", path_obsmat)

    mm_data = np.load(path_obsmat + "data.npy")
    mm_row_indices = np.load(path_obsmat + "row_indices.npy", )
    mm_column_indices = np.load(path_obsmat + "column_indices.npy")

    if mask_stacked is None:
        mask_stacked = np.ones(size_obsmat, dtype=bool)

    sparse_matrix = sp.sparse.csr_matrix(
        (mm_data, (mm_row_indices, mm_column_indices)),
        (size_obsmat,size_obsmat),
    )[:,mask_stacked][mask_stacked,:]

    return sparse_matrix


def load_all_obsmat(list_path_obsmat_prefix, size_obsmat, nstokes=None, kind='precomputations_indices', mask_stacked=None, return_transpose=False):
    """
    Load all obsmat from a list of paths.
    """

    assert kind in ['precomputations_indices', 'precomputations_scipy'], "The parameter 'kind' must be one of 'precomputations_scipy', 'precomputations_indices', corresponding respectively to the three different expected formats of the obsmat."

    if kind == 'precomputations_scipy':
        return [load_obsmat_mask(
            path_obsmat=str(path_obsmat),
            size_obsmat=size_obsmat,
            mask_stacked=mask_stacked
        ) for path_obsmat in list_path_obsmat_prefix]
    elif kind == 'precomputations_indices':
        assert nstokes is not None, "When using 'precomputations_indices', the parameter 'nstokes' must be provided so that size_obsmat // nstokes correspond to the number of pixel per observation matrix QQ, QU, UQ, UU for instance."
        
        label_stokes = ['q', 'u'] if nstokes == 2 else ['i', 'q', 'u']

        list_precomputations = [
            [
                [str(path_obsmat)  + f"_stokes_{label_stokes[i]}_stokes_{label_stokes[j]}.npz" for i in range(nstokes)]
                for j in range(nstokes)
            ]
            for path_obsmat in list_path_obsmat_prefix
        ]
        operator_output = to_QU_operators_from_files(list_precomputations, int(size_obsmat))
        if return_transpose:
            return operator_output.T
