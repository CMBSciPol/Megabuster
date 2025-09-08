import numpy as np
import healpy as hp
from jaxtyping import ArrayLike

from furax.core import DiagonalOperator, BlockColumnOperator, BlockRowOperator, BlockDiagonalOperator, BlockRowOperator, BroadcastDiagonalOperator, DenseBlockDiagonalOperator
from furax.obs.stokes import StokesQU

__all__ = [
    'get_maps_from_Stokes',
    'get_diagonal_operator_from_stokes_maps',
    'get_A_from_array',
    'get_preconditioner',
    'get_healpix_indices_patch_from_mask',
]

def get_maps_from_Stokes(final_maps):
    if isinstance(final_maps, StokesQU):
        return np.array([final_maps.q, final_maps.u])
    elif isinstance(final_maps, ArrayLike):
        return final_maps
    else:
        raise TypeError("final_maps must be a Stokes object or an array.")

def get_diagonal_operator_from_stokes_maps(stokes_maps, in_structure):
    """
    Given stokes maps, return a BlockDiagonalOperator that
    applies the diagonal operator to a StokesQU object.
    
    in_structure is the structure of the StokesQU object.

    Parameters
    ----------
    stokes_maps : np.ndarray
        The stokes maps, must have the last two shapes expressed as (..., n_stokes, n_pix).
    in_structure : ShapeDtypeStruct
        The structure of the StokesQU object.

    Returns
    -------
    BlockDiagonalOperator
        The diagonal operator as a BlockDiagonalOperator built from StokesQU pytree.
    """

    assert len(stokes_maps.shape) >= 2, "stokes_maps must have shape (..., n_stokes, n_pix)"

    nstokes = stokes_maps.shape[-2]

    tuple_diagonal_operators = tuple(
        DiagonalOperator(
            stokes_maps[..., num_stokes, :],
            in_structure=in_structure
        ) for num_stokes in range(nstokes)
    )
    return BlockDiagonalOperator(
            StokesQU(
                    *tuple_diagonal_operators
            )
        )

def get_preconditioner(preconditioner_matrix, in_structure):
    """
    Given a preconditioner matrix in terms of components, return a BlockColumnOperator that
    applies the preconditioner to a StokesQU object.
    
    in_structure must be as_structure(AOND['cmb'].q)

    Parameters
    ----------
    preconditioner_matrix : np.ndarray
        The preconditioner matrix in terms of components.
    in_structure : ShapeDtypeStruct
        The structure of the StokesQU object.

    Returns
    -------
    BlockColumnOperator
        The preconditioner as a BlockColumnOperator expressed with Pytrees using dictionaries of {'cmb', 'dust', 'synchrotron'}.
    """

    return BlockColumnOperator({
        component_0: BlockRowOperator({
            component_1: BlockDiagonalOperator(
                get_diagonal_operator_from_stokes_maps(
                    preconditioner_matrix[num_cpt_0, num_cpt_1, :, :],
                    in_structure
                )
            ) for num_cpt_1, component_1 in enumerate(['cmb', 'dust', 'synchrotron'])
        }) for num_cpt_0, component_0 in enumerate(['cmb', 'dust', 'synchrotron'])
    })

def get_dense_furax_operator_from_freq_array(matrix, in_structure):
    assert matrix.ndim == 4, "matrix must have shape (n_freq, n_stokes, n_stokes, n_pix, n_pix)"
    assert matrix.shape[1] == matrix.shape[2], "matrix must be square in the Stokes parameters"
    assert matrix.shape[3] == matrix.shape[4], "matrix must be square in the pixel space"
    nstokes = matrix.shape[1]
    assert nstokes == 2, "matrix must have 2 Stokes parameters (Q, U)"

    ops_Q = BlockRowOperator(
                StokesQU(
                    DenseBlockDiagonalOperator(matrix[:,0,0,...], in_structure, subscripts='fqp,fp->fq'), 
                    DenseBlockDiagonalOperator(matrix[:,0,1,...], in_structure, subscripts='fqp,fp->fq')
                )
            )
    ops_U = BlockRowOperator(
                StokesQU(
                    DenseBlockDiagonalOperator(matrix[:,1,0,...], in_structure, subscripts='fqp,fp->fq'), 
                    DenseBlockDiagonalOperator(matrix[:,1,1,...], in_structure, subscripts='fqp,fp->fq')
                )
            )
    list_QU_operators = [ops_Q, ops_U]
    return BlockColumnOperator(StokesQU(*list_QU_operators))

def get_A_from_array(matrix_A, in_structure_sed):
    if matrix_A.ndim == 2:
        matrix_to_build = matrix_A[..., None]
    else:
        matrix_to_build = matrix_A
    return BlockRowOperator(
        {component:BroadcastDiagonalOperator(matrix_to_build[num_cpt,...],in_structure=in_structure_sed,) for num_cpt, component in enumerate(['cmb', 'dust', 'synchrotron'])}
    )

def get_healpix_indices_patch_from_mask(mask, nside_patches, nest=False):
    """
    Get the healpix indices of the patches from a mask.

    Parameters
    ----------
    mask : np.ndarray
        The mask to get the indices from.
    nside_patches : int
        The nside of the patches.

    Returns
    -------
    np.ndarray
        The healpix indices of the patches.
    """
    if nest:
        order_in = 'NESTED'
        order_out = 'NESTED'
    else:
        order_in = 'RING'
        order_out = 'RING'
    
    all_indices = hp.ud_grade(np.arange(12*nside_patches**2), nside_out=hp.npix2nside(mask.size), order_in=order_in, order_out=order_out)

    values = np.unique((all_indices + 1) * mask)

    new_indices = np.zeros_like(all_indices, dtype=np.int32)
    for i, value in enumerate(values):
        location_indices = np.where((all_indices + 1) * mask == value)[0]
        if location_indices.size > 0: 
            new_indices[location_indices] = i

    return new_indices[mask != 0] - 1