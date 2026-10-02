import numpy as np
import healpy as hp
import jax.numpy as jnp
from jaxtyping import ArrayLike

from furax.core import BlockColumnOperator, BlockRowOperator, BroadcastDiagonalOperator, DenseBlockDiagonalOperator, DiagonalOperator
from furax.obs.stokes import StokesQU

__all__ = [
    'get_maps_from_Stokes',
    'get_diagonal_operator_from_stokes_maps',
    'get_A_from_array',
    'get_array_from_A',
    'get_preconditioner',
    'get_healpix_indices_patch_from_mask',
]

def get_maps_from_Stokes(final_maps):
    if isinstance(final_maps, StokesQU):
        return np.asarray(final_maps.data)
    elif isinstance(final_maps, ArrayLike):
        return final_maps
    else:
        raise TypeError("final_maps must be a Stokes object or an array.")

def get_diagonal_operator_from_stokes_maps(stokes_maps, in_structure):
    """
    Given stokes maps, return a DiagonalOperator that
    applies them elementwise to a StokesQU object.

    in_structure is the structure of the StokesQU object.

    Parameters
    ----------
    stokes_maps : np.ndarray
        The stokes maps, must have the last two shapes expressed as (..., n_stokes, n_pix).
    in_structure : StokesQU
        The structure of the StokesQU object, whose leaf has shape (n_stokes, ..., n_pix).

    Returns
    -------
    DiagonalOperator
        The diagonal operator acting on the StokesQU object.
    """

    assert len(stokes_maps.shape) >= 2, "stokes_maps must have shape (..., n_stokes, n_pix)"

    # Move the Stokes axis in front, as in the StokesQU backing array
    return DiagonalOperator(jnp.moveaxis(jnp.asarray(stokes_maps), -2, 0), in_structure=in_structure)

def get_preconditioner(preconditioner_matrix, in_structure):
    """
    Given a preconditioner matrix in terms of components, return a BlockColumnOperator that
    applies the preconditioner to a dictionary of StokesQU component maps.

    Parameters
    ----------
    preconditioner_matrix : np.ndarray
        The preconditioner matrix in terms of components, of shape (n_comp, n_comp, n_stokes, n_pix).
    in_structure : StokesQU
        The structure of a StokesQU component map, of shape (n_stokes, n_pix).

    Returns
    -------
    BlockColumnOperator
        The preconditioner as a BlockColumnOperator expressed with Pytrees using dictionaries of {'cmb', 'dust', 'synchrotron'}.
    """

    return BlockColumnOperator({
        component_0: BlockRowOperator({
            component_1: get_diagonal_operator_from_stokes_maps(
                preconditioner_matrix[num_cpt_0, num_cpt_1, :, :],
                in_structure
            ) for num_cpt_1, component_1 in enumerate(['cmb', 'dust', 'synchrotron'])
        }) for num_cpt_0, component_0 in enumerate(['cmb', 'dust', 'synchrotron'])
    })

def get_dense_furax_operator_from_freq_array(matrix):
    assert matrix.ndim == 5, "matrix must have shape (n_freq, n_stokes, n_stokes, n_pix, n_pix)"
    assert matrix.shape[1] == matrix.shape[2], "matrix must be square in the Stokes parameters"
    assert matrix.shape[3] == matrix.shape[4], "matrix must be square in the pixel space"
    nstokes = matrix.shape[1]
    assert nstokes == 2, "matrix must have 2 Stokes parameters (Q, U)"

    matrix = jnp.asarray(matrix)

    in_structure = StokesQU.structure_for((matrix.shape[0], matrix.shape[-1]), dtype=matrix.dtype)
    return DenseBlockDiagonalOperator(matrix, in_structure=in_structure, subscripts='fstqp,tfp->sfq')

def get_A_from_array(matrix_A, in_structure_sed):
    """
    Build a mixing matrix operator from an array of shape (n_comp, n_freq) or (n_comp, n_freq, n_pix),
    applied as the furax SED operators.
    """
    if matrix_A.ndim == 2:
        matrix_to_build = matrix_A[..., None]
    else:
        matrix_to_build = matrix_A
    return BlockRowOperator(
        {
            component: BroadcastDiagonalOperator(
                matrix_to_build[num_cpt, ...],
                axis_destination=(-2, -1),
                insert_axes=-2,
                in_structure=in_structure_sed,
            ) for num_cpt, component in enumerate(['cmb', 'dust', 'synchrotron'])
        }
    )

def get_array_from_A(A, n_pix, components=('cmb', 'dust', 'synchrotron')):
    """
    Return the mixing matrix operator as an array of shape (n_freq, n_comp, n_pix).

    The SEDs which do not depend on the pixel have shape (n_freq, 1) and are broadcast over the pixels.
    """
    return jnp.stack(
        [jnp.broadcast_to(A.blocks[component].diagonal, (A.blocks[component].diagonal.shape[0], n_pix)) for component in components],
        axis=1,
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