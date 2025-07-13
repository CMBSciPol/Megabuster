import numpy as np
from jaxtyping import ArrayLike

from furax.core import DiagonalOperator, BlockColumnOperator, BlockRowOperator, BlockDiagonalOperator
from furax.obs.stokes import StokesQU

__all__ = [
    'get_maps_from_Stokes',
    'get_diagonal_operator_from_stokes_maps',
    'get_preconditioner',
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
