import numpy as np
import jax
import jax.numpy as jnp
from jaxtyping import Inexact, PyTree, ArrayLike
import scipy
import equinox

from furax.obs.stokes import StokesQU
from furax.obs.operators._qu_rotations import QURotationOperator
from furax import AbstractLinearOperator, square
from furax.core import DiagonalOperator, IndexOperator, BlockColumnOperator, BlockRowOperator, BlockDiagonalOperator, SumOperator, TransposeOperator

__all__ = [
    'build_calibration_operator', 'StokesPyTree_to_list', 'list_to_stokes'
]
    
def build_calibration_operator(angles, shape) -> BlockDiagonalOperator:
    """
    Calibration Operator:
      StokesOp @ BlockColumn( BandpassRow @ HWPDiag @ ListOp @ AOp )
    """

    # Calibration matrix per frequency sample

    Calibration_Operator = QURotationOperator.create(
    shape=shape,
    stokes="QU",
    angles=angles,
    )
    #    for alpha in angles
    #]

    #Calibration_Operator = BlockDiagonalOperator(Calibration_list)

    return Calibration_Operator

def build_calibration_operator2(stokes_maps, angles, in_structure):
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

def StokesPyTree_to_list(stokes_pytree: PyTree) -> list[StokesQU]:
    """Convert a PyTree or Stokes-like object into a list of `StokesQU` per frequency.

    Accepts either:
    - a list/tuple of `StokesQU` objects (returned as-is), or
    - an object with attributes `.q` and `.u` where each is an array of shape
      (n_freq, n_pix). In that case returns [StokesQU(q=..., u=...), ...].

    Raises TypeError if the input shape is not understood.
    """
    # If already a list/tuple of StokesQU, return a shallow copy
    if isinstance(stokes_pytree, (list, tuple)):
        return list(stokes_pytree)

    # Try to fetch q/u attributes
    q = getattr(stokes_pytree, 'q', None)
    u = getattr(stokes_pytree, 'u', None)
    if q is None or u is None:
        raise TypeError("stokes_pytree must be a list of StokesQU or have 'q' and 'u' attributes")

    q = jnp.asarray(q)
    u = jnp.asarray(u)

    if q.ndim < 1 or u.ndim < 1 or q.shape[0] != u.shape[0]:
        raise ValueError("q and u must be at least 1D and have matching first dimension (n_freq)")

    n_freq = q.shape[0]
    return [StokesQU(q=q[i], u=u[i]) for i in range(n_freq)]


def list_to_stokes(sky_maps: list[StokesQU]) -> StokesQU:
    """Build a single `StokesQU` object from a list of per-frequency `StokesQU`.

    The returned object's `q` and `u` arrays will have shape (n_freq, n_pix).
    """
    if not isinstance(sky_maps, (list, tuple)):
        raise TypeError("sky_maps must be a list or tuple of StokesQU")

    # Stack using jax.numpy to keep types consistent with JAX-based code paths
    Q = jnp.stack([s.q for s in sky_maps])
    U = jnp.stack([s.u for s in sky_maps])
    return StokesQU(q=Q, u=U)