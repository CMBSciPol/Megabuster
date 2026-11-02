import numpy as np
import healpy as hp
import jax.numpy as jnp
from jaxtyping import ArrayLike
from typing import Callable

from furax.core import BlockColumnOperator, BlockRowOperator, BroadcastDiagonalOperator, DenseBlockDiagonalOperator, DiagonalOperator
from furax.obs.stokes import StokesQU

__all__ = [
    'Results',
    'build_furax_preconditioner',
    'get_maps_from_Stokes',
    'get_diagonal_operator_from_stokes_maps',
    'get_A_from_array',
    'get_array_from_A',
    'get_healpix_indices_patch_from_mask',
    'get_A_from_array_stokes'
]

class Results(object):
    """
    Class to store the results of the component separation.
    
    Attributes:
        x (ArrayLike):
            Parameters fitted, stored as an array.  
        s (ArrayLike):
            Sky map, stored as an array of shape (n_components, n_stokes, n_pixels).
        W_maxL (Callable):
            Function representing the mixing matrix operator at the parameters fitted.
        success (bool):
            Boolean indicating whether the optimization was successful.
    """
    params: list[str] = ['temp_dust', 'beta_dust', 'beta_pl']
    x: ArrayLike # parameters fitted
    s: ArrayLike # sky map
    W_maxL: Callable # W function at the parameters fitted
    A_maxL: ArrayLike # Mixing matrix operator at the parameters fitted
    success: bool # success of the optimization
    message: str
    W_params: Callable = None # Function to apply the mixing matrix operator at the parameters fitted
    
    def __init__(self):
        self.params = None
        self.x = None
        self.s = None
        self.W_maxL = None
        self.A_maxL = None
        self.success = False
        self.message = "No optimization performed yet."
        self.W_params = None

    @classmethod
    def from_compsep_results(
        cls, 
        name_params,
        final_params, 
        final_W_func, 
        final_A_maxL,
        final_maps, 
        number_iterations, 
        max_iter, 
        ordering_parameter=['beta_dust', 'beta_pl'], 
        ordering_component=['cmb', 'dust', 'synchrotron'],
        W_params=None
):
        """
        Create an instance of Results from the component separation results.
        
        Args:
            final_params (dict): 
                Final parameters after optimization.
            final_W_func (function): Function representing the final mixing matrix operator.
            final_maps (Stokes or ndarray): Final sky map.
            number_iterations (int): Number of iterations performed.
            max_iter (int): Maximum number of iterations allowed.
        
        Returns:
            Results: An instance of Results containing the final parameters, W function, and success status.
        """
        res = cls()
        res.params = name_params
        res.x = np.array([final_params[key] for key in ordering_parameter])

        def final_W(input_maps):
            """
            Function to apply the final mixing matrix operator to the input maps.
            """

            assert input_maps.ndim == 3, "Input maps must have shape (n_frequencies, n_stokes, n_pixels)."
            assert input_maps.shape[1] >= 2, "Input maps must contain at least Q and U Stokes parameters."
            

            input_maps_stokes = Stokes.from_stokes(
                q=hp.reorder(input_maps[:,-2,:],r2n=True), 
                u=hp.reorder(input_maps[:,-1,:],r2n=True)
            )
            return final_W_func(input_maps_stokes)
        
        if W_params is not None:
            def final_W_params(params, input_maps):
                """
                Function to apply the final mixing matrix operator to the input maps.
                """

                assert input_maps.ndim == 3, "Input maps must have shape (n_frequencies, n_stokes, n_pixels)."
                assert input_maps.shape[1] >= 2, "Input maps must contain at least Q and U Stokes parameters."
                

                input_maps_stokes = Stokes.from_stokes(
                    q=hp.reorder(input_maps[:,-2,:],r2n=True), 
                    u=hp.reorder(input_maps[:,-1,:],r2n=True)
                )
                return W_params(params, input_maps_stokes)
        else:
            final_W_params = None

        res.W_maxL = final_W
        res.A_maxL = final_A_maxL
        res.s = np.array([hp.reorder(final_maps[component], n2r=True) for component in range(final_maps.shape[0])])
        res.iterations_done = number_iterations
        res.success = number_iterations < max_iter
        res.message = f"Optimization finished after {number_iterations} iterations out of {max_iter} allowed."
        res.W_params = final_W_params
        return res
    

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

def build_furax_preconditioner(preconditioner_matrix, in_structure):
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

def get_A_from_array(matrix_A, structure_sed):
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
                in_structure=structure_sed,
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


def get_A_from_array_stokes(matrix_A, structure_sed, components=('cmb', 'dust', 'synchrotron')):
    """
    Build a mixing-matrix-like operator (components -> frequencies) whose coefficients
    depend on the Stokes parameter, unlike `megabuster.tools.get_A_from_array`.


    Parameters
    ----------
    matrix_A : array of shape (n_comp, n_stokes, n_freq, n_pix)
    structure_sed : StokesQU structure of a component map, of shape (n_stokes, n_pix).

    Returns
    -------
    BlockRowOperator mapping {'cmb', 'dust', 'synchrotron'} StokesQU maps of shape (n_stokes, n_pix)
    to a StokesQU frequency map of shape (n_stokes, n_freq, n_pix).
    """
    assert matrix_A.ndim == 4, "matrix_A must have shape (n_comp, n_stokes, n_freq, n_pix)"
    return BlockRowOperator(
        {
            component: BroadcastDiagonalOperator(
                matrix_A[num_cpt],
                axis_destination=(-3, -2, -1),
                insert_axes=-2,
                in_structure=structure_sed,
            ) for num_cpt, component in enumerate(components)
        }
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