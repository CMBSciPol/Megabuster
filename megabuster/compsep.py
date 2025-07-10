import numpy as np
import jax
import jax.numpy as jnp
import equinox
import lineax as lx
import healpy as hp
from jaxtyping import ArrayLike
from typing import Callable

import operator

from furax.core import IdentityOperator
from furax.tree import as_structure
from furax.obs.stokes import Stokes
# from furax import Config

from megabuster.minimizers import filter_optimize, minimize_likelihood
from megabuster.mixingmatrix import create_MixingMatrixOperator, create_MixingMatrixOperator_deriv
from megabuster.tools import get_diagonal_operator_from_stokes_maps, get_preconditioner, get_maps_from_Stokes

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
    x: ArrayLike # parameters fitted
    s: ArrayLike # sky map
    W_maxL: Callable # W function at the parameters fitted
    A_maxL: ArrayLike # Mixing matrix operator at the parameters fitted
    success: bool # success of the optimization
    
    def __init__(self):
        self.x = None
        self.s = None
        self.W_maxL = None
        self.A_maxL = None
        self.success = False

    @classmethod
    def from_compsep_results(
        cls, 
        final_params, 
        final_W_func, 
        final_A_maxL,
        final_maps, 
        number_iterations, 
        max_iter, 
        ordering_parameter=['beta_dust', 'beta_pl'], 
        ordering_component=['cmb', 'dust', 'synchrotron']
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
        res.x = np.array([final_params[key] for key in ordering_parameter])

        def final_W(input_maps):
            """
            Function to apply the final mixing matrix operator to the input maps.
            """

            assert input_maps.ndim == 3, "Input maps must have shape (n_frequencies, n_stokes, n_pixels)."
            assert input_maps.shape[1] >= 2, "Input maps must contain at least Q and U Stokes parameters."
            

            input_maps_stokes = Stokes.from_stokes(Q=input_maps[:,-2,:], U=input_maps[:,-1,:])
            output_map_stokes = final_W_func(input_maps_stokes)
            return np.array([get_maps_from_Stokes(output_map_stokes[key]) for key in ordering_component])

        res.W_maxL = final_W
        res.A_maxL = final_A_maxL
        res.s = np.array([get_maps_from_Stokes(final_maps[key]) for key in ordering_component])
        res.success = number_iterations < max_iter
        return res

def dot_2(x,y):
    """Scalar product of two Pytrees.

    If one of the leaves is complex, the hermitian scalar product is returned.

    Args:
        x: The first Pytree.
        y: The second Pytree.

    Example:
        >>> import furax as fx
        >>> x = {'a': jnp.array([1., 2, 3]), 'b': jnp.array([1, 0])}
        >>> y = {'a': jnp.array([2, -1, 1]), 'b': jnp.array([2, 0])}
        >>> fx.tree.dot(x, y)
        Array(5., dtype=float32)
    """
    
    xy = jax.tree.map(jnp.vdot, x, y)
    return jax.tree.reduce(operator.add, xy)
    

def perform_compsep(
    first_guess_params, 
    sky_map, 
    frequencies, 
    invN_matrix,
    fixed_params=dict(),
    binary_mask=None,
    dust_nu0=150.0, 
    synchrotron_nu0=20.0, 
    obs_mat_operator=None, 
    full_sky_transpose_obsmat_operator=None,
    use_preconditioner=True,
    sigma_perturbation=1e-6,
    diag_obsmat_matrices=None,
    optimize_func=filter_optimize,
    max_iter=1,
    tol=1e-5,
    ordering_parameter=['beta_dust', 'beta_pl'], 
    ordering_component=['cmb', 'dust', 'synchrotron']
    ):
    """
    Perform component separation using the given parameters and data.

    Parameters
    ----------
    first_guess_params: dict
        Dictionary of parameters. Must contain the following keys:
        - temp_dust: Dust temperature parameter.
        - beta_dust: Dust spectral index parameter.
        - beta_pl: Synchrotron spectral index parameter.
    sky_map: Stokes or ndarray
        StokesPyTree object or ndarray containing the sky map. Assumes to only contain Q and U Stokes parameters. If ndarray, the shape must be (n_frequencies, n_stokes, n_pixels).
    frequencies: jnp.ndarray
        Array of frequencies in GHz.
    invN_matrix:
        Inverse noise covariance matrix object with dimensions (n_frequencies, n_stokes, n_pixels). Currently implemented so that the Stokes parameters are only Q and U. 
    dust_nu0: float (optional)
        Dust reference frequency parameter in GHz. Default is 150.0 GHz.
    synchrotron_nu0: float (optional)
        Synchrotron reference frequency parameter in GHz. Default is 20.0 GHz.
    obs_mat_operator: AbstractLinearOperator (optional)
        Observation matrix operator. Default is None.
    use_preconditioner: bool (optional)
        Whether to use a preconditioner, currently does not account for the observation matrix. Default is True.

    
    Returns
    -------
    dict
        Final parameters stored as a dictionary with keys 'temp_dust', 'beta_dust', and 'beta_pl'.

    Notes
    -----
    All the inputs for sky map, inverse noise covariance matrix, and observation matrix operator are assumed to contain only the pixels observed and not the full sky, which otherwise leads to a significant increase in computation time. 
    """

    # Few tests
    assert isinstance(first_guess_params, dict), "First guess parameters must be stored as a dictionary."
    assert 'temp_dust' in first_guess_params or 'temp_dust' in fixed_params, "First guess parameters must contain 'temp_dust'."
    assert 'beta_dust' in first_guess_params or 'beta_dust' in fixed_params, "First guess parameters must contain 'beta_dust'."
    assert 'beta_pl' in first_guess_params or 'beta_pl' in fixed_params, "First guess parameters must contain 'beta_pl'."
    assert isinstance(sky_map, (Stokes, ArrayLike)), "Sky map must be a Stokes object or a ndarray."
    
    if binary_mask is not None:
        assert isinstance(binary_mask, ArrayLike), "Binary mask must be a ndarray."
        assert binary_mask.ndim == 1, "Binary mask must be a 1D array."
        pixels_to_retain = np.where(binary_mask != 0)[0]
    else:
        pixels_to_retain = np.ones_like(invN_matrix.shape[-1], dtype=bool)
    
    assert invN_matrix.ndim == 3, "Inverse noise covariance matrix must have shape (n_frequencies, n_stokes, n_pixels)."
    assert invN_matrix.shape[1] == 2, "Inverse noise covariance matrix must contain only Q and U Stokes parameters."
    

    invN_matrix = invN_matrix[..., pixels_to_retain]
    invN_matrix_nested = np.zeros_like(invN_matrix)
    
    for i in range(invN_matrix.shape[0]):
        invN_matrix_nested[i] = hp.reorder(invN_matrix[i], r2n=True)    
        
    if isinstance(sky_map, jnp.ndarray):
        assert sky_map.ndim == 3, "Sky map must have shape (n_frequencies, n_stokes, n_pixels)."
        assert sky_map.shape[1] == 2, "Sky map must contain only Q and U Stokes parameters."
        sky_map = Stokes.from_stokes(
            Q=hp.reorder(sky_map[:, -2], r2n=True)[..., pixels_to_retain], 
            U=hp.reorder(sky_map[:, -1], r2n=True)[..., pixels_to_retain]
        )
    else:
        assert sky_map.q.shape[0] == sky_map.u.shape[0], "Sky map must have the same number of Q and U Stokes parameters."
        assert sky_map.q.shape[1] == sky_map.u.shape[1], "Sky map must have the same number of pixels for Q and U Stokes parameters."
        # Retain only the pixels that are not masked
        sky_map = Stokes.from_stokes(
            Q=hp.reorder(sky_map.q, r2n=True)[..., pixels_to_retain], 
            U=hp.reorder(sky_map.u, r2n=True)[..., pixels_to_retain]
        )

    # Prepare the in_structure of the upcoming operators
    in_structure_sed = sky_map.structure_for((sky_map.shape[1],))
    in_structure_noise_cov = sky_map.structure.q

    invN = get_diagonal_operator_from_stokes_maps(invN_matrix, in_structure_noise_cov)

    # Prepare the observation matrix operator
    if obs_mat_operator is None:
        obs_mat_operator = IdentityOperator(invN.in_structure())
    
    if full_sky_transpose_obsmat_operator is None:
        full_sky_transpose_obsmat_operator = obs_mat_operator.T
    
    # Precompute part of the right-hand side of the CG equation
    ONd = full_sky_transpose_obsmat_operator(invN(sky_map))
    ONd.q.block_until_ready()

    

    if diag_obsmat_matrices is None:
        diagonal_ONO_array = jnp.copy(invN_matrix_nested)
    else:
            
        if diag_obsmat_matrices.shape[-1] == pixels_to_retain.size:
            print("Using diag obsmat matrices")
            diag_obsmat_matrices = diag_obsmat_matrices[..., pixels_to_retain]

        print("Using diag offdiag obsmat")
        diag_obsmat = jnp.array(diag_obsmat_matrices).reshape((len(frequencies), 2, invN_matrix_nested.shape[-1]))
        diagonal_ONO_array = jnp.einsum('fsp,fsp,fsp->fsp', diag_obsmat, invN_matrix_nested, diag_obsmat)
        
    number_components = 3 # CMB, dust, synchrotron

    def get_A_s_AOND(params, right_member=ONd):
        """
            Compute the log-proba given a set of parameters 
        """

        # Select parameters which are to be estimated 
        parameters_dict = dict()
        for param_name in ['temp_dust', 'beta_dust', 'beta_pl']:
            if param_name in params:
                parameters_dict[param_name] = params[param_name]
            else:
                parameters_dict[param_name] = fixed_params[param_name]

        # Mixing matrix operator
        A = create_MixingMatrixOperator(frequencies, parameters_dict, in_structure_sed, dust_nu0=dust_nu0, synchrotron_nu0=synchrotron_nu0)

        # Full right-hand side of the CG equation
        AOND = A.T(right_member)

        preconditioner = None
        if use_preconditioner:
            matrix_A = jnp.zeros((frequencies.size, number_components))
            for component in range(number_components):
                matrix_A = matrix_A.at[:,component].set(A.block_leaves[component]._diagonal.squeeze())
            
            # Compute the preconditioner matrix assuming that the observation matrix is the identity and the noise covariance matrix is diagonal in pixel domain
            preconditioner_matrix = jax.lax.stop_gradient(jnp.linalg.pinv(jnp.einsum('fc,fsp,fk->psck', matrix_A, diagonal_ONO_array, matrix_A)).T)

            # Prepare the preconditioner operator
            preconditioner = get_preconditioner(
                preconditioner_matrix, 
                in_structure=as_structure(AOND['cmb'].q)
            )


        diagonal_central_term = ( (A.T @ obs_mat_operator.T @ invN @ obs_mat_operator @ A) + sigma_perturbation * IdentityOperator(as_structure(AOND))).I(solver=lx.CG(rtol=1e-5, atol=1e-5, max_steps=200), throw=False, solver_throw=False, preconditioner=preconditioner)
        first_central_term =  diagonal_central_term(AOND)

        return A, first_central_term, AOND, diagonal_central_term

    @equinox.filter_custom_jvp
    def spectral_likelihood_custom_gradient(params):
        """
            Compute the log-proba given a set of parameters 
        """

        # Select parameters which are to be estimated 
        parameters_dict = dict()
        for param_name in ['temp_dust', 'beta_dust', 'beta_pl']:
            if param_name in params:
                parameters_dict[param_name] = params[param_name]
            else:
                parameters_dict[param_name] = fixed_params[param_name]

        _, map_s, AOND, _ = get_A_s_AOND(parameters_dict)

        logL = -dot_2(AOND, map_s)

        # Compute the negative log-likelihood
        return logL

    @spectral_likelihood_custom_gradient.def_jvp
    def custom_gradient(primals, tangents):
        """
        Custom JVP rule for the perturbative_negative_log_prob function.
        """
        params, = primals
        beta_tau, = tangents

        # Select parameters which are to be estimated 
        parameters_dict = dict()
        for param_name in ['temp_dust', 'beta_dust', 'beta_pl']:
            if param_name in params:
                parameters_dict[param_name] = params[param_name]
            else:
                parameters_dict[param_name] = fixed_params[param_name]

        A, map_s, AOND, _ = get_A_s_AOND(parameters_dict)

        logL = -dot_2(AOND, map_s)

        A_deriv = create_MixingMatrixOperator_deriv(
            frequencies, 
            parameters_dict, 
            in_structure_sed, 
            dust_nu0=dust_nu0, 
            synchrotron_nu0=synchrotron_nu0
        )

        left_hand_term = ONd - (obs_mat_operator.T @ invN @ obs_mat_operator @ A)(map_s)

        keys_params = params.keys()


        final_grad_log = 0
        for key in keys_params:
            final_grad_log += -2*dot_2(A_deriv[key](map_s), left_hand_term) * beta_tau[key]
        return logL, final_grad_log


    print("Launching minimization!!", flush=True)
    output_params, output_state = minimize_likelihood(first_guess_params, 
        spectral_likelihood_custom_gradient, 
        max_iter=max_iter, 
        tol=tol,
        optimize_func=optimize_func) # first output is the final parameters, second output is the final state of the optimizer 

    print("Minimization finished!! Now retrieving maps . . .", flush=True)
    A_maxL, final_maps = get_A_s_AOND(output_params, full_sky_transpose_obsmat_operator(invN(sky_map)))[:2]

    A_maxL_array = np.zeros((frequencies.size, number_components))
    for component in range(number_components):
        A_maxL_array = A_maxL_array.at[:,component].set(A_maxL.block_leaves[component]._diagonal.squeeze())
            
    def W_maxL(input_map, input_is_ring=False):
        assert input_map.ndim == 3, "Input map must have shape (n_frequencies, n_stokes, n_pixels)."
        
        if input_is_ring:
            new_input_map = np.zeros_like(input_map)
            for i in range(input_map.shape[0]):
                new_input_map[i] = hp.reorder(input_map[i], r2n=True)
            input_map = new_input_map    
            
        output_map_truncated = get_A_s_AOND(output_params, full_sky_transpose_obsmat_operator(invN(input_map[...,pixels_to_retain])))[1]

        output_map = np.zeros(((number_components,) + output_map_truncated.shape[-2:]), dtype=output_map_truncated.dtype)
        for j in range(number_components):
            output_map[j, :, pixels_to_retain] = output_map_truncated
            output_map[j] = hp.reorder(output_map[j], n2r=True)

        return output_map

    return Results.from_compsep_results(
        output_params, 
        W_maxL, 
        A_maxL_array,
        final_maps, 
        output_state[0].count, 
        max_iter, 
        ordering_parameter=ordering_parameter, 
        ordering_component=ordering_component
    )
