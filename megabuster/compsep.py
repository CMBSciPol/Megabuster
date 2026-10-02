import warnings
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
from furax.obs.stokes import Stokes

from furax_cs import minimize, SOLVER_NAMES

from megabuster.mixingmatrix import create_MixingMatrixOperator, create_MixingMatrixOperator_deriv
from megabuster.tools import (
    get_diagonal_operator_from_stokes_maps, 
    get_preconditioner, 
    get_A_from_array, 
    get_array_from_A,
    get_maps_from_Stokes, 
    get_dense_furax_operator_from_freq_array
)

__all__ = [
    'Results',
    'perform_compsep',
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
    do_minimization=True,
    fixed_params=dict(),
    binary_mask=None,
    dust_nu0=150.0, 
    synchrotron_nu0=20.0, 
    central_freq_op=None,
    obs_mat_operator=None, 
    obsmat_operator_rhs=None,
    use_preconditioner_diag=False,
    use_preconditioner_pinv=False,
    matrix_precond=None,
    dictionary_parameters_minimization: dict={
        'max_iter':1, 
        'tol':1e-5,
        'solver_name':"optax_lbfgs",
        'options':dict()
    },
    dictionary_parameters_CG: dict={'max_steps_CG':200, 'tol_CG':1e-6},
    ordering_parameter=['beta_dust', 'beta_pl'], 
    ordering_component=['cmb', 'dust', 'synchrotron'],
    patch_indices=None
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
        StokesPyTree object or ndarray containing the sky map. Assumes to only contain Q and U Stokes parameters. 
        
        If ndarray, the shape must be (n_frequencies, n_stokes, n_pixels).
    frequencies: jnp.ndarray
        Array of frequencies in GHz.
    invN_matrix:
        Inverse noise covariance matrix object with dimensions (n_frequencies, n_stokes, n_pixels). 
        Currently implemented so that the Stokes parameters are only Q and U. 
    do_minimization: bool (optional)
        Whether to perform the minimization or not. If False, the first_guess_params are used as output parameters. Default is True.
    fixed_params: dict (optional)
        Dictionary of fixed parameters. Can contain the following:
        * temp_dust: Dust temperature parameter.
        * beta_dust: Dust spectral index parameter.
        * beta_pl: Synchrotron spectral index parameter.
        If a parameter is in both first_guess_params and fixed_params, the value in fixed_params is used.
        Default is an empty dictionary.
    binary_mask: ndarray (optional)
        Binary mask to apply to the sky map and inverse noise covariance matrix. 
        Must be a 1D array of size n_pixels with values 0 or 1. Default is None, which means no mask is applied.
    dust_nu0: float (optional)
        Dust reference frequency parameter in GHz. Default is 150.0 GHz.
    synchrotron_nu0: float (optional)
        Synchrotron reference frequency parameter in GHz. Default is 20.0 GHz.
    central_freq_op: AbstractLinearOperator (optional)
        Central frequency operator corresponding to the matrix M which will be projected into component space and inverted as (A.T M A)^{-1]. 
        M must be a function compilable and jittable with JAX, ideally a FURAX operator, and applicable directly on nested masked Stokes objects.
        If None, it is taken to be the composition of the operator provided in obs_mat_operator (which is Identity of None was provided for obs_mat_operator) with invN and its transpose.
    obs_mat_operator: AbstractLinearOperator (optional)
        Observation matrix operator, which must be a function compilable and jittable with JAX, ideally a FURAX operator, and applicable directly on nested masked Stokes objects.
        Default is None and is taken to be the identity operator. Not used if central_freq_op is provided. 
    obsmat_operator_rhs: AbstractLinearOperator (optional)
        Right-hand side of the equation corresponding to the application of O.T to invN (d). WANRNING: The operator provided is assumed to be the a function applying O.T and not O. If None, it is taken to be the transpose of the operator provided in obs_mat_operator (which is Identity of None was provided for obs_mat_operator).
    use_preconditioner_diag: bool (optional)
        Whether to use a diagonal preconditioner for the central matrix (A.T M A). 
        If set to true, then matrix_precond must be provided as a matrix of shape (n_frequencies, n_stokes, n_pixels), and it will be assumed to be already in the NESTED pixel distribution.
        Default is False.
    use_preconditioner_pinv: bool (optional)
        Whether to use a pseudo-inverse preconditioner for the central matrix (A.T M A). 
        If set to true, then matrix_precond must be provided as a matrix of shape (n_frequencies, n_stokes, n_stokes, n_pixels, n_pixels), 
        and it will be assumed to be already in the NESTED pixel distribution. The preconditioner which will be used will be A_pinv.T @ matrix_precond @ A_pinv, where A_pinv is the pseudo-inverse of the mixing matrix at the current parameters. Default is False.
    matrix_precond: jnp.ndarray (optional)
        Matrix to use for the preconditioner. If use_preconditioner_diag is True, then it must be of shape (n_frequencies, n_stokes, n_pixels).
        If use_preconditioner_pinv is True, then it must be of shape (n_frequencies, n_stokes, n_stokes, n_pixels, n_pixels).
        In both case the matrix provided is assumed to be already in the NESTED pixel distribution.
        Default is None.
    optimize_func: function (optional)
        Function to use for the optimization. Must be a function that takes as input the first guess parameters, the function to minimize, the maximum number of iterations, and the tolerance.
        Default is filter_optimize.
    dictionary_parameters_minimization: dict (optional)
        Dictionary containing the parameters for the minimization. The keys must be 'max_iter' and 'tol'.
        The values must be integers and floats respectively.
        Default is {'max_iter': 1, 'tol': 1e-5}.
    dictionary_parameters_CG: dict (optional)
        Dictionary containing the parameters for the conjugate gradient solver. The keys must be 'max_steps_CG' and 'tol_CG'.
        The values must be integers and floats respectively.
        Default is {'max_steps_CG': 200, 'tol_CG': 1e-6}.
    ordering_parameter: list of str (optional)
        List of parameter names in the order they should be stored in the output Results object.
        Default is ['beta_dust', 'beta_pl'].
    ordering_component: list of str (optional)
        List of component names in the order they should be stored in the output Results object.
        Default is ['cmb', 'dust', 'synchrotron'].
    patch_indices: dict (optional)
        Dictionary containing the indices of the patches for each parameter to be estimated.
        The keys must be 'temp_dust_patches', 'beta_dust_patches', and 'beta_pl_patches'.
        The values must be 1D arrays of integers containing the indices of the patches for each parameter.
        If a parameter is not to be estimated in patches, its value must be None.
        Default is None, which means that all foregrounds parameters are estimated as single values over the full sky.

    
    
    Returns
    -------
    dict
        Final parameters stored as a dictionary with keys 'temp_dust', 'beta_dust', and 'beta_pl'.

    Notes
    -----
    All the inputs for sky map, inverse noise covariance matrix, and observation matrix operator are assumed to contain only the pixels observed and not the full sky, which otherwise leads to a significant increase in computation time. 
    """

    n_stokes = 2

    # Few tests
    assert isinstance(first_guess_params, dict), "First guess parameters must be stored as a dictionary."
    assert 'temp_dust' in first_guess_params or 'temp_dust' in fixed_params, "First guess parameters must contain 'temp_dust'."
    assert 'beta_dust' in first_guess_params or 'beta_dust' in fixed_params, "First guess parameters must contain 'beta_dust'."
    assert 'beta_pl' in first_guess_params or 'beta_pl' in fixed_params, "First guess parameters must contain 'beta_pl'."
    assert isinstance(sky_map, (Stokes, ArrayLike)), "Sky map must be a Stokes object or a ndarray."

    assert (use_preconditioner_diag != use_preconditioner_pinv) or (use_preconditioner_pinv == False), "Only one type of preconditioner (use_preconditioner_diag or use_preconditioner_pinv) can be used at a time."
    assert (use_preconditioner_pinv == (matrix_precond is not None)), "If the pseudo-inverse preconditioner (use_preconditioner_pinv) is set to True, then matrix_precond must be provided."
    
    if dictionary_parameters_minimization['tol'] < dictionary_parameters_CG['tol_CG'] and do_minimization:
        warnings.warn("The tolerance for the minimization is smaller than the tolerance for the conjugate gradient solver. This might lead to suboptimal results.")

    solver_name = dictionary_parameters_minimization.get('solver_name', 'optax_lbfgs')
    # assert solver_name in SOLVER_NAMES.__args__, f"Solver name must be one of {SOLVER_NAMES.__args__}."

    if do_minimization:
        print("The minimization will be performed with the following parameters:", dictionary_parameters_minimization, flush=True)

    if binary_mask is not None:
        assert isinstance(binary_mask, ArrayLike), "Binary mask must be a ndarray."
        assert binary_mask.ndim == 1, "Binary mask must be a 1D array."
        pixels_to_retain = np.where(binary_mask != 0)[0]
        pixels_to_retain_nested = np.where(hp.reorder(binary_mask, r2n=True) != 0)[0]
    else:
        pixels_to_retain = np.ones_like(invN_matrix.shape[-1], dtype=bool)
        pixels_to_retain_nested = np.ones_like(pixels_to_retain, dtype=bool)
        binary_mask = np.ones(invN_matrix.shape[-1])
    
    if patch_indices is not None:
        assert np.all(np.isin(list(patch_indices.keys()), ['temp_dust_patches', 'beta_dust_patches', 'beta_pl_patches'])), "Single patch indices must be a dictionary with keys 'temp_dust_patches', 'beta_dust_patches', and 'beta_pl_patches'."
        for key_patch in ['temp_dust_patches', 'beta_dust_patches', 'beta_pl_patches']:
            if key_patch not in patch_indices:
                patch_indices[key_patch] = None
            elif patch_indices[key_patch] is not None:
                full_size_patch_indices = np.zeros_like(binary_mask)
                full_size_patch_indices[pixels_to_retain] = patch_indices[key_patch]
                patch_indices[key_patch] = np.array(hp.reorder(full_size_patch_indices, r2n=True)[pixels_to_retain_nested], dtype=int)
                

    assert invN_matrix.ndim == 3, "Inverse noise covariance matrix must have shape (n_frequencies, n_stokes, n_pixels)."
    assert invN_matrix.shape[1] == n_stokes, "Inverse noise covariance matrix must contain only Q and U Stokes parameters."
    

    invN_matrix_nested = np.zeros(invN_matrix.shape[:2] + (pixels_to_retain_nested.size,))
    
    for i in range(invN_matrix.shape[0]):
        invN_matrix_nested[i] = hp.reorder(invN_matrix[i], r2n=True)[..., pixels_to_retain_nested]

    if isinstance(sky_map, ArrayLike):
        assert sky_map.ndim == 3, "Sky map must have shape (n_frequencies, n_stokes, n_pixels)."
        assert sky_map.shape[1] == n_stokes, "Sky map must contain only Q and U Stokes parameters."
        sky_map = Stokes.from_stokes(
            q=hp.reorder(sky_map[:, -2], r2n=True)[..., pixels_to_retain_nested], 
            u=hp.reorder(sky_map[:, -1], r2n=True)[..., pixels_to_retain_nested]
        )
    else:
        assert sky_map.q.shape[0] == sky_map.u.shape[0], "Sky map must have the same number of Q and U Stokes parameters."
        assert sky_map.q.shape[1] == sky_map.u.shape[1], "Sky map must have the same number of pixels for Q and U Stokes parameters."
        # Retain only the pixels that are not masked
        sky_map = Stokes.from_stokes(
            q=hp.reorder(sky_map.q, r2n=True)[..., pixels_to_retain_nested], 
            u=hp.reorder(sky_map.u, r2n=True)[..., pixels_to_retain_nested]
        )

    # Prepare the in_structure of the upcoming operators: component maps are (n_stokes, n_pix),
    # frequency maps are (n_stokes, n_freq, n_pix)
    in_structure_sed = sky_map.structure_for((sky_map.shape[1],))
    invN = get_diagonal_operator_from_stokes_maps(invN_matrix_nested, sky_map.structure)

    # Prepare the observation matrix operator
    if obs_mat_operator is None:
        obs_mat_operator = IdentityOperator(in_structure=invN.in_structure)
    
    if obsmat_operator_rhs is None:
        obsmat_operator_rhs = obs_mat_operator.T
    
    if central_freq_op is None:
        central_freq_op = obs_mat_operator.T @ invN @ obs_mat_operator

    # Precompute part of the right-hand side of the CG equation
    print("Precomputing the right-hand side of the CG equation . . .", flush=True)
    ONd = obsmat_operator_rhs(invN(sky_map))
    ONd.q.block_until_ready()
    print("Right-hand side of the CG equation precomputed!", flush=True)

    if matrix_precond is None:
        central_matrix_precond = jnp.copy(invN_matrix_nested)
    elif use_preconditioner_diag:
            
        if matrix_precond.shape[-1] == binary_mask.size:
            print("Using diag obsmat matrices")
            matrix_precond = matrix_precond[..., pixels_to_retain_nested]
        central_matrix_precond = jnp.einsum('fsp,fsp,fsp->fsp', matrix_precond, invN_matrix_nested, matrix_precond)

    elif use_preconditioner_pinv:
        assert matrix_precond.shape[:3] == (frequencies.size, n_stokes, n_stokes), "If we use a preconditioner based on the pseudo-inverse, the central preconditioner matrix must have its three first dimensions as (n_frequencies, n_stokes, n_stokes) with n_stokes=2."

        assert matrix_precond.shape[-1] == matrix_precond.shape[-2], "Central preconditioner matrix must have square submatrices."

        if matrix_precond.shape[-2:] == (binary_mask.size, binary_mask.size):
            central_matrix_precond = matrix_precond[..., pixels_to_retain_nested, pixels_to_retain_nested]
        else:
            central_matrix_precond = matrix_precond
        
        central_operator_precond = get_dense_furax_operator_from_freq_array(matrix_precond)
        
    number_components = 3 # CMB, dust, synchrotron
    n_pix = pixels_to_retain_nested.size

    # @equinox.filter_jit
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
        A = create_MixingMatrixOperator(frequencies, parameters_dict, in_structure_sed, dust_nu0=dust_nu0, synchrotron_nu0=synchrotron_nu0, patch_indices=patch_indices)

        # Full right-hand side of the CG equation
        AOND = A.T(right_member)

        preconditioner = None
        if use_preconditioner_diag:
            print("Using diagonal preconditioner")
            matrix_A = get_array_from_A(A, n_pix)

            # Compute the preconditioner matrix assuming that the observation matrix is the identity and the noise covariance matrix is diagonal in pixel domain
            preconditioner_matrix = jax.lax.stop_gradient(jnp.linalg.pinv(jnp.einsum('fcp,fsp,fkp->psck', matrix_A, central_matrix_precond, matrix_A)).T)

            # Prepare the preconditioner operator
            preconditioner = get_preconditioner(
                preconditioner_matrix, 
                in_structure=in_structure_sed
            )
        if use_preconditioner_pinv:
            print("Using pseudo-inverse preconditioner")
            matrix_A = get_array_from_A(A, n_pix)

            if patch_indices is None:
                print("Assuming no patches for the preconditioner computation")
                matrix_u, matrix_s, matrix_vh = jnp.linalg.svd(matrix_A[...,0].T, full_matrices=False)
            else:
                print("Assuming patches for the preconditioner computation")
                matrix_u, matrix_s, matrix_vh = jnp.linalg.svd(matrix_A.T, full_matrices=False)

            pseudo_inverse_At = jax.lax.stop_gradient(jnp.einsum(
                    '...ba,...b,...cb->...ac', 
                    matrix_vh, 
                    jnp.where(matrix_s!=0, 1./matrix_s, 0.),
                    matrix_u,
                )
            )

            pseudo_inverse_A_op = get_A_from_array(pseudo_inverse_At.T, in_structure_sed)
            preconditioner = pseudo_inverse_A_op.T @ central_operator_precond @ pseudo_inverse_A_op

        diagonal_central_term = (A.T @ central_freq_op @ A).I(
            solver=lx.CG(
                rtol=dictionary_parameters_CG['tol_CG'], 
                atol=dictionary_parameters_CG['tol_CG'], 
                max_steps=dictionary_parameters_CG['max_steps_CG']
            ), 
            preconditioner=preconditioner,
            callback=lambda x: print("Number of iterations in CG:", x.stats['num_steps'], flush=True)
        )
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
            synchrotron_nu0=synchrotron_nu0, 
            patch_indices=patch_indices
        )

        left_hand_term = ONd - (central_freq_op @ A)(map_s)

        keys_params = params.keys()

        final_grad_log = 0
        for key in keys_params:
            final_grad_log += -2*dot_2(A_deriv[key](map_s), left_hand_term) * beta_tau[key]
        return logL, final_grad_log

    if do_minimization:
        print("Launching minimization!!", flush=True)
        output_params, output_state = minimize(
            init_params=first_guess_params, 
            fn=spectral_likelihood_custom_gradient, 
            max_iter=dictionary_parameters_minimization['max_iter'], 
            rtol=dictionary_parameters_minimization['tol'],
            atol=dictionary_parameters_minimization['tol'],
            solver_name=solver_name,
            options=dictionary_parameters_minimization.get('options', None)
        ) # first output is the final parameters, second output is the final state of the optimizer 
        output_params[list(first_guess_params.keys())[0]].block_until_ready()
        print(output_params, flush=True)
        print("Minimization launched!! Preparing the retrieving of the maps . . .", flush=True)
        number_iterations = output_state.iter_num
        print("Finished minimization in {} iterations!".format(number_iterations), flush=True)
    else:
        print("Skipping minimization, using first guess parameters as output.", flush=True)
        output_params = first_guess_params
        number_iterations = 0

    A_maxL, final_maps = get_A_s_AOND(output_params, obsmat_operator_rhs(invN(sky_map)))[:2]

    A_maxL_array = np.asarray(get_array_from_A(A_maxL, n_pix))


    final_maps_nested = np.array([get_maps_from_Stokes(final_maps[key]) for key in ordering_component])

    final_maps_full_sky = np.zeros((number_components, final_maps_nested.shape[-2], binary_mask.shape[-1]), dtype=final_maps_nested.dtype)
    final_maps_full_sky[..., pixels_to_retain_nested] = final_maps_nested


    def W_maxL(input_map):
            
        output_map_truncated = get_A_s_AOND(output_params, obsmat_operator_rhs(invN(input_map[...,pixels_to_retain_nested])))[1]

        output_map_nested = np.array([get_maps_from_Stokes(output_map_truncated[key]) for key in ordering_component])

        output_map = np.zeros((number_components,output_map_nested.shape[-2], input_map.shape[-1]), dtype=output_map_nested.dtype)
        
        output_map[..., pixels_to_retain_nested] = output_map_nested
        for i in range(number_components):
            output_map[i] = hp.reorder(output_map[i], n2r=True)
        return output_map

    def W_params(params, input_map):

        output_map_truncated = get_A_s_AOND(params, obsmat_operator_rhs(invN(input_map[...,pixels_to_retain_nested])))[1]

        output_map_nested = np.array([get_maps_from_Stokes(output_map_truncated[key]) for key in ordering_component])

        output_map = np.zeros((number_components,output_map_nested.shape[-2], input_map.shape[-1]), dtype=output_map_nested.dtype)
        
        output_map[..., pixels_to_retain_nested] = output_map_nested
        for i in range(number_components):
            output_map[i] = hp.reorder(output_map[i], n2r=True)
        return output_map


    # clear JAX backend caches to free compiled buffers if necessary
    jax.clear_caches()

    return Results.from_compsep_results(
        list(output_params.keys()),
        output_params, 
        W_maxL, 
        A_maxL_array,
        final_maps_full_sky, 
        number_iterations, 
        dictionary_parameters_minimization['max_iter'], 
        ordering_parameter=ordering_parameter, 
        ordering_component=ordering_component,
        W_params=W_params
    )
