import warnings
import numpy as np
import jax
import jax.numpy as jnp
import equinox
import lineax as lx
import healpy as hp
import emcee
from jaxtyping import ArrayLike
from jax.flatten_util import ravel_pytree #Copilot Hessien estimation
from typing import Callable

import operator

from furax.core import IdentityOperator, DiagonalOperator
from furax.tree import as_structure
from furax.obs.stokes import Stokes
from furax.obs.operators._qu_rotations import QURotationOperator

# from furax import Config

from furax_cs import minimize, SOLVER_NAMES
from megatop.utils import logger
from megabuster.minimizers import filter_optimize, minimize_likelihood
from megabuster.mixingmatrix import create_MixingMatrixOperator, create_MixingMatrixOperator_deriv
from megabuster.calibrationmatrix import build_calibration_operator, StokesPyTree_to_list, list_to_stokes
from megabuster.tools import (
    get_diagonal_operator_from_stokes_maps, 
    get_preconditioner, 
    get_A_from_array, 
    get_maps_from_Stokes, 
    get_dense_furax_operator_from_freq_array,
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
        self.angles_uncertainties = None

    @classmethod
    def from__results(
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
        W_params=None,
        angles_uncertanties = None
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
            angles_uncertanties (list): list containing the uncertainties of the angles.
        
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
                Q=hp.reorder(input_maps[:,-2,:],r2n=True), 
                U=hp.reorder(input_maps[:,-1,:],r2n=True)
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
                    Q=hp.reorder(input_maps[:,-2,:],r2n=True), 
                    U=hp.reorder(input_maps[:,-1,:],r2n=True)
                )
                return W_params(params, input_maps_stokes)
        else:
            final_W_params = None

        res.W_maxL = final_W
        res.A_maxL = final_A_maxL
        res.s = np.array([hp.reorder(final_maps[component], n2r=True) for component in range(final_maps.shape[0])])
        res.success = number_iterations < max_iter
        res.message = f"Optimization finished after {number_iterations} iterations out of {max_iter} allowed."
        res.W_params = final_W_params
        res.angles_uncertainties = angles_uncertanties
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
    config,
    manager,
    first_guess_params, 
    sky_map, 
    frequencies, 
    invN_matrix,
    do_minimization=False,
    fixed_params=dict(),
    binary_mask=None,
    dust_nu0=150.0, 
    synchrotron_nu0=20.0,
    central_freq_op=None,
    use_calibration_matrix=True,
    angles_prior_dict=None,
    angles_names=None,
    obsmat_operator=None, 
    operator_rhs=None,
    use_preconditioner_diag=False,
    use_preconditioner_pinv=False,
    matrix_precond=None,
    solver_name="optax_lbfgs",
    dictionary_parameters_minimization: dict={'max_iter':1, 'tol':1e-5},
    dictionary_parameters_CG: dict={'max_steps_CG':400, 'tol_CG':1e-6},
    ordering_parameter=['beta_dust', 'beta_pl'], 
    ordering_component=['cmb', 'dust', 'synchrotron'],
    patch_indices=None
    ):
    """
    Perform component separation using the given parameters and data.

    Parameters
    ----------
    config : Congif
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
        If None, it is taken to be the composition of the operator provided in obs_mat_operator (which is Identity if None was provided for obs_mat_operator) with invN and its transpose.
    use_calibration_matrix: bool (optional)
    angles_prior_dict: Dictionary wich take angle prior uncertainty and their central value, respectively in keys "angle_uncertainty" and "angle_central_value" (optional)
    angles_names: If angles_prior_dict is no None, provide a list of str val with the angle names angle_i     
    obs_mat_operator: AbstractLinearOperator (optional)
        Observation matrix operator, which must be a function compilable and jittable with JAX, ideally a FURAX operator, and applicable directly on nested masked Stokes objects.
        Default is None and is taken to be the identity operator. Not used if central_freq_op is provided. 
    operator_rhs: AbstractLinearOperator (optional)
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

    # Safe defaults to avoid NameError when angle-related options are not provided
    if angles_names is None:
        angles_names = []
    # priors_dict / first_angles_list / prior_list are populated only if angles_prior_dict is set
    priors_dict = {}
    first_angles_list = []
    prior_list = []

    # Few tests
    assert isinstance(first_guess_params, dict), "First guess parameters must be stored as a dictionary."
    assert 'temp_dust' in first_guess_params or 'temp_dust' in fixed_params, "First guess parameters must contain 'temp_dust'."
    assert 'beta_dust' in first_guess_params or 'beta_dust' in fixed_params, "First guess parameters must contain 'beta_dust'."
    assert 'beta_pl' in first_guess_params or 'beta_pl' in fixed_params, "First guess parameters must contain 'beta_pl'."
    assert isinstance(sky_map, (Stokes, ArrayLike)), "Sky map must be a Stokes object or a ndarray."
    #assert isinstance(sky_map, (Stokes, np.ndarray, jnp.ndarray)), "Sky map must be a Stokes object or a ndarray." # Propal Copilot

    assert (use_preconditioner_diag != use_preconditioner_pinv) or (use_preconditioner_pinv == False), "Only one type of preconditioner (use_preconditioner_diag or use_preconditioner_pinv) can be used at a time."
    assert (use_preconditioner_pinv == (matrix_precond is not None)), "If the pseudo-inverse preconditioner (use_preconditioner_pinv) is set to True, then matrix_precond must be provided."
    
    if dictionary_parameters_minimization['tol'] < dictionary_parameters_CG['tol_CG'] and do_minimization:
        warnings.warn("The tolerance for the minimization is smaller than the tolerance for the conjugate gradient solver. This might lead to suboptimal results.")

    assert solver_name in SOLVER_NAMES.__args__, f"Solver name must be one of {SOLVER_NAMES.__args__}."

    if binary_mask is not None:
        assert isinstance(binary_mask, ArrayLike), "Binary mask must be a ndarray."
        assert binary_mask.ndim == 1, "Binary mask must be a 1D array."
        pixels_to_retain = np.where(binary_mask != 0)[0]
        pixels_to_retain_nested = np.where(hp.reorder(binary_mask, r2n=True) != 0)[0]
    else:
        #pixels_to_retain = np.ones_like(invN_matrix.shape[-1], dtype=bool)
        pixels_to_retain = np.ones(invN_matrix.shape[-1], dtype=bool) #Si erreur np>ones_like doit prendre une array, or invN_matrix.shape[-1] est une entier ?
        pixels_to_retain_nested = np.ones_like(pixels_to_retain, dtype=bool)
        #pixels_to_retain_nested = np.ones(pixels_to_retain, dtype=bool)
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
                #patch_indices[key_patch] = np.array(hp.reorder(full_size_patch_indices, r2n=True)[pixels_to_retain_nested], dtype=int)
                

    assert invN_matrix.ndim == 3, "Inverse noise covariance matrix must have shape (n_frequencies, n_stokes, n_pixels)."
    assert invN_matrix.shape[1] == n_stokes, "Inverse noise covariance matrix must contain only Q and U Stokes parameters."
    

    invN_matrix_nested = np.zeros(invN_matrix.shape[:2] + (pixels_to_retain_nested.size,))
    
    for i in range(invN_matrix.shape[0]):
        invN_matrix_nested[i] = hp.reorder(invN_matrix[i], r2n=True)[..., pixels_to_retain_nested]

    if isinstance(sky_map, ArrayLike):
        assert sky_map.ndim == 3, "Sky map must have shape (n_frequencies, n_stokes, n_pixels)."
        assert sky_map.shape[1] == n_stokes, "Sky map must contain only Q and U Stokes parameters."
        sky_map = Stokes.from_stokes(
            Q=hp.reorder(sky_map[:, -2], r2n=True)[..., pixels_to_retain_nested], 
            U=hp.reorder(sky_map[:, -1], r2n=True)[..., pixels_to_retain_nested]
        )
        #shape = sky_map.shape[-2:]
    else:
        assert sky_map.q.shape[0] == sky_map.u.shape[0], "Sky map must have the same number of Q and U Stokes parameters."
        assert sky_map.q.shape[1] == sky_map.u.shape[1], "Sky map must have the same number of pixels for Q and U Stokes parameters."
        # Retain only the pixels that are not masked
        sky_map = Stokes.from_stokes(
            Q=hp.reorder(sky_map.q, r2n=True)[..., pixels_to_retain_nested], 
            U=hp.reorder(sky_map.u, r2n=True)[..., pixels_to_retain_nested]
        )

    # Prepare the in_structure of the upcoming operators
    in_structure_sed = sky_map.structure_for((sky_map.shape[1],))
    in_structure_calibration = sky_map.structure_for((sky_map.shape))
    in_structure_noise_cov = sky_map.structure.q
    vect_shape = sky_map.q.shape
    print('vect_shape ! : ', vect_shape)

    invN = get_diagonal_operator_from_stokes_maps(invN_matrix_nested, in_structure_noise_cov)

    # Prepare the observation matrix operator

    #Calibration matrix operator
    if use_calibration_matrix:
        print('Using Calibration_matrix ...')
    else:
        if obsmat_operator is None:
            obsmat_operator = IdentityOperator(invN.in_structure())
    
        if operator_rhs is None:
            operator_rhs = obsmat_operator.T
    
        if central_freq_op is None:
            central_freq_op = obsmat_operator.T @ invN @ obsmat_operator
        # Precompute part of the right-hand side of the CG equation
        print("Precomputing the right-hand side of the CG equation . . .", flush=True)
        ONd = operator_rhs(invN(sky_map))
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

    print(type(invN(sky_map)))

    if angles_prior_dict is not None:
        # map the incoming key "central value" to the expected "central_value" parameter
        angles_dict = {}
        priors_dict = {}
        angles = angles_prior_dict.pop('angle_central_value')
        first_angles_list = []
        for i, angle in enumerate(angles):
            angles_dict[f'angle_{i}'] = angle
            first_angles_list.append(angle)
        angles_prior = angles_prior_dict.pop('angle_uncertainty')
        prior_list = []
        for i, prior in enumerate(angles_prior):
            priors_dict[f'angle_{i}'] = prior
            prior_list.append(prior)
        first_guess_params = {**first_guess_params,**angles_dict}
        angles_names = [f'angle_{i}' for i in range(len(angles))]
    else:
        first_guess_params = {**first_guess_params}

    c_right_member = invN(sky_map)

    # @equinox.filter_jit
    def get_A_s_AOND(params, invN = invN):
        """
            Compute the log-proba given a set of parameters 
        """

        # Select parameters which are to be estimated 
        parameters_dict = dict()
        #print('AOND params ! :', params)
        for param_name in ['temp_dust', 'beta_dust', 'beta_pl']+angles_names:
            if param_name in params:
                parameters_dict[param_name] = params[param_name]
            else:
                parameters_dict[param_name] = fixed_params[param_name]

        #parameters_dict = params

        if angles_prior_dict is not None:
            angles_list = [i for x, i in params.items() if x.startswith('angle_')]
            angles_jnp = jnp.array(angles_list)[:, None]
            C = build_calibration_operator(angles_jnp, vect_shape)
            operator_rhs = C.T
            right_member = operator_rhs(c_right_member) # Op ?
            c_central_freq_op = invN @ C
            central_freq_op = C.T @ invN @ C
        else:
            right_member = invN(sky_map)


        # Mixing matrix operator
        A = create_MixingMatrixOperator(frequencies, parameters_dict, in_structure_sed, dust_nu0=dust_nu0, synchrotron_nu0=synchrotron_nu0, patch_indices=patch_indices)

        # Full right-hand side of the CG equation
        AOND = A.T(operator_rhs(c_right_member))

        if not use_preconditioner_diag and not use_preconditioner_pinv:
            # Build the full operator (used downstream) but also build a
            # cheap diagonal preconditioner approximating diag(Op).
            Op = A.T @ C.T @ c_central_freq_op @ A
            preconditioner = IdentityOperator(Op.in_structure())

        diagonal_central_term = (A.T @ C.T @ c_central_freq_op @ A).I(
            solver=lx.CG(
                rtol=dictionary_parameters_CG['tol_CG'],
                atol=dictionary_parameters_CG['tol_CG'],
                max_steps=dictionary_parameters_CG['max_steps_CG']
            ),
            preconditioner=preconditioner,
        )

        first_central_term = diagonal_central_term(AOND)

        return A, first_central_term, AOND, c_central_freq_op, central_freq_op, right_member, diagonal_central_term

    @equinox.filter_custom_jvp
    def spectral_likelihood_custom_gradient(params):
        """
            Compute the log-proba given a set of parameters 
        """

        # Select parameters which are to be estimated 
        parameters_dict = dict()
        #print('Spectral_likelihood params ! :', params)
        for param_name in ['temp_dust', 'beta_dust', 'beta_pl']+angles_names:
            if param_name in params:
                parameters_dict[param_name] = params[param_name]
            else:
                parameters_dict[param_name] = fixed_params[param_name]
        #parameters_dict = params

        _, map_s, AOND, _, _, _, _ = get_A_s_AOND(parameters_dict)

        if angles_prior_dict is None:
            logL = -dot_2(AOND, map_s)
        else:
            angles_list = [i for x, i in params.items() if x.startswith('angle_')]
            angles_jnp = jnp.array(angles_list)
            first_angles_jnp =jnp.array(first_angles_list)
            prior_jnp = jnp.array(prior_list) 
            angles_diff = angles_jnp-first_angles_jnp
            term2 = jnp.sum((angles_diff**2)/(prior_jnp**2))
            logL = -dot_2(AOND, map_s) + term2

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
        #print('grad_likelihood params ! :', params)
        for param_name in ['temp_dust', 'beta_dust', 'beta_pl']+angles_names:
            if param_name in params:
                parameters_dict[param_name] = params[param_name]
            else:
                parameters_dict[param_name] = fixed_params[param_name]

        #parameters_dict = params

        #A, map_s, AOND, _ = get_A_s_AOND(parameters_dict)

        A, map_s, AOND, c_central_freq_op, central_freq_op, right_member, _ = get_A_s_AOND(parameters_dict)

        #logL = -dot_2(AOND, map_s)

        if angles_prior_dict is None:
            logL = -dot_2(AOND, map_s)
        else:
            angles_list = [i for x, i in params.items() if x.startswith('angle_')]
            angles_jnp = jnp.array(angles_list)
            first_angles_jnp =jnp.array(first_angles_list)
            prior_jnp = jnp.array(prior_list) 
            angles_diff = angles_jnp-first_angles_jnp
            term2 = jnp.sum((angles_diff**2)/(prior_jnp**2))
            logL = -dot_2(AOND, map_s) + term2

        A_deriv = create_MixingMatrixOperator_deriv(
            frequencies, 
            parameters_dict, 
            in_structure_sed, 
            dust_nu0=dust_nu0, 
            synchrotron_nu0=synchrotron_nu0, 
            patch_indices=patch_indices
        )

        c_left_hand_term = c_right_member - (c_central_freq_op @ A)(map_s)
        left_hand_term = C.T(c_left_hand_term)

        #left_hand_term = right_member - (central_freq_op @ A)(map_s)

        keys_params = params.keys()

        final_grad_log = 0
        Ax = A(map_s)
        i = 0

        for key in keys_params:
            if key.startswith("angle"):  # si le paramètre est un angle de calibration
                # On récupère l'index du bloc correspondant à cet angle
                # tu peux construire ce dict angle->index avant
                # Rotation infinitésimale via l'astuce +pi/4
                #X_shift = QURotationOperator.create(
                #    shape=Ax.q[i,:].shape,
                #    stokes="QU",
                #    angles=angles_jnp[i] + jnp.pi / 4
                #)
                Ax_idx = Ax[i,:]
                J_Ax = Stokes.from_stokes(Q = -Ax_idx.u, U = Ax_idx.q)
                X_op = QURotationOperator.create(shape=Ax_idx.q.shape, stokes="QU", angles=angles_jnp[i])
                dXAx = 2 * X_op(J_Ax)
                grad_cal = - 2*dot_2(dXAx, c_left_hand_term[i, :])
                # Construire le vecteur dXAx = (∂_{α_i} X) As
                #dXAx = dXAx.at[i].set(2.0 * X_shift(Ax[i]))
                #grad_cal = - * dot_2(X_shift(Ax[i,:]), left_hand_term[i,:])
                #grad_cal = - dot_2(X_shift(Ax[i,:]), left_hand_term[i,:])
                if priors_dict[key] != 0:
                    grad_prior = 2 * angles_diff[i] / priors_dict[key]**2
                else:
                    grad_prior = 0
                final_grad_log += (grad_cal + grad_prior)* beta_tau[key]
                i += 1
            else:
                final_grad_log += -2*dot_2(A_deriv[key](map_s), left_hand_term) * beta_tau[key]
        return logL, final_grad_log
    
    def grid2d(params):
            epsilon = jnp.array([-0.00004,0.00003,0.00002,0.00001,0,0.00001,0.00002,0.00003,0.00004,0.00004])
            n = len(epsilon)
            grid = jnp.zeros(n, 2)
            a0 = params['angle_0']
            a1 = params['angle_1']
            beta_tau = {key: 0.9985 for key in params_min}
            for i in range(n):
                print('i :', i)
                for j in range(n):
                    if i ==0:
                        key = 'angle_0'
                        params[key] = a0 + epsilon[j]
                    else:
                        key = 'angle_1'
                        params[key] = a1 + epsilon[j]
                    print('key, param :',key, params[key])
                    print('likelihood :', spectral_likelihood_custom_gradient(params))
                    logL, grad = custom_gradient((params_min,), (beta_tau,))
                    print('LogL, grad :', custom_gradient(params,beta_tau))
            return None

    theta_init = list(first_guess_params.values())
    dust_temp = [20.0]
    name_params = list(first_guess_params.keys())

    def angles_hessian_and_uncertainties_from_params(params, param_names=None, ridge_rel=1e-6, eps_rel=1e-7):
        flat_params, unravel = jax.flatten_util.ravel_pytree(params)
        flat_params = jnp.array(flat_params)
        n = len(flat_params)
        # Noms des paramètres pour le diagnostic
        if param_names is None:
            param_names = []
            for k, v in params.items():
                v = jnp.atleast_1d(v)
                if v.size == 1:
                    param_names.append(k)
                else:
                    param_names += [f"{k}[{i}]" for i in range(v.size)]
        
        eps_vec = jnp.sqrt(jnp.finfo(float).eps) * jnp.maximum(1.0, jnp.abs(flat_params))
        eps_vec = jnp.maximum(eps_vec, 1e-8)
        
        def neg_logpost_flat(flat): 
            return 2*spectral_likelihood_custom_gradient(unravel(flat))
        
        # Hessienne par différences finies
        H = jnp.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                ei = jnp.zeros(n).at[i].set(eps_vec[i])
                ej = jnp.zeros(n).at[j].set(eps_vec[j])
                fpp = neg_logpost_flat(flat_params + ei + ej)
                fpm = neg_logpost_flat(flat_params + ei - ej)
                fmp = neg_logpost_flat(flat_params - ei + ej)
                fmm = neg_logpost_flat(flat_params - ei - ej)
                d2f = (fpp - fpm - fmp + fmm) / (4 * eps_vec[i] * eps_vec[j])
                H = H.at[i, j].set(d2f)
                H = H.at[j, i].set(d2f)
                
            H = 0.5 * (H + H.T)

        # --- DIAGNOSTIC COMPLET ---
        print("\n=== DIAGNOSTIC HESSIENNE ===")
        print(f"{'Paramètre':<15} {'H[i,i]':>15} {'eps':>12}")
        for i, name in enumerate(param_names):
            print(f"{name:<15} {float(H[i,i]):>15.4e} {float(eps_vec[i]):>12.4e}")
            eigvals, eigvecs = jnp.linalg.eigh(H)
            print(f"\nValeurs propres : {eigvals}")
            print(f"Condition number : {float(jnp.linalg.cond(H)):.2e}")
            n_neg = int(jnp.sum(eigvals < 0))
            if n_neg > 0:
                print(f"\n⚠️ {n_neg} valeurs propres négatives :")
                for k in range(n):
                    if eigvals[k] < 0:
                        dominant = param_names[int(jnp.argmax(jnp.abs(eigvecs[:, k])))]
                        print(f" λ={float(eigvals[k]):.4e} — direction dominante : {dominant}")
                        print("→ La Hessienne n'est pas définie positive : le minimum n'est pas bien convergé")
                        print(" ou la likelihood est non-convexe dans ces directions.")
                        print(" Les incertitudes retournées sont NaN pour ces directions.\n")
        
        ridge = max(1e-12, float(ridge_rel) * float(jnp.mean(jnp.abs(jnp.diag(H)))))
        H_reg = H + ridge * jnp.eye(n)
        eigvals_reg, eigvecs_reg = jnp.linalg.eigh(H_reg) # Inverser uniquement les valeurs propres positives, NaN pour les négatives
        inv_eigvals = jnp.where(eigvals_reg > 0, 1.0 / eigvals_reg, jnp.nan)
        cov = eigvecs_reg @ jnp.diag(inv_eigvals) @ eigvecs_reg.T
        uncert = jnp.sqrt(jnp.diag(cov)) # NaN si direction non définie positive

        print("=== INCERTITUDES ===")
        for name, u in zip(param_names, uncert):
            if jnp.isnan(u):
                print(f" {name:<15} : NaN (direction non contrainte)")
            else: print(f" {name:<15} : {float(u):.4e}")
            
        return H, [float(x) for x in uncert]


    if do_minimization:
        print("Launching minimization!!", flush=True)

        fgp = {k: v * 0.9985 for k, v in first_guess_params.items()}
        #fgp = {k: v * 0.9 for k, v in first_guess_params.items()}

        print(fgp)

        output_params, output_state = minimize(
            init_params=fgp, 
            fn=spectral_likelihood_custom_gradient, 
            max_iter=dictionary_parameters_minimization['max_iter'], 
            rtol=dictionary_parameters_minimization['tol'],
            atol=dictionary_parameters_minimization['tol'],
            solver_name=solver_name
        ) # first output is the final parameters, second output is the final state of the optimizer 
        output_params[list(first_guess_params.keys())[0]].block_until_ready()

        print(output_params, flush=True)

        ref, gref = jax.value_and_grad(spectral_likelihood_custom_gradient)(output_params)
        grad_norm_ref = 0
        for key in gref:
            grad_norm_ref += gref[key]**2
        epsilon = 1.0e-12
        for k in output_params:
            print(k)
            output_params[k] += epsilon
            for i in range(3):
                print('param k :', output_params[k])
                print('params ! :', output_params)
                l, grad = jax.value_and_grad(spectral_likelihood_custom_gradient)(output_params)
                grad_norm = 0
                for key in grad:
                    grad_norm += grad[key]**2
                print('l value ?? :', l)
                #print('grad :', grad)
                print('grad_norm :', jnp.sqrt(grad_norm))
                #print('dgref :', jnp.sqrt(grad_norm) - gref_norm_ref)
                dref = l - ref
                print('likelihood dref :', dref)
                print(i)
                if i == 2:
                    output_params[k] += epsilon
                    continue
                output_params[k] += -epsilon

        number_iterations = output_state.iter_num
        #print('output_state 2 :',output_state.solver_state.state.opt_state[2].info)
        print(number_iterations)
        if priors_dict is not None:
            H, angles_uncertanties = angles_hessian_and_uncertainties_from_params(output_params)
            print("Diagonale H :", jnp.diag(H))
            print("Valeurs propres H :", jnp.linalg.eigvalsh(H))
            print("Condition number :", jnp.linalg.cond(H))
            print('angles_uncertanties ! :', angles_uncertanties)
            print('H :', H)
            a = 180/np.pi # conversion radians to degrees
            for key in name_params[2:]:
                output_params[key] = output_params[key] * a
        print(output_params, flush=True)
        #print("Minimization launched!! Preparing the retrieving of the maps . . .", flush=True)
        number_iterations = output_state.iter_num
    else:
        print("Skipping minimization, using first guess parameters as output.", flush=True)
        #output_params = first_guess_params
        #number_iterations = 0

        #for param_name in ['temp_dust', 'beta_dust', 'beta_pl']+angles_names:
        #    if param_name in params:
        #        parameters_dict[param_name] = params[param_name]
        #    else:
        #        parameters_dict[param_name] = fixed_params[param_name]

        # 1. init mcmc parameters:
        #theta_init_guess = dust_temp+theta_init
            
    A_maxL, final_maps = get_A_s_AOND(output_params)[:2]

    A_maxL_array = np.zeros((frequencies.size, number_components, n_pix))
    for component in range(number_components):
        A_maxL_array[:,component] = A_maxL.block_leaves[component]._diagonal


    final_maps_nested = np.array([get_maps_from_Stokes(final_maps[key]) for key in ordering_component])

    final_maps_full_sky = np.zeros((number_components, final_maps_nested.shape[-2], binary_mask.shape[-1]), dtype=final_maps_nested.dtype)
    final_maps_full_sky[..., pixels_to_retain_nested] = final_maps_nested


    def W_maxL(input_map):
            
        output_map_truncated = get_A_s_AOND(output_params)[1]

        output_map_nested = np.array([get_maps_from_Stokes(output_map_truncated[key]) for key in ordering_component])

        output_map = np.zeros((number_components,output_map_nested.shape[-2], input_map.shape[-1]), dtype=output_map_nested.dtype)
        
        output_map[..., pixels_to_retain_nested] = output_map_nested
        for i in range(number_components):
            output_map[i] = hp.reorder(output_map[i], n2r=True)
        return output_map

    def W_params(params, input_map):

        output_map_truncated = get_A_s_AOND(params)[1]

        output_map_nested = np.array([get_maps_from_Stokes(output_map_truncated[key]) for key in ordering_component])

        output_map = np.zeros((number_components,output_map_nested.shape[-2], input_map.shape[-1]), dtype=output_map_nested.dtype)
        
        output_map[..., pixels_to_retain_nested] = output_map_nested
        for i in range(number_components):
            output_map[i] = hp.reorder(output_map[i], n2r=True)
        return output_map

    return Results.from__results(
        name_params,
        output_params, 
        W_maxL, 
        A_maxL_array,
        final_maps_full_sky, 
        number_iterations, 
        dictionary_parameters_minimization['max_iter'], 
        ordering_parameter=ordering_parameter, 
        ordering_component=ordering_component,
        W_params=W_params,
        angles_uncertanties = None
    )
