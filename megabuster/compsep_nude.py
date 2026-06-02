import warnings
import shutil
import numpy as np
import jax
jax.config.update("jax_traceback_filtering", "off")
import jax.numpy as jnp
import equinox
import lineax as lx
import healpy as hp
import jax.scipy.stats as stats
import blackjax
import matplotlib.pyplot as plt
import corner
import arviz as az

from getdist import MCSamples, plots
from jaxtyping import ArrayLike
from typing import Callable

import operator

from furax.core import IdentityOperator
from furax.tree import as_structure
from furax.obs.stokes import Stokes
from furax.obs.operators._qu_rotations import QURotationOperator
from furax.obs.operators._beam_operator import BeamOperator

from furax_cs import minimize, SOLVER_NAMES

from megabuster.mixingmatrix import create_MixingMatrixOperator, create_MixingMatrixOperator_deriv
from megabuster.calibrationmatrix import build_calibration_operator, StokesPyTree_to_list, list_to_stokes
from megabuster.tools import (
    get_diagonal_operator_from_stokes_maps, 
    get_preconditioner,
    get_preconditioner2,
    get_A_from_array, 
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
    Cov: ArrayLike = None # Covariance matrix of the parameters fitted, if available
    params_names: list = None # Params_names
    #mean_cov: list = None # Mean values of estimated parameters
    mc_samples = None # Results samples from HMC
    mean_deg = None # Mean values of estimated parameters in degrees if angles were estimated
    def __init__(self):
        self.params = None
        self.x = None
        self.s = None
        self.W_maxL = None
        self.A_maxL = None
        self.success = False
        self.message = "No optimization performed yet."
        self.W_params = None
        self.Cov = None
        self.params_names = None
        self.mc_samples = None
        self.mean_deg = None

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
        W_params=None,
        Cov=None,
        params_names = None,
        mc_samples = None,
        mean_deg = None
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
        res.x = np.array([final_params[key] for key in name_params])

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
        res.Cov = Cov
        res.params_names = params_names
        res.mc_samples = mc_samples
        res.mean_deg = mean_deg
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
    use_obsmat=False,
    angles_prior_dict=None,
    angles_names=None,
    obs_mat_operator=None, 
    obsmat_operator_rhs=None,
    operator_rhs=None,
    use_preconditioner_diag=False,
    use_preconditioner_pinv=False,
    matrix_precond=None,
    solver_name="ADABK0",
    dictionary_parameters_minimization: dict={'max_iter':1, 'tol':1e-5},
    dictionary_parameters_CG: dict={'max_steps_CG':400, 'tol_CG':1e-6},
    ordering_parameter=['beta_dust', 'beta_pl'], 
    ordering_component=['cmb', 'dust', 'synchrotron'],
    patch_indices=None,
    use_hessienne = False,
    use_hmc = False,
    n_warm = 300,
    n_samples = 10000,
    is_noise_map = False,
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

    #assert solver_name in SOLVER_NAMES.__args__, f"Solver name must be one of {SOLVER_NAMES.__args__}."

    if binary_mask is not None:
        assert isinstance(binary_mask, ArrayLike), "Binary mask must be a ndarray."
        assert binary_mask.ndim == 1, "Binary mask must be a 1D array."
        pixels_mask = np.where(binary_mask != 0)[0]
        npix_full = invN_matrix.shape[-1]
        pixels_to_retain = np.arange(npix_full)
        pixels_to_retain_nested = np.where(hp.reorder(binary_mask, r2n=True) != 0)[0]
    else:
        pixels_to_retain = np.ones_like(invN_matrix.shape[-1], dtype=bool)
        pixels_to_retain_nested = np.ones_like(pixels_to_retain, dtype=bool)
        binary_mask = np.ones(invN_matrix.shape[-1])             

    assert invN_matrix.ndim == 3, "Inverse noise covariance matrix must have shape (n_frequencies, n_stokes, n_pixels)."
    assert invN_matrix.shape[1] == n_stokes, "Inverse noise covariance matrix must contain only Q and U Stokes parameters."
    
    invN_matrix_nested = np.zeros(invN_matrix.shape[:2] + (len(pixels_to_retain),))

    for i in range(invN_matrix.shape[0]):
        invN_matrix_nested[i] = invN_matrix[i][..., pixels_to_retain]

    mask_2d = np.zeros(npix_full, dtype=bool)
    mask_2d[pixels_to_retain] = True

    if isinstance(sky_map, ArrayLike):
        assert sky_map.ndim == 3, "Sky map must have shape (n_frequencies, n_stokes, n_pixels)."
        assert sky_map.shape[1] == n_stokes, "Sky map must contain only Q and U Stokes parameters."
        sky_map = Stokes.from_stokes(
            Q=sky_map[:, -2][..., pixels_to_retain], 
            U=sky_map[:, -1][..., pixels_to_retain]
        )
    else:
        assert sky_map.q.shape[0] == sky_map.u.shape[0], "Sky map must have the same number of Q and U Stokes parameters."
        assert sky_map.q.shape[1] == sky_map.u.shape[1], "Sky map must have the same number of pixels for Q and U Stokes parameters."
        sky_map = Stokes.from_stokes(
            Q=sky_map.q[..., pixels_to_retain], 
            U=sky_map.u[..., pixels_to_retain]
        )

    # Prepare the in_structure of the upcoming operators
    in_structure_sed = sky_map.structure_for((sky_map.shape[1],))
    in_structure_noise_cov = sky_map.structure.q
    vect_shape = sky_map.q.shape

    invN = get_diagonal_operator_from_stokes_maps(invN_matrix_nested, in_structure_noise_cov)
    

    # Create miscalibration dict
    angles_dict = {}
    priors_dict = {}
    first_angles_list = []
    prior_list = []

    # Recover angles central values and priors from the config
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

    # Create initial parameters positions dict
    first_guess_params = {**first_guess_params,**angles_dict}
    angles_names = [f'angle_{i}' for i in range(len(angles))]
        
    number_components = 3 # CMB, dust, synchrotron
    n_pix = pixels_to_retain_nested.size

     # LOAD BEAM OPERATOR

    beams = config.beams
    beam_fl = jnp.array([
        hp.gauss_beam(fwhm=np.radians(fwhm / 60), lmax=2 * config.nside + config.map2cl_pars.delta_ell)
        for fwhm in beams
    ])  # shape (6, lmax+1)

    B = BeamOperator(lmax=2 * config.nside + config.map2cl_pars.delta_ell, beam_fl=beam_fl, in_structure=invN.in_structure)
    
    # @equinox.filter_jit
    def get_A_s_AOND(params):
        """
            Compute the log-proba given a set of parameters 
        """

        # Select parameters which are to be estimated 
        parameters_dict = dict()
        for param_name in ['temp_dust', 'beta_dust', 'beta_pl']+angles_names:
            if param_name in params:
                parameters_dict[param_name] = params[param_name]
            else:
                parameters_dict[param_name] = fixed_params[param_name]

        c_right_member = invN(sky_map)
        right_member = c_right_member # Op ?
        c_central_freq_op = invN
        central_freq_op = invN

        # Mixing matrix operator
        A = create_MixingMatrixOperator(frequencies, parameters_dict, in_structure_sed, dust_nu0=dust_nu0, synchrotron_nu0=synchrotron_nu0, patch_indices=patch_indices)

        # Full right-hand side of the CG equation
        AOND = A.T(right_member)

        ## Calculate preconditioners :

        Op = A.T @ central_freq_op @ A
        preconditioner = IdentityOperator(in_structure=Op.in_structure)
        diagonal_central_term = (A.T @ central_freq_op @ A).I(
            solver=lx.CG(
                rtol=dictionary_parameters_CG['tol_CG'], 
                atol=dictionary_parameters_CG['tol_CG'], 
                max_steps=dictionary_parameters_CG['max_steps_CG']
            ), 
            preconditioner=preconditioner,
            callback=lambda x: print("Number of iterations in CG:", x.stats['num_steps'], flush=True)
        )
        first_central_term = diagonal_central_term(AOND)

        return A, first_central_term, AOND, c_central_freq_op, central_freq_op, c_right_member, right_member, diagonal_central_term

    @equinox.filter_custom_jvp
    def spectral_likelihood_custom_gradient(params):
        """
            Compute the log-proba given a set of parameters 
        """

        # Select parameters which are to be estimated 
        parameters_dict = dict()
        for param_name in ['temp_dust', 'beta_dust', 'beta_pl']+angles_names:
            if param_name in params:
                parameters_dict[param_name] = params[param_name]
            else:
                parameters_dict[param_name] = fixed_params[param_name]

        _, map_s, AOND, _, _, _, _, _ = get_A_s_AOND(parameters_dict)

        if use_calibration_matrix:
            n_angles = len([k for k in parameters_dict.keys() if k.startswith('angle_')])
            angles_jnp = jnp.array([parameters_dict[f'angle_{i}'] for i in range(n_angles)])
            first_angles_jnp = jnp.array([first_angles_list[i] for i in range(n_angles)])
            prior_jnp = jnp.array([prior_list[i] if prior_list[i] is not None else float('inf') for i in range(n_angles)])
            angles_diff = angles_jnp - first_angles_jnp
            term2 = jnp.sum((angles_diff**2)/(prior_jnp**2))
            logL = -dot_2(AOND, map_s) + term2
        else:
            logL = -dot_2(AOND, map_s)

        return logL

    @spectral_likelihood_custom_gradient.def_jvp
    def custom_gradient(primals, tangents):
        """
        Custom JVP rule for the perturbative_negative_log_prob function.
        """
        params, = primals
        beta_tau, = tangents

        jax.debug.print("params = {}", params)

        # Select parameters which are to be estimated 
        parameters_dict = dict()
        for param_name in ['temp_dust', 'beta_dust', 'beta_pl']+angles_names:
            if param_name in params:
                parameters_dict[param_name] = params[param_name]
            else:
                parameters_dict[param_name] = fixed_params[param_name]

        A, map_s, AOND, c_central_freq_op, central_freq_op, c_right_member, right_member, _ = get_A_s_AOND(parameters_dict)

        logL = -dot_2(AOND, map_s)

        A_deriv = create_MixingMatrixOperator_deriv(
            frequencies, 
            parameters_dict, 
            in_structure_sed, 
            dust_nu0=dust_nu0, 
            synchrotron_nu0=synchrotron_nu0, 
            patch_indices=patch_indices
        )

        keys_params = params.keys()
        final_grad_log = 0

        ONd = invN(sky_map)
        left_hand_term = ONd - (central_freq_op @ A)(map_s)
        for key in keys_params:
            if key.startswith("angle"):
                final_grad_log += 0
            else:
                final_grad_log += -2*dot_2(A_deriv[key](map_s), left_hand_term) * beta_tau[key]

        return logL, final_grad_log
    
    if do_minimization:

        print("Launching minimization!!", flush=True)
        output_params, output_state = minimize(
        spectral_likelihood_custom_gradient,   # 1er argument positionnel, plus fn=
        init_params=first_guess_params,
        solver_name=solver_name,
        max_iter=config.parametric_sep_pars.megabuster_options.max_iter,
        rtol=1e-25,
        atol=0.0,
        )
        print('output_params :', output_params)
        number_iterations = output_state.iter_num
        print('number ofiterations :', number_iterations)
        output_params[list(first_guess_params.keys())[0]].block_until_ready()
    else:
        print("Skipping minimization, using first guess parameters as output.", flush=True)
        output_params = first_guess_params
        number_iterations = 0

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
        W_params=W_params,
        Cov = cov_np,
        params_names = params_names,
        mc_samples = mc_samples,
        mean_deg = mean_np,
    )