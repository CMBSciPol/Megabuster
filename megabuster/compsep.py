import warnings
import shutil
import numpy as np
import jax
jax.config.update("jax_traceback_filtering", "off")
jax.config.update("jax_enable_x64", True)
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
from megabuster.BandpassOperator import BandpassOperator
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
    diagonal_central_term = None
    W_array=None,
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
        self.diagonal_central_term = None
        self.W_array=None,

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
        mean_deg = None,
        diagonal_central_term=None,
        W_array=None,
        
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
                q=input_maps[:,-2,:], 
                u=input_maps[:,-1,:]
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
                    q=input_maps[:,-2,:], 
                    u=input_maps[:,-1,:]
                )
                return W_params(params, input_maps_stokes)
        else:
            final_W_params = None

        res.W_maxL = final_W
        res.A_maxL = final_A_maxL
        res.s = np.array(final_maps)
        res.success = number_iterations < max_iter
        res.message = f"Optimization finished after {number_iterations} iterations out of {max_iter} allowed."
        res.W_params = final_W_params
        res.Cov = Cov
        res.params_names = params_names
        res.mc_samples = mc_samples
        res.mean_deg = mean_deg
        res.diagonal_central_term = diagonal_central_term
        res.W_array = W_array
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
    real_invN_matrix=None,
    real_sky_map = None,
    do_minimization=False,
    fixed_params=dict(),
    binary_mask=None,
    dust_nu0=150.0, 
    synchrotron_nu0=150.0,
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
    #solver_name="scipy_tnc",
    #solver_name="optax_lbfgs",
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
    use_bandpass = False,
    distribute = None,
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
    distribute: str | bool | None (optional)
        Distribution of the frequency channels over the devices, using the furax ensemble
        engine: the per-channel contributions to the likelihood are evaluated on the
        devices and summed across them. Default is None: when several devices are visible
        and the computation supports it (no patches, observation matrix, real beams, HMC or
        Hessian), a mesh is installed over the largest number of devices that divides the
        channels and the computation is distributed; on a single device it runs as before.
        False disables the distribution; the name of a mesh axis forces it over that axis
        (`jax.set_mesh` must then be active around the call).

    
    
    Returns
    -------
    dict
        Final parameters stored as a dictionary with keys 'temp_dust', 'beta_dust', and 'beta_pl'.

    Notes
    -----
    All the inputs for sky map, inverse noise covariance matrix, and observation matrix operator are assumed to contain only the pixels observed and not the full sky, which otherwise leads to a significant increase in computation time. 
    """

    if isinstance(invN_matrix, dict):
        any_arr = next(iter(invN_matrix.values()))
        npix_invN = any_arr.shape[-1]
        invN_matrix_full = np.zeros((len(config.map_sets), any_arr.shape[1], npix_invN), dtype=any_arr.dtype)
        for exp, arr in invN_matrix.items():
            idx = [i for i, m in enumerate(config.map_sets) if m.exp_tag == exp]
            invN_matrix_full[idx] = arr
        invN_matrix = invN_matrix_full

    if isinstance(real_invN_matrix, dict):
        any_arr_real = next(iter(real_invN_matrix.values()))
        npix_real = any_arr_real.shape[-1]
        real_invN_matrix_full = np.zeros(
            (len(config.map_sets), any_arr_real.shape[1], npix_real), dtype=any_arr_real.dtype
        )
        for exp, arr in real_invN_matrix.items():
            idx = [i for i, m in enumerate(config.map_sets) if m.exp_tag == exp]
            real_invN_matrix_full[idx] = arr
        real_invN_matrix = real_invN_matrix_full

    if isinstance(binary_mask, dict):
        any_mask = next(iter(binary_mask.values()))
        binary_mask_union = np.zeros_like(any_mask)
        for arr in binary_mask.values():
            binary_mask_union = np.maximum(binary_mask_union, arr)
        binary_mask = binary_mask_union

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

    use_full_sky = True
    if binary_mask is not None:
        assert isinstance(binary_mask, ArrayLike), "Binary mask must be a ndarray."
        assert binary_mask.ndim == 1, "Binary mask must be a 1D array."
        pixels_mask = np.where(binary_mask != 0)[0]
        if use_full_sky:
            #npix_full = invN_matrix.shape[-1]
            #unobserved_pixels = np.where(binary_mask == 0)[0]
            #pixels_to_retain = np.arange(npix_full)
            #n_pix = pixels_to_retain.size
            npix_full = invN_matrix.shape[-1]
            n_pix = npix_full
            pixels_to_retain = np.arange(npix_full)
            pixels_to_retain_t = np.where(binary_mask != 0)[0]
            pixels_to_retain_nested = np.arange(npix_full)
            unobserved_pixels = np.where(binary_mask == 0)[0] 
            binary_mask = np.ones(npix_full)
            #unobserved_pixels = np.array([], dtype=int)
        else :
            pixels_to_retain = np.where(binary_mask != 0)[0]
            pixels_to_retain_t = np.where(binary_mask != 0)[0]
            n_pix = pixels_to_retain.size
            #unobserved_pixels = None
        #unobserved_pixels = np.where(binary_mask == 0)[0]
        #pixels_to_retain = np.where(binary_mask != 0)[0]
        pixels_to_retain_nested = np.where(hp.reorder(binary_mask, r2n=True) != 0)[0]
    else:
        npix_full = invN_matrix.shape[-1]
        n_pix = npix_full
        pixels_to_retain = np.arange(npix_full)
        pixels_to_retain_t = np.where(binary_mask != 0)[0]
        pixels_to_retain_nested = np.arange(npix_full)
        binary_mask = np.ones(npix_full)
        unobserved_pixels = np.array([], dtype=int)
    
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
    

    #invN_matrix_nested = np.zeros(invN_matrix.shape[:2] + (pixels_to_retain_nested.size,))
    invN_matrix_ring = np.zeros(invN_matrix.shape[:2] + (pixels_to_retain.size,))
    if config.pre_proc_pars.use_real_beams:
        real_invN_matrix_ring = np.zeros(real_invN_matrix.shape[:2] + (pixels_to_retain.size,))
    if use_full_sky:
        invN_matrix_ring[..., unobserved_pixels] = 0.0
        if config.pre_proc_pars.use_real_beams:
            real_invN_matrix_ring[..., unobserved_pixels] = 0.0

    for i in range(invN_matrix.shape[0]):
        #invN_matrix_nested[i] = hp.reorder(invN_matrix[i], r2n=True)[..., pixels_to_retain_nested]
        invN_matrix_ring[i] = invN_matrix[i][..., pixels_to_retain]
        if config.pre_proc_pars.use_real_beams:
            real_invN_matrix_ring[i] = real_invN_matrix[i][..., pixels_to_retain]
        # If noiseless tests create identity noise matrix divided by a factor 100:
        #ones_matrix = np.ones(invN_matrix.shape[1:])/100  # shape (n_stokes, n_pixels)
        #invN_matrix_nested[i] = hp.reorder(ones_matrix, r2n=True)[..., pixels_to_retain_nested]

    #mask_2d = np.zeros(npix_full, dtype=bool)
    #mask_2d[pixels_to_retain] = True

    if isinstance(sky_map, ArrayLike):
        assert sky_map.ndim == 3, "Sky map must have shape (n_frequencies, n_stokes, n_pixels)."
        assert sky_map.shape[1] == n_stokes, "Sky map must contain only Q and U Stokes parameters."
        #sky_map = Stokes.from_stokes(
        #    q=hp.reorder(sky_map[:, -2], r2n=True)[..., pixels_to_retain_nested], 
        #    u=hp.reorder(sky_map[:, -1], r2n=True)[..., pixels_to_retain_nested]
        #)
        sky_map = Stokes.from_stokes(
            q=sky_map[:, -2][..., pixels_to_retain], 
            u=sky_map[:, -1][..., pixels_to_retain]
        )
        if config.pre_proc_pars.use_real_beams:
            real_sky_map = Stokes.from_stokes(
                q=real_sky_map[:, -2][..., pixels_to_retain], 
                u=real_sky_map[:, -1][..., pixels_to_retain]
            )
    else:
        assert sky_map.q.shape[0] == sky_map.u.shape[0], "Sky map must have the same number of Q and U Stokes parameters."
        assert sky_map.q.shape[1] == sky_map.u.shape[1], "Sky map must have the same number of pixels for Q and U Stokes parameters."
        # Retain only the pixels that are not masked
        #sky_map = Stokes.from_stokes(
        #    q=hp.reorder(sky_map.q, r2n=True)[..., pixels_to_retain_nested], 
        #    u=hp.reorder(sky_map.u, r2n=True)[..., pixels_to_retain_nested]
        #)
        sky_map = Stokes.from_stokes(
            q=sky_map.q[..., pixels_to_retain], 
            u=sky_map.u[..., pixels_to_retain]
        )
        if config.pre_proc_pars.use_real_beams:
            real_sky_map = Stokes.from_stokes(
                q=real_sky_map.q[..., pixels_to_retain], 
                u=real_sky_map.u[..., pixels_to_retain]
            )
    # Prepare the in_structure of the upcoming operators
    in_structure_sed = sky_map.structure_for((sky_map.shape[1],))
    in_structure_noise_cov = sky_map.structure
    vect_shape = sky_map.q.shape

    invN = get_diagonal_operator_from_stokes_maps(invN_matrix_ring, in_structure_noise_cov)
    real_invN = get_diagonal_operator_from_stokes_maps(real_invN_matrix_ring, in_structure_noise_cov) if config.pre_proc_pars.use_real_beams else None
    
    #Noiseless case for tests:
    #invN = IdentityOperator(in_structure=invN2.in_structure)

    # create miscalibration dict
    angles_dict = {}
    priors_dict = {}
    first_angles_list = []
    prior_list = []

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
    if do_minimization:
        # Minimisation : les angles de départ sont les valeurs centrales du prior
        first_guess_params = {**first_guess_params, **angles_dict}
    else:
        # Pas de minimisation : on garde les angles fournis dans first_guess_params
        # (ex. angles estimés), et on ne prend angle_central_value que pour ceux qui manquent
        first_guess_params = {**angles_dict, **first_guess_params}
    angles_names = [f'angle_{i}' for i in range(len(angles))]

    if use_obsmat:
        # Prepare the observation matrix operator
        # Precompute part of the right-hand side of the CG equation
        print("Precomputing the right-hand side of the CG equation . . .", flush=True)
        ONd = obsmat_operator_rhs(invN(sky_map))
        ONd.q.block_until_ready()
        print("Right-hand side of the CG equation precomputed!", flush=True)

    if matrix_precond is None:
        central_matrix_precond = jnp.copy(invN_matrix_ring)
    elif use_preconditioner_diag:
            
        if matrix_precond.shape[-1] == binary_mask.size:
            print("Using diag obsmat matrices")
            matrix_precond = matrix_precond[..., pixels_to_retain]
        central_matrix_precond = jnp.einsum('fsp,fsp,fsp->fsp', matrix_precond, invN_matrix_ring, matrix_precond)

    elif use_preconditioner_pinv:
        assert matrix_precond.shape[:3] == (frequencies.size, n_stokes, n_stokes), "If we use a preconditioner based on the pseudo-inverse, the central preconditioner matrix must have its three first dimensions as (n_frequencies, n_stokes, n_stokes) with n_stokes=2."

        assert matrix_precond.shape[-1] == matrix_precond.shape[-2], "Central preconditioner matrix must have square submatrices."

        if matrix_precond.shape[-2:] == (binary_mask.size, binary_mask.size):
            central_matrix_precond = matrix_precond[..., pixels_to_retain_ring, pixels_to_retain]
        else:
            central_matrix_precond = matrix_precond
        
        central_operator_precond = get_dense_furax_operator_from_freq_array(matrix_precond)

    # LOAD BANDPASS OPERATOR

    if use_bandpass:
        if any(map_set.frequency is None for map_set in config.map_sets):
            raise ValueError("use_bandpass=True requires the passbands of the map sets (map_set.frequency, map_set.weight).")
        bandpasses = [(map_set.frequency, map_set.weight) for map_set in config.map_sets]
        frequencies_mixing = BandpassOperator.integration_frequencies(bandpasses)
        Bp = BandpassOperator.create_from_samples(
            bandpasses,
            in_structure=jax.tree.map(
            lambda leaf: jax.ShapeDtypeStruct(leaf.shape[:-1] + (frequencies_mixing.size,) + leaf.shape[-1:], leaf.dtype),
            in_structure_sed,
            ),
        )
    else:
        frequencies_mixing = frequencies
        Bp = None

    # LOAD BEAM OPERATOR

    #beams = config.beams
    #beam_fl = jnp.array([
    #    hp.gauss_beam(fwhm=np.radians(fwhm / 60), lmax=config.lmax)
    #    for fwhm in beams
    #])  # shape (6, lmax+1)

    #print('Beams, lmax = ', beams, config.lmax)
    #print("beam_fl = ", beam_fl)
    #print('Checkpoint')

    #B = BeamOperator(
    #    lmax=config.lmax,
    #    beam_fl=beam_fl,
    #    in_structure=invN.in_structure,
    #    unobserved_pixels=unobserved_pixels  # np.where(binary_mask == 0)[0]
    #    )
    
    #Op_rhs = B.T
    #print('invN(sky_map) = ', jnp.sum(invN(sky_map).q**2))
    #Test Beam Operator

    #B_sky = B(sky_map)
    #B_inv_B_sky = B.I(B_sky)  # ou B.T si BeamOperator n'a pas de .I et que B est juste une convolution (auquel cas B.I doit exister si B est inversible, vérifie l'implémentation de BeamOperator)

    #diff = jax.tree.map(lambda a, b: a - b, B(B_inv_B_sky), B(sky_map))
    #residual_norm = jnp.sqrt(dot_2(diff, diff))
    #jax.debug.print("résidu B.I(B(x)) - x = {:.6e}", residual_norm)

    #print('sky_map = ', sky_map)
    #print('invN = ', invN)
    ONd = invN(sky_map)
    if config.pre_proc_pars.use_real_beams:
        #ONd_Beam = Op_rhs(real_invN(real_sky_map))
        ONd_Beam = real_invN(real_sky_map)
        #ONd_Beam = invN(sky_map)
        #ONd_Beam = Op_rhs(invN(sky_map))
        reals = real_invN(real_sky_map)
        falses = invN(sky_map)
        #reals = np.array(get_maps_from_Stokes(real_invN(real_sky_map)))[pixels_to_retain_t]
        print('reals', reals)
        print('falses', falses)
        print('ONd_Beam', ONd_Beam)
    else:
        ONd_Beam = invN(sky_map)
    #Test Beam Operator

    #B_sky = B(sky_map)
    #B_inv_B_sky = B.I(B_sky)  # ou B.T si BeamOperator n'a pas de .I et que B est juste une convolution (auquel cas B.I doit exister si B est inversible, vérifie l'implémentation de BeamOperator)

    #diff = jax.tree.map(lambda a, b: a - b, B(B_inv_B_sky), B(sky_map))
    #residual_norm = jnp.sqrt(dot_2(diff, diff))
    #jax.debug.print("résidu B.I(B(x)) - x = {:.6e}", residual_norm)

    #print("ONd.q = ", ONd.q)
    #jax.debug.print("NaN in ONd ?: {}", jnp.any(jnp.isnan(ONd.q)))

    number_components = 3 # CMB, dust, synchrotron

    # Distribution of the frequency channels over the devices of a mesh axis: the channels
    # are the members of a furax ensemble, and their contributions to the likelihood are
    # summed across the devices by the engine (see megabuster/distributed.py). By default
    # (distribute=None), the channels are distributed whenever several devices are visible
    # and the computation supports it; False disables it, a mesh axis name forces it.
    if distribute is None:
        supported = (
            (patch_indices is None or all(v is None for v in patch_indices.values()))
            and not use_obsmat
            and not config.pre_proc_pars.use_real_beams
            and not use_hmc
            and not use_hessienne
        )
        if supported:
            from megabuster.distributed import auto_mesh_axis

            distribute = auto_mesh_axis(len(frequencies))
    elif distribute is False:
        distribute = None

    if distribute is not None:
        from megabuster.BandpassOperator import channel_bandpasses
        from megabuster.distributed import (
            build_channel_ensemble,
            distributed_A_s_AOND,
            distributed_gradients,
            minimize_scipy_on_calling_thread,
        )

        if patch_indices is not None and any(v is not None for v in patch_indices.values()):
            raise NotImplementedError('distribute does not support patch indices yet.')
        if use_obsmat or config.pre_proc_pars.use_real_beams:
            raise NotImplementedError(
                'distribute does not support the observation matrix or the real beams yet.'
            )
        if use_bandpass:
            channel_nus, channel_weights = channel_bandpasses(
                [(map_set.frequency, map_set.weight) for map_set in config.map_sets]
            )
        else:
            channel_nus = np.asarray(frequencies, dtype=float)[:, None]
            channel_weights = np.ones_like(channel_nus)
        ensemble = build_channel_ensemble(
            sky_map,
            invN_matrix_ring,
            channel_nus,
            channel_weights,
            axis=distribute,
        )
        if use_full_sky and unobserved_pixels.size:
            observed_mask = np.ones(n_pix, dtype=bool)
            observed_mask[unobserved_pixels] = False
            observed_mask = jnp.asarray(observed_mask)
        else:
            observed_mask = None

        def distributed_angles(parameters_dict):
            if not use_calibration_matrix:
                return jnp.zeros(len(angles_names))
            return jnp.array([parameters_dict[name] for name in angles_names])

        def distributed_spectral(parameters_dict):
            return {key: parameters_dict[key] for key in ('temp_dust', 'beta_dust', 'beta_pl')}

        def distributed_s_AOND(parameters_dict):
            return distributed_A_s_AOND(
                ensemble,
                distributed_angles(parameters_dict),
                distributed_spectral(parameters_dict),
                observed_mask,
                dust_nu0,
                synchrotron_nu0,
            )

    def build_preconditioner2(invN, matrix_A, angles_jnp, Op, unobserved_pixels=None):
        """
        Build a diagonal preconditioner for Op = A.T @ C.T @ invN @ C @ A.
        
        Since invN_Q = invN_U (same variance for Q and U), C.T @ invN @ C 
        simplifies back to invN (cross terms vanish).

        Parameters
        ----------
        invN : DiagonalOperator
            Inverse noise covariance operator, _diagonal[0] = Q, _diagonal[1] = U.
            Each has shape (n_freq, n_pix).
        matrix_A : jnp.ndarray
            Mixing matrix diagonal, shape (n_freq, n_components, n_pix).
        angles_jnp : jnp.ndarray
            HWP angles, shape (n_freq,).
        Op : dict
            used to get in_structure.

        Returns
        -------
        preconditioner : BlockColumnOperator
            Diagonal preconditioner operator ready to pass to CG.
        """

        # 1. Extraire la diagonale de invN
        invN_Q = invN._diagonal[0]
        invN_U = invN._diagonal[1]

        # 2. Calcul de C.T @ invN @ C
        cos_2a = jnp.cos(2 * angles_jnp).reshape(-1, 1)
        sin_2a = jnp.sin(2 * angles_jnp).reshape(-1, 1)

        CtinvNC_QQ = cos_2a**2 * invN_Q + sin_2a**2 * invN_U
        CtinvNC_UU = sin_2a**2 * invN_Q + cos_2a**2 * invN_U
        CtinvNC_QU = cos_2a * sin_2a * (invN_U - invN_Q)

        CtinvNC = jnp.stack([
        jnp.stack([CtinvNC_QQ, CtinvNC_QU], axis=1),
        jnp.stack([CtinvNC_QU, CtinvNC_UU], axis=1),
        ], axis=1)

        # 3. Contraction avec A
        matrix = jnp.einsum('fcp, fsjp, fkp -> cksjp', matrix_A, CtinvNC, matrix_A)
        # -> (n_comp, n_comp, n_stokes, n_stokes, n_pix)
        # Reshape en (n_pix, n_comp*n_stokes, n_comp*n_stokes)
        n_comp, _, n_stokes, _, n_pix = matrix.shape
        matrix_2d = matrix.transpose(4, 0, 2, 1, 3).reshape(n_pix, n_comp*n_stokes, n_comp*n_stokes)

        # 4. Regularisation : remplacer les pixels non-observés par l'identité avant inversion
        eye = jnp.eye(n_comp * n_stokes)

        if unobserved_pixels is not None:
            matrix_2d = matrix_2d.at[unobserved_pixels].set(eye)

        #jax.debug.print("NaN in matrix_2d after regularization: {}", jnp.any(jnp.isnan(matrix_2d)))
        #jax.debug.print("Inf in matrix_2d after regularization: {}", jnp.any(jnp.isinf(matrix_2d)))
        #jax.debug.print("NaN in matrix_A: {}", jnp.any(jnp.isnan(matrix_A)))
        #jax.debug.print("NaN in invN_Q: {}", jnp.any(jnp.isnan(invN_Q)))
        #jax.debug.print("NaN in invN_U: {}", jnp.any(jnp.isnan(invN_U)))
        #jax.debug.print("NaN in CtinvNC: {}", jnp.any(jnp.isnan(CtinvNC)))

        # 5. Inversion (plus de singularité)
        inv_matrix_2d = jnp.linalg.inv(matrix_2d) # (n_pix, n_comp*n_stokes, n_comp*n_stokes)
        #jax.debug.print("NaN in inv_matrix_2d: {}", jnp.any(jnp.isnan(inv_matrix_2d)))
        # 6. Remettre à zéro les pixels non-observés
        #inv_matrix_2d = inv_matrix_2d.at[unobserved_pixels].set(0.0)
        if unobserved_pixels is not None:
            inv_matrix_2d = inv_matrix_2d.at[unobserved_pixels].set(0.0)

        # Reshape back
        preconditioner_matrix = inv_matrix_2d.reshape(n_pix, n_comp, n_stokes, n_comp, n_stokes).transpose(1, 3, 2, 4, 0)
        # (n_comp, n_comp, n_stokes, n_stokes, n_pix)

        # 4. Construire l'opérateur Furax
        return get_preconditioner2(
            preconditioner_matrix,
            in_structure=as_structure(Op.in_structure['cmb'])
        )
 
    def build_preconditioner(invN, matrix_A, Op, unobserved_pixels=None):
        invN_Q = invN._diagonal[0]
        invN_U = invN._diagonal[1]

        M_Q = jnp.einsum('fcp, fp, fkp -> ckp', matrix_A, invN_Q, matrix_A)
        M_U = jnp.einsum('fcp, fp, fkp -> ckp', matrix_A, invN_U, matrix_A)

        # Regulariser les pixels non-observés
        if unobserved_pixels is not None:
            eye = jnp.eye(M_Q.shape[0])  # (n_comp, n_comp)
            M_Q = M_Q.at[:, :, unobserved_pixels].set(eye[..., None])
            M_U = M_U.at[:, :, unobserved_pixels].set(eye[..., None])

        inv_M_Q = jnp.linalg.inv(M_Q.transpose(2, 0, 1)).transpose(1, 2, 0)
        inv_M_U = jnp.linalg.inv(M_U.transpose(2, 0, 1)).transpose(1, 2, 0)

        # Remettre à zéro les pixels non-observés
        if unobserved_pixels is not None:
            inv_M_Q = inv_M_Q.at[:, :, unobserved_pixels].set(0.0)
            inv_M_U = inv_M_U.at[:, :, unobserved_pixels].set(0.0)

        preconditioner_matrix = jnp.stack([inv_M_Q, inv_M_U], axis=2)

        return get_preconditioner(
            preconditioner_matrix,
            as_structure(Op.in_structure['cmb'])
        )
    
    def get_A_s_AOND_Beam(params, right_member):
        """
            Compute the log-proba given a set of parameters 
        """

        # Select parameters which are to be estimated 
        parameters_dict = dict()
        for param_name in ['temp_dust', 'beta_dust', 'beta_pl']+angles_names:
            if param_name in params:
                if param_name in angles_names:
                    if not use_calibration_matrix:
                        parameters_dict[param_name] = 0.0
                    else:
                        parameters_dict[param_name] = params[param_name]
                else:
                    parameters_dict[param_name] = params[param_name]
            else:
                parameters_dict[param_name] = fixed_params[param_name]

        if use_calibration_matrix:
            n_angles = len([k for k in parameters_dict.keys() if k.startswith('angle_')])
            angles_jnp = jnp.array([parameters_dict[f'angle_{i}'] for i in range(n_angles)])[:, None]
            C = build_calibration_operator(-angles_jnp, vect_shape)
            #C = IdentityOperator(in_structure=in_structure_noise_cov)
            operator_rhs = C.T
            c_right_member = invN(sky_map)
            right_member = operator_rhs(c_right_member) # Op ?
            c_central_freq_op = invN @ C
            central_freq_op = C.T @ invN @ C
        elif not use_obsmat:
            #C = IdentityOperator(in_structure=invN.in_structure)
            #operator_rhs = C
            c_right_member = invN(sky_map)
            #right_member = c_right_member # Op ?
            c_central_freq_op = invN
            #central_freq_op = B.T @ real_invN @ B
            #central_freq_op = B.T @ invN @ B
            #central_freq_op = invN
            central_freq_op = real_invN

        # Mixing matrix operator
        #A = create_MixingMatrixOperator(frequencies, parameters_dict, in_structure_sed, dust_nu0=dust_nu0, synchrotron_nu0=synchrotron_nu0, patch_indices=patch_indices)
        A = create_MixingMatrixOperator(frequencies_mixing, parameters_dict, in_structure_sed, dust_nu0=dust_nu0, synchrotron_nu0=synchrotron_nu0, patch_indices=patch_indices)
        if use_bandpass:
              A = (Bp @ A).reduce()
        # Full right-hand side of the CG equation
        AOND = A.T(right_member)

        print('AOND : ', AOND)

        if use_calibration_matrix:
            Op = A.T @ central_freq_op @ A
            # Construction de matrix_A
            matrix_A = jnp.zeros((frequencies.size, number_components, n_pix))
            for i, component in enumerate(A.block_leaves):
                matrix_A = matrix_A.at[:, i, :].set(component._diagonal)
            
            if use_full_sky:
                diagonal_central_term = build_preconditioner2(invN, matrix_A, angles_jnp, Op, unobserved_pixels=unobserved_pixels)
            else:
                diagonal_central_term = build_preconditioner2(invN, matrix_A, angles_jnp, Op)
            first_central_term = diagonal_central_term(AOND)

        elif not use_obsmat:
            #Op = A.T @ central_freq_op @ A
            matrix_A = jnp.zeros((frequencies.size, number_components, n_pix))
            for i, component in enumerate(A.block_leaves):
                matrix_A = matrix_A.at[:, i, :].set(component._diagonal)

            #if use_full_sky:
            #    diagonal_central_term = build_preconditioner(invN, matrix_A, Op, unobserved_pixels=unobserved_pixels)
            #else:
            #    diagonal_central_term = build_preconditioner(invN, matrix_A, Op)

            Op = A.T @ real_invN @ A
            #Op = A.T @ invN @ A

            preconditioner = build_preconditioner(real_invN, matrix_A, Op, unobserved_pixels=unobserved_pixels)
            #preconditioner = build_preconditioner(invN, matrix_A, Op, unobserved_pixels=unobserved_pixels)

            diagonal_central_term = (A.T @ central_freq_op @ A).I(
                solver=lx.GMRES(
                    rtol=dictionary_parameters_CG['tol_CG'], 
                    atol=dictionary_parameters_CG['tol_CG'], 
                    max_steps=dictionary_parameters_CG['max_steps_CG']
                ), 
                preconditioner=preconditioner,
                callback=lambda x: print("Number of iterations in CG:", x.stats['num_steps'], flush=True)
            )
            first_central_term = diagonal_central_term(AOND)

        #jax.debug.print("'first_central_term 2 = {}", first_central_term)

        return A, first_central_term, AOND, c_central_freq_op, central_freq_op, c_right_member, right_member, diagonal_central_term


    # @equinox.filter_jit
    def get_A_s_AOND(params, right_member = ONd):
        """
            Compute the log-proba given a set of parameters 
        """

        # Select parameters which are to be estimated 
        parameters_dict = dict()
        for param_name in ['temp_dust', 'beta_dust', 'beta_pl']+angles_names:
            if param_name in params:
                if param_name in angles_names:
                    if not use_calibration_matrix:
                        parameters_dict[param_name] = 0.0
                    else:
                        parameters_dict[param_name] = params[param_name]
                else:
                    parameters_dict[param_name] = params[param_name]
            else:
                parameters_dict[param_name] = fixed_params[param_name]

        if use_calibration_matrix:
            n_angles = len([k for k in parameters_dict.keys() if k.startswith('angle_')])
            angles_jnp = jnp.array([parameters_dict[f'angle_{i}'] for i in range(n_angles)])[:, None]
            C = build_calibration_operator(-angles_jnp, vect_shape)
            #C = IdentityOperator(in_structure=in_structure_noise_cov)
            operator_rhs = C.T
            c_right_member = invN(sky_map)
            right_member = operator_rhs(c_right_member) # Op ?
            c_central_freq_op = invN @ C
            central_freq_op = C.T @ invN @ C
        elif not use_obsmat:
            #C = IdentityOperator(in_structure=invN.in_structure)
            #operator_rhs = C
            c_right_member = invN(sky_map)
            right_member = c_right_member # Op ?
            c_central_freq_op = invN
            central_freq_op = invN
            C = IdentityOperator(in_structure=invN.in_structure)

        # Mixing matrix operator
        #A = create_MixingMatrixOperator(frequencies, parameters_dict, in_structure_sed, dust_nu0=dust_nu0, synchrotron_nu0=synchrotron_nu0, patch_indices=patch_indices)
        A = create_MixingMatrixOperator(frequencies_mixing, parameters_dict, in_structure_sed, dust_nu0=dust_nu0, synchrotron_nu0=synchrotron_nu0, patch_indices=patch_indices)
        if use_bandpass:
            A = (Bp @ A).reduce()
        # Full right-hand side of the CG equation
        AOND = A.T(right_member)

        ## Calculate preconditioners :
        preconditioner = None

        if use_obsmat:
            c_right_member = None
            c_central_freq_op = None
            if use_preconditioner_diag:
                print("Using diagonal preconditioner")
                matrix_A = jnp.zeros((frequencies.size, number_components, n_pix))
                for component in range(number_components):
                    matrix_A = matrix_A.at[:,component,:].set(A.block_leaves[component]._diagonal)
                
                # Compute the preconditioner matrix assuming that the observation matrix is the identity and the noise covariance matrix is diagonal in pixel domain
                preconditioner_matrix = jax.lax.stop_gradient(jnp.linalg.pinv(jnp.einsum('fcp,fsp,fkp->psck', matrix_A, central_matrix_precond, matrix_A)).T)

                # Prepare the preconditioner operator
                preconditioner = get_preconditioner(
                    preconditioner_matrix, 
                    as_structure(AOND['cmb'])
                )
        if use_preconditioner_pinv:
            print("Using pseudo-inverse preconditioner")
            matrix_A = jnp.zeros((frequencies.size, number_components, n_pix))
            for component in range(number_components):
                matrix_A = matrix_A.at[:,component,:].set(A.block_leaves[component]._diagonal)
            
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

        if use_calibration_matrix:
            Op = A.T @ central_freq_op @ A
            # Construction de matrix_A
            matrix_A = jnp.zeros((frequencies.size, number_components, n_pix))
            for i, component in enumerate(A.block_leaves):
                matrix_A = matrix_A.at[:, i, :].set(component._diagonal)
            
            if use_full_sky:
                diagonal_central_term = build_preconditioner2(invN, matrix_A, angles_jnp, Op, unobserved_pixels=unobserved_pixels)
                #diagonal_central_term = build_preconditioner2(invN, matrix_A, angles_jnp, Op)
            else:
                diagonal_central_term = build_preconditioner2(invN, matrix_A, angles_jnp, Op)
            first_central_term = diagonal_central_term(AOND)
            #jax.debug.print("NaN in first_central_term ?: {}", jnp.any(jnp.isnan(first_central_term['cmb'].q)))

        elif not use_obsmat:
            Op = A.T @ central_freq_op @ A
            matrix_A = jnp.zeros((frequencies.size, number_components, n_pix))
            for i, component in enumerate(A.block_leaves):
                matrix_A = matrix_A.at[:, i, :].set(component._diagonal)

            if use_full_sky:
                diagonal_central_term = build_preconditioner(invN, matrix_A, Op, unobserved_pixels=unobserved_pixels)
            else:
                diagonal_central_term = build_preconditioner(invN, matrix_A, Op)

            #diagonal_central_term = build_preconditioner(invN, matrix_A, Op, unobserved_pixels=unobserved_pixels)
            #preconditioner = IdentityOperator(in_structure=Op.in_structure)
            #diagonal_central_term = (A.T @ central_freq_op @ A).I(
            #    solver=lx.CG(
            #        rtol=dictionary_parameters_CG['tol_CG'], 
            #        atol=dictionary_parameters_CG['tol_CG'], 
            #        max_steps=dictionary_parameters_CG['max_steps_CG']
            #    ), 
            #    preconditioner=preconditioner,
            #    callback=lambda x: print("Number of iterations in CG:", x.stats['num_steps'], flush=True)
            #)
            first_central_term = diagonal_central_term(AOND)
            #print('first_central_term 1 = ', first_central_term)
            #jax.debug.print("'first_central_term 1 = {}", first_central_term)

        #jax.debug.print("'first_central_term 2 = {}", first_central_term)

        return A, first_central_term, AOND, c_central_freq_op, central_freq_op, c_right_member, right_member, diagonal_central_term, C

    @equinox.filter_custom_jvp
    def spectral_likelihood_custom_gradient(params):
        """
            Compute the log-proba given a set of parameters 
        """

        # Select parameters which are to be estimated 
        parameters_dict = dict()
        for param_name in ['temp_dust', 'beta_dust', 'beta_pl']+angles_names:
            if param_name in params:
                if param_name in angles_names:
                    #if not use_calibration_matrix:
                    #    parameters_dict[param_name] = 0.0
                    #else:
                    parameters_dict[param_name] = params[param_name]
                else:
                    parameters_dict[param_name] = params[param_name]
            else:
                parameters_dict[param_name] = fixed_params[param_name]

        if distribute is not None:
            map_s, AOND = distributed_s_AOND(parameters_dict)
        else:
            _, map_s, AOND, _, _, _, _, _, _ = get_A_s_AOND(parameters_dict)

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
                if param_name in angles_names:
                    #if not use_calibration_matrix:
                    #    parameters_dict[param_name] = 0.0
                    #else:
                    parameters_dict[param_name] = params[param_name]
                else:
                    parameters_dict[param_name] = params[param_name]
            else:
                parameters_dict[param_name] = fixed_params[param_name]

        if distribute is not None:
            map_s, AOND = distributed_s_AOND(parameters_dict)
        else:
            A, map_s, AOND, c_central_freq_op, central_freq_op, c_right_member, right_member, _ , _  = get_A_s_AOND(parameters_dict)

        if use_calibration_matrix:
            n_angles = len([k for k in parameters_dict.keys() if k.startswith('angle_')])
            angles_jnp = jnp.array([parameters_dict[f'angle_{i}'] for i in range(n_angles)])
            first_angles_jnp = jnp.array([first_angles_list[i] for i in range(n_angles)])
            prior_jnp = jnp.array([prior_list[i] if prior_list[i] is not None else float('inf') for i in range(n_angles)])
            angles_diff = angles_jnp - first_angles_jnp
            term2 = jnp.sum((angles_diff**2)/(prior_jnp**2))
            logL = -dot_2(AOND, map_s) + term2
            prior_jnp = jnp.array([prior_list[i] if prior_list[i] is not None else float('inf') for i in range(n_angles)])
            
        else:
            logL = -dot_2(AOND, map_s)

        #A_deriv = create_MixingMatrixOperator_deriv(
        #    frequencies, 
        #    parameters_dict, 
        #    in_structure_sed, 
        #    dust_nu0=dust_nu0, 
        #    synchrotron_nu0=synchrotron_nu0, 
        #    patch_indices=patch_indices
        #)

        A_deriv = create_MixingMatrixOperator_deriv(frequencies_mixing, parameters_dict, in_structure_sed, dust_nu0=dust_nu0, synchrotron_nu0=synchrotron_nu0, patch_indices=patch_indices)
        if use_bandpass:
            A_deriv = {key: (Bp @ op).reduce() for key, op in A_deriv.items()}

        keys_params = params.keys()
        final_grad_log = 0

        if distribute is not None:
            # The per-channel contributions to the gradients, summed over the members by
            # the ensemble engine; the angle gradients come back with one entry per channel.
            grads = distributed_gradients(
                ensemble,
                distributed_angles(parameters_dict),
                distributed_spectral(parameters_dict),
                map_s,
                dust_nu0,
                synchrotron_nu0,
                use_calibration_matrix,
            )
            for key in keys_params:
                if key.startswith('angle'):
                    if not use_calibration_matrix:
                        continue
                    i = int(key.split('_')[1])
                    if priors_dict[key] is not None:
                        grad_prior = 2 * angles_diff[i] / priors_dict[key]**2
                    else:
                        grad_prior = 0
                    final_grad_log += (grads['angle'][i] + grad_prior) * beta_tau[key]
                else:
                    final_grad_log += grads[key] * beta_tau[key]
            return logL, final_grad_log

        if use_calibration_matrix:
            c_left_hand_term = c_right_member - (c_central_freq_op @ A)(map_s)
            left_hand_term = right_member - (central_freq_op @ A)(map_s)
            Ax = A(map_s)
            for key in keys_params:
                if key.startswith("angle"):  # si le paramètre est un angle de calibration
                    # On récupère l'index du bloc correspondant à cet angle
                    # tu peux construire ce dict angle->index avant
                    # Rotation infinitésimale via l'astuce +pi/4 en faisant attention l'operateur R(θ)
                    # est définit par une rotation de -θ
                    i = int(key.split("_")[1])
                    Ax_idx = Ax[i,:]
                    X_op = QURotationOperator.create(shape=Ax_idx.q.shape, stokes="QU", angles=-angles_jnp[i] - jnp.pi/4)
                    dXAx = 2 * X_op(Ax_idx)
                    grad_cal = -2*dot_2(dXAx, c_left_hand_term[i, :])
                    if priors_dict[key] is not None:
                        grad_prior = 2 * angles_diff[i] / priors_dict[key]**2
                    else:
                        grad_prior = 0
                    final_grad_log += (grad_cal + grad_prior)* beta_tau[key]
                else:
                    final_grad_log += -2*dot_2(A_deriv[key](map_s), left_hand_term) * beta_tau[key]
        else:
            ONd = invN(sky_map)
            left_hand_term = ONd - (central_freq_op @ A)(map_s)
            for key in keys_params:
                if key.startswith("angle"):
                    final_grad_log += 0
                else:
                    final_grad_log += -2*dot_2(A_deriv[key](map_s), left_hand_term) * beta_tau[key]

        return logL, final_grad_log

    def spectral_likelihood(params):
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

        _, map_s, AOND, _, _, _, _, _, _ = get_A_s_AOND(parameters_dict)

        if angles_prior_dict is None:
            logL = -dot_2(AOND, map_s)
        else:
            n_angles = len([k for k in parameters_dict.keys() if k.startswith('angle_')])
            angles_jnp = jnp.array([parameters_dict[f'angle_{i}'] for i in range(n_angles)])
            first_angles_jnp = jnp.array([first_angles_list[i] for i in range(n_angles)])
            prior_jnp = jnp.array([prior_list[i] if prior_list[i] is not None else float('inf') for i in range(n_angles)])
            angles_diff = angles_jnp - first_angles_jnp
            term2 = jnp.sum((angles_diff**2)/(prior_jnp**2))
            logL = -dot_2(AOND, map_s) + term2

        Npix = 1

        # Compute the negative log-likelihood
        return logL/Npix

    def Hessienne_plus_corner_plot(params):

        fact = 180/np.pi # radians to degrees factor

        params_names = [p for p in angles_names + ['temp_dust', 'beta_dust', 'beta_pl']
                    if p in params]

        # 2. Convertir dict -> vecteur plat
        def params_to_vec(params_dict):
            return jnp.array([params_dict[k] for k in params_names])

        # 3. Convertir vecteur plat -> dict
        def vec_to_params(vec):
            return {k: vec[i] for i, k in enumerate(params_names)}

        def likelihood_vec(vec):
            return spectral_likelihood(vec_to_params(vec))

        vec_params = params_to_vec(params)

        H = jax.hessian(lambda p: likelihood_vec(p))(vec_params)
        cov = jnp.linalg.inv(H)
        variances = jnp.diag(cov)
        std = jnp.sqrt(variances)
        eigvals = jnp.linalg.eigvals(H)

        # 1. Convertir en numpy
        cov_np = np.array(cov)

        # Indices des paramètres qui sont des angles
        angle_indices = [i for i, k in enumerate(params_names) if k in angles_names]

        # Vecteur de mise à l'échelle : fact pour les angles, 1 pour les autres
        scale = np.ones(len(params_names))
        scale[angle_indices] = fact  # fact = 180/pi

        output_params = np.array([params[k] for k in params_names])  # radians

        mean_np = output_params.copy()
        mean_np[angle_indices] *= fact  # angles → degrés

        D = np.diag(scale)
        cov_np_deg = D @ cov_np @ D  # covariance en degrés²

        return cov_np_deg, mean_np, params_names, mean_np

    def run_inference(rng_key, initial_position, logprob_fn, num_warmup=100):
            warmup = blackjax.window_adaptation(
                blackjax.hmc,
                logprob_fn,
                num_integration_steps=20,
            )
            (state, parameters), _ = warmup.run(
                rng_key,
                initial_position,
                num_steps=num_warmup
            )
            print("Learned step size:", parameters["step_size"])
            print("Learned inverse mass matrix:", parameters["inverse_mass_matrix"])
        
            hmc = blackjax.hmc(logprob_fn, **parameters)
            kernel = jax.jit(hmc.step)
            return state, kernel

    def inference_loop(rng_key, kernel, initial_state, num_samples):
        @jax.jit
        def one_step(state, rng_key):
            state, info = kernel(rng_key, state)
            return state, (state, info)
        
        keys = jax.random.split(rng_key, num_samples)
        final_state, (states, infos) = jax.lax.scan(one_step, initial_state, keys)
        return states, infos

    def log_prob(params):
        return -spectral_likelihood_custom_gradient(params)

    def do_hmc(n_warmup=1000,n_samples = 10000):
        rng_key = jax.random.PRNGKey(1)
        rng_key, warmup_key, sample_key = jax.random.split(rng_key, 3)
        
        initial_state, hmc_kernel = run_inference(
            warmup_key,
            first_guess_params,
            log_prob,
            n_warmup,
        )
        
        states, infos = inference_loop(
            sample_key,
            hmc_kernel,
            initial_state,
            n_samples,
        )
        
        print("Acceptance rate:", np.mean(infos.acceptance_rate))

        position_degrees = {}
        output_params = {}
        for name, samples in states.position.items():
            output_params[name] = np.array(samples)
            if name.startswith("angle"):
                position_degrees[name] = np.array(samples) * 180 / np.pi
            else:
                position_degrees[name] = np.array(samples)

        samples_array = np.column_stack([v for v in position_degrees.values()])
        
        #samples_array = np.column_stack([np.array(v) for v in states.position.values()])
        params_names = list(states.position.keys())
        
        means = np.mean(samples_array, axis=0)
        stds = np.std(samples_array, axis=0)

        # ✅ Ajout ArviZ : construire idata depuis states.position
        idata = az.from_dict(
            posterior={
                name: np.expand_dims(np.array(samples), axis=0)  # (1, n_draws)
                for name, samples in states.position.items()
            }
        )
        
        # ✅ Diagnostics de convergence
        summary = az.summary(idata, round_to=4)
        print(summary)

        # Construire les labels avec mean ± std
        labels_with_stats = [
            f"{name} = {mean:.3f} ± {std:.3f}"
            for name, mean, std in zip(params_names, means, stds)
        ]

        stats_params_dict = {
            name: {
                "mean": float(np.mean(np.array(samples))),
                "mean_of_std": float(np.std(np.array(samples))),       # std de la chaîne
                "std_of_mean": float(np.std(np.array(samples)) / np.sqrt(len(np.array(samples))))  # std de la moyenne
            }
            for name, samples in states.position.items()
        }
        
        mc_samples = MCSamples(
            samples=samples_array,
            names=params_names,
            labels=labels_with_stats
        )

        return mc_samples, states, output_params

    mc_samples = None
    params_names = None
    cov_np = None
    mean_np = None
    
    if do_minimization:
        print("Launching minimization!!", flush=True)
        if distribute is not None and solver_name.startswith('scipy'):
            # The host callback of the scipy solvers of furax_cs evaluates the likelihood
            # from a thread that does not see the mesh of the ensemble: run the same jaxopt
            # solver on the calling thread instead.
            output_params, output_state = minimize_scipy_on_calling_thread(
                spectral_likelihood_custom_gradient,
                first_guess_params,
                solver_name,
                config.parametric_sep_pars.megabuster_options.max_iter,
            )
        else:
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
        # Ensure these are defined on all code paths to avoid UnboundLocalError
        if use_hessienne:
            print('Use Hessienne ou quoi ?')
            cov_np, mean_np, params_names, results = Hessienne_plus_corner_plot(output_params)
            output_params = {k: float(results[i]) for i, k in enumerate(params_names)}
        elif use_hmc:
            mc_samples, states, output_params = do_hmc(n_warm, n_samples)
            output_params = {k: float(np.mean(np.array(v))) for k, v in states.position.items()}
            params_names = [p for p in angles_names + ['temp_dust', 'beta_dust', 'beta_pl']
                    if p in output_params]
            number_iterations = 1
        else:
            params_names = [p for p in angles_names + ['temp_dust', 'beta_dust', 'beta_pl']
                    if p in output_params]
    else:
        print("Skipping minimization, using first guess parameters as output.", flush=True)
        output_params = first_guess_params
        number_iterations = 0
        if use_hessienne:
            cov_np, mean_np, params_names, results = Hessienne_plus_corner_plot(output_params)
            #output_params = {k: float(results[i]) for i, k in enumerate(params_names)}
        else:
            params_names = [p for p in angles_names + ['temp_dust', 'beta_dust', 'beta_pl']
                    if p in output_params]
            
    #output_params = first_guess_params
    print('Oh')
    print("Final output parameters:", output_params, flush=True)

    def resolve_parameters(params):
        parameters_dict = dict()
        for param_name in ['temp_dust', 'beta_dust', 'beta_pl'] + angles_names:
            if param_name in params:
                parameters_dict[param_name] = params[param_name]
            else:
                parameters_dict[param_name] = fixed_params[param_name]
        return parameters_dict

    def compute_component_maps(params):
        """The component maps at `params`, whichever the execution mode."""
        if distribute is not None:
            return distributed_s_AOND(resolve_parameters(params))[0]
        return get_A_s_AOND(params)[1]

    #ONd = invN(sky_map)
    if distribute is not None:
        final_maps = compute_component_maps(output_params)
        # The mixing matrix at the fitted parameters, only used for the output A_maxL array
        A_maxL = create_MixingMatrixOperator(
            frequencies_mixing,
            resolve_parameters(output_params),
            in_structure_sed,
            dust_nu0=dust_nu0,
            synchrotron_nu0=synchrotron_nu0,
            patch_indices=patch_indices,
        )
        if use_bandpass:
            A_maxL = (Bp @ A_maxL).reduce()
    elif config.pre_proc_pars.use_real_beams:
        A_maxL, final_maps, AOND, _, central_freq_op, _, _, diagonal_central_term = get_A_s_AOND_Beam(output_params, right_member=ONd_Beam)
        #calculate residuals
        #Op = A_maxL.T @ central_freq_op @ A_maxL
        #Op_x = Op(final_maps)
        #diff = jax.tree.map(lambda a, b: a - b, Op_x, AOND)
        #res_norm = jnp.sqrt(dot_2(diff, diff))
        #aond_norm = jnp.sqrt(dot_2(AOND, AOND))
        #jax.block_until_ready(res_norm)
        #jax.debug.print("résidu absolu = {:.6e}", res_norm)
        #jax.debug.print("||AOND||      = {:.6e}", aond_norm)
        #jax.debug.print("résidu relatif = {:.6e}", res_norm / aond_norm)
        #AON = A_maxL.T @ B.T @ real_invN
        #W = diagonal_central_term @ AON
        #W = None
    else :
        A_maxL, final_maps, AOND, c_central_freq_op, central_freq_op, _, _, diagonal_central_term, C = get_A_s_AOND(output_params, right_member=ONd)
        #AON = A_maxL.T @ C.T @ invN
        #W = diagonal_central_term @ AON

    #W = None

    n_pix_retain = pixels_to_retain.size
            

    A_maxL_array = np.zeros((frequencies.size, number_components, n_pix_retain))
    for component in range(number_components):
        A_maxL_array[:,component] = A_maxL.block_leaves[component]._diagonal


    final_maps_ring = np.array([get_maps_from_Stokes(final_maps[key]) for key in ordering_component])

    final_maps_full_sky = np.zeros((number_components, final_maps_ring.shape[-2], binary_mask.shape[-1]), dtype=final_maps_ring.dtype)
    final_maps_full_sky[..., pixels_to_retain] = final_maps_ring

    #n_angles_W = len([k for k in output_params.keys() if k.startswith('angle_')])
    #angles_jnp_W = jnp.array(
    #    [output_params[f'angle_{i}'] for i in range(n_angles_W)]
    #)[:, None]  # (n_freq, 1)

    #invN_Q = invN_matrix_ring[:, 0, :]  # (n_freq, n_pix)
    #invN_U = invN_matrix_ring[:, 1, :]  # (n_freq, n_pix)

    #cos_2a = jnp.cos(2 * angles_jnp_W)
    #sin_2a = jnp.sin(2 * angles_jnp_W)

    #CtinvN_QQ = cos_2a * invN_Q
    #CtinvN_QU = sin_2a * invN_U
    #CtinvN_UQ = -sin_2a * invN_Q
    #CtinvN_UU = cos_2a * invN_U

    #CtinvN = jnp.stack([
    #    jnp.stack([CtinvN_QQ, CtinvN_QU], axis=0),
    #    jnp.stack([CtinvN_UQ, CtinvN_UU], axis=0),
    #], axis=0)  # (s_out, s_in, n_freq, n_pix)

    #matrix_A_W = jnp.asarray(A_maxL_array)  # (n_freq, n_comp, n_pix)

    #M = jnp.einsum('fcp,sSfp,fCp->cCsSp', matrix_A_W, CtinvN, matrix_A_W)
    #n_comp_W = matrix_A_W.shape[1]
    #n_pix_W = matrix_A_W.shape[-1]
    #M_2d = M.transpose(4, 0, 2, 1, 3).reshape(n_pix_W, n_comp_W * 2, n_comp_W * 2)

    #eye_W = jnp.eye(n_comp_W * 2)
    #if unobserved_pixels is not None and len(unobserved_pixels) > 0:
    #    M_2d = M_2d.at[unobserved_pixels].set(eye_W)

    #M_inv_2d = jnp.linalg.inv(M_2d)
    #if unobserved_pixels is not None and len(unobserved_pixels) > 0:
    #    M_inv_2d = M_inv_2d.at[unobserved_pixels].set(0.0)

    #M_inv_W = M_inv_2d.reshape(n_pix_W, n_comp_W, 2, n_comp_W, 2).transpose(1, 3, 2, 4, 0)

    #AON_W = jnp.einsum('fCp,SifP->CSfiP', matrix_A_W, CtinvN)

    #W_array = np.array(
    #    jnp.einsum('ccsSp,cSfip->csfip', M_inv_W, AON_W)
    #)  # (n_comp, s_out, n_freq, s_in, n_pix_retain)

    W_array = None

    def W_maxL(input_map):

        output_map_truncated = jax.tree.map(lambda x: x[..., pixels_to_retain_t], compute_component_maps(output_params))

        output_map_ring = np.array([get_maps_from_Stokes(output_map_truncated[key]) for key in ordering_component])

        output_map = np.zeros((number_components,output_map_ring.shape[-2], input_map.shape[-1]), dtype=output_map_ring.dtype)
        
        output_map[..., pixels_to_retain_t] = output_map_ring

        for i in range(number_components):
            output_map[i] = output_map[i]
        return output_map

    def W_params(params, input_map):

        output_map_truncated = jax.tree.map(lambda x: x[..., pixels_to_retain_t], compute_component_maps(params))

        output_map_ring = np.array([get_maps_from_Stokes(output_map_truncated[key]) for key in ordering_component])

        output_map = np.zeros((number_components,output_map_ring.shape[-2], input_map.shape[-1]), dtype=output_map_ring.dtype)
        
        output_map[..., pixels_to_retain_t] = output_map_ring
        print('nan dans les output_map2 ?', jnp.any(jnp.isnan(output_map)))
        for i in range(number_components):
            output_map[i] = output_map[i]
        return output_map
    
    

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
        #diagonal_central_term = W,
        W_array = W_array,
    )