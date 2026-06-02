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
        #pixels_to_retain = np.where(binary_mask != 0)[0]
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
    

    #invN_matrix_nested = np.zeros(invN_matrix.shape[:2] + (pixels_to_retain_nested.size,))
    invN_matrix_nested = np.zeros(invN_matrix.shape[:2] + (len(pixels_to_retain),))

    for i in range(invN_matrix.shape[0]):
        #invN_matrix_nested[i] = hp.reorder(invN_matrix[i], r2n=True)[..., pixels_to_retain_nested]
        invN_matrix_nested[i] = invN_matrix[i][..., pixels_to_retain]
        # If noiseless tests create identity noise matrix divided by a factor 100:
        #ones_matrix = np.ones(invN_matrix.shape[1:])/100  # shape (n_stokes, n_pixels)
        #invN_matrix_nested[i] = hp.reorder(ones_matrix, r2n=True)[..., pixels_to_retain_nested]

    mask_2d = np.zeros(npix_full, dtype=bool)
    mask_2d[pixels_to_retain] = True

    if isinstance(sky_map, ArrayLike):
        assert sky_map.ndim == 3, "Sky map must have shape (n_frequencies, n_stokes, n_pixels)."
        assert sky_map.shape[1] == n_stokes, "Sky map must contain only Q and U Stokes parameters."
        #sky_map = Stokes.from_stokes(
        #    Q=hp.reorder(sky_map[:, -2], r2n=True)[..., pixels_to_retain_nested], 
        #    U=hp.reorder(sky_map[:, -1], r2n=True)[..., pixels_to_retain_nested]
        #)
        sky_map = Stokes.from_stokes(
            Q=sky_map[:, -2][..., pixels_to_retain], 
            U=sky_map[:, -1][..., pixels_to_retain]
        )
    else:
        assert sky_map.q.shape[0] == sky_map.u.shape[0], "Sky map must have the same number of Q and U Stokes parameters."
        assert sky_map.q.shape[1] == sky_map.u.shape[1], "Sky map must have the same number of pixels for Q and U Stokes parameters."
        # Retain only the pixels that are not masked
        #sky_map = Stokes.from_stokes(
        #    Q=hp.reorder(sky_map.q, r2n=True)[..., pixels_to_retain_nested], 
        #    U=hp.reorder(sky_map.u, r2n=True)[..., pixels_to_retain_nested]
        #)
        sky_map = Stokes.from_stokes(
            Q=sky_map.q[..., pixels_to_retain], 
            U=sky_map.u[..., pixels_to_retain]
        )

    # Prepare the in_structure of the upcoming operators
    in_structure_sed = sky_map.structure_for((sky_map.shape[1],))
    in_structure_noise_cov = sky_map.structure.q
    vect_shape = sky_map.q.shape

    invN = get_diagonal_operator_from_stokes_maps(invN_matrix_nested, in_structure_noise_cov)
    
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
    first_guess_params = {**first_guess_params,**angles_dict}
    angles_names = [f'angle_{i}' for i in range(len(angles))]

    if use_obsmat:
        # Prepare the observation matrix operator
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

    def build_preconditioner(invN, matrix_A, angles_jnp, Op):
        """
        Build a diagonal preconditioner for Op = A.T @ C.T @ invN @ C @ A.
        
        Since invN_Q = invN_U (same variance for Q and U), C.T @ invN @ C 
        simplifies back to invN (cross terms vanish).

        Parameters
        ----------
        invN : BlockDiagonalOperator
            Inverse noise covariance operator, block_leaves[0] = Q, block_leaves[1] = U.
            Each block has diagonal of shape (n_freq, n_pix).
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
        invN_Q = invN.block_leaves[0]._diagonal
        invN_U = invN.block_leaves[1]._diagonal

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

        matrix = jnp.einsum('fcp, fsjp, fkp -> cksjp', matrix_A, CtinvNC, matrix_A)
        # (n_comp, n_comp, n_stokes, n_stokes, n_pix)

        # Reshape en (n_pix, n_comp*n_stokes, n_comp*n_stokes)
        n_comp, _, n_stokes, _, n_pix = matrix.shape
        matrix_2d = matrix.transpose(4, 0, 2, 1, 3).reshape(n_pix, n_comp*n_stokes, n_comp*n_stokes)

        inv_matrix_2d = jnp.linalg.inv(matrix_2d)  # (n_pix, n_comp*n_stokes, n_comp*n_stokes)

        # Reshape back
        preconditioner_matrix = inv_matrix_2d.reshape(n_pix, n_comp, n_stokes, n_comp, n_stokes).transpose(1, 3, 2, 4, 0)
        # (n_comp, n_comp, n_stokes, n_stokes, n_pix)

        # 4. Construire l'opérateur Furax
        return get_preconditioner2(
            preconditioner_matrix,
            in_structure=as_structure(Op.in_structure['cmb'])
        )
 
    # @equinox.filter_jit
    def get_A_s_AOND(params):
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

        # Mixing matrix operator
        A = create_MixingMatrixOperator(frequencies, parameters_dict, in_structure_sed, dust_nu0=dust_nu0, synchrotron_nu0=synchrotron_nu0, patch_indices=patch_indices)

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
                    in_structure=as_structure(AOND['cmb'].q)
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
            
            diagonal_central_term = build_preconditioner(invN, matrix_A, angles_jnp, Op)
            first_central_term = diagonal_central_term(AOND)

        elif not use_obsmat:
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
            #print('first_central_term 1 = ', first_central_term)
            #jax.debug.print("'first_central_term 1 = {}", first_central_term)

        #jax.debug.print("'first_central_term 2 = {}", first_central_term)

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
                if param_name in angles_names:
                    #if not use_calibration_matrix:
                    #    parameters_dict[param_name] = 0.0
                    #else:
                    parameters_dict[param_name] = params[param_name]
                else:
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
                if param_name in angles_names:
                    #if not use_calibration_matrix:
                    #    parameters_dict[param_name] = 0.0
                    #else:
                    parameters_dict[param_name] = params[param_name]
                else:
                    parameters_dict[param_name] = params[param_name]
            else:
                parameters_dict[param_name] = fixed_params[param_name]

        A, map_s, AOND, c_central_freq_op, central_freq_op, c_right_member, right_member, _ = get_A_s_AOND(parameters_dict)

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

        _, map_s, AOND, _, _, _, _, _ = get_A_s_AOND(parameters_dict)

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

        return cov_np_deg, mean_np, params_names, output_params

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

    def do_hmc(n_warmup=300,n_samples = 10000):
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
            cov_np, mean_np, params_names, results = Hessienne_plus_corner_plot(output_params)
            output_params = {k: float(results[i]) for i, k in enumerate(params_names)}
        elif use_hmc:
            mc_samples, states, output_params = do_hmc(n_warm, n_samples)
            #output_params = {k: float(np.mean(np.array(v))) for k, v in states.position.items()}
            params_names = [p for p in angles_names + ['temp_dust', 'beta_dust', 'beta_pl']
                    if p in output_params]
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

    A_maxL, final_maps = get_A_s_AOND(output_params)[:2]

    A_maxL_array = np.zeros((frequencies.size, number_components, n_pix))
    for component in range(number_components):
        A_maxL_array[:,component] = A_maxL.block_leaves[component]._diagonal


    final_maps_nested = np.array([get_maps_from_Stokes(final_maps[key]) for key in ordering_component])

    final_maps_full_sky = np.zeros((number_components, final_maps_nested.shape[-2], binary_mask.shape[-1]), dtype=final_maps_nested.dtype)
    final_maps_full_sky[..., pixels_to_retain_nested] = final_maps_nested


    def W_maxL(input_map):
            
        output_map_truncated = get_A_s_AOND(output_params)[1][...,pixels_to_retain_nested]

        output_map_nested = np.array([get_maps_from_Stokes(output_map_truncated[key]) for key in ordering_component])

        output_map = np.zeros((number_components,output_map_nested.shape[-2], input_map.shape[-1]), dtype=output_map_nested.dtype)
        
        output_map[..., pixels_to_retain_nested] = output_map_nested
        for i in range(number_components):
            output_map[i] = hp.reorder(output_map[i], n2r=True)
        return output_map

    def W_params(params, input_map):

        output_map_truncated = get_A_s_AOND(params)[1][...,pixels_to_retain_nested]

        output_map_nested = np.array([get_maps_from_Stokes(output_map_truncated[key]) for key in ordering_component])

        output_map = np.zeros((number_components,output_map_nested.shape[-2], input_map.shape[-1]), dtype=output_map_nested.dtype)
        
        output_map[..., pixels_to_retain_nested] = output_map_nested
        for i in range(number_components):
            output_map[i] = hp.reorder(output_map[i], n2r=True)
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
    )