import logging, warnings
import numpy as np
import jax
import jax.numpy as jnp
import healpy as hp
from jaxtyping import ArrayLike

from furax.core import IdentityOperator
from furax.obs.stokes import Stokes

from furax_cs import minimize, SOLVER_NAMES

from megabuster.likelihood import Likelihood
from megabuster.preconditioner import get_preconditioner
from megabuster.tools import (
    Results,
    get_diagonal_operator_from_stokes_maps, 
    get_array_from_A,
    get_maps_from_Stokes, 
    get_dense_furax_operator_from_freq_array
)

logger = logging.getLogger(__name__)

__all__ = [
    'perform_compsep',
]


def perform_compsep(
    first_guess_params, 
    sky_map, 
    frequencies, 
    invN_matrix,
    do_minimization=True,
    fixed_params=dict(),
    binary_mask=None,
    dust_nu0=150.0, 
    synchrotron_nu0=150.0, 
    matrix_central=None, #TODO: Merge all 3 matrix_central, matrix_precond, common_filtered_eigenvector into one preconditioner object
    matrix_precond=None,
    common_filtered_eigenvector=None,
    obs_mat_operator=None, 
    obsmat_operator_rhs=None,
    use_preconditioner_diag=False,
    use_preconditioner_pinv=False,
    use_preconditioner_qr=False,
    use_preconditioner_common_basis=False,
    indices_freq_not_filtered=None,
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
    
    if dictionary_parameters_minimization['tol'] < dictionary_parameters_CG['tol_CG'] and do_minimization:
        warnings.warn("The tolerance for the minimization is smaller than the tolerance for the conjugate gradient solver. This might lead to suboptimal results.")

    solver_name = dictionary_parameters_minimization.get('solver_name', 'optax_lbfgs')
    # assert solver_name in SOLVER_NAMES.__args__, f"Solver name must be one of {SOLVER_NAMES.__args__}."

    if do_minimization:
        logger.info(f"The minimization will be performed with the following parameters: {dictionary_parameters_minimization}")

    if binary_mask is not None:
        assert isinstance(binary_mask, ArrayLike), "Binary mask must be a ndarray."
        assert binary_mask.ndim == 1, "Binary mask must be a 1D array."
        pixels_to_retain = np.where(binary_mask != 0)[0]
        pixels_to_retain_nested = np.where(hp.reorder(binary_mask, r2n=True) != 0)[0]
    else:
        pixels_to_retain = np.arange(invN_matrix.shape[-1])
        pixels_to_retain_nested = np.arange(invN_matrix.shape[-1])
        binary_mask = np.ones(invN_matrix.shape[-1])
    
    if patch_indices is not None:
        raise ValueError("Patch indices are not yet implemented. Please set patch_indices to None.")

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
    structure_sed = sky_map.structure_for((sky_map.shape[1],))
    invN = get_diagonal_operator_from_stokes_maps(invN_matrix_nested, sky_map.structure)


    # Prepare the observation matrix operator
    if obs_mat_operator is None:
        obs_mat_operator = IdentityOperator(in_structure=invN.in_structure)
    
    if obsmat_operator_rhs is None:
        obsmat_operator_rhs = obs_mat_operator.T
    
    if matrix_central is None:
        central_freq_op = obs_mat_operator.T @ invN @ obs_mat_operator
    else:
        assert matrix_central.shape[:3] == (frequencies.size, n_stokes, n_stokes), "If we use a precomputed matrix for the likelihood, it must have its three first dimensions as (n_frequencies, n_stokes, n_stokes) with n_stokes=2."
        assert matrix_central.shape[-1] == matrix_central.shape[-2], "Precomputed matrix for the likelihood must have square submatrices."
        assert matrix_central.ndim == 5, "Precomputed matrix for the likelihood must have 5 dimensions (n_freq, n_stokes, n_stokes, n_pix, n_pix)."

        central_freq_op = get_dense_furax_operator_from_freq_array(
                matrix_central
            )

    if indices_freq_not_filtered is not None:
        assert np.isin(indices_freq_not_filtered, np.arange(frequencies.size)).all(), "The indices of frequencies not filtered must be in the range of the number of frequencies."

        logger.warning(f"Not computing eigenspectra for frequencies indices {indices_freq_not_filtered}, corresponding to frequencies {[frequencies[i] for i in indices_freq_not_filtered]} ")
        indices_freq_not_filtered = np.array(indices_freq_not_filtered, dtype=int)

    # Precompute part of the right-hand side of the CG equation
    logger.info("Precomputing the right-hand side of the CG equation . . .")
    ONd = obsmat_operator_rhs(invN(sky_map))
    ONd.q.block_until_ready()
    logger.info("Right-hand side of the CG equation precomputed!")

    # if use_preconditioner_diag and matrix_precond is not None and matrix_precond.shape[-1] == binary_mask.size:
    #     logger.info("Using diag obsmat matrices")
    #     matrix_precond = matrix_precond[..., pixels_to_retain_nested]

    preconditioner = get_preconditioner(
        use_preconditioner_diag=use_preconditioner_diag,
        use_preconditioner_pinv=use_preconditioner_pinv,
        use_preconditioner_qr=use_preconditioner_qr,
        use_preconditioner_common_basis=use_preconditioner_common_basis,
        invN_matrix=invN_matrix_nested, 
        matrix_precond=matrix_precond,
        matrix_central=matrix_central,
        common_filtered_eigenvector=common_filtered_eigenvector,
        indices_freq_not_filtered=indices_freq_not_filtered, 
        structure_sed=structure_sed,
        structure_sky_map=sky_map.structure,
        patch_indices=patch_indices,
    )
        
    number_components = 3 # CMB, dust, synchrotron
    n_pix = pixels_to_retain_nested.size

    likelihood_object = Likelihood(
        frequencies = frequencies,
        fixed_params = fixed_params,
        structure_sed = structure_sed,
        central_freq_op = central_freq_op,
        right_member = ONd,
        dictionary_parameters_CG = dictionary_parameters_CG,
        dictionary_parameters_minimization = dictionary_parameters_minimization,
        preconditioner = preconditioner,
        helpers_mixing_matrix = {
            'dust_nu0': dust_nu0,
            'synchrotron_nu0': synchrotron_nu0,
            'patch_indices': patch_indices
        }
    )

    if do_minimization:
        logger.info("Launching minimization!!")
        output_params, output_state = minimize(
            init_params=first_guess_params, 
            fn=likelihood_object.get_spectral_likelihood(), 
            max_iter=dictionary_parameters_minimization['max_iter'], 
            rtol=dictionary_parameters_minimization['tol'],
            atol=dictionary_parameters_minimization['tol'],
            solver_name=solver_name,
            options=dictionary_parameters_minimization.get('options', None)
        ) # first output is the final parameters, second output is the final state of the optimizer 
        output_params[list(first_guess_params.keys())[0]].block_until_ready()
        logger.info(f"Final parameters: {output_params}")
        logger.debug(f"Final state: {output_state}")
        logger.info("Minimization launched!! Preparing the retrieving of the maps . . .")
        number_iterations = output_state.iter_num
        logger.info(f"Finished minimization in {number_iterations} iterations!")
    else:
        logger.info("Skipping minimization, using first guess parameters as output.")
        output_params = first_guess_params
        number_iterations = 0

    A_maxL, final_maps = likelihood_object.get_A_s_AOND(output_params, obsmat_operator_rhs(invN(sky_map)))[:2]

    A_maxL_array = np.asarray(get_array_from_A(A_maxL, n_pix))


    final_maps_nested = np.array([get_maps_from_Stokes(final_maps[key]) for key in ordering_component])

    final_maps_full_sky = np.zeros((number_components, final_maps_nested.shape[-2], binary_mask.shape[-1]), dtype=final_maps_nested.dtype)
    final_maps_full_sky[..., pixels_to_retain_nested] = final_maps_nested


    def W_maxL(input_map):
            
        output_map_truncated = likelihood_object.get_A_s_AOND(output_params, obsmat_operator_rhs(invN(input_map[...,pixels_to_retain_nested])))[1]

        output_map_nested = np.array([get_maps_from_Stokes(output_map_truncated[key]) for key in ordering_component])

        output_map = np.zeros((number_components,output_map_nested.shape[-2], input_map.shape[-1]), dtype=output_map_nested.dtype)
        
        output_map[..., pixels_to_retain_nested] = output_map_nested
        for i in range(number_components):
            output_map[i] = hp.reorder(output_map[i], n2r=True)
        return output_map

    def W_params(params, input_map):

        output_map_truncated = likelihood_object.get_A_s_AOND(params, obsmat_operator_rhs(invN(input_map[...,pixels_to_retain_nested])))[1]

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
