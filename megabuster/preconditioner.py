import logging
import numpy as np
import jax
import jax.numpy as jnp
import jax.lax
from furax.core import IdentityOperator, DenseBlockDiagonalOperator

from megabuster.tools import (
    build_furax_preconditioner,
    get_array_from_A,
    get_A_from_array,
    get_A_from_array_stokes,
    get_dense_furax_operator_from_freq_array,
    get_diagonal_operator_from_stokes_maps,
)

logger = logging.getLogger(__name__)

__all__ = [
    'get_preconditioner',
    'PreconditionerNone',
    'PreconditionerDiagonal',
    'PreconditionerSVD',
    'PreconditionerQR',
    'PreconditionerSVD_test'
]


def _check_matrix_precond_dense(matrix_precond, n_freq, n_stokes):
    """Check the shape (n_freq, n_stokes, n_stokes, n_pix, n_pix) of a dense central preconditioner matrix."""

    assert matrix_precond.shape[:3] == (n_freq, n_stokes, n_stokes), "If we use a preconditioner based on a precomputed matrix, it must have its three first dimensions as (n_frequencies, n_stokes, n_stokes) with n_stokes=2."
    assert matrix_precond.shape[-1] == matrix_precond.shape[-2], "Central preconditioner matrix must have square submatrices."
    assert matrix_precond.ndim == 5, "Central preconditioner matrix must have 5 dimensions (n_freq, n_stokes, n_stokes, n_pix, n_pix)."


def _get_frequency_weights(invN_matrix, indices_freq_not_filtered):
    """
    Weights W applied to the mixing matrix, B = W A,
    with W = N^{-1/2} for the frequencies not filtered and 1 for the others, and their inverse.

    """
    n_freq = invN_matrix.shape[0]
    if indices_freq_not_filtered is None:
        indices_freq_not_filtered = np.empty(0, dtype=int)

    sqrt_invN = jnp.sqrt(jnp.asarray(invN_matrix)[indices_freq_not_filtered])

    weights = jnp.ones_like(invN_matrix).at[indices_freq_not_filtered].set(sqrt_invN)
    inverse_weights = jnp.ones_like(invN_matrix).at[indices_freq_not_filtered].set(
        jnp.where(sqrt_invN > 0, 1.0 / jnp.where(sqrt_invN > 0, sqrt_invN, 1.0), 0.0)
    )
    return weights, inverse_weights


class PreconditionerNone:
    def __init__(self, *args, **kwargs):
        logger.warning("Setting up no preconditioner")
        
    def get(self, *args, **kwargs):
        logger.warning("Using no preconditioner")
        return None


class PreconditionerDiagonal:
    def __init__(self, invN_matrix, structure_sed, *args, **kwargs):
        logger.info("Setting up diagonal preconditioner")
        self.central_matrix_precond = jnp.array(invN_matrix)
        self.n_pix = invN_matrix.shape[-1]

        self.structure_sed = structure_sed

    def get(self, A):
        logger.info("Using diagonal preconditioner")
        
        matrix_A = get_array_from_A(A, self.n_pix)

        # Compute the preconditioner matrix assuming that the observation matrix is the identity and the noise covariance matrix is diagonal in pixel domain
        preconditioner_matrix = jax.lax.stop_gradient(
            jnp.linalg.pinv(
                jnp.einsum('fcp,fsp,fkp->psck', matrix_A, self.central_matrix_precond, matrix_A)
            ).T
        )

        # Prepare the preconditioner operator
        return build_furax_preconditioner(
            preconditioner_matrix, 
            in_structure=self.structure_sed
        )

class PreconditionerSVD:
    def __init__(self, matrix_precond, structure_sed, invN_matrix, patch_indices=None, *args, **kwargs):
        logger.info("Setting up pseudo-inverse preconditioner")
        
        self.structure_sed = structure_sed

        assert (matrix_precond is not None), "If the pseudo-inverse preconditioner (use_preconditioner_pinv) is set to True, then matrix_precond must be provided."

        _check_matrix_precond_dense(matrix_precond, n_freq=invN_matrix.shape[0], n_stokes=invN_matrix.shape[1])

        self.n_pix = matrix_precond.shape[-1]

        self.patch_indices = patch_indices

        self.central_operator_precond = get_dense_furax_operator_from_freq_array(matrix_precond)

    def get(self, A):
        logger.info("Using pseudo-inverse preconditioner for the preconditioner computation")
        
        matrix_A = get_array_from_A(A, self.n_pix)
        
        if self.patch_indices is None:
            matrix_u, matrix_s, matrix_vh = jnp.linalg.svd(matrix_A[...,0].T, full_matrices=False)
        else:
            matrix_u, matrix_s, matrix_vh = jnp.linalg.svd(matrix_A.T, full_matrices=False)

        pseudo_inverse_At = jax.lax.stop_gradient(jnp.einsum(
                '...ba,...b,...cb->...ac', 
                matrix_vh, 
                jnp.where(matrix_s!=0, 1./matrix_s, 0.),
                matrix_u,
            )
        )

        pseudo_inverse_A_op = get_A_from_array(pseudo_inverse_At.T, self.structure_sed)

        return pseudo_inverse_A_op.T @ self.central_operator_precond @ pseudo_inverse_A_op
    

class PreconditionerQR:
    def __init__(
        self, 
        invN_matrix, 
        matrix_precond,
        indices_freq_not_filtered, 
        structure_sed,
        structure_sky_map,
        *args,
        **kwargs
    ):
        logger.info("Setting up QR decomposition preconditioner")

        prefactor_pixel_weight_A, inverse_prefactor_pixel_weight_A = _get_frequency_weights(invN_matrix, indices_freq_not_filtered)

        self.prefactor_pixel_weight_A = prefactor_pixel_weight_A
        self.structure_sed = structure_sed

        self.n_pix = invN_matrix.shape[-1]

        if matrix_precond is not None:
            _check_matrix_precond_dense(matrix_precond, n_freq=invN_matrix.shape[0], n_stokes=invN_matrix.shape[1])

            operator_inverse_prefactor = get_diagonal_operator_from_stokes_maps(
                inverse_prefactor_pixel_weight_A,
                structure_sky_map
            )
            self.central_operator_precond = operator_inverse_prefactor @ get_dense_furax_operator_from_freq_array(matrix_precond) @ operator_inverse_prefactor
        else:
            self.central_operator_precond = IdentityOperator(in_structure=structure_sky_map)


    def get(self, A):
        logger.info("Using QR decomposition for the preconditioner computation")

        matrix_A = get_array_from_A(A, self.n_pix)

        # B = N^{-1/2} A for each (stokes, pixel), shape (n_stokes, n_pix, n_freq, n_comp)
        weighted_A = jnp.einsum('fcp,fsp->spfc', matrix_A, self.prefactor_pixel_weight_A)
        q_matrix, r_matrix = jnp.linalg.qr(weighted_A)

        pseudo_inverse_B = jax.lax.stop_gradient(jnp.einsum(
            '...ab,...cb->...ac',
            jnp.linalg.pinv(r_matrix),
            q_matrix
        ))

        pseudo_inverse_A_op = get_A_from_array_stokes(
            jnp.einsum('spcf->csfp', pseudo_inverse_B),
            self.structure_sed
        )

        return pseudo_inverse_A_op.T @ self.central_operator_precond @ pseudo_inverse_A_op

class PreconditionerQR_test:
    def __init__(
        self, 
        matrix_precond,
        invN_matrix,
        structure_sed,
        structure_sky_map,
        *args,
        **kwargs
    ):
        logger.info("Setting up QR decomposition preconditioner")

        self.structure_sed = structure_sed

        self.n_pix = structure_sky_map.shape[-1]

        if matrix_precond is not None:
            _check_matrix_precond_dense(matrix_precond, n_freq=invN_matrix.shape[0], n_stokes=invN_matrix.shape[1])

            self.central_operator_precond = get_dense_furax_operator_from_freq_array(matrix_precond)
        else:
            #TODO: This is not correct, would maybe need invN, but for now good enough (177 iter without matrix_precond in easy case)
            self.central_operator_precond = IdentityOperator(in_structure=structure_sky_map)


    def get(self, A):
        logger.info("Using QR decomposition for the preconditioner computation")

        matrix_A = get_array_from_A(A, self.n_pix)

        q_matrix, r_matrix = jnp.linalg.qr(matrix_A[...,0].T)

        pseudo_inverse_At = jax.lax.stop_gradient(jnp.einsum(
            '...ab,...cb->...ac',
            jnp.linalg.pinv(r_matrix),
            q_matrix
        ))

        pseudo_inverse_A_op = get_A_from_array(pseudo_inverse_At.T, self.structure_sed)

        return pseudo_inverse_A_op.T @ self.central_operator_precond @ pseudo_inverse_A_op

class PreconditionerQR_test_pixels:
    def __init__(
        self, 
        matrix_precond,
        invN_matrix,
        structure_sed,
        structure_sky_map,
        *args,
        **kwargs
    ):
        logger.info("Setting up QR decomposition preconditioner")

        self.structure_sed = structure_sed

        self.n_pix = structure_sky_map.shape[-1]

        if matrix_precond is not None:
            _check_matrix_precond_dense(matrix_precond, n_freq=invN_matrix.shape[0], n_stokes=invN_matrix.shape[1])

            self.central_operator_precond = get_dense_furax_operator_from_freq_array(matrix_precond)
        else:
            #TODO: This is not correct, would maybe need invN, but for now good enough (177 iter without matrix_precond in easy case)
            self.central_operator_precond = IdentityOperator(in_structure=structure_sky_map)


    def get(self, A):
        logger.info("Using QR decomposition for the preconditioner computation")

        matrix_A = get_array_from_A(A, self.n_pix)

        weighted_A = jnp.einsum('fcp,s->spfc', matrix_A, jnp.ones(2))
        q_matrix, r_matrix = jnp.linalg.qr(weighted_A)

        pseudo_inverse_B = jax.lax.stop_gradient(jnp.einsum(
            '...ab,...cb->...ac',
            jnp.linalg.pinv(r_matrix),
            q_matrix
        ))

        pseudo_inverse_A_op = get_A_from_array_stokes(
            jnp.einsum('spcf->csfp', pseudo_inverse_B),
            self.structure_sed
        )
        return pseudo_inverse_A_op.T @ self.central_operator_precond @ pseudo_inverse_A_op


class PreconditionerSVD_test:
    def __init__(
        self, 
        invN_matrix, 
        matrix_precond,
        indices_freq_not_filtered, 
        structure_sed,
        structure_sky_map,
        *args,
        **kwargs
    ):
        logger.info("Setting up SVD decomposition test preconditioner")

        prefactor_pixel_weight_A, inverse_prefactor_pixel_weight_A = _get_frequency_weights(invN_matrix, indices_freq_not_filtered)

        self.prefactor_pixel_weight_A = prefactor_pixel_weight_A
        self.structure_sed = structure_sed

        self.n_pix = invN_matrix.shape[-1]

        if matrix_precond is not None:
            _check_matrix_precond_dense(matrix_precond, n_freq=invN_matrix.shape[0], n_stokes=invN_matrix.shape[1])

            operator_inverse_prefactor = get_diagonal_operator_from_stokes_maps(
                inverse_prefactor_pixel_weight_A,
                structure_sky_map
            )
            self.central_operator_precond = operator_inverse_prefactor @ get_dense_furax_operator_from_freq_array(matrix_precond) @ operator_inverse_prefactor
        else:
            self.central_operator_precond = IdentityOperator(in_structure=structure_sky_map)

    def get(self, A):
        logger.info("Using SVD decomposition test for the preconditioner computation")

        matrix_A = get_array_from_A(A, self.n_pix)

        weighted_A = jnp.einsum('fcp,fsp->spfc', matrix_A, self.prefactor_pixel_weight_A)
        matrix_u, matrix_s, matrix_vh = jnp.linalg.svd(weighted_A, full_matrices=False)

        pseudo_inverse_B = jax.lax.stop_gradient(jnp.einsum(
                '...ba,...b,...cb->...ac', 
                matrix_vh, 
                jnp.where(matrix_s!=0, 1./matrix_s, 0.),
                matrix_u,
            )
        )

        pseudo_inverse_A_op = get_A_from_array_stokes(
            jnp.einsum('spcf->csfp', pseudo_inverse_B),
            self.structure_sed
        )

        return pseudo_inverse_A_op.T @ self.central_operator_precond @ pseudo_inverse_A_op

class PreconditionerCommonBasis:
    """
    Preconditioner for M = sum_f A_f^T C_f A_f, with C_f = O_f^T N_f^-1 O_f the central block of
    frequency f (dense, given as matrix_central = load_matrix_precond(..., power_diagonal=1)).

        P = V [sum_f a_f a_f^T lambda_{f,k}]^-1 V^T,   lambda_{f,k} = (V^T C_f V)_kk

    i.e. all the (whitened) C_f are approximated by their diagonal in a common orthonormal basis V, where M is
    then block diagonal with one 3x3 block per mode k, inverted exactly. 

    common_filtered_eigenvector: V, e.g. the precomputed eigenvectors of C_f for one filtered frequency
        (no new eigendecomposition). If None, V is the eigenbasis of sum_f C~_f / tr(C~_f), C~_f = W^-1 C_f W^-1.
      
    Built with furax operators: P = V_op @ B^-1_op @ V_op.T @, with V_op.T a DenseBlockDiagonalOperator
    applying V^T to each component map (the mode index is reshaped as (n_stokes, n_pix)), and B^-1_op the per mode
    3x3 inverses as a BlockColumnOperator of DiagonalOperators (build_furax_preconditioner, as for the diagonal
    preconditioner, with modes instead of pixels).
    """

    components = ('cmb', 'dust', 'synchrotron')
    ncomponents = len(components)

    def __init__(
        self, 
        common_filtered_eigenvector, 
        matrix_central, 
        invN_matrix, 
        structure_sed, 
        indices_freq_not_filtered=None, 
        *args, 
        **kwargs
    ):
        logger.info("Setting up common basis preconditioner")
        assert matrix_central is not None, "The common basis preconditioner needs matrix_central, the dense central blocks C_f (power_diagonal=1)."
        _check_matrix_precond_dense(matrix_central, n_freq=invN_matrix.shape[0], n_stokes=invN_matrix.shape[1])

        n_freq, n_stokes, _, n_pix, _ = matrix_central.shape

        self.n_stokes = n_stokes
        self.structure_sed = structure_sed
        self.n_pix = n_pix
        
        n_tot = n_stokes * n_pix

        assert common_filtered_eigenvector is not None, "The common basis preconditioner needs common_filtered_eigenvector, the eigenvectors of one filtered frequency (or of sum_f C~_f / tr(C~_f))."
        assert common_filtered_eigenvector.shape == (n_stokes * n_pix, n_stokes * n_pix), f"common_filtered_eigenvector must have shape {(n_stokes * n_pix, n_stokes * n_pix)}"

        logger.info("Common basis preconditioner: using the eigenvectors provided")
        assert common_filtered_eigenvector.shape == (n_tot, n_tot), f"common_filtered_eigenvector must have shape {(n_tot, n_tot)}"

        # lambda_{f,k} = (V^T C~_f V)_{kk}
        # self.eigenvalues_per_freq = jnp.stack([jnp.sum(common_filtered_eigenvector * (whitened_block(f) @ common_filtered_eigenvector), axis=0) for f in range(n_freq)])
        common_filtered_eigenvector = common_filtered_eigenvector.reshape(n_stokes, n_pix, n_tot)
        self.eigenvalues_per_freq = jnp.stack([
            jnp.einsum(
                'sak,stab,tbk->k',
                 common_filtered_eigenvector, 
                 matrix_central[f], 
                 common_filtered_eigenvector
                )
            for f in range(n_freq)
        ])

        structure_components = {component: structure_sed for component in self.components}
        # V^T applied to each component map: y[t, b] = sum_{s, a} V[(s, a), (t, b)] x[s, a]
        self.operator_eigenvectors_T = DenseBlockDiagonalOperator(
            common_filtered_eigenvector.reshape(n_stokes, n_pix, n_stokes, n_pix),
            in_structure=structure_components,
            subscripts='satb,sa->tb',
        )

    def get(self, A):
        logger.info("Using common basis preconditioner")

        matrix_A = get_array_from_A(A, self.n_pix)[..., 0]                 # (n_freq, 3), no patches
        cov_diagonalized = jnp.einsum('fc,fd,fk->kcd', matrix_A, matrix_A, self.eigenvalues_per_freq)
        inverse_cov_diagonalized = jax.lax.stop_gradient(jnp.linalg.inv(cov_diagonalized))     # (n_modes, 3, 3)

        operator_inverse_covariance_diag = build_furax_preconditioner(
            jnp.transpose(inverse_cov_diagonalized, (1, 2, 0)).reshape(
                self.ncomponents, 
                self.ncomponents, 
                self.n_stokes, 
                self.n_pix
            ),
            in_structure=self.structure_sed,
        )
        return (
            self.operator_eigenvectors_T.T
            @ operator_inverse_covariance_diag
            @ self.operator_eigenvectors_T
        )


def get_preconditioner(
    use_preconditioner_diag: bool = False,
    use_preconditioner_pinv: bool = False,
    use_preconditioner_qr: bool = False,
    use_preconditioner_qr_test: bool = False,
    use_preconditioner_qr_test_pixels: bool = False,
    use_preconditioner_svd_test: bool = False,
    use_preconditioner_common_basis: bool = False,
    *args,
    **kwargs
):
    list_of_flags = [use_preconditioner_diag, use_preconditioner_pinv, use_preconditioner_qr, use_preconditioner_qr_test, use_preconditioner_qr_test_pixels, use_preconditioner_svd_test, use_preconditioner_common_basis]

    number_preconditioners = sum(bool(flag) for flag in list_of_flags)
    assert number_preconditioners <= 1, "Only one type of preconditioner (use_preconditioner_diag, use_preconditioner_pinv, use_preconditioner_qr or use_preconditioner_svd_test) can be set to True."

    if use_preconditioner_diag:
        return PreconditionerDiagonal(
            *args,
            **kwargs
        )
    elif use_preconditioner_pinv:
        return PreconditionerSVD(
            *args,
            **kwargs
        )
    elif use_preconditioner_qr:
        return PreconditionerQR(
            *args,
            **kwargs
        )
    elif use_preconditioner_qr_test:
        return PreconditionerQR_test(
            *args,
            **kwargs
        )
    elif use_preconditioner_qr_test_pixels:
        return PreconditionerQR_test_pixels(
            *args,
            **kwargs
        )
    elif use_preconditioner_svd_test:
        return PreconditionerSVD_test(
            *args,
            **kwargs
        )
    elif use_preconditioner_common_basis:
        return PreconditionerCommonBasis(
            *args,
            **kwargs
        )
    else:
        return PreconditionerNone()
