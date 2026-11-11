import jax.numpy as jnp
from furax.tree import as_structure
from megabuster.tools import (
    get_preconditioner,
    get_preconditioner2,
)

J_QU = jnp.array([[0., -1.], [1., 0.]])   # J(Q, U) = (−U, Q), indices [stokes_out, stokes_in]
    
def _hwp_AtNA(matrix_A, matrix_A_b, CtinvNC):
    """A.T N' A pixel par pixel, avec A_fc = α_fc·1 + β_fc·J (HWP + bande passante déjà intégrées).

    matrix_A = α, matrix_A_b = β : (n_freq, n_comp, n_pix).  CtinvNC = N' : (n_freq, 2, 2, n_pix).
    Renvoie (n_comp, n_comp, n_stokes, n_stokes, n_pix).
    """
    NJ = jnp.einsum('fsSp,Su->fsup', CtinvNC, J_QU)      # N' J
    JtN = jnp.einsum('Ss,fSup->fsup', J_QU, CtinvNC)     # Jᵀ N'
    JtNJ = jnp.einsum('Ss,fSup->fsup', J_QU, NJ)         # Jᵀ N' J
    a, b = matrix_A, matrix_A_b
    return (
        jnp.einsum('fcp,fkp,ftup->cktup', a, a, CtinvNC)
        + jnp.einsum('fcp,fkp,ftup->cktup', a, b, NJ)
        + jnp.einsum('fcp,fkp,ftup->cktup', b, a, JtN)
        + jnp.einsum('fcp,fkp,ftup->cktup', b, b, JtNJ)
    )

def _invert_per_pixel(matrix, Op, unobserved_pixels=None):
    """Inverse de chaque bloc (2 n_comp × 2 n_comp) par pixel, mise en forme pour get_preconditioner2."""
    n_comp, _, n_stokes, _, n_pix = matrix.shape
    matrix_2d = matrix.transpose(4, 0, 2, 1, 3).reshape(n_pix, n_comp*n_stokes, n_comp*n_stokes)
    if unobserved_pixels is not None:
        matrix_2d = matrix_2d.at[unobserved_pixels].set(jnp.eye(n_comp * n_stokes))
    inv_matrix_2d = jnp.linalg.inv(matrix_2d)
    if unobserved_pixels is not None:
        inv_matrix_2d = inv_matrix_2d.at[unobserved_pixels].set(0.0)
    preconditioner_matrix = inv_matrix_2d.reshape(n_pix, n_comp, n_stokes, n_comp, n_stokes).transpose(1, 3, 2, 4, 0)
    return get_preconditioner2(
        preconditioner_matrix,
        in_structure=as_structure(Op.in_structure['cmb'])
    )

def _CtinvNC(invN, angles_jnp):
    """N' = C.T @ invN @ C, de forme (n_freq, 2, 2, n_pix)."""
    invN_Q = invN._diagonal[0]
    invN_U = invN._diagonal[1]
    cos_2a = jnp.cos(2 * angles_jnp).reshape(-1, 1)
    sin_2a = jnp.sin(2 * angles_jnp).reshape(-1, 1)
    CtinvNC_QQ = cos_2a**2 * invN_Q + sin_2a**2 * invN_U
    CtinvNC_UU = sin_2a**2 * invN_Q + cos_2a**2 * invN_U
    CtinvNC_QU = cos_2a * sin_2a * (invN_U - invN_Q)
    return jnp.stack([
        jnp.stack([CtinvNC_QQ, CtinvNC_QU], axis=1),
        jnp.stack([CtinvNC_QU, CtinvNC_UU], axis=1),
    ], axis=1)

def build_preconditioner_HWP_NC(invN, matrix_A, matrix_A_b, Op, unobserved_pixels=None):
    """HWP + bande passante, SANS matrice de calibration : inverse exacte de A.T @ invN @ A.

    N' = invN, diagonal en Stokes : diag(invN_Q, invN_U).
    """
    invN_Q = invN._diagonal[0]
    invN_U = invN._diagonal[1]
    zeros = jnp.zeros_like(invN_Q)
    N = jnp.stack([
        jnp.stack([invN_Q, zeros], axis=1),
        jnp.stack([zeros, invN_U], axis=1),
    ], axis=1)                                           # (n_freq, 2, 2, n_pix)
    return _invert_per_pixel(_hwp_AtNA(matrix_A, matrix_A_b, N), Op, unobserved_pixels)

def build_preconditioner_NHWP_NC(invN, matrix_A, Op, unobserved_pixels=None):
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

def build_preconditioner_HWP_C(invN, matrix_A, matrix_A_b, angles_jnp, Op, unobserved_pixels=None):
    """HWP + bande passante + matrice de calibration."""
    return _invert_per_pixel(_hwp_AtNA(matrix_A, matrix_A_b, _CtinvNC(invN, angles_jnp)), Op, unobserved_pixels)

def build_preconditioner_NHWP_C(invN, matrix_A, angles_jnp, Op, unobserved_pixels=None):
    """Sans HWP, avec matrice de calibration : inverse exacte de A.T @ C.T @ invN @ C @ A."""
    matrix = jnp.einsum('fcp,fkp,ftup->cktup', matrix_A, matrix_A, _CtinvNC(invN, angles_jnp))
    return _invert_per_pixel(matrix, Op, unobserved_pixels)