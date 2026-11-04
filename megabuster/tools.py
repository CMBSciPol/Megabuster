import numpy as np
import healpy as hp
import jax
import jax.numpy as jnp
import jax_healpy as jhp
from jaxtyping import ArrayLike

from furax.core import DiagonalOperator, BlockColumnOperator, BlockRowOperator, BlockDiagonalOperator, BroadcastDiagonalOperator, DenseBlockDiagonalOperator
from furax.obs.stokes import StokesQU
from furax import AbstractLinearOperator, symmetric

__all__ = [
    'get_maps_from_Stokes',
    'get_diagonal_operator_from_stokes_maps',
    'get_A_from_array',
    'get_preconditioner',
    'get_healpix_indices_patch_from_mask',
    'HarmonicPreconditionerOperator',
    'HybridPreconditionerOperator'
]

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
    applies the diagonal operator to a StokesQU object.

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
        The diagonal operator acting on the StokesQU object, which multiplies its Q and U
        components by stokes_maps[..., 0, :] and stokes_maps[..., 1, :].
    """

    assert len(stokes_maps.shape) >= 2, "stokes_maps must have shape (..., n_stokes, n_pix)"

    # Move the Stokes axis in front, as in the StokesQU backing array
    return DiagonalOperator(jnp.moveaxis(jnp.asarray(stokes_maps), -2, 0), in_structure=in_structure)

def get_preconditioner(preconditioner_matrix, in_structure):
    """
    Given a preconditioner matrix in terms of components, return a BlockColumnOperator that
    applies the preconditioner to a StokesQU object.
    
    in_structure must be as_structure(AOND['cmb'])

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

class StokesLinearOperator(AbstractLinearOperator):
    stokes_maps: jnp.ndarray

    def mv(self, x):
        #print('self.stokes_maps[:, :, 0]:', self.stokes_maps[:, :, 0])
        q = self.stokes_maps[0, 0] * x.q + self.stokes_maps[0, 1] * x.u
        u = self.stokes_maps[1, 0] * x.q + self.stokes_maps[1, 1] * x.u
        return StokesQU(q, u)

def get_diagonal_operator_from_stokes_maps2(stokes_maps, in_structure):
    assert stokes_maps.shape[0] == 2 and stokes_maps.shape[1] == 2
    return StokesLinearOperator(stokes_maps=stokes_maps, in_structure=in_structure)
    
def get_preconditioner2(preconditioner_matrix, in_structure):
    """
    Given a preconditioner matrix in terms of components, return a BlockColumnOperator that
    applies the preconditioner to a StokesQU object.
    
    in_structure must be as_structure(AOND['cmb'])

    Parameters
    ----------
    preconditioner_matrix : np.ndarray
        The preconditioner matrix in terms of components, of shape
        (n_comp, n_comp, n_stokes, n_stokes, n_pix).
    in_structure : StokesQU
        The structure of the StokesQU object of a component.

    Returns
    -------
    BlockColumnOperator
        The preconditioner as a BlockColumnOperator expressed with Pytrees using dictionaries of {'cmb', 'dust', 'synchrotron'}.
    """

    return BlockColumnOperator({
        component_0: BlockRowOperator({
            component_1: BlockDiagonalOperator(
                get_diagonal_operator_from_stokes_maps2(
                    preconditioner_matrix[num_cpt_0, num_cpt_1, :, :, :],  # (n_stokes, n_stokes, n_pix),
                    in_structure
                )
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

    # The StokesQU object stores Q and U stacked on its leading axis, of shape (n_stokes, n_freq, n_pix):
    # out[s, f, q] = sum_{t, p} matrix[f, s, t, q, p] * x[t, f, p]
    in_structure = StokesQU.structure_for((matrix.shape[0], matrix.shape[-1]), matrix.dtype)
    return DenseBlockDiagonalOperator(matrix, in_structure=in_structure, subscripts='fstqp,tfp->sfq')

def get_A_from_array(matrix_A, in_structure_sed):
    """
    Build a mixing matrix operator from an array of shape (n_comp, n_freq) or (n_comp, n_freq, n_pix),
    applied as the furax SED operators.
    """
    if matrix_A.ndim == 2:
        matrix_to_build = matrix_A[..., None]
    else:
        matrix_to_build = matrix_A
    return BlockRowOperator(
        {component:BroadcastDiagonalOperator(
            matrix_to_build[num_cpt,...],
            axis_destination=(-2, -1),
            insert_axes=-2,
            in_structure=in_structure_sed,
        ) for num_cpt, component in enumerate(['cmb', 'dust', 'synchrotron'])}
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

@symmetric
class HarmonicPreconditionerOperator(AbstractLinearOperator):
    _M_inv: jnp.ndarray          # (lmax+1, n_comp, n_comp)
    _nside: int
    _lmax: int
    _n_comp_names: list
    _unobserved_pixels: jnp.ndarray
    def mv(self, x):
        lmax = self._lmax
        L = lmax + 1
        n_comp = len(self._n_comp_names)

        x_Q = jnp.stack([x[c].q for c in self._n_comp_names], axis=0)  # (n_comp, n_pix)
        x_U = jnp.stack([x[c].u for c in self._n_comp_names], axis=0)

        # map2alm retourne (L, 2L-1) en format s2fft
        alms_Q = jax.vmap(lambda m: jhp.map2alm(m, lmax=lmax))(x_Q)  # (n_comp, L, 2L-1)
        alms_U = jax.vmap(lambda m: jhp.map2alm(m, lmax=lmax))(x_U)

        def apply_M_inv(alms):
            # alms : (n_comp, L, 2L-1)
            # Pour chaque ell, M_inv[ell] est (n_comp, n_comp)
            # On applique M_inv[ell] sur alms[:, ell, :] pour chaque ell
            # En vectorisant sur ell via vmap
            
            def apply_one_ell(M_inv_ell, alms_ell, valid_ell):
                # M_inv_ell : (n_comp, n_comp)
                # alms_ell : (n_comp, 2L-1)
                # valid_ell : (2L-1,) booléen
                rotated = M_inv_ell @ alms_ell  # (n_comp, 2L-1)
                return jnp.where(valid_ell[None, :], rotated, 0.0)

            # Construire le masque de validité pour chaque ell : shape (L, 2L-1)
            m_range = jnp.arange(2 * L - 1)  # 0..2L-2
            ell_range = jnp.arange(L)         # 0..L-1
            # m valide si L-1-ell <= m_idx <= L-1+ell
            valid = (m_range[None, :] >= (L - 1 - ell_range[:, None])) & \
                    (m_range[None, :] <= (L - 1 + ell_range[:, None]))
            # valid : (L, 2L-1)

            # vmap sur ell
            alms_out = jax.vmap(apply_one_ell)(
                self._M_inv,   # (L, n_comp, n_comp)
                alms.transpose(1, 0, 2),  # (L, n_comp, 2L-1)
                valid          # (L, 2L-1)
            )
            # alms_out : (L, n_comp, 2L-1) -> retransposer en (n_comp, L, 2L-1)
            return alms_out.transpose(1, 0, 2)

        alms_Q_out = apply_M_inv(alms_Q)
        alms_U_out = apply_M_inv(alms_U)

        # alm2map attend (L, 2L-1) et nside
        out_Q = jnp.real(jax.vmap(lambda alm: jhp.alm2map(alm, nside=self._nside, lmax=lmax))(alms_Q_out))
        out_U = jnp.real(jax.vmap(lambda alm: jhp.alm2map(alm, nside=self._nside, lmax=lmax))(alms_U_out))

        if self._unobserved_pixels is not None:
            out_Q = out_Q.at[:, self._unobserved_pixels].set(0.0)
            out_U = out_U.at[:, self._unobserved_pixels].set(0.0)

        return {
            c: StokesQU(out_Q[i], out_U[i])
            for i, c in enumerate(self._n_comp_names)
        }
@symmetric
class HybridPreconditionerOperator(AbstractLinearOperator):
    _P_sqrt_inv: jnp.ndarray   # (n_pix, n_comp, n_comp)
    _M_inv: jnp.ndarray        # (lmax+1, n_comp, n_comp)
    _nside: int
    _lmax: int
    _n_comp_names: list
    _unobserved_pixels: jnp.ndarray

    def mv(self, x):
        lmax = self._lmax
        L = lmax + 1

        def apply_P_sqrt_inv(x):
            # x : dict de StokesQU
            # Pour Q et U séparément
            x_Q = jnp.stack([x[c].q for c in self._n_comp_names], axis=0)  # (n_comp, n_pix)
            x_U = jnp.stack([x[c].u for c in self._n_comp_names], axis=0)

            # P_sqrt_inv : (n_pix, n_comp, n_comp)
            # out[c, p] = sum_k P_sqrt_inv[p, c, k] * x[k, p]
            out_Q = jnp.einsum('pck, kp -> cp', self._P_sqrt_inv, x_Q)
            out_U = jnp.einsum('pck, kp -> cp', self._P_sqrt_inv, x_U)

            return {c: StokesQU(out_Q[i], out_U[i]) for i, c in enumerate(self._n_comp_names)}

        def apply_M_inv_harm(x):
            x_Q = jnp.stack([x[c].q for c in self._n_comp_names], axis=0)
            x_U = jnp.stack([x[c].u for c in self._n_comp_names], axis=0)

            alms_Q = jax.vmap(lambda m: jhp.map2alm(m, lmax=lmax))(x_Q)
            alms_U = jax.vmap(lambda m: jhp.map2alm(m, lmax=lmax))(x_U)

            m_range = jnp.arange(2 * L - 1)
            ell_range = jnp.arange(L)
            valid = (m_range[None, :] >= (L - 1 - ell_range[:, None])) & \
                    (m_range[None, :] <= (L - 1 + ell_range[:, None]))

            def apply_one_ell(M_inv_ell, alms_ell, valid_ell):
                rotated = M_inv_ell @ alms_ell
                return jnp.where(valid_ell[None, :], rotated, 0.0)

            def apply_M(alms):
                alms_out = jax.vmap(apply_one_ell)(
                    self._M_inv,
                    alms.transpose(1, 0, 2),
                    valid
                )
                return alms_out.transpose(1, 0, 2)

            alms_Q_out = apply_M(alms_Q)
            alms_U_out = apply_M(alms_U)

            out_Q = jnp.real(jax.vmap(lambda alm: jhp.alm2map(alm, nside=self._nside, lmax=lmax))(alms_Q_out))
            out_U = jnp.real(jax.vmap(lambda alm: jhp.alm2map(alm, nside=self._nside, lmax=lmax))(alms_U_out))

            if self._unobserved_pixels is not None:
                out_Q = out_Q.at[:, self._unobserved_pixels].set(0.0)
                out_U = out_U.at[:, self._unobserved_pixels].set(0.0)

            return {c: StokesQU(out_Q[i], out_U[i]) for i, c in enumerate(self._n_comp_names)}

        # P^{-1} x = P_sqrt_inv @ M_inv_harm @ P_sqrt_inv @ x
        x1 = apply_P_sqrt_inv(x)
        x2 = apply_M_inv_harm(x1)
        x3 = apply_P_sqrt_inv(x2)
        return x3