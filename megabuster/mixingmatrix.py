import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from furax import BlockColumnOperator, BlockRowOperator, DiagonalOperator, ZeroOperator
from furax.core import StackedEnsemble, lift
from furax.obs.operators import CMBOperator, DustOperator, SynchrotronOperator

from megabuster.distributed import shard_ensemble

__all__ = [
    'create_mixing_matrix_operator',
    'create_mixing_matrix_operator_deriv',
    'member_mixing_operator',
    'member_mixing_operator_deriv',
    'DustDerivOperator',
    'SynchrotronDerivOperator',
]

single_patch_indices = {
    'temp_dust_patches': None,
    'beta_dust_patches': None,
    'beta_pl_patches': None,
}
def _member_block_row(operators, structure):
    """The one-channel operator: the SED diagonals of `operators`, as a BlockRow."""
    # the SED operators hold a diagonal of shape (1, 1), or (1, n_pix) with patch indices;
    # a ZeroOperator has none and goes through unchanged
    return BlockRowOperator(
        {
            component: operator
            if isinstance(operator, ZeroOperator)
            else DiagonalOperator(operator._diagonal[0], in_structure=structure)
            for component, operator in operators.items()
        }
    )


def member_mixing_operator(
    nu, parameters_dict, structure, dust_nu0, synchrotron_nu0, patch_indices
):
    """The mixing matrix of one channel: component maps -> the channel's map.

    The SEDs are those of the furax operators at the channel's frequency; the operator maps
    `{'cmb', 'dust', 'synchrotron'}` component maps of the member structure to a single
    channel map, without a frequency axis.
    """
    if patch_indices is None:
        patch_indices = single_patch_indices

    cmb = CMBOperator(frequencies=nu, in_structure=structure)
    dust = DustOperator(
        frequencies=nu,
        frequency0=dust_nu0,
        temperature=parameters_dict['temp_dust'],
        beta=parameters_dict['beta_dust'],
        temperature_patch_indices=patch_indices['temp_dust_patches'],
        beta_patch_indices=patch_indices['beta_dust_patches'],
        in_structure=structure,
    )
    synchrotron = SynchrotronOperator(
        frequencies=nu,
        frequency0=synchrotron_nu0,
        beta_pl=parameters_dict['beta_pl'],
        beta_pl_patch_indices=patch_indices['beta_pl_patches'],
        in_structure=structure,
    )
    return _member_block_row({'cmb': cmb, 'dust': dust, 'synchrotron': synchrotron}, structure)


def member_mixing_operator_deriv(
    nu, parameters_dict, structure, dust_nu0, synchrotron_nu0, patch_indices
):
    """The derivatives of the mixing matrix of one channel, keyed by spectral parameter.

    The derivative of the CMB SED with respect to any foreground spectral index is zero, so
    each derivative operator keeps a single non-zero block: the others are `ZeroOperator`s,
    whose application XLA eliminates entirely.
    """
    if patch_indices is None:
        patch_indices = single_patch_indices

    zero = ZeroOperator(in_structure=structure)
    dust_derivative = DustDerivOperator(
        frequencies=nu,
        frequency0=dust_nu0,
        temperature=parameters_dict['temp_dust'],
        beta=parameters_dict['beta_dust'],
        temperature_patch_indices=patch_indices['temp_dust_patches'],
        beta_patch_indices=patch_indices['beta_dust_patches'],
        in_structure=structure,
    )
    synchrotron_derivative = SynchrotronDerivOperator(
        frequencies=nu,
        frequency0=synchrotron_nu0,
        beta_pl=parameters_dict['beta_pl'],
        beta_pl_patch_indices=patch_indices['beta_pl_patches'],
        in_structure=structure,
    )
    return {
        'beta_dust': _member_block_row(
            {'cmb': zero, 'dust': dust_derivative, 'synchrotron': zero}, structure
        ),
        'beta_pl': _member_block_row(
            {'cmb': zero, 'dust': zero, 'synchrotron': synchrotron_derivative}, structure
        ),
    }


def _channel_ensemble(
    member_operator,
    frequencies,
    parameters_dict,
    in_structure_sed,
    dust_nu0,
    synchrotron_nu0,
    patch_indices,
    axis,
):
    """The ensemble of the per-channel operators, laid out over the mesh axis, if any."""
    frequencies = np.asarray(frequencies, dtype=np.float64)
    channels = StackedEnsemble(frequencies.size, {'nu': frequencies[:, None]})
    return shard_ensemble(
        lift(
            lambda member: member_operator(
                member['nu'],
                parameters_dict,
                in_structure_sed,
                dust_nu0,
                synchrotron_nu0,
                patch_indices,
            ),
            channels,
        ),
        axis,
    )


def create_mixing_matrix_operator(
    frequencies,
    parameters_dict,
    in_structure_sed,
    dust_nu0=150.0,
    synchrotron_nu0=20.0,
    patch_indices=None,
    axis=None,
):
    """
    Create the mixing matrix operator, as a column of per-channel blocks.

    The blocks are the members of a furax ensemble, laid out over the devices of the mesh axis
    `axis`, if any. The block column is built with `axes=1`, so that its output keeps the
    layout of the multi-frequency maps, `(n_stokes, n_freq, n_pix)`.

    Parameters
    ----------
    frequencies: ndarray
        Array of frequencies in GHz.
    parameters_dict: dict
        Dictionary of parameters. Must contain the following
        keys: 'temp_dust', 'beta_dust', and 'beta_pl'.
    in_structure_sed: equinox.Structure
        Structure of the component maps, which is also the one of a single-channel map.
    dust_nu0: float (optional)
        Dust reference frequency parameter in GHz. Default is 150.0 GHz.
    synchrotron_nu0: float (optional)
        Synchrotron reference frequency parameter in GHz. Default is 20.0 GHz.
    patch_indices: dict (optional)
        Patch index of each pixel for the keys 'temp_dust_patches', 'beta_dust_patches' and
        'beta_pl_patches', or None for a parameter uniform over the sky.
    axis: str (optional)
        Mesh axis over which the channels are distributed. Default is None, a single device.

    Returns
    -------
    BlockColumnOperator
        Mixing matrix operator, mapping a dict of component maps to a multi-frequency map.
    """
    return BlockColumnOperator(
        _channel_ensemble(
            member_mixing_operator,
            frequencies,
            parameters_dict,
            in_structure_sed,
            dust_nu0,
            synchrotron_nu0,
            patch_indices,
            axis,
        ),
        axes=1,
    )


def create_mixing_matrix_operator_deriv(
    frequencies,
    parameters_dict,
    in_structure_sed,
    dust_nu0=150.0,
    synchrotron_nu0=20.0,
    patch_indices=None,
    axis=None,
):
    """
    Create the derivatives of the mixing matrix operator with respect to the spectral indices.

    Each derivative is a column of per-channel blocks, built as the mixing matrix operator of
    `create_mixing_matrix_operator`, with which it shares its parameters.

    Returns
    -------
    dict
        Mixing matrix derivative operators, keyed by 'beta_dust' and 'beta_pl'.
    """
    members = _channel_ensemble(
        member_mixing_operator_deriv,
        frequencies,
        parameters_dict,
        in_structure_sed,
        dust_nu0,
        synchrotron_nu0,
        patch_indices,
        axis,
    )
    return {
        key: BlockColumnOperator(lift(lambda member, key=key: member[key], members), axes=1)
        for key in ('beta_dust', 'beta_pl')
    }



class DustDerivOperator(DustOperator):
    """
    Derivative of the dust SED operator with respect to the spectral index beta:
    d SED / d beta = log(nu / nu0) * SED.

    Takes the same arguments as furax `DustOperator`.
    """

    def sed(self) -> Float[Array, 'freq pix'] | Float[Array, 'freq 1']:
        return jnp.log(self.frequencies / self.frequency0)[:, None] * super().sed()


class SynchrotronDerivOperator(SynchrotronOperator):
    """
    Derivative of the synchrotron SED operator with respect to the spectral index beta_pl:
    d SED / d beta_pl = log(nu / nu0) * SED.

    Takes the same arguments as furax `SynchrotronOperator`.
    """

    def sed(self) -> Float[Array, 'freq pix'] | Float[Array, 'freq 1']:
        return jnp.log(self.frequencies / self.frequency0)[:, None] * super().sed()
