import jax.numpy as jnp
from jaxtyping import Array, Float

from furax import BlockRowOperator, DiagonalOperator
from furax.obs.operators import CMBOperator, DustOperator, SynchrotronOperator

__all__ = [
    'member_mixing_operator',
    'member_mixing_operator_deriv',
    'CMBDerivOperator',
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
    # the SED operators hold a diagonal of shape (1, 1), or (1, n_pix) with patch indices
    return BlockRowOperator(
        {
            component: DiagonalOperator(operator._diagonal[0], in_structure=structure)
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

    cmb = CMBOperator(nu, in_structure=structure)
    dust = DustOperator(
        nu,
        frequency0=dust_nu0,
        temperature=parameters_dict['temp_dust'],
        beta=parameters_dict['beta_dust'],
        temperature_patch_indices=patch_indices['temp_dust_patches'],
        beta_patch_indices=patch_indices['beta_dust_patches'],
        in_structure=structure,
    )
    synchrotron = SynchrotronOperator(
        nu,
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
    each derivative operator keeps a single non-zero block.
    """
    if patch_indices is None:
        patch_indices = single_patch_indices

    zero = CMBDerivOperator(frequencies=nu, in_structure=structure)
    dust = DustDerivOperator(
        nu,
        frequency0=dust_nu0,
        temperature=parameters_dict['temp_dust'],
        beta=parameters_dict['beta_dust'],
        temperature_patch_indices=patch_indices['temp_dust_patches'],
        beta_patch_indices=patch_indices['beta_dust_patches'],
        in_structure=structure,
    )
    synchrotron = SynchrotronDerivOperator(
        nu,
        frequency0=synchrotron_nu0,
        beta_pl=parameters_dict['beta_pl'],
        beta_pl_patch_indices=patch_indices['beta_pl_patches'],
        in_structure=structure,
    )
    return {
        'beta_dust': _member_block_row(
            {'cmb': zero, 'dust': dust, 'synchrotron': zero}, structure
        ),
        'beta_pl': _member_block_row(
            {'cmb': zero, 'dust': zero, 'synchrotron': synchrotron}, structure
        ),
    }



class CMBDerivOperator(CMBOperator):
    """
    Zero operator with the structure of the CMB SED operator, used as the derivative of the
    CMB SED with respect to any foreground spectral parameter.
    """

    def sed(self) -> Float[Array, 'freq 1']:
        return jnp.zeros_like(super().sed())


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
