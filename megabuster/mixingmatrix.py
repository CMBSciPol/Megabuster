import jax.numpy as jnp
from jaxtyping import Array, Float

from furax.obs.operators import CMBOperator, DustOperator, SynchrotronOperator, mixing_matrix

__all__ = [
    'create_MixingMatrixOperator',
    'create_MixingMatrixOperator_deriv',
    'CMBDerivOperator',
    'DustDerivOperator',
    'SynchrotronDerivOperator',
]

single_patch_indices = {
    'temp_dust_patches': None,
    'beta_dust_patches': None,
    'beta_pl_patches': None,
}
def create_MixingMatrixOperator(frequencies, parameters_dict, in_structure_sed, dust_nu0=150.0, synchrotron_nu0=20.0, patch_indices=None):
    """
    Create the mixing matrix operator using the provided parameters.

    Parameters
    ----------
    frequencies: jnp.ndarray
        Array of frequencies in GHz.
    parameters_dict: dict
        Dictionary of parameters. Must contain the following
        keys: 'temp_dust', 'beta_dust', and 'beta_pl'.
    in_structure_sed: equinox.Structure
        Structure to be provided to the SED object.
    dust_nu0: float (optional)
        Dust reference frequency parameter in GHz. Default is 150.0 GHz.
    synchrotron_nu0: float (optional)
        Synchrotron reference frequency parameter in GHz. Default is 20.0 GHz.

    Returns
    -------
    AbstractLinearOperator
        Mixing matrix operator, mapping a dict of component StokesQU maps to a frequency StokesQU map.

    """
    if patch_indices is None:
        patch_indices = single_patch_indices

    cmb = CMBOperator(frequencies, in_structure=in_structure_sed)
    dust = DustOperator(
        frequencies,
        frequency0=dust_nu0,
        temperature=parameters_dict['temp_dust'],
        beta=parameters_dict['beta_dust'],
        temperature_patch_indices=patch_indices['temp_dust_patches'],
        beta_patch_indices=patch_indices['beta_dust_patches'],
        in_structure=in_structure_sed
    )
    synchrotron = SynchrotronOperator(
        frequencies, 
        frequency0=synchrotron_nu0, 
        beta_pl=parameters_dict['beta_pl'], 
        beta_pl_patch_indices=patch_indices['beta_pl_patches'],
        in_structure=in_structure_sed
    )

    # Mixing matrix operator
    return mixing_matrix(cmb=cmb, dust=dust, synchrotron=synchrotron)

def create_MixingMatrixOperator_deriv(
        frequencies, 
        parameters_dict, 
        in_structure_sed, 
        dust_nu0=150.0, 
        synchrotron_nu0=20.0, 
        patch_indices=None
    ):
    """
    Create the derivatives of the mixing matrix operator with respect to the spectral indices.

    Parameters
    ----------
    frequencies: jnp.ndarray
        Array of frequencies in GHz.
    parameters_dict: dict
        Dictionary of parameters. Must contain the following
        keys: 'temp_dust', 'beta_dust', and 'beta_pl'.
    in_structure_sed: equinox.Structure
        Structure to be provided to the SED object.
    dust_nu0: float (optional)
        Dust reference frequency parameter in GHz. Default is 150.0 GHz.
    synchrotron_nu0: float (optional)
        Synchrotron reference frequency parameter in GHz. Default is 20.0 GHz.

    Returns
    -------
    dict
        Mixing matrix derivative operators, keyed by 'beta_dust' and 'beta_pl'.

    """

    if patch_indices is None:
        patch_indices = single_patch_indices

    cmb_derivative = CMBDerivOperator(
        frequencies=frequencies, 
        in_structure=in_structure_sed,
    )
    dust_derivative = DustDerivOperator(
        frequencies,
        frequency0=dust_nu0,
        temperature=parameters_dict['temp_dust'],
        beta=parameters_dict['beta_dust'],
        temperature_patch_indices=patch_indices['temp_dust_patches'],
        beta_patch_indices=patch_indices['beta_dust_patches'],
        in_structure=in_structure_sed,
    )
    synchrotron_derivative = SynchrotronDerivOperator(
        frequencies, 
        frequency0=synchrotron_nu0, 
        beta_pl=parameters_dict['beta_pl'], 
        beta_pl_patch_indices=patch_indices['beta_pl_patches'],
        in_structure=in_structure_sed
    )

    # Mixing matrix operator
    return {'beta_dust':mixing_matrix(cmb=cmb_derivative, dust=dust_derivative, synchrotron=cmb_derivative),
            'beta_pl':mixing_matrix(cmb=cmb_derivative, dust=cmb_derivative, synchrotron=synchrotron_derivative)}



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
