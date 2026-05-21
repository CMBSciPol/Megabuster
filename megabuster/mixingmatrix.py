import jax.numpy as jnp
from astropy.cosmology import Planck15
from jaxtyping import Array, Float, Int, PyTree
from scipy import constants

from furax.obs import CMBOperator, DustOperator, SynchrotronOperator
from furax.obs.operators._seds import MixingMatrixOperator, _H_OVER_K_GHZ, CMBOperator, DustOperator, SynchrotronOperator

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
    Create a MixingMatrixOperator object using the provided parameters.

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
    MixingMatrixOperator
        Mixing matrix operator object.

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
    return MixingMatrixOperator(cmb=cmb, dust=dust, synchrotron=synchrotron)

def create_MixingMatrixOperator_deriv(
        frequencies, 
        parameters_dict, 
        in_structure_sed, 
        dust_nu0=150.0, 
        synchrotron_nu0=20.0, 
        patch_indices=None
    ):
    """
    Create a MixingMatrixOperator object using the provided parameters.

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
    MixingMatrixOperator
        Mixing matrix operator object.

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
    return {'beta_dust':MixingMatrixOperator(cmb=cmb_derivative, dust=dust_derivative, synchrotron=cmb_derivative),
            'beta_pl':MixingMatrixOperator(cmb=cmb_derivative, dust=cmb_derivative, synchrotron=synchrotron_derivative)}



class CMBDerivOperator(CMBOperator):
    """
    Operator for Cosmic Microwave Background (CMB) spectral energy distribution.

    Args:
        frequencies (Array): Array of frequencies.
        in_structure (PyTree): Input structure describing the shape and dtype of the input.
        units (str, optional): Units for the operator ('K_CMB' or 'K_RJ'). Defaults to 'K_CMB'.

    Example:
        >>> from furax.operators.seds import CMBOperator
        >>> import jax.numpy as jnp
        >>> nu = jnp.array([30, 40, 100])  # Frequencies in GHz
        >>> in_structure = ...  # Define input structure (e.g., using HealpixLandscape)
        >>> sky_map = ...  # Define sky map
        >>> cmbOp = CMBOperator(
        ...     frequencies=nu,
        ...     in_structure=in_structure,
        ...     units='K_CMB',
        ... )
        >>> result = cmbOp(sky_map)
        >>> print(result)
    """

    def __init__(self, *args, **keywords):
        super().__init__(*args, **keywords)

    def sed(self) -> Float[Array, '...']:
        """
        Compute the spectral energy distribution for the CMB.

        Returns:
            Float[Array, '...']: The SED for the CMB.
        """
        return jnp.zeros_like(self.frequencies) / jnp.expand_dims(self.factor, axis=-1)


class DustDerivOperator(DustOperator):
    """
    Operator for dust spectral energy distribution.

    Args:
        frequencies (Array): Array of frequencies.
        frequency0 (float, optional): Reference frequency. Defaults to 100.
        temperature (float | Array): Dust temperature.
        units (str, optional): Units for the operator ('K_CMB' or 'K_RJ'). Defaults to 'K_CMB'.
        temperature_patch_indices (Array | None, optional): Indices for patch-based temperature.
        beta (float | Array): Spectral index beta.
        beta_patch_indices (Array | None, optional): Indices for patch-based beta.
        in_structure (PyTree): Input structure.

    Attributes:
        temperature (Array): Dust temperature.
        beta (Array): Spectral index beta.
        factor (Array | float): Conversion factor based on the unit type.

    Example:
        >>> from furax.operators.seds import SynchrotronOperator
        >>> import jax.numpy as jnp
        >>> nu = jnp.array([30, 40, 100])  # Frequencies in GHz
        >>> in_structure = ...  # Define input structure (e.g., using HealpixLandscape)
        >>> beta_dust = 1.54  # Spectral index
        >>> temperature = 20.0  # Dust temperature
        >>> sky_map = ...  # Define sky map
        >>> dustOperator = SynchrotronOperator(
        ...     frequencies=nu,
        ...     frequency0=20.0,
        ...     beta=beta_dust,
        ...     temperature=temperature
        ...     in_structure=in_structure,
        ...     units='K_CMB',
        ... )
        >>> result = dustOperator(sky_map)
        >>> print(result)
    """

    def __init__(self, *args, **keywords):
        super().__init__(*args, **keywords)

    def sed(self) -> Float[Array, '...']:
        t = self._get_at(
            jnp.expm1(self.frequency0 / self.temperature * _H_OVER_K_GHZ)
            / jnp.expm1(self.frequencies / self.temperature * _H_OVER_K_GHZ),
            self.temperature_patch_indices,
        )
        b = self._get_at(
            jnp.log(self.frequencies / self.frequency0) * (self.frequencies / self.frequency0) ** (1 + self.beta), self.beta_patch_indices
        )
        sed = (t * b) * jnp.expand_dims(self.factor, axis=-1)
        return sed


class SynchrotronDerivOperator(SynchrotronOperator):
    """Spectral Energy Distribution (SED) operator for synchrotron emission.

    This operator models synchrotron emission based on a power-law SED
    with optional running of the spectral index.

    Attributes:
        beta_pl (Float[Array, '...']): Power-law spectral index values.
        beta_pl_patch_indices (Int[Array, '...'] | None):
             Optional indices for patch-specific beta values.
        nu_pivot (float): Pivot frequency in GHz for the running spectral index. Default is 1.0 GHz.
        running (float): Running of the spectral index. Default is 0.0.
        units (str): Output unit for the operator, either 'K_CMB' or 'K_RJ'.
        factor (Float[Array, '...'] | float): Conversion factor between units.

    Args:
        frequencies (Float[Array, '...']): Frequencies in GHz.
        frequency0 (float): Reference frequency for the SED. Default is 100 GHz.
        nu_pivot (float): Pivot frequency for running spectral index. Default is 1.0 GHz.
        running (float): Running of the spectral index. Default is 0.0.
        units (str): Units of the output, either 'K_CMB' or 'K_RJ'. Default is 'K_CMB'.
        beta_pl (float | Float[Array, '...']):
            Power-law spectral index or an array of indices.
        beta_pl_patch_indices (Int[Array, '...'] | None):
            Optional indices for patch-specific beta values.
        in_structure (PyTree[jax.ShapeDtypeStruct]): Input structure defining the shape of the data.

    Example:
        >>> from furax.operators.seds import SynchrotronOperator
        >>> import jax.numpy as jnp
        >>> nu = jnp.array([30, 40, 100])  # Frequencies in GHz
        >>> in_structure = ...  # Define input structure (e.g., using HealpixLandscape)
        >>> sky_map = ...  # Define sky map
        >>> beta_pl = -3.0  # Spectral index
        >>> synchrotron_operator = SynchrotronOperator(
        ...     frequencies=nu,
        ...     frequency0=20.0,
        ...     beta_pl=beta_pl,
        ...     in_structure=in_structure,
        ...     units='K_CMB',
        ... )
        >>> result = synchrotron_operator(sky_map)
        >>> print(result)
    """

    def __init__(self, *args, **keywords):
        super().__init__(*args, **keywords)

    def sed(self) -> Float[Array, '...']:
        sed = self._get_at(
            (
                (self.frequencies / self.frequency0)
                ** (self.beta_pl + self.running * jnp.log(self.frequencies / self.nu_pivot))
            ),
            self.beta_pl_patch_indices,
        )

        sed = self._get_at(
            jnp.log(self.frequencies / self.frequency0) * (self.frequencies / self.frequency0) ** (self.beta_pl), self.beta_pl_patch_indices
        )
        sed *= jnp.expand_dims(self.factor, axis=-1)

        return sed