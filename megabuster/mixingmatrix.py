import jax.numpy as jnp
from jaxtyping import Array, Float

from furax.obs.operators._seds import CMBOperator, DustOperator, SynchrotronOperator, mixing_matrix
from furax.obs.stokes import StokesQU

__all__ = [
    'create_MixingMatrixOperator',
    'create_MixingMatrixOperator_deriv',
    'effective_seds',
    'CMBDerivOperator',
    'DustDerivOperator',
    'SynchrotronDerivOperator',
]

single_patch_indices = {
    'temp_dust_patches': None,
    'beta_dust_patches': None,
    'beta_pl_patches': None,
}
def create_MixingMatrixOperator(frequencies, parameters_dict, in_structure_sed, dust_nu0=150.0, synchrotron_nu0=150.0, patch_indices=None):
    """
    Create the mixing matrix operator of the CMB, dust and synchrotron, using the provided parameters.

    Parameters
    ----------
    frequencies: jnp.ndarray
        Array of frequencies in GHz.
    parameters_dict: dict
        Dictionary of parameters. Must contain the following
        keys: 'temp_dust', 'beta_dust', and 'beta_pl'.
    in_structure_sed: PyTree[jax.ShapeDtypeStruct]
        Structure of the component maps, without frequency axis, e.g. sky_map.structure_for((n_pix,)).
    dust_nu0: float (optional)
        Dust reference frequency parameter in GHz. Default is 150.0 GHz.
    synchrotron_nu0: float (optional)
        Synchrotron reference frequency parameter in GHz. Default is 150.0 GHz.
    patch_indices: dict (optional)
        Patch index of each pixel for the keys 'temp_dust_patches', 'beta_dust_patches' and
        'beta_pl_patches', or None for a parameter uniform over the sky.
        Default is None, which means that all the parameters are uniform over the sky.

    Returns
    -------
    AbstractLinearOperator
        Mixing matrix operator, mapping the component maps {'cmb', 'dust', 'synchrotron'} to the
        frequency maps.

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
        synchrotron_nu0=150.0, 
        patch_indices=None
    ):
    """
    Create the derivatives of the mixing matrix with respect to the spectral indices, using the
    provided parameters.

    Parameters
    ----------
    frequencies: jnp.ndarray
        Array of frequencies in GHz.
    parameters_dict: dict
        Dictionary of parameters. Must contain the following
        keys: 'temp_dust', 'beta_dust', and 'beta_pl'.
    in_structure_sed: PyTree[jax.ShapeDtypeStruct]
        Structure of the component maps, without frequency axis, e.g. sky_map.structure_for((n_pix,)).
    dust_nu0: float (optional)
        Dust reference frequency parameter in GHz. Default is 150.0 GHz.
    synchrotron_nu0: float (optional)
        Synchrotron reference frequency parameter in GHz. Default is 20.0 GHz.
    patch_indices: dict (optional)
        Patch index of each pixel for the keys 'temp_dust_patches', 'beta_dust_patches' and
        'beta_pl_patches', or None for a parameter uniform over the sky.
        Default is None, which means that all the parameters are uniform over the sky.

    Returns
    -------
    dict
        Mixing matrix operators of the derivatives with respect to 'beta_dust' and 'beta_pl'.
        In each of them, the components whose SED does not depend on the parameter have a zero SED.

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


def effective_seds(nu, w, spectral, dust_nu0, synchrotron_nu0, derivatives=False):
    """The SED of each component integrated over a channel, and its derivatives.

    The SEDs are those of the furax operators, integrated against the channel weights:
    `sed_c = sed_c(nu) @ w`, matching `(Bp @ A).reduce()` (and the plain mixing matrix for
    a single frequency of weight 1). The components are ordered cmb, dust, synchrotron.
    """
    structure = StokesQU.structure_for((1,))
    cmb = CMBOperator(nu, in_structure=structure)
    dust = DustOperator(
        nu,
        frequency0=dust_nu0,
        temperature=spectral['temp_dust'],
        beta=spectral['beta_dust'],
        in_structure=structure,
    )
    synchrotron = SynchrotronOperator(
        nu,
        frequency0=synchrotron_nu0,
        beta_pl=spectral['beta_pl'],
        in_structure=structure,
    )
    sed = jnp.stack([op._diagonal[:, 0] @ w for op in (cmb, dust, synchrotron)])
    if not derivatives:
        return sed, None, None
    # the dust and synchrotron SEDs are proportional to (nu / nu0)^beta:
    # their derivatives with respect to beta are log(nu / nu0) * SED
    zero = jnp.zeros(())
    log_nu_dust = jnp.log(nu / dust_nu0)
    log_nu_sync = jnp.log(nu / synchrotron_nu0)
    dust_derivative = jnp.stack([zero, (log_nu_dust * dust._diagonal[:, 0]) @ w, zero])
    synchrotron_derivative = jnp.stack([zero, zero, (log_nu_sync * synchrotron._diagonal[:, 0]) @ w])
    return sed, dust_derivative, synchrotron_derivative



class CMBDerivOperator(CMBOperator):
    """
    Operator for the derivative of the Cosmic Microwave Background (CMB) spectral energy
    distribution with respect to the spectral parameters.

    The CMB SED does not depend on the spectral parameters, so that its derivative is zero. It is
    also used for the components which do not depend on the parameter of a derivative.

    Args:
        frequencies (Array): Array of frequencies in GHz.
        in_structure (PyTree): Input structure describing the shape and dtype of the input.
        units (str, optional): Units for the operator ('K_CMB' or 'K_RJ'). Defaults to 'K_CMB'.

    Example:
        >>> import jax.numpy as jnp
        >>> from furax.obs.landscapes import HealpixLandscape
        >>> from megabuster.mixingmatrix import CMBDerivOperator
        >>> nu = jnp.array([30.0, 40.0, 100.0])  # Frequencies in GHz
        >>> landscape = HealpixLandscape(nside=8, stokes='QU')
        >>> sky_map = landscape.ones()
        >>> cmbDerivOp = CMBDerivOperator(
        ...     frequencies=nu,
        ...     in_structure=landscape.structure,
        ...     units='K_CMB',
        ... )
        >>> result = cmbDerivOp(sky_map)
        >>> result.q.shape
        (3, 768)
        >>> bool(jnp.all(result.q == 0))
        True
    """

    def sed(self) -> Float[Array, 'freq 1']:
        """
        Compute the derivative of the spectral energy distribution of the CMB, which is zero.

        Returns:
            Float[Array, 'freq 1']: The derivative of the SED of the CMB.
        """
        return jnp.zeros_like(self.frequencies)[:, None]


class DustDerivOperator(DustOperator):
    """
    Operator for the derivative of the dust spectral energy distribution with respect to the
    spectral index beta.

    The dust SED is proportional to (nu / nu0)^beta, so that its derivative is
    log(nu / nu0) * SED.

    Args:
        frequencies (Array): Array of frequencies in GHz.
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

    Example:
        >>> import jax.numpy as jnp
        >>> from furax.obs.landscapes import HealpixLandscape
        >>> from megabuster.mixingmatrix import DustDerivOperator
        >>> nu = jnp.array([30.0, 40.0, 100.0])  # Frequencies in GHz
        >>> landscape = HealpixLandscape(nside=8, stokes='QU')
        >>> beta_dust = 1.54  # Spectral index
        >>> temperature = 20.0  # Dust temperature
        >>> sky_map = landscape.ones()
        >>> dustDerivOp = DustDerivOperator(
        ...     frequencies=nu,
        ...     frequency0=150.0,
        ...     beta=beta_dust,
        ...     temperature=temperature,
        ...     in_structure=landscape.structure,
        ...     units='K_CMB',
        ... )
        >>> result = dustDerivOp(sky_map)
        >>> result.q.shape
        (3, 768)
    """

    def sed(self) -> Float[Array, 'freq pix'] | Float[Array, 'freq 1']:
        # the SED is proportional to (nu / nu0)^beta: its derivative is log(nu / nu0) * SED
        nu = self.frequencies[:, None]
        return jnp.log(nu / self.frequency0) * super().sed()


class SynchrotronDerivOperator(SynchrotronOperator):
    """Operator for the derivative of the synchrotron Spectral Energy Distribution (SED) with
    respect to the spectral index beta_pl.

    The synchrotron emission is modelled by a power-law SED with optional running of the spectral
    index, proportional to (nu / nu0)^(beta_pl + running * log(nu / nu_pivot)), so that its
    derivative is log(nu / nu0) * SED.

    Attributes:
        beta_pl (Float[Array, '...']): Power-law spectral index values.
        beta_pl_patch_indices (Int[Array, '...'] | None):
             Optional indices for patch-specific beta values.
        nu_pivot (float): Pivot frequency in GHz for the running spectral index. Default is 1.0 GHz.
        running (float): Running of the spectral index. Default is 0.0.
        units (str): Output unit for the operator, either 'K_CMB' or 'K_RJ'.

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
        >>> import jax.numpy as jnp
        >>> from furax.obs.landscapes import HealpixLandscape
        >>> from megabuster.mixingmatrix import SynchrotronDerivOperator
        >>> nu = jnp.array([30.0, 40.0, 100.0])  # Frequencies in GHz
        >>> landscape = HealpixLandscape(nside=8, stokes='QU')
        >>> sky_map = landscape.ones()
        >>> beta_pl = -3.0  # Spectral index
        >>> synchrotron_deriv_operator = SynchrotronDerivOperator(
        ...     frequencies=nu,
        ...     frequency0=20.0,
        ...     beta_pl=beta_pl,
        ...     in_structure=landscape.structure,
        ...     units='K_CMB',
        ... )
        >>> result = synchrotron_deriv_operator(sky_map)
        >>> result.q.shape
        (3, 768)
    """

    def sed(self) -> Float[Array, 'freq pix'] | Float[Array, 'freq 1']:
        # the SED is proportional to (nu / nu0)^(beta_pl + running * log(nu / nu_pivot)):
        # its derivative is log(nu / nu0) * SED
        nu = self.frequencies[:, None]
        return jnp.log(nu / self.frequency0) * super().sed()