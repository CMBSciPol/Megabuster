"""Bandpass operator: integrates frequency maps over the bandpass of each channel."""

from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Float, PyTree

from furax import AbstractLinearOperator, BlockRowOperator, BroadcastDiagonalOperator
from furax.core import TransposeOperator
from furax.core.rules import AbstractCompositionRule
from furax.obs.operators._seds import AbstractSEDOperator, _check_units, k_rj_to_k_cmb

__all__ = ['BandpassOperator']


class BandpassOperator(AbstractLinearOperator):
    r"""Operator integrating frequency maps over the bandpass of each channel.

    In the data model $d = B A s + n$, the mixing matrix $A$ evaluates the sky at the integration
    frequencies $\nu_k$, and $B$ integrates it over the transmission $B_c(\nu)$ of each channel $c$
    with the trapezoidal rule:

    $$
    d_c = \int B_c(\nu)\, (A s)_\nu\, d\nu \approx \sum_k W_{ck}\, (A s)_{\nu_k},
    \qquad W_{ck} = B_c(\nu_k)\, \Delta\nu_k,
    $$

    where $\Delta\nu_k$ are the trapezoidal weights of the integration frequencies. The
    transmission is neither normalized nor converted to other units: it must be expressed for the
    units of $A s$.

    The frequency axis is the second-to-last axis of each Stokes component, e.g. `(n_nu, n_pix)`,
    and the output has the channel axis in its place, `(n_channels, n_pix)`.

    Use `BandpassOperator.create(...)` to build it from a bandpass function, or
    `BandpassOperator.create_from_samples(...)` to build it from sampled bandpasses with the
    convention of PySM.

    Algebraic rules: once reduced, the composition of $B$ with an SED operator, or with a mixing
    matrix made of SED operators, is a diagonal operator with the SED integrated over each
    bandpass, $W \, \text{SED}$. Applying `(B @ A).reduce()` then costs one multiplication per
    channel instead of one per integration frequency.

    Attributes:
        weights: Integration weights $W_{ck}$, of shape `(n_channels, n_nu)`.
    """

    weights: Float[Array, 'channel nu']

    def __post_init__(self) -> None:
        super().__post_init__()
        n_nu = self.weights.shape[1]
        for leaf in jax.tree.leaves(self.in_structure):
            if len(leaf.shape) < 2 or leaf.shape[-2] != n_nu:
                raise ValueError(
                    f'The input maps must have {n_nu} integration frequencies on their '
                    f'second-to-last axis, got shape {leaf.shape}.'
                )

    @classmethod
    def create(
        cls,
        bandpass: Callable[[Float[Array, ' nu'], float], Float[Array, ' nu']],
        frequencies: Float[Array, ' nu'],
        channel_frequencies: Float[Array, ' channel'],
        in_structure: PyTree[jax.ShapeDtypeStruct],
    ) -> 'BandpassOperator':
        """Create the operator from a bandpass function.

        Args:
            bandpass: Transmission of a channel, called as `bandpass(nu, nu_ref)` with the
                integration frequencies and the reference frequency of the channel [GHz].
            frequencies: Integration frequencies, in increasing order [GHz].
            channel_frequencies: Reference frequency of each channel [GHz].
            in_structure: Structure of the frequency maps $A s$.

        Examples:
            >>> from furax.obs.landscapes import HealpixLandscape
            >>> from furax.obs.operators import DustOperator
            >>> landscape = HealpixLandscape(nside=8, stokes='IQU')
            >>> nu = jnp.linspace(70.0, 180.0, 221)
            >>> A = DustOperator(
            ...     nu, frequency0=150.0, temperature=20.0, beta=1.54,
            ...     in_structure=landscape.structure,
            ... )
            >>> def top_hat(nu, nu_ref):
            ...     return jnp.where(jnp.abs(nu / nu_ref - 1) <= 0.15, 1 / (0.3 * nu_ref), 0.0)
            >>> B = BandpassOperator.create(
            ...     top_hat, nu, jnp.array([90.0, 150.0]), in_structure=A.out_structure
            ... )
            >>> (B @ A)(landscape.ones()).shape
            (2, 768)
        """
        frequencies = jnp.asarray(frequencies)
        transmission = jnp.stack([bandpass(frequencies, nu_ref) for nu_ref in channel_frequencies])
        return cls(transmission * _trapezoid_weights(frequencies), in_structure=in_structure)

    @classmethod
    def create_from_samples(
        cls,
        bandpasses: Sequence[tuple[ArrayLike, ArrayLike | None]],
        in_structure: PyTree[jax.ShapeDtypeStruct],
        *,
        units: str = 'K_CMB',
    ) -> 'BandpassOperator':
        r"""Create the operator from sampled bandpasses, with the convention of PySM.

        As in `pysm3.Sky.get_emission(frequencies, weights)`, the weights $w_c(\nu)$ of a channel
        are a transmission in power units: they apply to the specific intensity and are normalized
        to unit integral. The channel map is

        $$
        d_c = \frac{\int w_c(\nu)\, g(\nu)\, (A s)_\nu\, d\nu}{\int w_c(\nu)\, g(\nu)\, d\nu},
        $$

        where $g$ is proportional to the derivative of the specific intensity with respect to the
        temperature in the units of $A s$: $g(\nu) = \nu^2$ in $K_{RJ}$, and
        $g(\nu) = \nu^2 / f(\nu)$ in $K_{CMB}$, with $f$ the conversion factor from $K_{RJ}$ to
        $K_{CMB}$. A map that does not depend on the frequency, such as the CMB in $K_{CMB}$, is
        unchanged.

        The integrals use the trapezoidal rule on the frequencies of each channel, and a channel
        sampled at a single frequency is monochromatic.

        Args:
            bandpasses: For each channel, its frequencies in increasing order [GHz], and its
                weights, or `None` for a top-hat bandpass.
            in_structure: Structure of the frequency maps $A s$, with $A$ evaluated at
                `BandpassOperator.integration_frequencies(bandpasses)`.
            units: Units of $A s$, `'K_CMB'` or `'K_RJ'`.

        Examples:
            >>> from furax.obs.landscapes import HealpixLandscape
            >>> from furax.obs.operators import DustOperator
            >>> landscape = HealpixLandscape(nside=8, stokes='IQU')
            >>> bandpasses = [
            ...     (jnp.linspace(77.0, 106.0, 30), None),
            ...     (jnp.linspace(128.0, 169.0, 42), None),
            ... ]
            >>> nu = BandpassOperator.integration_frequencies(bandpasses)
            >>> A = DustOperator(
            ...     nu, frequency0=150.0, temperature=20.0, beta=1.54,
            ...     in_structure=landscape.structure,
            ... )
            >>> B = BandpassOperator.create_from_samples(bandpasses, in_structure=A.out_structure)
            >>> (B @ A)(landscape.ones()).shape
            (2, 768)
        """
        _check_units(units)
        frequencies = cls.integration_frequencies(bandpasses)
        weights = jnp.zeros((len(bandpasses), frequencies.size))
        for channel, bandpass in enumerate(bandpasses):
            nu, channel_weights = _channel_samples(*bandpass)
            support = _support(nu, channel_weights)
            row = _normalized_weights(nu, channel_weights, units)
            indices = jnp.searchsorted(frequencies, nu[support])
            weights = weights.at[channel, indices].set(row[support])
        return cls(weights, in_structure=in_structure)

    @staticmethod
    def integration_frequencies(
        bandpasses: Sequence[tuple[ArrayLike, ArrayLike | None]],
    ) -> Float[Array, ' nu']:
        """Return the frequencies at which to evaluate the sky for `create_from_samples`.

        These are the frequencies of the channels, except those that do not contribute to any
        channel: zero frequency, where the conversion to $K_{CMB}$ and some SEDs are undefined,
        and frequencies of zero weight.

        Args:
            bandpasses: For each channel, its frequencies [GHz] and its weights.

        Returns:
            The frequencies, sorted and without repetition [GHz].
        """
        frequencies = []
        for bandpass in bandpasses:
            nu, weights = _channel_samples(*bandpass)
            frequencies.append(nu[_support(nu, weights)])
        return jnp.unique(jnp.concatenate(frequencies))

    def mv(self, x: PyTree[Float[Array, '... nu pix']]) -> PyTree[Float[Array, '... channel pix']]:
        return jax.tree.map(self._integrate, x)

    def transpose(self) -> AbstractLinearOperator:
        return BandpassTransposeOperator(operator=self)

    def _integrate(self, x: Float[Array, '... nu pix']) -> Float[Array, '... channel pix']:
        return jnp.einsum('cn,...np->...cp', self.weights, x)


class BandpassTransposeOperator(TransposeOperator):
    r"""Transpose of `BandpassOperator`.

    It spreads each channel map over the integration frequencies, following the bandpass of the
    channel: $(B^T y)_{\nu_k} = \sum_c W_{ck}\, y_c$.
    """

    operator: BandpassOperator

    def mv(self, x: PyTree[Float[Array, '... channel pix']]) -> PyTree[Float[Array, '... nu pix']]:
        return jax.tree.map(self._spread, x)

    def _spread(self, y: Float[Array, '... channel pix']) -> Float[Array, '... nu pix']:
        return jnp.einsum('cn,...cp->...np', self.operator.weights, y)


class BandpassSEDRule(AbstractCompositionRule):
    """Binary rule for `B @ A = diag(W SED)`, where `A` is an SED operator."""

    left_operator_class = BandpassOperator
    right_operator_class = AbstractSEDOperator

    def apply(
        self, left: AbstractLinearOperator, right: AbstractLinearOperator
    ) -> list[AbstractLinearOperator]:
        assert isinstance(left, BandpassOperator)  # ty assert
        assert isinstance(right, AbstractSEDOperator)  # ty assert
        return [
            BroadcastDiagonalOperator(
                left.weights @ right._diagonal,
                axis_destination=right.axis_destination,
                insert_axes=right.insert_axes,
                in_structure=right.in_structure,
            )
        ]


class BandpassBlockRowRule(AbstractCompositionRule):
    """Binary rule for `B @ [A₁ | A₂ | ...] = [B @ A₁ | B @ A₂ | ...]`."""

    left_operator_class = BandpassOperator
    right_operator_class = BlockRowOperator

    def apply(
        self, left: AbstractLinearOperator, right: AbstractLinearOperator
    ) -> list[AbstractLinearOperator]:
        assert isinstance(right, BlockRowOperator)  # ty assert
        return [BlockRowOperator(right._tree_map(lambda block: left @ block)).reduce()]


def _channel_samples(
    nu: ArrayLike, weights: ArrayLike | None
) -> tuple[Float[Array, ' nu'], Float[Array, ' nu']]:
    """Return the frequencies and weights of a channel as arrays, top-hat if weights is None."""
    nu = jnp.atleast_1d(jnp.asarray(nu, dtype=float))
    if nu.size == 1:
        return nu, jnp.ones(1)
    weights = jnp.ones_like(nu) if weights is None else jnp.asarray(weights, dtype=float)
    if weights.shape != nu.shape:
        raise ValueError(
            f'The bandpass has {nu.size} frequencies but weights of shape {weights.shape}.'
        )
    return nu, weights


def _support(nu: Float[Array, ' nu'], weights: Float[Array, ' nu']) -> Array:
    """Return the mask of the frequencies of a channel that have a nonzero integration weight."""
    if nu.size == 1:
        return jnp.ones(1, dtype=bool)
    return (nu > 0) & (weights != 0)


def _normalized_weights(
    nu: Float[Array, ' nu'], weights: Float[Array, ' nu'], units: str
) -> Float[Array, ' nu']:
    """Return the integration weights of a channel sampled at `nu`, with unit sum."""
    if nu.size == 1:
        return jnp.ones(1)
    gain = nu**2 * _trapezoid_weights(nu)
    if units == 'K_CMB':
        # the conversion factor is undefined at zero frequency, where the gain tends to zero
        positive = nu > 0
        gain = jnp.where(positive, gain / k_rj_to_k_cmb(jnp.where(positive, nu, 1.0)), 0.0)
    weights = weights * gain
    return weights / weights.sum()


def _trapezoid_weights(nu: Float[Array, ' nu']) -> Float[Array, ' nu']:
    """Return the weights of the trapezoidal rule on the grid `nu`."""
    steps = jnp.diff(nu)
    return 0.5 * (jnp.pad(steps, (1, 0)) + jnp.pad(steps, (0, 1)))