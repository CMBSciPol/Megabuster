"""Distribution of the component separation over the frequency channels.

The spectral likelihood of `perform_compsep` is a sum over the frequency channels: with
A the mixing matrix, C the calibration rotations and N the noise covariance, both the
right-hand side

    AOND = A.T C.T N^-1 d = sum_f A_f.T C_f.T N_f^-1 d_f

and the central term

    A.T C.T N^-1 C A = sum_f A_f.T C_f.T N_f^-1 C_f A_f    (diagonal in the pixels)

are sums of per-channel contributions. The channels are the members of a furax
`StackedEnsemble`: `map_reduce` evaluates one traced body on every member and reduces the
`'sum'` outputs over the members and across the devices of the mesh axis the ensemble is
sharded over. The per-pixel inversion of the summed central term and the scalar products of
the likelihood run replicated on every device.

A member carries the data of one channel:

- `d`: its sky map, `StokesQU` of shape `(n_pix,)`;
- `invN`: the diagonal of its inverse noise covariance, `(2, n_pix)` for Q and U;
- `nu`, `w`: its integration frequencies and weights, `(n_samp,)` each — a single
  frequency of weight 1 without bandpass, the normalized bandpass weights otherwise, so
  that the effective SED of a component is `sed(nu) @ w`, as in `(Bp @ A).reduce()`;
- `index`: the channel index, to pick its angle in the fitted parameters.

The structure is ready for a per-channel beam operator `B_f` (harmonic convolution): it
would act inside the member body, between the SEDs and the noise weighting, leaving the
member layout and the reductions unchanged.
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from furax.core import Policy, StackedEnsemble, map_reduce
from furax.obs.operators._qu_rotations import rotate_qu
from furax.obs.operators._seds import CMBOperator, DustOperator, SynchrotronOperator
from furax.obs.stokes import StokesQU

from megabuster.BandpassOperator import _normalized_weights

__all__ = [
    'auto_mesh_axis',
    'build_channel_ensemble',
    'channel_bandpasses',
    'distributed_A_s_AOND',
    'distributed_gradients',
    'minimize_scipy_on_calling_thread',
]

COMPONENTS = ('cmb', 'dust', 'synchrotron')


def auto_mesh_axis(n_channels):
    """Install a 1D device mesh and return its axis name, or None on a single device.

    The channels are split over the largest number of devices that divides them. A mesh
    already installed by the caller is reused when its one axis divides the channels. The
    mesh is set globally (`jax.set_mesh`), so that later evaluations — including from the
    functions returned in the results — find it.
    """
    mesh = jax.sharding.get_mesh()
    if not mesh.empty:
        if len(mesh.axis_names) == 1 and n_channels % mesh.devices.size == 0:
            return mesh.axis_names[0] if mesh.devices.size > 1 else None
        print(
            f'The installed mesh {mesh} cannot lay out the {n_channels} channels '
            f'(one axis dividing them is needed): running on a single device.',
            flush=True,
        )
        return None
    n_devices = max(d for d in range(1, jax.device_count() + 1) if n_channels % d == 0)
    if n_devices == 1:
        return None
    devices = jax.devices()[:n_devices]
    print(
        f'Distributing the {n_channels} channels over {n_devices} '
        f'{devices[0].platform.upper()} device(s)'
        + (f' out of {jax.device_count()}' if n_devices < jax.device_count() else ''),
        flush=True,
    )
    jax.set_mesh(jax.make_mesh((n_devices,), ('freq',), devices=devices))
    return 'freq'


def minimize_scipy_on_calling_thread(fn, init_params, solver_name, max_iter):
    """The scipy solver of `furax_cs.minimize`, without the host-callback indirection.

    `furax_cs.minimize` wraps the scipy solvers in `jax.pure_callback` (for vmap support):
    the objective is then evaluated from the runtime's callback thread, which does not see
    the mesh set by `jax.set_mesh` — the mesh is thread-local, and it cannot be set during
    the tracing of `value_and_grad`. Calling the same jaxopt solver directly keeps the
    evaluations on the calling thread, where the mesh of the ensemble is visible.

    The objective and its gradient are also jitted (`furax_cs` evaluates them eagerly):
    the engine and the collectives are traced once instead of at every evaluation.

    Returns:
        The optimized parameters and the solver state (with `iter_num`), like
        `furax_cs.minimize`.
    """
    from jaxopt import ScipyBoundedMinimize

    method = solver_name.split('_', 1)[1]
    solver = ScipyBoundedMinimize(
        fun=lambda params: fn(params),
        method=method,
        jit=True,
        maxiter=max_iter,
        options={'disp': False},
    )
    result = solver.run(init_params, bounds=None)
    return result.params, result.state


def channel_bandpasses(bandpasses, units='K_CMB'):
    """Return the per-channel integration frequencies and normalized weights.

    Each channel integrates over its own frequency samples, with the weights of
    `BandpassOperator.create_from_samples` (PySM convention): the effective SED of a
    component is `sed(nu) @ w`, identical to the rows of `(Bp @ A).reduce()` restricted
    to the frequencies of the channel.

    Args:
        bandpasses: For each channel, its frequencies in increasing order [GHz], and its
            weights, or `None` for a top-hat bandpass. All the channels must have the same
            number of samples, so that they form an ensemble.

    Returns:
        The frequencies and the weights, two arrays of shape `(n_channels, n_samp)`.
    """
    nus, ws = [], []
    for nu, weight in bandpasses:
        nu = np.atleast_1d(np.asarray(nu, dtype=float))
        if np.any(nu <= 0):
            raise ValueError('the per-channel integration requires positive frequencies.')
        weight = np.ones_like(nu) if weight is None else np.asarray(weight, dtype=float)
        nus.append(nu)
        ws.append(np.asarray(_normalized_weights(jnp.asarray(nu), jnp.asarray(weight), units)))
    if len({nu.size for nu in nus}) != 1:
        raise ValueError(
            'the channels must have the same number of bandpass samples to form an ensemble, '
            f'got {[nu.size for nu in nus]}.'
        )
    return np.stack(nus), np.stack(ws)


def build_channel_ensemble(sky_map, invN_diagonal, nus, weights, *, axis=None, batch=None):
    """Build the ensemble of the frequency channels, laid out over a mesh axis.

    Args:
        sky_map: The sky maps, `StokesQU` with leaves of shape `(n_channels, n_pix)`.
        invN_diagonal: The diagonals of the inverse noise covariances, of shape
            `(n_channels, 2, n_pix)` with Q and U.
        nus: The integration frequencies of each channel, `(n_channels, n_samp)` [GHz].
        weights: The integration weights of each channel, `(n_channels, n_samp)`.
        axis: The mesh axis the channels are split over (`jax.set_mesh` must be active),
            or None to keep the ensemble on a single device.
        batch: The channels of one device evaluated together, vectorized. None evaluates
            all of them at once, like the non-distributed code.

    Returns:
        The `StackedEnsemble` of the channels.
    """
    n_channels = nus.shape[0]
    stacked = {
        # a Stokes object is backed by one array of shape (n_stokes, ...): the member axis
        # is moved in front and the member rebuilds its StokesQU with `from_array`
        'd': jnp.stack([jnp.asarray(sky_map.q), jnp.asarray(sky_map.u)], axis=1),
        'invN': jnp.asarray(invN_diagonal),
        'nu': jnp.asarray(nus, dtype=float),
        'w': jnp.asarray(weights, dtype=float),
        'index': jnp.arange(n_channels),
    }
    devices = 1
    if axis is not None:
        mesh = jax.sharding.get_abstract_mesh()
        if mesh.empty or axis not in mesh.shape:
            raise ValueError(f'no mesh axis {axis!r} is set: use `jax.set_mesh`.')
        devices = mesh.shape[axis]
        if n_channels % devices:
            raise ValueError(f'{devices} devices do not divide {n_channels} channels.')
    per_device = n_channels // devices
    if batch is None:
        batch = per_device
    if per_device % batch:
        raise ValueError(f'rounds of {batch} members do not divide {per_device} members.')
    ensemble = StackedEnsemble(n_channels, stacked, policy=Policy(batch=batch))
    if axis is not None:
        ensemble = ensemble.shard(axis)
    return ensemble


def _effective_seds(nu, w, spectral, dust_nu0, synchrotron_nu0, derivatives=False):
    """The SED of each component integrated over the channel, and its derivatives.

    The SEDs are those of the furax operators, integrated against the channel weights:
    `sed_c = sed_c(nu) @ w`, matching `(Bp @ A).reduce()` (and the plain mixing matrix for
    a single frequency of weight 1).
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
    dsed_dust = jnp.stack([zero, (log_nu_dust * dust._diagonal[:, 0]) @ w, zero])
    dsed_sync = jnp.stack([zero, zero, (log_nu_sync * synchrotron._diagonal[:, 0]) @ w])
    return sed, dsed_dust, dsed_sync


def _dot(x: StokesQU, y: StokesQU):
    return jnp.vdot(x.q, y.q) + jnp.vdot(x.u, y.u)


def _combine(seds, s):
    """The frequency map of the channel, `sum_c sed_c * s_c`."""
    return StokesQU(
        sum(seds[i] * s[c].q for i, c in enumerate(COMPONENTS)),
        sum(seds[i] * s[c].u for i, c in enumerate(COMPONENTS)),
    )


def _primal_member(member, angles, spectral, *, dust_nu0, synchrotron_nu0):
    """The contributions of one channel to AOND and to the central term.

    The calibration operator is `C_f = R(-angle_f)` (`build_calibration_operator` with the
    negated angles), so the right-hand side member is `C_f.T N_f^-1 d_f = R(angle_f) N_f^-1 d_f`
    and the central term member is `C_f.T N_f^-1 C_f` weighted by the SEDs, as assembled in
    `build_preconditioner2`.
    """
    sed, _, _ = _effective_seds(
        member['nu'], member['w'], spectral, dust_nu0, synchrotron_nu0
    )
    angle = angles[member['index']]
    d = StokesQU.from_array(member['d'])
    invN_q, invN_u = member['invN'][0], member['invN'][1]
    nd = StokesQU(invN_q * d.q, invN_u * d.u)
    right_member = rotate_qu(nd, angle)  # C.T (N^-1 d)
    aond = {
        c: StokesQU(sed[i] * right_member.q, sed[i] * right_member.u)
        for i, c in enumerate(COMPONENTS)
    }
    # C.T @ N^-1 @ C in the (Q, U) Stokes basis
    cos_2a = jnp.cos(2 * angle)
    sin_2a = jnp.sin(2 * angle)
    ctinvnc = jnp.stack(
        [
            jnp.stack(
                [
                    cos_2a**2 * invN_q + sin_2a**2 * invN_u,
                    cos_2a * sin_2a * (invN_u - invN_q),
                ]
            ),
            jnp.stack(
                [
                    cos_2a * sin_2a * (invN_u - invN_q),
                    sin_2a**2 * invN_q + cos_2a**2 * invN_u,
                ]
            ),
        ]
    )  # (n_stokes, n_stokes, n_pix)
    central = jnp.einsum('c,sjp,k->cksjp', sed, ctinvnc, sed)
    return {'aond': aond, 'central': central}


def _grad_member(member, angles, spectral, s, *, dust_nu0, synchrotron_nu0, use_calibration):
    """The contributions of one channel to the gradients of the likelihood.

    With `lht_f = C_f.T N_f^-1 d_f - C_f.T N_f^-1 C_f A_f s` the left-hand term of the
    channel and `c_lht_f` its uncalibrated counterpart, the gradients are
    `-2 <dA_f/dbeta s, lht_f>` for the spectral indices and
    `-2 <2 R(-angle_f - pi/4) A_f s, c_lht_f>` for the angle of the channel.
    """
    sed, dsed_dust, dsed_sync = _effective_seds(
        member['nu'], member['w'], spectral, dust_nu0, synchrotron_nu0, derivatives=True
    )
    angle = angles[member['index']]
    d = StokesQU.from_array(member['d'])
    invN_q, invN_u = member['invN'][0], member['invN'][1]
    a_s = _combine(sed, s)  # A s
    ca_s = rotate_qu(a_s, -angle)  # C (A s)
    nd = StokesQU(invN_q * d.q, invN_u * d.u)
    nca_s = StokesQU(invN_q * ca_s.q, invN_u * ca_s.u)
    c_lht = StokesQU(nd.q - nca_s.q, nd.u - nca_s.u)  # N^-1 d - N^-1 C A s
    lht = rotate_qu(c_lht, angle)  # C.T (...)
    grads = {
        'beta_dust': -2 * _dot(_combine(dsed_dust, s), lht),
        'beta_pl': -2 * _dot(_combine(dsed_sync, s), lht),
    }
    if use_calibration:
        dxax = rotate_qu(a_s, -angle - jnp.pi / 4)
        grads['angle'] = -2 * _dot(2 * dxax, c_lht)
    else:
        grads['angle'] = jnp.zeros(())
    return grads


def _invert_central(central, observed):
    """The per-pixel inverse of the summed central term.

    The unobserved pixels are replaced by the identity before the inversion and zeroed
    after, as in `build_preconditioner2`, with a mask instead of a scatter.

    Args:
        central: The summed central term, `(n_comp, n_comp, n_stokes, n_stokes, n_pix)`.
        observed: Boolean mask of the observed pixels, `(n_pix,)`, or None.

    Returns:
        The inverse, `(n_pix, n_comp, n_stokes, n_comp, n_stokes)`.
    """
    n_comp, _, n_stokes, _, n_pix = central.shape
    n = n_comp * n_stokes
    matrix = central.transpose(4, 0, 2, 1, 3).reshape(n_pix, n, n)
    if observed is not None:
        matrix = jnp.where(observed[:, None, None], matrix, jnp.eye(n))
    inverse = jnp.linalg.inv(matrix)
    if observed is not None:
        inverse = jnp.where(observed[:, None, None], inverse, 0.0)
    return inverse.reshape(n_pix, n_comp, n_stokes, n_comp, n_stokes)


def _apply_central_inverse(inverse, aond):
    """The component maps `s = (A.T C.T N^-1 C A)^-1 AOND`."""
    x = jnp.stack([jnp.stack([aond[c].q, aond[c].u]) for c in COMPONENTS])  # (n_comp, 2, n_pix)
    y = jnp.einsum('pcskt,ktp->csp', inverse, x)
    return {c: StokesQU(y[i, 0], y[i, 1]) for i, c in enumerate(COMPONENTS)}


def distributed_A_s_AOND(ensemble, angles, spectral, observed, dust_nu0, synchrotron_nu0):
    """AOND and the component maps s, summed over the channel members.

    The two `'sum'` outputs of the engine — AOND and the central term — are reduced across
    the devices and held by all of them; the per-pixel inversion and the application run
    replicated.

    Args:
        ensemble: The ensemble of the channels, from `build_channel_ensemble`.
        angles: The calibration angles, `(n_channels,)` (zeros without calibration).
        spectral: The spectral parameters, keys 'temp_dust', 'beta_dust' and 'beta_pl'.
        observed: Boolean mask of the observed pixels, `(n_pix,)`, or None.
        dust_nu0: Dust reference frequency [GHz].
        synchrotron_nu0: Synchrotron reference frequency [GHz].

    Returns:
        The component maps s and AOND, two dictionaries of `StokesQU` keyed by component.
    """
    apply = partial(_primal_member, dust_nu0=dust_nu0, synchrotron_nu0=synchrotron_nu0)
    outputs, _ = map_reduce(
        apply, ensemble, {'aond': 'sum', 'central': 'sum'}, angles, spectral
    )
    inverse = _invert_central(outputs['central'], observed)
    s = _apply_central_inverse(inverse, outputs['aond'])
    return s, outputs['aond']


def distributed_gradients(
    ensemble, angles, spectral, s, dust_nu0, synchrotron_nu0, use_calibration
):
    """The gradients of the likelihood, summed over the channel members.

    Returns:
        A dictionary with the scalar gradients 'beta_dust' and 'beta_pl', summed over the
        channels, and 'angle', the gradient of the angle of each channel, `(n_channels,)`
        (without the prior terms).
    """
    apply = partial(
        _grad_member,
        dust_nu0=dust_nu0,
        synchrotron_nu0=synchrotron_nu0,
        use_calibration=use_calibration,
    )
    outputs, _ = map_reduce(
        apply,
        ensemble,
        {'beta_dust': 'sum', 'beta_pl': 'sum', 'angle': 1},
        angles,
        spectral,
        s,
    )
    if ensemble.policy.axis is not None:
        # the stride-1 angle gradients stay laid out over the devices: replicate them
        # (n_channels scalars) so that they can be indexed per parameter
        outputs['angle'] = jax.sharding.reshard(outputs['angle'], jax.sharding.PartitionSpec())
    return outputs
