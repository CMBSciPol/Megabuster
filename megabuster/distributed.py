"""Distribution of the frequency channels over the devices.

There is a single code path: the mixing matrix and the noise covariance are always built as
furax ensembles of per-channel operators — `A = BlockColumnOperator(ensemble of A_f)`,
`N⁻¹ = BlockDiagonalOperator(ensemble of N_f⁻¹)` — and the block algebra reduces the central
term to `AdditionOperator(ensemble of A_fᵀ N_f⁻¹ A_f)`, whose conjugate gradient sums the
per-channel contributions through the ensemble engine. When several devices are visible,
`auto_mesh_axis` installs a 1D mesh and the ensembles are sharded over it: each device then
evaluates its own channels and the sums become cross-device reductions. On a single device
the same expressions run vectorized, as the sequential code did.

A multi-channel map keeps one layout throughout, the one of the public API of
`perform_compsep`: a `Stokes` container whose data is `(n_stokes, n_channels, n_pix)`. The
block composites are built with `axes=1`, the member axis of their stacked sides, so the
engine slices, stacks and shards the channel axis in place: nothing is transposed, and a
user-provided operator — an observation matrix, a central operator — applies to the maps
directly.
"""

import jax

__all__ = [
    'auto_mesh_axis',
    'shard_ensemble',
    'minimize_scipy_on_calling_thread',
]


def auto_mesh_axis(n_channels):
    """Install a 1D device mesh and return its axis name, or None on a single device.

    The channels are split over the largest number of devices that divides them. A mesh
    already installed by the caller is reused when its one axis divides the channels and its
    axes are `Auto` (the conjugate gradient goes through lineax, which does not know about
    the explicit-sharding avals). The mesh is set globally (`jax.set_mesh`), so that later
    evaluations — including from the functions returned in the results — find it.
    """
    auto = jax.sharding.AxisType.Auto
    mesh = jax.sharding.get_mesh()
    if not mesh.empty:
        if (
            len(mesh.axis_names) == 1
            and mesh.axis_types == (auto,)
            and n_channels % mesh.devices.size == 0
        ):
            return mesh.axis_names[0] if mesh.devices.size > 1 else None
        print(
            f'The installed mesh {mesh} cannot lay out the {n_channels} channels '
            f'(one Auto axis dividing them is needed): the channels are not distributed.',
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
    jax.set_mesh(jax.make_mesh((n_devices,), ('freq',), devices=devices, axis_types=(auto,)))
    return 'freq'


def shard_ensemble(ensemble, axis):
    """The ensemble laid out over the mesh axis, unchanged when no axis is given."""
    if axis is None:
        return ensemble
    return ensemble.shard(axis)


def minimize_scipy_on_calling_thread(fn, init_params, solver_name, max_iter, tol=None):
    """The scipy solver of `furax_cs.minimize`, without the host-callback indirection.

    `furax_cs.minimize` wraps the scipy solvers in `jax.pure_callback` (for vmap support):
    the objective is then evaluated from the runtime's callback thread, which does not see
    the mesh set by `jax.set_mesh` — the mesh is thread-local, and it cannot be set during
    the tracing of `value_and_grad`. Calling the same jaxopt solver directly keeps the
    evaluations on the calling thread, where the mesh is visible, and jits the objective
    and its gradient, traced once instead of at every evaluation.

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
        tol=tol,
        options={'disp': False},
    )
    result = solver.run(init_params, bounds=None)
    return result.params, result.state
