"""Distribution of the frequency channels over the devices.

The likelihood of `perform_compsep` is evaluated over the ensemble of the frequency
channels (see megabuster/likelihood.py), on a single device or laid out over the devices of
a mesh axis. This module holds what is about the distribution itself: the automatic mesh
installation and the scipy solver run on the calling thread, where the mesh is visible.
"""

import jax

__all__ = [
    'auto_mesh_axis',
    'minimize_scipy_on_calling_thread',
]


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
