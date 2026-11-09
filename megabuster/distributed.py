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


def minimize_scipy_on_calling_thread(
    fn,
    init_params,
    solver_name,
    max_iter=1000,
    rtol=1e-8,
    atol=1e-8,
    lower_bound=None,
    upper_bound=None,
    precondition=False,
    solver_options={},
    refresh_steps=10,
    **fn_kwargs,
):
    """`furax_cs.minimize` for its scipy solvers, without the host-callback indirection.

    `furax_cs.minimize` wraps the scipy solvers in `jax.pure_callback` (for vmap support):
    the objective is then evaluated from the runtime's callback thread, which does not see
    the mesh set by `jax.set_mesh` — the mesh is thread-local, and it cannot be set during
    the tracing of `value_and_grad`. Calling the same jaxopt solver directly keeps the
    evaluations on the calling thread, where the mesh is visible, and jits the objective
    and its gradient, traced once instead of at every evaluation.

    The signature and the returned state are those of `furax_cs.minimize`, so that either
    can be called with the same arguments. The tolerances and the scipy options are
    honoured, which the scipy path of `furax_cs.minimize` drops: `solver_options['options']`
    holds the options of the scipy method, and `rtol` and `atol` set its tolerances
    (`ftol`, `gtol` and `xtol` for TNC, `ftol` and `gtol` for L-BFGS-B, `final_tr_radius`
    for COBYQA) unless these options give them explicitly. `refresh_steps` only concerns the
    optimistix solvers.

    Returns:
        The optimized parameters and the `furax_cs` `UnifiedState` of the solver.
    """
    from furax_cs.optim import condition
    from furax_cs.optim.minimize import UnifiedState
    from furax_cs.optim.solvers import SELFCONDITIONED_SOLVERS
    from jaxopt import ScipyBoundedMinimize

    if precondition and solver_name in SELFCONDITIONED_SOLVERS:
        print(f"Solver '{solver_name}' is self-conditioned: preconditioning ignored.", flush=True)
        precondition = False
    if precondition:
        fn, to_opt, from_opt = condition(
            fn,
            lower=lower_bound,
            upper=upper_bound,
            scale_function=precondition,
            init_params=init_params,
            **fn_kwargs,
        )
        init_params = to_opt(init_params)
        lower_bound = to_opt(lower_bound) if lower_bound is not None else None
        upper_bound = to_opt(upper_bound) if upper_bound is not None else None
    else:
        from_opt = lambda params: params

    method = solver_name.split('_', 1)[1]
    options = {'disp': False, **((solver_options or {}).get('options') or {})}
    if method == 'tnc':
        options.setdefault('ftol', atol)
        options.setdefault('gtol', rtol)
        options.setdefault('xtol', atol)
    elif method == 'l-bfgs-b':
        options.setdefault('ftol', atol)
        options.setdefault('gtol', rtol)
    elif method == 'cobyqa':
        options.setdefault('final_tr_radius', atol)

    solver = ScipyBoundedMinimize(
        fun=lambda params, **kwargs: fn(params, **kwargs),
        method=method,
        jit=True,
        maxiter=max_iter,
        options=options,
    )
    if lower_bound is None and upper_bound is None:
        bounds = None
    else:
        bounds = (lower_bound, upper_bound)
    result = solver.run(init_params, bounds=bounds, **fn_kwargs)
    params = from_opt(result.params)
    state = UnifiedState(
        best_loss=result.state.fun_val,
        best_y=params,
        iter_num=result.state.iter_num,
        solver_state=result.state,
    )
    return params, state
