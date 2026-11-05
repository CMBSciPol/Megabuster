"""Distribution of the frequency channels over the devices.

There is a single code path: the mixing matrix and the noise covariance are always built as
furax ensembles of per-channel operators — `A = BlockColumnOperator(ensemble of A_f)`,
`N⁻¹ = BlockDiagonalOperator(ensemble of N_f⁻¹)` — and the block algebra reduces the central
term to `AdditionOperator(ensemble of A_fᵀ N_f⁻¹ A_f)`, whose conjugate gradient sums the
per-channel contributions through the ensemble engine. When several devices are visible,
`auto_mesh_axis` installs a 1D mesh and the ensembles are sharded over it: each device then
evaluates its own channels and the sums become cross-device reductions. On a single device
the same expressions run vectorized, as the sequential code did.

Two layouts of a multi-channel map meet here. The public API of `perform_compsep` is
Stokes-first, data `(n_stokes, n_channels, n_pix)`, and keeps it. The block composites hand
each block its slice of their input through `Ensemble.zip`, which takes the member axis
leading, so the maps they consume and return are channel-first,
`(n_channels, n_stokes, n_pix)`, one member per entry. (`StackedEnsemble` itself accepts
`axes`, the member axis of each leaf, so an ensemble built here could read a Stokes-first
map in place; only the composites still require the leading axis.) A user-provided operator
acting on Stokes-first maps — an observation matrix, a central operator — is therefore
conjugated by furax's `MoveAxisOperator`, `R.T @ M @ R`; it also forces the single-device
path, since it couples the channels. The permutation preserves the sharding, so it is
device-local.
"""

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

from furax.core import (
    BlockRowOperator,
    DiagonalOperator,
    StackedEnsemble,
)

from megabuster.mixingmatrix import (
    create_MixingMatrixOperator,
    create_MixingMatrixOperator_deriv,
)

__all__ = [
    'auto_mesh_axis',
    'channel_ensemble',
    'shard_ensemble',
    'stack_channel_maps',
    'member_mixing_operator',
    'member_mixing_operator_deriv',
    'sed_matrix_from_ensemble',
    'minimize_scipy_on_calling_thread',
]

COMPONENTS = ('cmb', 'dust', 'synchrotron')


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
            f'(one Auto axis dividing them is needed): running on a single device.',
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


def channel_ensemble(arrays):
    """The ensemble of the channels from a dict of arrays with a leading channel axis.

    The ensemble is not laid out over the mesh: `lift` vectorizes its body by fetching one
    member, which an already-sharded ensemble cannot provide. Shard the lifted result with
    `shard_ensemble` instead.
    """
    n_channels = len(next(iter(arrays.values())))
    return StackedEnsemble(n_channels, {k: jnp.asarray(v) for k, v in arrays.items()}, 1)


def shard_ensemble(ensemble, axis):
    """The ensemble laid out over the mesh axis, unchanged when no axis is given."""
    if axis is None:
        return ensemble
    return ensemble.shard(axis)


def stack_channel_maps(stokes, axis):
    """A Stokes-first map of the channels, re-laid out with the channel axis first.

    The returned container is the stacked form of the ensemble of the channel maps: its
    data has shape `(n_channels, n_stokes, n_pix)`, laid out over the mesh axis when given.
    """
    # built leaf-wise: the stacked container is not a valid Stokes object (its leading axis
    # is the channel, not the Stokes axis), it is the member pytree with stacked leaves
    stacked = jax.tree.map(lambda leaf: jnp.moveaxis(jnp.asarray(leaf), 1, 0), stokes)
    if axis is not None:
        stacked = jax.tree.map(lambda leaf: jax.device_put(leaf, P(axis)), stacked)
    return stacked


def _member_diagonals(operator_blocks):
    """The per-component SED diagonals of a one-frequency mixing operator, as a dict."""
    # the SED operators hold a diagonal of shape (1, 1), or (1, n_pix) with patch indices
    return {component: block._diagonal[0] for component, block in operator_blocks.items()}


def member_mixing_operator(
    nu, parameters_dict, structure, dust_nu0, synchrotron_nu0, patch_indices
):
    """The mixing matrix of one channel: component maps -> the channel's map.

    The SEDs are those of `create_MixingMatrixOperator` evaluated at the channel's
    frequency; the operator maps `{'cmb', 'dust', 'synchrotron'}` component maps of the
    member structure to a single channel map, without a frequency axis.
    """
    multi_channel = create_MixingMatrixOperator(
        nu,
        parameters_dict,
        structure,
        dust_nu0=dust_nu0,
        synchrotron_nu0=synchrotron_nu0,
        patch_indices=patch_indices,
    )
    return BlockRowOperator(
        {
            component: DiagonalOperator(diagonal, in_structure=structure)
            for component, diagonal in _member_diagonals(multi_channel.blocks).items()
        }
    )


def member_mixing_operator_deriv(
    nu, parameters_dict, structure, dust_nu0, synchrotron_nu0, patch_indices
):
    """The derivatives of the mixing matrix of one channel, keyed by spectral parameter."""
    multi_channel = create_MixingMatrixOperator_deriv(
        nu,
        parameters_dict,
        structure,
        dust_nu0=dust_nu0,
        synchrotron_nu0=synchrotron_nu0,
        patch_indices=patch_indices,
    )
    return {
        key: BlockRowOperator(
            {
                component: DiagonalOperator(diagonal, in_structure=structure)
                for component, diagonal in _member_diagonals(operator.blocks).items()
            }
        )
        for key, operator in multi_channel.items()
    }


def sed_matrix_from_ensemble(a_ensemble, n_pix, components=COMPONENTS):
    """The mixing matrix of an ensemble of per-channel operators, as a replicated array.

    Returns the SEDs as an array of shape `(n_channels, n_components, n_pix)`, the layout of
    `get_array_from_A`, so that the preconditioner computations apply unchanged: the mesh
    axes are `Auto`, the compiler inserts the gathers where needed.
    """
    blocks = a_ensemble.stacked.blocks
    return jnp.stack(
        [
            jnp.broadcast_to(blocks[component].diagonal, (a_ensemble.size, n_pix))
            for component in components
        ],
        axis=1,
    )


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
