import jax
import jax.numpy as jnp
from jaxtyping import Array
import equinox
import optax

from typing import Any, Callable, Optional

from jax_grid_search._optimizers import OptimizerState, _debug_callback 
from jax_grid_search._progressbar import ProgressBar 

__all__ = [
    'filter_optimize',
    'new_filter_optimize',
    'minimize_likelihood'
]

# @equinox.filter_jit
def filter_optimize(init_params, fun, opt, max_iter, tol, **kwargs):
    """ 
    Define a function that computes both value and gradient of the objective.

    Parameters
    ----------
    init_params: dict
        Initial parameters.
    fun: callable
        Objective function.
    opt: optax object
        Optimizer.
    max_iter: int
        Maximum number of iterations.
    tol: float
        Tolerance.
    kwargs: dict
        Additional keyword arguments.

    Returns
    -------
    final_params: dict
        Final parameters stored as a dictionary with keys 'temp_dust', 'beta_dust', and 'beta_pl'.
    final_state: dict
        Final state of the optimizer.
    """
    # value_and_grad_fun = jax.value_and_grad(fun)
    value_and_grad_fun = equinox.filter_value_and_grad(
        fun
    )

    # value_fn = lambda params: value_and_grad_fun(params, **kwargs)[0]  # Compute value only
    # value_fn = equinox.filter_jit(fun)
    value_fn = fun
    
    # Single optimization step.
    # @equinox.filter_jit
    def step(carry):
        params, state, _ = carry
        value, grad = value_and_grad_fun(params, **kwargs)  # Compute value and gradient
        updates, state = opt.update(
            grad, state, params, value=value, grad=grad, value_fn=value_fn, **kwargs
        )  # Perform update
        params = optax.apply_updates(params, updates)  # Update params
        return (params, state, updates)

    # Stopping condition.
    # @equinox.filter_jit
    def continuing_criterion(carry):
        _, state, updates = carry
        iter_num = optax.tree_utils.tree_get(state, 'count')  # Get iteration count from optimizer state
        iter_num = 0 if iter_num is None else iter_num
        update_norm = optax.tree_utils.tree_l2_norm(updates)  # Compute update norm
        return (iter_num == 0) | ((iter_num < max_iter) & (update_norm >= tol))

    # Initialize optimizer state.
    init_carry = (init_params, opt.init(init_params), init_params)

    # Run the while loop.
    final_params, final_state, _ = jax.lax.while_loop(continuing_criterion, step, init_carry)

    return final_params, final_state


# @partial(jax.jit, static_argnums=(1, 2, 3, 5, 9))
def new_filter_optimize(
    init_params: Array,
    fun: Callable[[Array], Array],
    opt: optax._src.base.GradientTransformationExtraArgs,
    max_iter: int,
    tol: Array,
    progress: Optional[ProgressBar] = None,
    progress_id: int = 0,
    upper_bound: Optional[Array] = None,
    lower_bound: Optional[Array] = None,
    log_updates: bool = False,
    **kwargs: Any,
) -> tuple[Array, OptimizerState]:
    """
    Code inspired from github.com/ASKabalan/jax-grid-search/blob/main/src/jax_grid_search/_optimizers.py 
    and adapted to equinox
    """
    # Define a function that computes both value and gradient of the objective.
    # value_and_grad_fun = jax.value_and_grad(fun)
    value_and_grad_fun = equinox.filter_value_and_grad(fun)
    
    update_history = jnp.zeros((max_iter, 2)) if log_updates else None

    # Single optimization step.
    def step(carry: OptimizerState) -> OptimizerState:
        value, grad = value_and_grad_fun(carry.params, **kwargs)  # Compute value and gradient
        updates, state = opt.update(grad, carry.state, carry.params, value=carry.value, grad=grad, value_fn=fun, **kwargs)  # Perform update
        update_norm = optax.tree_utils.tree_l2_norm(updates)  # Compute update norm
        params = optax.apply_updates(carry.params, updates)  # Update params
        if upper_bound is not None and lower_bound is not None:
            params = optax.projections.projection_box(params, lower_bound, upper_bound)  # Apply box constraints
        if log_updates and carry.update_history is not None:
            iter_num = optax.tree_utils.tree_get(carry.state, "count")
            to_log = jnp.array([update_norm, value])
            carry = carry._replace(update_history=carry.update_history.at[iter_num].set(to_log))

        best_params = jax.tree.map(
            lambda x, y: jnp.where((carry.best_val < value) | jnp.isnan(value), x, y),
            carry.best_params,
            carry.params,
        )
        best_val = jnp.where((carry.best_val < value) | jnp.isnan(value), carry.best_val, value)

        if progress:
            iter_num = optax.tree_utils.tree_get(carry.state, "count")
            progress.update(progress_id, (update_norm, tol, iter_num, carry.value, max_iter), desc_cb=_debug_callback, total=max_iter)

        return carry._replace(
            params=params,
            state=state,
            updates=updates,
            value=value,
            best_val=best_val,
            best_params=best_params,
            update_norm=update_norm,
        )

    # Stopping condition.
    def continuing_criterion(carry: OptimizerState) -> Any:
        iter_num = optax.tree_utils.tree_get(carry.state, "count")  # Get iteration count from optimizer state
        iter_num = 0 if iter_num is None else iter_num
        update_norm = carry.update_norm
        return (iter_num == 0) | ((iter_num < max_iter) & (update_norm >= tol))

    # Initialize optimizer state.
    init_state = OptimizerState(init_params, opt.init(init_params), init_params, jnp.inf, jnp.inf, jnp.inf, init_params, update_history)

    # Run the while loop.
    if progress:
        progress.create_task(progress_id, total=max_iter)
    final_opt_state = jax.lax.while_loop(continuing_criterion, step, init_state)
    if progress:
        progress.finish(progress_id, total=max_iter)

    # Was the last evaluation better than the best?
    best_params = jax.tree.map(
        lambda x, y: jnp.where((final_opt_state.best_val < final_opt_state.value) | jnp.isnan(final_opt_state.value), x, y),
        final_opt_state.best_params,
        final_opt_state.params,
    )
    best_value: float = jnp.where(
        (final_opt_state.best_val < final_opt_state.value) | jnp.isnan(final_opt_state.value),
        final_opt_state.best_val,
        final_opt_state.value,
    )  # type: ignore[assignment]
    final_opt_state = final_opt_state._replace(best_params=best_params, best_val=best_value)

    return final_opt_state.best_params, final_opt_state

def minimize_likelihood(first_guess_params, 
        fun, 
        max_iter=100, 
        tol=1e-5,
        optimize_func=filter_optimize):
    
    solver = optax.lbfgs()
    return optimize_func(
        first_guess_params, 
        fun, 
        solver, 
        max_iter=max_iter, 
        tol=tol
    )
