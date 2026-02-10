import jax
import jax.numpy as jnp
from jaxtyping import Array
import equinox
import optax
from optax._src import linesearch

from typing import Any, Callable, Optional

# from jax_grid_search._optimizers import OptimizerState, _debug_callback 
# from jax_grid_search._progressbar import ProgressBar 
# from furax_cs import minimize

__all__ = [
    'filter_optimize',
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


def minimize_likelihood(first_guess_params, 
        fun, 
        max_iter=100, 
        tol=1e-5,
        optimize_func=filter_optimize):
    
    solver = optax.lbfgs(
        linesearch.scale_by_zoom_linesearch(
            max_linesearch_steps=50,
            initial_guess_strategy='one',
            verbose=True
        )
    )
    return optimize_func(
        first_guess_params, 
        fun, 
        solver, 
        max_iter=max_iter, 
        tol=tol
    )
