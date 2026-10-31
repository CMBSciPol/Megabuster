import logging, warnings
import operator
import jax
import jax.numpy as jnp
import equinox
import lineax as lx

from megabuster.mixingmatrix import create_MixingMatrixOperator, create_MixingMatrixOperator_deriv
from megabuster.tools import (
    get_A_from_array, 
    get_array_from_A,
)

logger = logging.getLogger(__name__)

__all__ = [
    'Likelihood',
]

def dot_2(x,y):
    """Scalar product of two Pytrees.

    If one of the leaves is complex, the hermitian scalar product is returned.

    Args:
        x: The first Pytree.
        y: The second Pytree.

    Example:
        >>> import furax as fx
        >>> x = {'a': jnp.array([1., 2, 3]), 'b': jnp.array([1, 0])}
        >>> y = {'a': jnp.array([2, -1, 1]), 'b': jnp.array([2, 0])}
        >>> fx.tree.dot(x, y)
        Array(5., dtype=float32)
    """
    
    xy = jax.tree.map(jnp.vdot, x, y)
    return jax.tree.reduce(operator.add, xy)

class Likelihood:
    def __init__(
        self,
        frequencies,
        fixed_params,
        structure_sed,
        central_freq_op,
        right_member,
        dictionary_parameters_CG,
        dictionary_parameters_minimization,
        preconditioner = None,
        helpers_mixing_matrix: dict = {
            'dust_nu0': 150.0,
            'synchrotron_nu0': 150.0,
            'patch_indices': None
        }
    ):
        self.fixed_params = fixed_params
        self.frequencies = frequencies
        self.structure_sed = structure_sed
        self.dictionary_mixing_matrix = {
            'frequencies': frequencies,
            'structure_sed': structure_sed,
            **helpers_mixing_matrix
        }

        self.patch_indices = self.dictionary_mixing_matrix.get('patch_indices', None)

        self.central_freq_op = central_freq_op
        self.dictionary_parameters_CG = dictionary_parameters_CG
        self.right_member = right_member

        self.n_pix = right_member.shape[-1]

        self.dictionary_parameters_minimization = dictionary_parameters_minimization
        self.preconditioner = preconditioner

    def complete_params(self, params):
        """Return the full parameter dictionary: the parameters fitted, completed by the fixed ones."""
        parameters_dict = dict()
        for param_name in ['temp_dust', 'beta_dust', 'beta_pl']:
            if param_name in params:
                parameters_dict[param_name] = params[param_name]
            else:
                parameters_dict[param_name] = self.fixed_params[param_name]
        return parameters_dict

    # @equinox.filter_jit
    def get_A_s_AOND(self, params, right_member=None):
        """
            Compute the log-proba given a set of parameters
        """
        if right_member is None:
            right_member = self.right_member

        # Select parameters which are to be estimated
        parameters_dict = self.complete_params(params)

        # Mixing matrix operator
        A = create_MixingMatrixOperator(
            parameters_dict=parameters_dict, 
            **self.dictionary_mixing_matrix
        )

        # Full right-hand side of the CG equation
        AOND = A.T(right_member)

        preconditioner = self.preconditioner.get(A)

        options_preconditioner = {} if preconditioner is None else {'preconditioner': preconditioner}

        diagonal_central_term = (A.T @ self.central_freq_op @ A).I(
            solver=lx.CG(
                rtol=self.dictionary_parameters_CG['tol_CG'], 
                atol=self.dictionary_parameters_CG['tol_CG'], 
                max_steps=self.dictionary_parameters_CG['max_steps_CG']
            ), 
            callback=lambda x: print("Number of iterations in CG:", x.stats['num_steps'], flush=True),
            **options_preconditioner
        )
        first_central_term = diagonal_central_term(AOND)

        return A, first_central_term, AOND, diagonal_central_term

    def get_spectral_likelihood(self):
        """
            Compute the log-proba given a set of parameters 
        """

        get_A_s_AOND = self.get_A_s_AOND

        @equinox.filter_custom_jvp
        def spectral_likelihood_custom_gradient(params):
            """
                Compute the log-proba given a set of parameters 
            """

            # Select parameters which are to be estimated 
            parameters_dict = self.complete_params(params)

            _, map_s, AOND, _ = get_A_s_AOND(params=parameters_dict, right_member=self.right_member)

            logL = -dot_2(AOND, map_s)

            # Compute the negative log-likelihood
            return logL

        @spectral_likelihood_custom_gradient.def_jvp
        def custom_gradient(primals, tangents):
            """
            Custom JVP rule for the perturbative_negative_log_prob function.
            """
            params, = primals
            beta_tau, = tangents

            # Select parameters which are to be estimated 
            parameters_dict = self.complete_params(params)

            A, map_s, AOND, _ = get_A_s_AOND(params=parameters_dict, right_member=self.right_member)

            logL = -dot_2(AOND, map_s)

            A_deriv = create_MixingMatrixOperator_deriv(
                parameters_dict=parameters_dict,
                **self.dictionary_mixing_matrix,   
            )

            left_hand_term = self.right_member - (self.central_freq_op @ A)(map_s)

            keys_params = params.keys()

            final_grad_log = 0
            for key in keys_params:
                patch_key = f'{key}_patches'
                if self.patch_indices is not None and self.patch_indices.get(patch_key) is not None:
                    raise Exception("Multi-patch version not validated")
                
                    # Per-patch parameter: weight each pixel by the tangent of its patch
                    tangent_per_pixel = beta_tau[key][self.patch_indices[patch_key]]
                    A_deriv_tangent = get_A_from_array(
                        jnp.moveaxis(get_array_from_A(A_deriv[key], self.n_pix) * tangent_per_pixel, 1, 0),
                        self.structure_sed
                    )
                    final_grad_log += -2*dot_2(A_deriv_tangent(map_s), left_hand_term)
                else:
                    final_grad_log += -2*dot_2(A_deriv[key](map_s), left_hand_term) * beta_tau[key]
            return logL, final_grad_log

        return spectral_likelihood_custom_gradient
