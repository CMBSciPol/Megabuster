import numpy as np
import jax
import jax.numpy as jnp
from jaxtyping import Inexact, PyTree, ArrayLike
import scipy

from furax.obs.stokes import StokesQU
from furax import AbstractLinearOperator, square
from furax.core import DiagonalOperator, IndexOperator, BlockColumnOperator, BlockRowOperator, BlockDiagonalOperator, SumOperator, TransposeOperator

__all__ = [
    'get_data_ind_j',
    'assemble_ind_j',
    'ObsMatOperator',
    'to_QU_operators',
    'to_QU_operators_from_files',
]
    
def get_data_ind_j(n_pix, list_matrices: list, return_operator=True):
    """ Retrieve the data and indices of the non-zero elements of a sparse matrix
    and return them in a format that can be used to create a furax operator.
    """
    all_values_i = []
    all_values_j = []
    all_counts_i = []
    all_counts_j = []
    all_first_indices_i = []
    all_first_indices_j = []
    all_ind_i = []
    all_ind_j = []
    all_data = []

    max_counts_i = 0
    max_counts_j = 0

    max_len_values_i = 0
    max_len_values_j = 0

    n_freq = len(list_matrices)

    for matrix in list_matrices:
        
        ind_i, ind_j, data = scipy.sparse.find(matrix)
        
        
        values_i, first_indices_i, counts_i = np.unique(ind_i, return_counts=True, return_index=True)
        
        values_j, first_indices_j, counts_j = np.unique(ind_j, return_counts=True, return_index=True)
        

        all_values_i.append(values_i)
        all_values_j.append(values_j)
        all_counts_i.append(counts_i)
        all_counts_j.append(counts_j)
        all_first_indices_i.append(first_indices_i)
        all_first_indices_j.append(first_indices_j)
        all_ind_i.append(ind_i)
        all_ind_j.append(ind_j)
        all_data.append(data)
        
        if counts_i.size != 0:
            max_counts_i = max(np.max(counts_i),max_counts_i)
            max_counts_j = max(np.max(counts_j), max_counts_j)

            max_len_values_i = max(len(values_i), max_len_values_i)
            max_len_values_j = max(len(values_j), max_len_values_j)

    max_counts_i = max(max_counts_i, max_counts_j)
    max_counts_j = max(max_counts_i, max_counts_j)

    max_len_values_i = max(max_len_values_i, max_len_values_j)
    max_len_values_j = max(max_len_values_i, max_len_values_j)
        
    
    new_ind_i = np.zeros((n_freq, max_len_values_i, max_counts_i), dtype=np.int32)
    new_data_i = np.zeros((n_freq, max_len_values_i, max_counts_i), dtype=np.float64) # Filling the diagonal data with 0
    new_data_j = np.zeros((n_freq, max_len_values_j, max_counts_j), dtype=np.float64) # Filling the diagonal data with 0
    new_ind_j = np.zeros((n_freq, max_len_values_j, max_counts_j), dtype=np.int64)

    for i, matrix in enumerate(list_matrices):
        n_data = len(all_values_i[i])
        
        #n_pix = matrix.shape[0]
        
        first_indices_i = np.append(all_first_indices_i[i], all_ind_j[i].size + 1)
        first_indices_j = np.append(all_first_indices_j[i], all_ind_i[i].size + 1)
        # print("## Entering loop", flush=True)
        # for j in range(2*n_pix):
        # for j in tqdm(range(n_data)): # Loop on the indices in the mask
        for j in range(n_data): # Loop on the indices in the mask

            new_ind_i[i, j, :all_counts_i[i][j]] = all_ind_j[i][first_indices_i[j]:first_indices_i[j+1]]
            new_data_i[i, j, :all_counts_i[i][j]] = all_data[i][first_indices_i[j]:first_indices_i[j+1]]
            
            indices_ = np.where(all_ind_j[i] == all_values_j[i][j])[0]
            new_ind_j[i, j, :all_counts_j[i][j]] = all_ind_i[i][indices_]
            new_data_j[i, j, :all_counts_j[i][j]] = all_data[i][indices_]


    # print("Exiting loop", flush=True)
    new_ind_ib = jnp.broadcast_to(jnp.arange(n_freq, dtype=jnp.int32), (max_counts_i, max_len_values_i, n_freq)).T
    new_ind_jb = jnp.broadcast_to(jnp.arange(n_freq, dtype=jnp.int32), (max_counts_i, max_len_values_i, n_freq)).T
    new_ind_i = jnp.array(new_ind_i, dtype=np.int32)
    new_ind_j = jnp.array(new_ind_j, dtype=np.int32)

    new_ind_i = (new_ind_ib, new_ind_i)
    new_ind_j = (new_ind_jb, new_ind_j)

    new_data_i = jnp.array(new_data_i, dtype=np.float64)
    # new_ind_i = jnp.array(new_ind_i, dtype=np.int32)

    new_data_j = jnp.array(new_data_j, dtype=np.float64)
    # new_ind_j = jnp.array(new_ind_j, dtype=np.int32)

    if return_operator:
        return get_operators_obsmat(n_freq, n_pix, new_data_i, new_ind_i, new_data_j, new_ind_j)
    return new_data_i, new_ind_i, new_data_j, new_ind_j, all_values_i, all_values_j, all_counts_i, all_counts_j

def assemble_ind_j(list_obs_mat_path, n_pix):
    """ Retrieve the data and indices of the non-zero elements of a sparse matrix
    and return them in a format that can be used to create a furax operator.
    """

    n_freq = len(list_obs_mat_path)

    all_values_i = []
    all_values_j = []
    all_counts_i = []
    all_counts_j = []
    all_new_ind_i = []
    all_new_ind_j = []
    all_new_data_i = []
    all_new_data_j = []

    max_counts_i = 0
    max_counts_j = 0

    max_len_values_i = 0
    max_len_values_j = 0


    for i in range(n_freq):
        decomposed_arrays = np.load(list_obs_mat_path[i])
        
        all_values_i.append(decomposed_arrays['values_i'])
        all_values_j.append(decomposed_arrays['values_j'])
        all_counts_i.append(decomposed_arrays['counts_i'])
        all_counts_j.append(decomposed_arrays['counts_j'])
        all_new_ind_i.append(decomposed_arrays['new_ind_i'])
        all_new_ind_j.append(decomposed_arrays['new_ind_j'])
        all_new_data_i.append(decomposed_arrays['new_data_i'])
        all_new_data_j.append(decomposed_arrays['new_data_j'])

        if all_counts_i[i].size != 0:
            max_counts_i = max(np.max(all_counts_i[i]),max_counts_i)
            max_counts_j = max(np.max(all_counts_j[i]), max_counts_j)

            max_len_values_i = max(len(all_values_i[i]), max_len_values_i)
            max_len_values_j = max(len(all_values_j[i]), max_len_values_j)
    
    max_counts_i = max(max_counts_i, max_counts_j)
    max_counts_j = max(max_counts_i, max_counts_j)

    max_len_values_i = max(max_len_values_i, max_len_values_j)
    max_len_values_j = max(max_len_values_i, max_len_values_j)
    
    new_ind_i = np.zeros((n_freq, max_len_values_i, max_counts_i), dtype=np.int32)
    new_data_i = np.zeros((n_freq, max_len_values_i, max_counts_i), dtype=np.float64) # Filling the diagonal data with 0
    new_data_j = np.zeros((n_freq, max_len_values_j, max_counts_j), dtype=np.float64) # Filling the diagonal data with 0
    new_ind_j = np.zeros((n_freq, max_len_values_j, max_counts_j), dtype=np.int64)

    for i in range(n_freq):
        n_data = len(all_values_i[i])
        
        for j in range(n_data): # Loop on the indices in the mask

            new_ind_i[i, j, :all_counts_i[i][j]] = all_new_ind_i[i][j, :all_counts_i[i][j]]
            new_data_i[i, j, :all_counts_i[i][j]] = all_new_data_i[i][j, :all_counts_i[i][j]]

            new_ind_j[i, j, :all_counts_j[i][j]] = all_new_ind_j[i][j, :all_counts_j[i][j]]
            new_data_j[i, j, :all_counts_j[i][j]] = all_new_data_j[i][j, :all_counts_j[i][j]]


    # print("Exiting loop", flush=True)
    new_ind_ib = jnp.broadcast_to(jnp.arange(n_freq, dtype=jnp.int32), (max_counts_i, max_len_values_i, n_freq)).T
    new_ind_jb = jnp.broadcast_to(jnp.arange(n_freq, dtype=jnp.int32), (max_counts_i, max_len_values_i, n_freq)).T
    new_ind_i = jnp.array(new_ind_i, dtype=np.int32)
    new_ind_j = jnp.array(new_ind_j, dtype=np.int32)

    new_ind_i = (new_ind_ib, new_ind_i)
    new_ind_j = (new_ind_jb, new_ind_j)

    new_data_i = jnp.array(new_data_i, dtype=np.float64)
    # new_ind_i = jnp.array(new_ind_i, dtype=np.int32)

    new_data_j = jnp.array(new_data_j, dtype=np.float64)
    # new_ind_j = jnp.array(new_ind_j, dtype=np.int32)
    
    return get_operators_obsmat(n_freq, n_pix, new_data_i, new_ind_i, new_data_j, new_ind_j)


def get_operators_obsmat(n_freq, n_pix, new_data_i, new_ind_i, new_data_j, new_ind_j):

    max_counts_i = new_data_i.shape[-1]
    max_counts_j = new_data_j.shape[-1]
    if max_counts_i != 0:
        new_data_i = jnp.array(new_data_i, dtype=np.float64) # np
        new_data_j = jnp.array(new_data_j, dtype=np.float64) # np
    else:
        # Case where there are no non-zero elements in the matrix
        max_counts_i = 0
        max_counts_j = 0

        new_ind_i = 2*(jnp.zeros((n_freq, n_pix, 0), dtype=np.int32),)
        new_ind_j = 2*(jnp.zeros((n_freq, n_pix, 0), dtype=np.int32),)
        new_data_i = jnp.zeros((n_freq, n_pix, max_counts_i), dtype=np.float64) # Filling the diagonal data with 0 # np
        new_data_j = jnp.zeros((n_freq, n_pix, max_counts_j), dtype=np.float64) # Filling the diagonal data with 0 # np
        
    
    # print("## Creating first batch of operators", flush=True)
    sum_i_op = SumOperator(axis=-1, in_structure=jax.ShapeDtypeStruct((n_freq, n_pix, max_counts_i), dtype=new_data_i.dtype))
    data_i_op = DiagonalOperator(new_data_i, in_structure=jax.ShapeDtypeStruct((n_freq, n_pix, max_counts_i), dtype=new_data_i.dtype))
    indices_i_op = IndexOperator(new_ind_i, in_structure=jax.ShapeDtypeStruct((n_freq, n_pix,), dtype=new_data_i.dtype), out_structure=jax.ShapeDtypeStruct((n_freq, n_pix, max_counts_i), dtype=new_data_i.dtype))

    # print("## Creating second batch of operators", flush=True)
    sum_j_op = SumOperator(axis=-1, in_structure=jax.ShapeDtypeStruct((n_freq, n_pix, max_counts_j), dtype=new_data_i.dtype))
    data_j_op = DiagonalOperator(new_data_j, in_structure=jax.ShapeDtypeStruct((n_freq, n_pix, max_counts_j), dtype=new_data_i.dtype))
    indices_j_op = IndexOperator(new_ind_j, in_structure=jax.ShapeDtypeStruct((n_freq, n_pix,), dtype=new_data_i.dtype), out_structure=jax.ShapeDtypeStruct((n_freq, n_pix, max_counts_j), dtype=new_data_i.dtype))

    return sum_i_op @ data_i_op @ indices_i_op, sum_j_op @ data_j_op @ indices_j_op


@square
class ObsMatOperator(AbstractLinearOperator):
    """
    """
    
    operator_i: AbstractLinearOperator
    operator_j: AbstractLinearOperator

    def __init__(
        self,
        operator_i: ArrayLike,
        operator_j: ArrayLike,
    ):  
        # print("New ObsMatOperator", flush=True)
        self.operator_i = operator_i
        self.operator_j = operator_j
    
    @classmethod
    def from_list_scipysparse(cls, n_pix, list_matrices: list):
        """ Create a new ObsMatOperator from a lsit of scipy sparse matrix """
        print("Building observation matrix operator!", flush=True)
        operator_i, operator_j = get_data_ind_j(n_pix, list_matrices, return_operator=True)
        return cls(operator_i=operator_i, operator_j=operator_j)

    @classmethod
    def from_path_decomposed_arrays(cls, list_obs_mat_path, n_pix):
        """ Create a new ObsMatOperator from decomposed arrays stored in files """
        operator_i, operator_j = assemble_ind_j(list_obs_mat_path, n_pix)
        return cls(operator_i=operator_i, operator_j=operator_j)

    def in_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self.operator_i.in_structure()
    
    def transpose(self) -> AbstractLinearOperator:
        return ObsMatOperatorTransposeOperator(self)

    @jax.jit
    def mv(self, x: PyTree[Inexact[jax.Array, '...']]) -> Inexact[jax.Array, '...']:
        return self.operator_i(x)

class ObsMatOperatorTransposeOperator(TransposeOperator):
    operator: ObsMatOperator

    @jax.jit
    def mv(self, x: PyTree[Inexact[jax.Array, ' _a']]) -> PyTree[Inexact[jax.Array, ' _a']]:
        return self.operator.operator_j(x)


def chop_up(list_obs_mat, n_stokes, n_pix):
    """ Chop up obs_mat into (n_stokes, n_stokes) chunks
    and convert each to furax operators. """
    n_pix = list_obs_mat[0].shape[0] // n_stokes

    
    list_all_op = []
    for i in range(n_stokes):
        intermediary = []
        for j in range(n_stokes):
            intermediary.append([obs_mat[i*n_pix:(i+1)*n_pix,:][:,j*n_pix:(j+1)*n_pix] for obs_mat in list_obs_mat])
        list_all_op.append(intermediary)
    


    return [[ObsMatOperator.from_list_scipysparse(n_pix, list_all_op[i][j])
                for j in range(n_stokes)] for i in range(n_stokes)]



def to_QU_operators(obs_mats, n_pix):
    all_ops = chop_up(obs_mats, n_stokes=2, n_pix=n_pix)

    ops_Q = BlockRowOperator(StokesQU(all_ops[0][0], all_ops[0][1]))
    ops_U = BlockRowOperator(StokesQU(all_ops[1][0], all_ops[1][1]))
    
    list_QU_operators = [ops_Q, ops_U]

    return BlockColumnOperator(StokesQU(*list_QU_operators))

def to_QU_operators_from_files(list_obs_mat_path, n_pix, n_stokes=2):


    new_list_paths = [[[list_obs_mat_path[f][i][j] for f in range(len(list_obs_mat_path))] for i in range(n_stokes)] for j in range(n_stokes)]

    all_ops = []

    for i in range(n_stokes):
        ops_list = []
        for j in range(n_stokes):
            ops_list.append(ObsMatOperator.from_path_decomposed_arrays(new_list_paths[i][j], n_pix))
        all_ops.append(ops_list)

    ops_Q = BlockRowOperator(StokesQU(all_ops[0][0], all_ops[0][1]))
    ops_U = BlockRowOperator(StokesQU(all_ops[1][0], all_ops[1][1]))
    
    # list_QU_operators = [BlockDiagonalOperator(ops_Q), BlockDiagonalOperator(ops_U)]
    list_QU_operators = [ops_Q, ops_U]

    return BlockColumnOperator(StokesQU(*list_QU_operators))
