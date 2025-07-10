import numpy as np
import scipy
from scipy.sparse import csr_matrix

def get_data_ind_j(matrix: scipy.sparse.sparray, path_save):
    """ Retrieve the data and indices of the non-zero elements of a sparse matrix
    and return them in a format that can be used to create a furax operator.
    """

    print("## Getting data and indices", flush=True)
    ind_i, ind_j, data = scipy.sparse.find(matrix)
    
    print("## Getting unique values and counts", flush=True)
    values_i, first_indices_i, counts_i = np.unique(ind_i, return_counts=True, return_index=True)
    print("## Getting unique values and counts -- 2", flush=True)
    values_j, first_indices_j, counts_j = np.unique(ind_j, return_counts=True, return_index=True)
    print("## Getting unique values and counts -- 3", flush=True)
    
    n_data = len(values_i)
    n_pix = n_data//2
    if counts_i.size != 0:
        max_counts_i = np.max(counts_i)
        max_counts_j = np.max(counts_j)
    else:
        max_counts_i = 0
        max_counts_j = 0
        
    new_ind_i = np.zeros((len(values_i), max_counts_i), dtype=np.int32)
    new_data_i = np.zeros((len(values_i), max_counts_i), dtype=np.float64) # Filling the diagonal data with 0
    new_data_j = np.zeros((len(values_j), max_counts_j), dtype=np.float64) # Filling the diagonal data with 0
    new_ind_j = np.zeros((len(values_j), max_counts_j), dtype=np.int64)

    first_indices_i = np.append(first_indices_i, ind_j.size + 1)
    first_indices_j = np.append(first_indices_j, ind_i.size + 1)
    print("## Entering loop", flush=True)
    # for j in range(2*n_pix):
    for j in tqdm(range(n_data)): # Loop on the indices in the mask

        #if j % 2000 == 0:
        #    print(f"------ Looping {j}/{n_data}", flush=True)
        new_ind_i[j,:counts_i[j]] = ind_j[first_indices_i[j]:first_indices_i[j+1]]
        new_data_i[j,:counts_i[j]] = data[first_indices_i[j]:first_indices_i[j+1]]
        
        indices_ = np.where(ind_j == values_j[j])[0]
        new_ind_j[j,:counts_j[j]] = ind_i[indices_]
        new_data_j[j,:counts_j[j]] = data[indices_]
        
    print("Exiting loop", flush=True)


    new_data_i = np.array(new_data_i, dtype=np.float64)
    new_ind_i = np.array(new_ind_i, dtype=np.int32)

    new_data_j = np.array(new_data_j, dtype=np.float64)
    new_ind_j = np.array(new_ind_j, dtype=np.int32)

    print(f"## Saving to {path_save}", flush=True)
    np.savez(path_save, new_data_i=new_data_i, new_ind_i=new_ind_i, new_data_j=new_data_j, new_ind_j=new_ind_j, values_i=values_i, values_j=values_j, counts_i=counts_i, counts_j=counts_j)
