import argparse
import os, time

# Set the environment variable to avoid printing the error message
os.environ["EQX_ON_ERROR"] = "nan"

import numpy as np
import scipy as sp
import toml
from tqdm import tqdm
import jax
import jax.numpy as jnp
import healpy as hp

import megabuster

jax.config.update("jax_enable_x64", True)

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument(
    'params_toml', metavar='N', type=str, nargs='+', help='a toml file path to be added to the config'
)
args = parser.parse_args()
path_toml_params = args.params_toml[0]

print('Parsing from :', path_toml_params, flush=True)

assert os.path.exists(path_toml_params), "A path for the toml file containing parameters for the run and the sampler object must be provided"

with open(path_toml_params) as f:
    dictionary_parameters = toml.load(f)
f.close()

dictionary_path_inputs = dictionary_parameters['path_inputs']

path_precomputations = dictionary_path_inputs['path_precomputations']

path_nhits = dictionary_path_inputs['path_nhits']
# The nhits is used to build the analysis mask (here without inhomogeneities), which must be compatible with the observation matrices 

path_N = dictionary_path_inputs['path_N']

path_input_MSS2_obsmat = dictionary_path_inputs['path_directory_input_MSS2_obsmat']

# Prepare the paths of observation matrices 

path_save_obsmat_file = "sobs_RC1.r01_SAT{}_mission_f{:03d}_4way_{}_sky_obsmat_healpix"


all_incomplete_path_save_obsmat_file = [
    path_save_obsmat_file.format(4, freq, 'coadd') for freq in [30, 40]
]
all_incomplete_path_save_obsmat_file += [
    path_save_obsmat_file.format(1, freq, 'coadd') for freq in [90, 150]
]
all_incomplete_path_save_obsmat_file += [
    path_save_obsmat_file.format(3, freq, 'coadd') for freq in [230, 290]
]

all_path_save_obsmat_file = [
    path_input_MSS2_obsmat + path for path in all_incomplete_path_save_obsmat_file
]

# Preparing output paths

path_eigen_decomp_fname = [path_precomputations + path.replace('healpix', 'egeinvalue_precomputations.npz') for path in all_incomplete_path_save_obsmat_file]

list_obsmat_operator_fname = [path_precomputations + path + '_masked' for path in all_incomplete_path_save_obsmat_file]

all_path_save_obsmat_file_masked = [
    path_precomputations + path + '_masked' for path in all_incomplete_path_save_obsmat_file
]


# Setting up the parameters of the problems

nside = dictionary_parameters['nside']
npix = 12*nside**2
nstokes = dictionary_parameters['nstokes']

frequencies = np.array(dictionary_parameters['frequencies'])

n_freq = len(frequencies)

instrument = megabuster.helper.get_instrument('SO_SAT')

# Retrieving the binary mask from the nhits map

def make_mask(nhits_path, zero_threshold, nside):
    nhits_map = hp.read_map(nhits_path)
    nhits_smoothed = hp.smoothing(
        hp.ud_grade(nhits_map, nside, power=-2, dtype=np.float64),
        fwhm=np.pi/180)
    nhits_smoothed[nhits_smoothed < 0] = 0
    nhits_smoothed /= np.amax(nhits_smoothed)
    binary_mask = np.zeros_like(nhits_smoothed)
    binary_mask[nhits_smoothed > zero_threshold] = 1
    fsky = np.sum(binary_mask) / len(binary_mask)

    return binary_mask, fsky

nside_out = dictionary_parameters['nside'] # Output nside for the mask

zero_threshold = 0.45
mask_B, fsky = make_mask(path_nhits, zero_threshold, nside=nside_out)

theta, phi = hp.pix2ang(nside=nside_out, ipix=np.arange(mask_B.size), lonlat=True)

mask_south = np.zeros_like(mask_B)

mask_south[theta < 115] = 1
mask_south[theta > 280] = 1
mask_south[phi < -35] = 1
mask_south[phi > -15] = 0

mask_B_ring = mask_B * mask_south
mask_B_nest = hp.reorder(mask_B_ring, r2n=True)

mask_indices = mask_B_nest != 0

indices_mask = np.arange(npix)[mask_B_nest != 0]

mask_stacked_nest = np.hstack((indices_mask + npix, indices_mask + 2 * npix))

n_pix = int(mask_B_nest.sum())

f_sky = mask_B_nest.sum() / mask_B_nest.size

# Loading the noise covariance matrix

# Preparing the inverse noise covariance matrix
print("Loading the inverse noise covariance matrix from", path_N, flush=True)
def load_N_matrix(path_N, nside):
    N_matrix = np.load(path_N)
    if N_matrix.ndim != 2:
        output = []
        for i in range(N_matrix.shape[0]):
            output.append(hp.ud_grade(N_matrix[i], nside_out=nside, power=2))
        N_matrix = np.array(output)
    return N_matrix

N_matrix = load_N_matrix(path_N, nside)

noisecov_QU_masked = N_matrix[:,1:,:] * mask_B_ring
inverse_noisecov_QU_masked = np.zeros_like(noisecov_QU_masked)
inverse_noisecov_QU_masked[noisecov_QU_masked != 0] = 1./noisecov_QU_masked[noisecov_QU_masked != 0]

inverse_noisecov_QU_masked_nested = np.zeros_like(inverse_noisecov_QU_masked)
for i in range(n_freq):
    inverse_noisecov_QU_masked_nested[i,...] = hp.reorder(inverse_noisecov_QU_masked[i], r2n=True)

inverse_noisecov_QU_masked_nested = jnp.array(inverse_noisecov_QU_masked_nested)

# Preparing observation matrices for the precomputations --------------------------------

mask_B_nest_x3 = np.array([mask_B_nest,mask_B_nest,mask_B_nest]).flatten()

mask_diag = sp.sparse.diags(mask_B_nest_x3)

time_start = time.time()
for path in tqdm(all_path_save_obsmat_file): #[:2]

    print(path_input_MSS2_obsmat+path)
    filename_only = path.split("/")[-1]
    fname_output = path_precomputations + filename_only + "_masked.npz"
    print(fname_output)

    if os.path.exists(fname_output):
        print(f"## File {fname_output} already exists, skipping masking.", flush=True)
        continue

    full_matrix = sp.sparse.load_npz(path + ".npz")
    # full_matrix = megabuster.io.load_all_obsmat(
    #         [path_input_MSS2_obsmat + path + '.npz'],
    #         size_obsmat=3 * npix,
    #         mask_stacked=mask_stacked_nest,
    #         kind="precomputations_scipy",
    #     )
    masked_obsmat = mask_diag @ full_matrix @ mask_diag
    
    print('Saving masked obsmat...')
    sp.sparse.save_npz(fname_output, masked_obsmat)

print("Time taken to save masked obsmat files:", time.time() - time_start, "seconds")

time_start = time.time()
print("Starting saving obsmat precomputation...", flush=True)
megabuster.precomputations.save_obsmat_precomputation(
            list_path_obsmat_input=all_path_save_obsmat_file_masked, #all_path_save_obsmat_file_masked,
            list_path_obsmat_output=list_obsmat_operator_fname, # obsmat_operator_fname_unmasked -- not unmasked
        )

print("Time taken to save obsmat precomputation:", time.time() - time_start, "seconds")

time_start = time.time()
print("Loading the observation matrices...", flush=True)
obsmat_sp = megabuster.io.load_all_obsmat(
            list_obsmat_operator_fname,
            size_obsmat=3 * npix,
            mask_stacked=mask_stacked_nest,
            kind="precomputations_scipy",
        )
print("Time taken to load obsmat precomputation:", time.time() - time_start, "seconds")

for idx_freq in range(n_freq):
    assert (hp.reorder(inverse_noisecov_QU_masked[idx_freq], r2n=True)[...,mask_B_nest!=0].ravel() != 0).all(), "The inverse noise covariance matrix contains zero values, which will cause issues in the eigendecomposition. Please check the input noise covariance matrix and the mask."
print(f"Min value noise covariance matrix on the masked pixels: {np.min(inverse_noisecov_QU_masked[...,mask_B_ring!=0])}")

time_start = time.time()
print("Starting computing eigenspectrum of frequency central operator...", flush=True)
megabuster.precomputations.compute_eigenspectrum_from_matrices(
    obsmat_sp,
    inverse_noisecov_QU_masked,
    mask_B_nest,
    path_output="",
    list_name_output=path_eigen_decomp_fname,
)
print("Time taken to compute eigenspectrum:", time.time() - time_start, "seconds")
print("Precomputations done.")