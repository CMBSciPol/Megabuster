import argparse
import os, time

# Set the environment variable to avoid printing the error message
os.environ["EQX_ON_ERROR"] = "nan"

import numpy as np
import scipy as sp
import toml
import jax
import jax.numpy as jnp
import healpy as hp
from opt_einsum import contract

from fgbuster import get_observation, get_noise_realization

from furax.obs.stokes import Stokes

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

# 

name_version = dictionary_parameters['name_version']

dictionary_minimiser = dictionary_parameters['dictionary_minimiser']

dictionary_path_inputs = dictionary_parameters['path_inputs']

dictionary_parameters_CG = dictionary_parameters.get('dictionary_parameters_CG', dict())

# Prepare the inputs from the config file

options_minimizer = dictionary_minimiser.get('options', dict())

print("Options for the minimizer:", options_minimizer, flush=True)

path_output = dictionary_path_inputs['path_output'] + f'/{name_version}/'

if not os.path.exists(path_output):
    print(f"Creating output directory at {path_output}")
    os.makedirs(path_output)
else:
    print(f"Output directory {path_output} already exists, files might be overwritten!")

# Prepare the paths of the input data


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

path_precomputations = dictionary_path_inputs['path_precomputations']
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

# Generating input maps

np.random.seed(42)
input_foregrounds_maps = get_observation(instrument, 'd0s0', nside=nside, noise=False)

np.random.seed(42)
input_cmb = get_observation(instrument, 'c1', nside=nside, noise=False)

if dictionary_parameters['add_noise']:
    np.random.seed(41)
    input_noise = get_noise_realization(
        instrument=instrument,
        nside=nside
    )[...,1:,:]
    if dictionary_parameters.get('filter_noise', False):
        print("The noise maps will be filtered with the observation matrices for the filtered case...")

else:
    input_noise = 0

input_freq_maps = input_cmb + input_foregrounds_maps

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
# ERROR WAS HERE!!!! Maybe add check that mask indices and mask stacked is compatible with obsmat?
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



time_start = time.time()
print("Loading the observation matrices...", flush=True)
obsmat_sp = megabuster.io.load_all_obsmat(
            list_obsmat_operator_fname,
            size_obsmat=3 * npix,
            mask_stacked=mask_stacked_nest,
            kind="precomputations_scipy",
        )
print("Time taken to load obsmat precomputation:", time.time() - time_start, "seconds")


# Building FURAX operators
freq_obsmat_operator_T = megabuster.io.build_obsmat_operator_from_flattened_matrices(
                obsmat_sp,
                nstokes=2,
                return_transpose=True,
            )

inverse_full_matrix_central =  megabuster.io.load_matrix_precond(
    path_eigen_decomp_fname, 
    power_diagonal=-1
)

central_freq_op =  megabuster.tools.get_dense_furax_operator_from_freq_array(
                megabuster.io.load_matrix_precond(
                    path_eigen_decomp_fname, 
                    power_diagonal=1
                )
            )


print("Filtering the input maps...", flush=True)
freq_maps_preprocessed_QU_masked = np.zeros((n_freq, 3, npix))
input_noise_for_filtered_case = np.zeros((n_freq, 3, npix))

for i in range(n_freq):
    print(i)

    filtered_maps = obsmat_sp[i].dot(
            hp.reorder(
                    input_freq_maps[i,1:,...],r2n=True
                )[...,mask_B_nest!=0].ravel()
            ).reshape((2,n_pix))    
    filtered_maps_extended = np.zeros((3, npix))
    filtered_maps_extended[1:,mask_B_nest!=0] = filtered_maps
    freq_maps_preprocessed_QU_masked[i] = hp.reorder(filtered_maps_extended, n2r=True)

    if dictionary_parameters['add_noise'] and dictionary_parameters.get('filter_noise', False):
        filtered_noise = obsmat_sp[i].dot(
                hp.reorder(
                        input_noise[i,:,...],r2n=True
                    )[...,mask_B_nest!=0].ravel()
                ).reshape((2,n_pix))    
        filtered_noise_extended = np.zeros((3, npix))
        filtered_noise_extended[1:,mask_B_nest!=0] = filtered_noise
        input_noise_for_filtered_case[i] = hp.reorder(filtered_noise_extended, n2r=True)

if dictionary_parameters['add_noise'] and not dictionary_parameters.get('filter_noise', False):
    input_noise_for_filtered_case = input_noise

freq_maps_preprocessed_QU_masked = freq_maps_preprocessed_QU_masked[:,1:,:] * mask_B_ring

if dictionary_parameters['add_noise'] and dictionary_parameters.get('filter_noise', False):
    input_noise_for_filtered_case = input_noise_for_filtered_case[:,1:,:] * mask_B_ring


print("Performing component separation 'sanity' with noise...")

time_start = time.time()
res_sanity = megabuster.compsep.perform_compsep(
        first_guess_params={'beta_dust': np.array(1.54), 'beta_pl': np.array(-3.0)},
        fixed_params={'temp_dust': 20.0},
        sky_map=input_freq_maps[:,1:,:] * mask_B_ring + input_noise,
        frequencies=frequencies,
        invN_matrix=inverse_noisecov_QU_masked,
        do_minimization=True,
        binary_mask=mask_B_ring,
        obs_mat_operator=None, 
        obsmat_operator_rhs=None,
        use_preconditioner_diag=True,
        use_preconditioner_pinv=False,
        matrix_precond=None,
        central_freq_op=None,
        dictionary_parameters_minimization={
            'max_iter':dictionary_minimiser.get('max_iter', 20),
            'tol':dictionary_minimiser.get('tol', 1e-5),
            'solver_name': dictionary_minimiser.get('solver_name', "optax_lbfgs"),
            'options': options_minimizer
        },
        dictionary_parameters_CG={
            'max_steps_CG':dictionary_parameters_CG.get('max_steps_CG', 200), 
            'tol_CG':dictionary_parameters_CG.get('tol_CG', 1e-6)
        },
        ordering_parameter=['beta_dust', 'beta_pl'], 
        ordering_component=['cmb', 'dust', 'synchrotron'],
    )
print("Time taken for component separation sanity test:", time.time() - time_start, "seconds")

print("Results of component separation for sanity test:", res_sanity.x, flush=True)

print("Performing component separation filtered without correction...")
time_start = time.time()
res_filtered_no_correction = megabuster.compsep.perform_compsep(
        first_guess_params={
            'beta_dust': np.array(1.54), 'beta_pl': np.array(-3.0)
        },
        fixed_params={'temp_dust': 20.0},
        sky_map=freq_maps_preprocessed_QU_masked + input_noise_for_filtered_case,
        frequencies=frequencies,
        invN_matrix=inverse_noisecov_QU_masked,
        do_minimization=True,
        binary_mask=mask_B_ring,
        obs_mat_operator=None, 
        obsmat_operator_rhs=None,
        use_preconditioner_diag=True,
        use_preconditioner_pinv=False,
        matrix_precond=None,
        central_freq_op=None,
        dictionary_parameters_minimization={
            'max_iter':dictionary_minimiser.get('max_iter', 20),
            'tol':dictionary_minimiser.get('tol', 1e-5),
            'solver_name': dictionary_minimiser.get('solver_name', "optax_lbfgs"),
            'options': options_minimizer
        },
        dictionary_parameters_CG={
            'max_steps_CG':200, 'tol_CG':1e-6
        },
        ordering_parameter=['beta_dust', 'beta_pl'], 
        ordering_component=['cmb', 'dust', 'synchrotron'],
    )
print("Time taken for component separation filtered without correction:", time.time() - time_start, "seconds")

print("Results of component separation filtered without correction:", res_filtered_no_correction.x, flush=True)

time_start = time.time()
res_with_correction = megabuster.compsep.perform_compsep( 
        first_guess_params={
            'beta_dust': np.array(1.54), 'beta_pl': np.array(-3.0)
        },
        fixed_params={'temp_dust': 20.0},
        sky_map=freq_maps_preprocessed_QU_masked + input_noise_for_filtered_case,
        frequencies=frequencies,
        invN_matrix=inverse_noisecov_QU_masked,
        do_minimization=True,
        binary_mask=mask_B_ring,
        obs_mat_operator=None, 
        obsmat_operator_rhs=freq_obsmat_operator_T, 
        use_preconditioner_diag=False,
        use_preconditioner_pinv=True,
        matrix_precond=inverse_full_matrix_central,
        central_freq_op=central_freq_op,
        dictionary_parameters_minimization={
            'max_iter':dictionary_minimiser.get('max_iter', 20),
            'tol':dictionary_minimiser.get('tol', 1e-5),
            'solver_name': dictionary_minimiser.get('solver_name', "optax_lbfgs"),
            'options': options_minimizer
        },
        dictionary_parameters_CG={
            'max_steps_CG':200, 'tol_CG':1e-6
        },
        ordering_parameter=['beta_dust', 'beta_pl'], 
        ordering_component=['cmb', 'dust', 'synchrotron'],
    )
print("Time taken for component separation with correction:", time.time() - time_start, "seconds")

print("Results of component separation filtered with correction:", res_with_correction.x, flush=True)

dictionary_output_maps = {
    'sanity': res_sanity.s,
    'filtered_no_correction': res_filtered_no_correction.s,
    'filtered_with_correction': res_with_correction.s
}
dictionary_output_params = {
    'sanity': res_sanity.x,
    'filtered_no_correction': res_filtered_no_correction.x,
    'filtered_with_correction': res_with_correction.x
}

print("Saving the output maps to", path_output+ "results_compsep_filtered_compsep_maps.npz", flush=True)
np.savez(path_output + "results_compsep_filtered_compsep_maps.npz", **dictionary_output_maps)

print("Saving the output parameters to", path_output+ "results_compsep_filtered_compsep_params.npz", flush=True)
np.savez(path_output + "results_compsep_filtered_compsep_params.npz", **dictionary_output_params)
