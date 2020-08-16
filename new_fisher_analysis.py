from physics import Physics
from database_management import RunDatabaseManager
from run_camb_parallel import CAMBRun
from run_mpv_parallel import MPVRun
import os
from mpi4py import MPI
import time
import pandas as pd
import numpy as np
import dill
from mean_pairwise_velocity import mean_pairwise_velocity
from covariance import covariance_matrix as Cov
from numerics import interpolate
from generate_parameters import ParameterGenerator
from helpers import MyProcessPool, execute
import MPI_error_handler

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
name = MPI.Get_processor_name()


def chunk_evenly(lst, number_of_chunks):
    base_length = len(lst) // (number_of_chunks)
    remainder = len(lst) % number_of_chunks

    index = 0
    chunks = []
    for i in range(number_of_chunks):
        if remainder == 0:
            chunks.append(lst[index:index + base_length])
            index += base_length
        else:
            chunks.append(lst[index:index + base_length + 1])
            remainder -= 1
            index += base_length + 1

    return chunks


def load_results_file(path, success):
    if success:
        return np.load(path, allow_pickle=True)
    else:
        return None


def get_derivatives(mpv_database, ids_to_load, paramGen, r_vals, z_vals, interpolation_poly_deg):
    results_tmp = []
    paths=list(mpv_database['results_path'].loc[ids_to_load])
    success=list(mpv_database['successful_TF'].loc[ids_to_load])
    with MyProcessPool(min([12, len(ids_to_load)])) as p:
        results_tmp = list(p.imap(lambda i:load_results_file(paths[i], success[i]), list(range(len(paths)))))

    results = paramGen.regroup_results(results_tmp)
    derivatives = paramGen.get_derivatives(results, r_vals_eval=r_vals, z_vals=z_vals, deg=interpolation_poly_deg)
    return derivatives,results


def compute_fisher_matrix(mass_id, derivatives, inverse_covariance, Nz, Nr, n_params=6, mass_param_id=0):

    fisher_matrix = np.empty((n_params, n_params))
    for p in range(0, n_params):  # loop over different parameters
        for q in range(p, n_params):
            fisher = 0.0
            for k in range(Nz):
                for m in range(Nr):
                    for n in range(Nr):
                        if p == mass_param_id:
                            d1 = derivatives[p][mass_id][k][m]
                        else:
                            d1 = derivatives[p][k][m]

                        if q == mass_param_id:
                            d2 = derivatives[q][mass_id][k][n]
                        else:
                            d2 = derivatives[q][k][n]

                        fisher += d1 * inverse_covariance[k, m, n] * d2
            fisher_matrix[p, q], fisher_matrix[q, p] = fisher, fisher
    return fisher_matrix


def get_fisher_matrices(mpv_db, ids_to_load, paramGens_info, r_vals, z_vals, interpolation_poly_deg, axion_masses):
    fisher_matrices=[]
    derivatives=[]
    velocities=[]
    for i in range(len(paramGens_info)):
        paramGen=ParameterGenerator(*paramGens_info[i])
        ds,vs=get_derivatives(mpv_db, ids_to_load[i], paramGen, r_vals, z_vals, interpolation_poly_deg)
        derivatives.append(ds)
        velocities.append(vs)

        fisher_matrices_tmp=None
        with MyProcessPool(min([12, len(axion_masses)])) as p:
            fisher_matrices_tmp=list(p.imap(lambda mass_id: compute_fisher_matrix(mass_id, ds, inverse_cov_matrices, len(z_vals), len(r_vals), n_params=6, mass_param_id=0), list(range(len(axion_masses)))))
        fisher_matrices.append(fisher_matrices_tmp)

    return fisher_matrices,derivatives,velocities

STAGE_II=0
STAGE_III=1
STAGE_IV=2

if rank==0:
    run_code = "2020-08-02_sharp_k_lin_n=10_stageII"
    window = mean_pairwise_velocity.SHARP_K
    stage = STAGE_II

    run_database = RunDatabaseManager(os.getcwd() + "/camb_db.dat", os.getcwd() + "/mpv_db.dat")

    axion_masses = np.logspace(-33, -21, 12*10+1)
    n_steps=10
    step_sizes_log = np.logspace(-4, 0.0, 21)
    step_sizes_lin = np.linspace(0.0, 1.0, 21)[1:]
    step_sizes = np.unique(np.concatenate((step_sizes_log, step_sizes_lin), 0))
    interpolation_poly_deg=1

    phys = Physics(True, True, READ_H=False)

    OmegaMh2 = phys.Omega0 * phys.h ** 2
    OmegaBh2 = phys.OmegaB * phys.h ** 2

    omegaMh2_vals = np.linspace(OmegaMh2 - 2 * OmegaMh2 * 0.05, OmegaMh2 + 2 * OmegaMh2 * 0.05, 5)
    omegaBh2_vals = np.linspace(OmegaBh2 - 2 * OmegaBh2 * 0.05, OmegaBh2 + 2 * OmegaBh2 * 0.05, 5)
    h_vals = np.linspace(phys.h - 2 * phys.h * 0.05, phys.h + 2 * phys.h * 0.05, 5)
    ns_vals = np.linspace(phys.n - 2 * 0.005, phys.n + 2 * 0.005, 5)
    logAs_1010_vals = np.linspace(phys.logAs_1010 - 2 * phys.logAs_1010 * 0.05, phys.logAs_1010 + 2 * phys.logAs_1010 * 0.05, 5)

    #defining survey stages
    sigma_v_vals = [[310.0, 460.0, 560.0], [160.0, 200.0, 230.0], [120.0, 120.0, 120.0, 120.0, 130.0]]
    minClusterMass = [1.0e14, 1.0e14, 0.6e14]
    overlap_area = [4000.0, 6000.0, 10000.0]
    z_bin_no = [3,3,5]
    zmin_vals = [0.1, 0.1, 0.1]
    zmax_vals = [0.4, 0.4, 0.6]

    delta_r = 2.0
    r_vals = np.arange(20.0, 180.0, delta_r)/phys.h
    delta_r/=phys.h
    print(len(r_vals))
    zmin = zmin_vals[stage]
    zmax = zmax_vals[stage]
    Nz = z_bin_no[stage]
    z_step = (zmax - zmin) / Nz
    z_vals = np.linspace(zmin + z_step / 2, zmax - z_step / 2, Nz)

    print("Getting covariance matrix.")
    start_cov=time.time()
    filename = "../../data/axion_frac=0.000_matterpower_out.dat"

    P_CAMB = np.loadtxt(filename)
    p_interp = interpolate(P_CAMB[:, 0] * phys.h, P_CAMB[:, 1] / phys.h ** 3, phys.P_cdm, phys.P_cdm)



    try:
        cov = Cov.load_covariance("covariance_matrix_{}_{}.dat".format(['top-hat', 'gaussian', 'sharp-k', 'no-filter'][window], ["stageII", "stageIII", "stageIV"][stage]))
    except (IOError, OSError) as e:
        print("Covariance matrix not found. Recomputing...")
        cov = Cov(p_interp, None, zmin, zmax, Nz, r_vals, delta_r, overlap_area[stage] / 129600 * np.pi, sigma_v_vals[stage], kmin=min(P_CAMB[:, 0] * phys.h), kmax=max(P_CAMB[:, 0] * phys.h), mMin=minClusterMass[stage], mMax=1e16, physics=phys, window_function=window)
        cov.save_covariance("covariance_matrix_{}_{}.dat".format(['top-hat', 'gaussian', 'sharp-k', 'no-filter'][window], ["stageII", "stageIII", "stageIV"][stage]))

    inverse_cov_matrices=np.empty((Nz, len(r_vals), len(r_vals)))
    for i in range(Nz):
        inverse_cov_matrices[i]=cov.get_inverted_covariance(i)

    print("Total time take to get covariance matrices: {:.2}s".format(start_cov-time.time()))

    print("Generating paramters and accessing database")
    start_gen = time.time()
    run_database_ids=[]
    #parameter_generators=[]
    parameter_generators_info=[]

    for i in range(len(step_sizes)):
        run_database_ids_tmp = []
        axion_frac_vals = np.arange(0.0, min([step_sizes[i]*n_steps+1e-5, 1.0+1e-5]), step_sizes[i])
        #axion_frac_vals = np.linspace(0.0, step_sizes[i]*n_steps, n_steps + 1)

        paramGen_params=(axion_masses, axion_frac_vals, phys.f_axion, omegaMh2_vals, OmegaMh2, omegaBh2_vals, OmegaBh2, h_vals, phys.h, ns_vals, phys.n, logAs_1010_vals, phys.logAs_1010)
        parameter_generators_info.append(paramGen_params)

        paramGen = ParameterGenerator(*paramGen_params)
        #parameter_generators.append(paramGen)
        for param in paramGen.get_params():
            run_entry=run_database.add_run(param, z_vals, minClusterMass[stage], high_res=True, high_res_multiplier=5, jenkins_mass_function=False, window_function=window)
            run_database_ids_tmp.append(run_entry.name)

        run_database_ids.append(run_database_ids_tmp)

    run_database.save_database()
    print("Total time take to generate parameter sets: {:.2}s".format(start_gen - time.time()))


    camb_run_module = CAMBRun(run_database)
    camb_run_module.run()

    mpv_run_module = MPVRun(run_database)
    mpv_run_module.run()

    print("Computing derivatives and Fisher matrices")
    start_fisher=time.time()

    chunk = lambda lst, n: [lst[i:min([i + n, len(lst)])] for i in range(0, len(lst), n)]

    db_path = run_database.get_mpv_db_path()
    db_path = comm.bcast(db_path, root=0)

    derivative_info = (r_vals, z_vals, interpolation_poly_deg)
    derivative_info = comm.bcast(derivative_info, root=0)

    axion_masses = comm.bcast(axion_masses, root=0)

    inverse_cov_matrices = comm.bcast(inverse_cov_matrices, root=0)

    ids_to_load = chunk_evenly(run_database_ids, size)
    ids_to_load = comm.scatter(ids_to_load, root=0)

    paramGens_info = chunk_evenly(parameter_generators_info, size)
    paramGens_info = comm.scatter(paramGens_info, root=0)

    fisher_matrices, derivatives, velocities = get_fisher_matrices(run_database.get_mpv_database(), ids_to_load, paramGens_info, r_vals, z_vals, interpolation_poly_deg, axion_masses)

    flatten = lambda l: [item for sublist in l for item in sublist]
    velocities = comm.gather(velocities, root=0)
    velocities = flatten(velocities)

    derivatives = comm.gather(derivatives, root=0)
    derivatives = flatten(derivatives)

    fisher_matrices = comm.gather(fisher_matrices, root=0)
    fisher_matrices = flatten(fisher_matrices)
    print("Total time take to compute Fisher matrices: {:.2}s".format(start_fisher - time.time()))

    np.save("run_information_{}".format(run_code), (axion_masses, step_sizes, n_steps, interpolation_poly_deg))
    np.save("fisher_matrices_results_{}".format(run_code), fisher_matrices)
    np.save("derivatives_results_{}".format(run_code), derivatives)
    np.save("velocities_results_{}".format(run_code), velocities)

else:
    CAMBRun.run_child_nodes()
    MPVRun.run_child_nodes()

    db_path = None
    db_path = comm.bcast(db_path, root=0)

    derivative_info = None
    derivative_info = comm.bcast(derivative_info, root=0)
    r_vals, z_vals, interpolation_poly_deg = derivative_info

    axion_masses = None
    axion_masses = comm.bcast(axion_masses, root=0)

    inverse_cov_matrices = None
    inverse_cov_matrices = comm.bcast(inverse_cov_matrices, root=0)

    ids_to_load = None
    ids_to_load = comm.scatter(ids_to_load, root=0)

    paramGens_info = None
    paramGens_info = comm.scatter(paramGens_info, root=0)

    mpv_db = None
    with open(db_path, "rb") as f:
        mpv_db = dill.load(f)

    fisher_matrices, derivatives, velocities = get_fisher_matrices(mpv_db, ids_to_load, paramGens_info, r_vals, z_vals, interpolation_poly_deg, axion_masses)

    velocities = comm.gather(velocities, root=0)
    derivatives = comm.gather(derivatives, root=0)
    fisher_matrices = comm.gather(fisher_matrices, root=0)

    assert derivatives is None
    assert fisher_matrices is None


