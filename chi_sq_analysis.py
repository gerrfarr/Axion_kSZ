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
from helpers import MyProcessPool
import MPI_error_handler
from scipy.interpolate import interp1d

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

def compute_chi_sq(mpv_database, fiducial_vals, id_to_load, cov_path, r_eval_vals, Nz):
    path=mpv_database['results_path'].loc[id_to_load]
    success = mpv_database['successful_TF'].loc[id_to_load]

    if not success:
        return np.nan

    vals = load_results_file(path, success)
    cov = Cov.load_covariance(cov_path)
    chi_sq=0
    for i in range(Nz):
        inv_cov = cov.get_inverted_covariance(i)
        result_vs = interp1d(vals[0], vals[1][i])(r_eval_vals)
        chi_sq+=np.dot((fiducial_vals[i]-result_vs)**2, np.dot(inv_cov, (fiducial_vals[i]-result_vs)**2))/Nz

    return chi_sq


def get_chi_sq_vals(mpv_database, base_run_ids, run_ids, cov_path, r_eval_vals, Nz):
    chi_sq_vals = []
    for i in range(len(base_run_ids)):
        base_run_id=base_run_ids[i]
        fiducial_path = mpv_database['results_path'].loc[base_run_id]
        fiducial_success = mpv_database['successful_TF'].loc[base_run_id]

        if not fiducial_success:
            return None

        fiducial_vals=load_results_file(fiducial_path, fiducial_success)
        fiducial_vs = [interp1d(fiducial_vals[0], fiducial_vals[1][j])(r_eval_vals) for j in range(Nz)]

        with MyProcessPool(min([12, len(run_ids[i])])) as p:
            chi_sq_vals.append(list(p.imap(lambda run_id: compute_chi_sq(mpv_database, fiducial_vs, run_id, cov_path, r_eval_vals, Nz), run_ids[i])))

    return chi_sq_vals




STAGE_II=0
STAGE_III=1
STAGE_IV=2

if rank==0:
    run_code = "2020-07-29_sharp_k_lin_n=10_stageIV"
    window = mean_pairwise_velocity.SHARP_K
    stage = STAGE_IV

    run_database = RunDatabaseManager(os.getcwd() + "/camb_db.dat", os.getcwd() + "/mpv_db.dat")

    axion_masses = np.logspace(-27, -21, 6*10+1)
    axion_frac_vals_lin = np.linspace(0.0, 1.0, 15)
    axion_frac_vals_log = np.logspace(1e-3, 0.0, 15)
    axion_frac_vals = np.unique(np.concatenate((axion_frac_vals_lin, axion_frac_vals_log), 0))

    phys = Physics(True, True, READ_H=False)

    OmegaMh2 = phys.Omega0 * phys.h**2
    OmegaBh2 = phys.OmegaB * phys.h**2

    #defining survey stages
    sigma_v_vals = [[310.0, 460.0, 560.0], [160.0, 200.0, 230.0], [120.0, 120.0, 120.0, 120.0, 130.0]]
    minClusterMass = [1.0e14, 1.0e14, 0.6e14]
    overlap_area = [4000.0, 6000.0, 10000.0]
    z_bin_no = [3,3,5]
    zmin_vals = [0.1, 0.1, 0.1]
    zmax_vals = [0.4, 0.4, 0.6]

    delta_r = 2.0
    r_vals = np.arange(20.0, 180.0, delta_r) / phys.h
    delta_r /= phys.h
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

    cov_path="covariance_matrix_{}_{}.dat".format(['top-hat', 'gaussian', 'sharp-k', 'no-filter'][window], ["stageII", "stageIII", "stageIV"][stage])
    try:
        cov = Cov.load_covariance(cov_path)
    except (IOError, OSError) as e:
        print("Covariance matrix not found. Recomputing...")
        cov = Cov(p_interp, None, zmin, zmax, Nz, r_vals, delta_r, overlap_area[stage] / 129600 * np.pi, sigma_v_vals[stage], kmin=min(P_CAMB[:, 0] * phys.h), kmax=max(P_CAMB[:, 0] * phys.h), mMin=minClusterMass[stage], mMax=1e16, physics=phys, window_function=window)
        cov.save_covariance(cov_path)

    inverse_cov_matrices=np.empty((Nz, len(r_vals), len(r_vals)))
    for i in range(Nz):
        inverse_cov_matrices[i]=cov.get_inverted_covariance(i)

    print("Total time take to get covariance matrices: {:.f2}s".format(time.time()-start_cov))

    print("Generating parameters and accessing database")
    start_gen = time.time()
    run_database_ids=[]
    base_run_ids=[]

    number_of_runs=0
    for mass in axion_masses:
        base_run_id = None
        run_database_ids_tmp = []
        for axion_frac in axion_frac_vals:
            phys = Physics.create_parameter_set(mass, axion_frac, OmegaMh2, OmegaBh2, phys.h, phys.n, phys.logAs_1010, print=False)

            run_entry=run_database.add_run(phys, z_vals, minClusterMass[stage], high_res=True, high_res_multiplier=5, jenkins_mass_function=False, window_function=window)

            if phys.f_axion==0.0:
                base_run_id=run_entry.name
            else:
                run_database_ids_tmp.append(run_entry.name)

            number_of_runs+=1

        run_database_ids.append(run_database_ids_tmp)
        base_run_ids.append(base_run_id)

    run_database.save_database()
    print("Total time take to generate parameter sets: {:.f2}s".format(time.time()-start_gen))
    print(np.shape(run_database_ids))

    camb_run_module = CAMBRun(run_database)
    camb_run_module.run()

    mpv_run_module = MPVRun(run_database)
    mpv_run_module.run()

    print("Computing {} Chi-sq values".format(number_of_runs))
    start_chi_sq=time.time()

    print("Broadcasting db path...")
    db_path = run_database.get_mpv_db_path()
    db_path = comm.bcast(db_path, root=0)
    print("Broadcasting covariance...")
    cov_path = comm.bcast(cov_path, root=0)
    r_vals, z_vals = comm.bcast((r_vals, z_vals), root=0)

    print("Scattering computations...")
    ids_to_load = chunk_evenly(run_database_ids, size)
    ids_to_load = comm.scatter(ids_to_load, root=0)

    base_ids_to_load = chunk_evenly(base_run_ids, size)
    base_ids_to_load = comm.scatter(base_ids_to_load, root=0)

    print("Starting computation...")
    chi_sq_vals = get_chi_sq_vals(run_database.get_mpv_database(), base_ids_to_load, ids_to_load, cov_path, r_vals, len(z_vals))

    flatten = lambda l: [item for sublist in l for item in sublist]

    chi_sq_vals = comm.gather(chi_sq_vals, root=0)
    chi_sq_vals = flatten(chi_sq_vals)

    print("Total time take to compute Chi-sq values: {:.f2}s".format(time.time()-start_chi_sq))

    np.save("chi-sq_results_{}".format(run_code), (axion_masses, axion_frac_vals, chi_sq_vals))

else:
    CAMBRun.run_child_nodes()
    MPVRun.run_child_nodes()

    db_path = None
    db_path = comm.bcast(db_path, root=0)

    cov_path = None
    cov_path = comm.bcast(cov_path, root=0)

    r_vals, z_vals = None, None
    r_vals, z_vals = comm.bcast((r_vals, z_vals), root=0)

    ids_to_load = None
    ids_to_load = comm.scatter(ids_to_load, root=0)

    base_ids_to_load = None
    base_ids_to_load = comm.scatter(base_ids_to_load, root=0)

    mpv_db = None
    with open(db_path, "rb") as f:
        mpv_db = dill.load(f)

    chi_sq_vals = get_chi_sq_vals(mpv_db, base_ids_to_load, ids_to_load, cov_path, r_vals, len(z_vals))

    chi_sq_vals = comm.gather(chi_sq_vals, root=0)

    assert chi_sq_vals is None

