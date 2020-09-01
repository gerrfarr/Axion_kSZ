import time
from mean_pairwise_velocity import *
from mpi4py import MPI
from helpers import MyProcessPool, execute
import dill
import os
import shutil
import pandas as pd
from save_output import SaveOutput
from database_management import RunDatabaseManager

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
name = MPI.Get_processor_name()

class MPVRun:

    def __init__(self, database):
        """

        :type database: RunDatabaseManager
        """
        print("Initializing MPVRun")
        self.__database = database

    @staticmethod
    def do_work(mpv_db_ID, mpv_db_path):
        mpv_db=None
        with open(mpv_db_path, "rb") as f:
            mpv_db = dill.load(f)

        db_entry=mpv_db.loc[mpv_db_ID]

        with open(db_entry['logs_path'], "a+", buffering=1) as log_file:
            with SaveOutput(log_file):
                try:
                    startMPV = time.time()

                    physics_params = None
                    try:
                        with open(db_entry['params_path'], 'rb') as file:
                            physics_params = dill.load(file)
                    except FileNotFoundError:
                        print("Parameter file does not exist\n")
                        return False
                        # raise Exception("Parameter file with ID {} does not exist".format(camb_db_ID))
                    physics_params.print_params()
                    file_root = db_entry['CAMB_files_root']

                    P_CAMB = np.loadtxt(file_root + '_matterpower_out.dat')
                    p_interp = interpolate(P_CAMB[:, 0] * physics_params.h, P_CAMB[:, 1] / physics_params.h ** 3, physics_params.P_cdm, physics_params.P_cdm)
                    gInterpolator = GrowthInterpolation(file_root, physics=physics_params)

                    mpv = mean_pairwise_velocity(p_interp, gInterpolator, kmin=min(P_CAMB[:, 0] * physics_params.h), kmax=max(P_CAMB[:, 0] * physics_params.h), mMin=db_entry['minClusterMass'], mMax=1e16, physics=physics_params, AXIONS=True, jenkins_mass=db_entry['use_jenkins_mass_function_TF'], window_function=db_entry['window_function_ID'])

                    v_vals_array = []
                    r_vals = []
                    for z in db_entry['z_vals']:
                        rs, vs = mpv.compute(z, do_unbiased=False, high_res=db_entry['high_res_TF'], high_res_multi=db_entry['high_res_mulitplier'])
                        v_vals_array.append(vs)
                        r_vals = rs
                    np.save(db_entry['results_path'], (r_vals, v_vals_array))
                    print("Time taken for mean pairwise velocity computation: {:.2f} s\n".format(time.time() - startMPV))
                    print("___\n")

                except OSError as ex:
                    print(str(ex) + "\n")
                    print("Velocity Spectra with ID {} did not compute because CAMB failed!".format(mpv_db_ID))
                    return False

                except Exception as ex:
                    print(str(ex) + "\n")
                    print("Something went wrong in process {}!\n".format(mpv_db_ID))
                    return False

                else:
                    return True


    def run(self):

        assert(rank==0)

        start=time.time()

        ids_to_be_run=list(self.__database.get_all_new_mpv_runs().index)
        np.random.shuffle(ids_to_be_run)

        print("MPV: Scattering {} tasks to {} nodes...".format(len(ids_to_be_run), size))

        id_vals_chunked=np.array_split(ids_to_be_run, size)

        db_path=self.__database.get_mpv_db_path()

        db_path=comm.bcast(db_path, root=0)
        id_vals=comm.scatter(id_vals_chunked, root=0)

        with MyProcessPool(min([12,len(id_vals)])) as p:
            results=list(p.imap(lambda ID: self.do_work(ID, db_path), id_vals))

        results = comm.gather(results, root=0)
        flatten = lambda l: [item for sublist in l for item in sublist]
        results = flatten(results)
        print("Gathered {} results".format(len(results)))

        successful_count=0
        for i in range(len(ids_to_be_run)):
            self.__database.set_mpv_run_outcome(ids_to_be_run[i], results[i])
            if results[i]:
                successful_count+=1
        self.__database.save_mpv_database()

        print("{}/{} mean-pairwise-velocity runs successful.".format(successful_count, len(ids_to_be_run)))

        print("Total time take by MPV: {:.3f}".format(time.time() - start))

    @staticmethod
    def run_child_nodes():
        assert(rank!=0)
        id_vals_chunked = None
        db_path = None

        db_path = comm.bcast(db_path, root=0)
        id_vals = comm.scatter(id_vals_chunked, root=0)

        with MyProcessPool(min([12,len(id_vals)])) as p:
            results = list(p.imap(lambda ID: MPVRun.do_work(ID, db_path), id_vals))

        results = comm.gather(results, root=0)

        assert results is None
