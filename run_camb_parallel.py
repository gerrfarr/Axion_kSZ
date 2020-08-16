import time
from mean_pairwise_velocity import *
from mpi4py import MPI
from helpers import MyProcessPool, execute
import dill
import os
import shutil
import pandas as pd
from database_management import RunDatabaseManager

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
name = MPI.Get_processor_name()

class CAMBRun:

    def __init__(self, database):
        """

        :type database: RunDatabaseManager
        """
        print("Initializing CAMBRun")
        self.__database = database

    @staticmethod
    def run_axionCAMB(fileroot, phys, log_path=None):
        start_time = time.time()
        try:
            if log_path is not None:
                subprocess.call("/home/gerrit/kSZ/axionCAMB/camb /home/gerrit/kSZ/axionCAMB/inifiles/params.ini 1 {} 2 {} 3 {} 4 {} 5 {} 6 {} 7 {} 8 T 9 {} > {}".format(phys.Omega_Bh2, max([(phys.Omega0 - phys.OmegaB) * phys.h ** 2 * (1 - phys.f_axion), 1.0e-6]), max([(phys.Omega0 - phys.OmegaB) * phys.h ** 2 * phys.f_axion, 1.0e-6]), phys.m_axion, 100 * phys.h, phys.n, np.exp(phys.logAs_1010) / 1.0e10, fileroot, log_path), shell=True)
            else:
                subprocess.call("/home/gerrit/kSZ/axionCAMB/camb /home/gerrit/kSZ/axionCAMB/inifiles/params.ini 1 {} 2 {} 3 {} 4 {} 5 {} 6 {} 7 {} 8 T 9 {}".format(phys.Omega_Bh2, max([(phys.Omega0 - phys.OmegaB) * phys.h ** 2 * (1 - phys.f_axion), 1.0e-6]), max([(phys.Omega0 - phys.OmegaB) * phys.h ** 2 * phys.f_axion, 1.0e-6]), phys.m_axion, 100 * phys.h, phys.n, np.exp(phys.logAs_1010) / 1.0e10, fileroot), shell=True)
        except Exception as ex:
            print(str(ex))
            print("AxionCAMB failed!")

        return time.time() - start_time

    @staticmethod
    def do_work(camb_db_ID, camb_db_path):
        camb_db=None
        with open(camb_db_path, "rb") as f:
            camb_db = dill.load(f)

        db_entry=camb_db.loc[camb_db_ID]

        with open(db_entry['logs_path'], "a+", buffering=1) as log_file:

            physics_params = None
            try:
                with open(db_entry['params_path'], 'rb') as file:
                    physics_params = dill.load(file)
            except FileNotFoundError:
                log_file.write("Parameter file does not exist\n")
                return False
                #raise Exception("Parameter file with ID {} does not exist".format(camb_db_ID))

            try:
                camb_dir, camb_root = os.path.split(db_entry['path_root'])
                time=CAMBRun.run_axionCAMB(camb_root, physics_params, db_entry['logs_path'])
                log_file.write("CAMB took {:.2f}s\n".format(time))

                shutil.move(os.getcwd() + "/" + camb_root + "_params.ini", db_entry['path_root'] + "_params.ini")
                shutil.move(os.getcwd() + "/" + camb_root + "_matterpower_out.dat", db_entry['path_root'] + "_matterpower_out.dat")
                shutil.move(os.getcwd() + "/" + camb_root + "_evolution.dat", db_entry['path_root'] + "_evolution.dat")
                shutil.move(os.getcwd() + "/" + camb_root + "_devolution.dat", db_entry['path_root'] + "_devolution.dat")
                shutil.move(os.getcwd() + "/" + camb_root + "_a_vals.dat", db_entry['path_root'] + "_a_vals.dat")
                shutil.move(os.getcwd() + "/" + camb_root + "_transfer_out.dat", db_entry['path_root'] + "_transfer_out.dat")
            except Exception as ex:
                log_file.write("AxionCAMB failed! Got following exception: "+str(ex)+"\n")
                return False
            else:
                log_file.write("Succefully ran CAMB and moved files.\n")

        return True


    def run(self):

        assert(rank==0)

        start=time.time()

        ids_to_be_run=self.__database.get_all_new_camb_runs().index

        print("CAMB: Scattering {} tasks to {} nodes...".format(len(ids_to_be_run), size))

        id_vals_chunked=np.array_split(ids_to_be_run, size)

        db_path=self.__database.get_camb_db_path()

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
            self.__database.set_camb_run_outcome(ids_to_be_run[i], results[i])
            if results[i]:
                successful_count+=1
        self.__database.save_camb_database()

        print("{}/{} CAMB runs successful.".format(successful_count, len(ids_to_be_run)))

        print("Total time take b CAMB: {:.3f}".format(time.time() - start))

    @staticmethod
    def run_child_nodes():
        assert(rank!=0)
        id_vals_chunked = None
        db_path = None

        db_path = comm.bcast(db_path, root=0)
        id_vals = comm.scatter(id_vals_chunked, root=0)

        with MyProcessPool(min([12,len(id_vals)])) as p:
            results = list(p.imap(lambda ID: CAMBRun.do_work(ID, db_path), id_vals))

        results = comm.gather(results, root=0)

        assert results is None
