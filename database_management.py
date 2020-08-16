import numpy as np
import dill
from physics import Physics
import os
import pandas as pd
from mean_pairwise_velocity import mean_pairwise_velocity


class RunDatabaseManager:

    def __init__(self, data_base_of_camb_runs, data_base_of_mpv_runs, root_dir=None):
        if root_dir is None:
            self.__root_dir = os.getcwd()
        else:
            self.__root_dir = root_dir

        self.__camb_db_path = data_base_of_camb_runs
        if os.path.exists(data_base_of_camb_runs):
            self.__camb_database = self.load_camb_database()
        else:
            self.__camb_database = pd.DataFrame(columns=['ID', 'path_root', 'params_path', 'logs_path', 'hash_value', 'ran_TF', 'successful_TF'])
            self.__camb_database.set_index('ID', inplace=True)
            self.save_camb_database()

        self.__mpv_database_path = data_base_of_mpv_runs
        if os.path.exists(self.__mpv_database_path):
            self.__mpv_database = self.load_mpv_database()
        else:
            self.__mpv_database = pd.DataFrame(columns=['ID', 'results_path', 'logs_path', 'CAMB_run_ID', 'CAMB_files_root', 'params_path', 'z_vals', 'minClusterMass', 'high_res_TF', 'high_res_mulitplier', 'use_jenkins_mass_function_TF', 'window_function_ID', 'hash_value', 'ran_TF', 'successful_TF'])
            self.__mpv_database.set_index('ID', inplace=True)
            self.save_mpv_database()

    def save_camb_database(self, db=None):
        if db is not None:
            self.__camb_database = db
        """self.__camb_database.to_pickle(self.__camb_db_path)"""
        with open(self.__camb_db_path, "wb") as f:
            dill.dump(self.__camb_database, f)

    def load_camb_database(self):
        """return pd.read_pickle(self.__camb_db_path)"""
        data = None
        with open(self.__camb_db_path, "rb") as f:
            data = dill.load(f)
        return data

    def get_camb_db_path(self):
        return self.__camb_db_path

    def save_mpv_database(self, db=None):
        if db is not None:
            self.__mpv_database = db
        """self.__mpv_database.to_pickle(self.__mpv_database_path)"""
        with open(self.__mpv_database_path, "wb") as f:
            dill.dump(self.__mpv_database, f)

    def load_mpv_database(self):
        """return pd.read_pickle(self.__mpv_database_path)"""
        data = None
        with open(self.__mpv_database_path, "rb") as f:
            data = dill.load(f)
        return data

    def get_mpv_db_path(self):
        return self.__mpv_database_path

    def save_database(self):
        self.save_camb_database()
        self.save_mpv_database()

    @staticmethod
    def save_physics_params(path, physics_params):
        with open(path, "wb") as f:
            dill.dump(physics_params, f)

    def get_CAMB_dataset(self, ID):
        return self.__camb_database.loc[ID]

    def set_camb_run_outcome(self, ID, successful):
        self.__camb_database.loc[ID, 'ran_TF'] = True
        self.__camb_database.loc[ID, 'successful_TF']=successful

    def set_mpv_run_outcome(self, ID, successful):
        self.__mpv_database.loc[ID, 'ran_TF'] = True
        self.__mpv_database.loc[ID, 'successful_TF']=successful


    def get_mpv_database(self):
        return self.__mpv_database

    def get_camb_database(self):
        return self.__camb_database

    def add_run(self, physics_params, z_vals, minClusterMass, high_res=True, high_res_multiplier=5, jenkins_mass_function=False, window_function=mean_pairwise_velocity.GAUSSIAN):
        camb_db_entry = None
        mpv_db_entry = None
        if np.any(hash(physics_params) == self.__camb_database['hash_value']):
            camb_db_entry = self.__camb_database.loc[hash(physics_params) == self.__camb_database['hash_value']].iloc[0]
        else:
            max_ID = len(self.__camb_database)
            self.save_physics_params(self.__root_dir + "/physics_param_files/params_ID={}.dat".format(max_ID), physics_params)
            self.__camb_database = self.__camb_database.append({'params_path': self.__root_dir + "/physics_param_files/params_ID={}.dat".format(max_ID), 'logs_path': self.__root_dir + "/camb_logs/camb_log_ID={}.dat".format(max_ID), 'hash_value': hash(physics_params), 'path_root': self.__root_dir + "/camb_files/camb_run_ID={}".format(max_ID), 'ran_TF': False}, ignore_index=True)
            camb_db_entry = self.__camb_database.iloc[-1]

        max_ID = len(self.__mpv_database)
        dict_mpv_run_info = {'results_path': self.__root_dir + "/mpv_results/mpv_run_ID={}.npy".format(max_ID), 'logs_path': self.__root_dir + "/mpv_logs/mpv_log_ID={}.txt".format(max_ID), 'params_path': camb_db_entry['params_path'], 'CAMB_run_ID': camb_db_entry.name, 'CAMB_files_root': camb_db_entry['path_root'], 'z_vals': z_vals, 'minClusterMass': minClusterMass, 'high_res_TF': high_res, 'high_res_mulitplier': high_res_multiplier, 'use_jenkins_mass_function_TF': jenkins_mass_function, 'window_function_ID': window_function, 'ran_TF': False}

        if np.any(self.__mpv_database['hash_value'] == self.hash_mpv_db_entry(dict_mpv_run_info)):
            mpv_db_entry = self.__mpv_database.loc[self.__mpv_database['hash_value'] == self.hash_mpv_db_entry(dict_mpv_run_info)].iloc[0]
        else:
            dict_mpv_run_info['hash_value']=self.hash_mpv_db_entry(dict_mpv_run_info)
            self.__mpv_database = self.__mpv_database.append(dict_mpv_run_info, ignore_index=True)
            mpv_db_entry = self.__mpv_database.iloc[-1]

        return mpv_db_entry

    def get_all_new_camb_runs(self):
        return self.__camb_database.loc[np.logical_not(self.__camb_database['ran_TF'])]

    def get_all_new_mpv_runs(self):
        return self.__mpv_database.loc[np.logical_not(self.__mpv_database['ran_TF'])]

    @staticmethod
    def comp_np_arrays(a, b):
        tf_vals = []
        for x in a:
            try:
                if np.all(x == b):
                    tf_vals.append(True)
                else:
                    tf_vals.append(False)
            except ValueError:
                tf_vals.append(False)
        return tf_vals

    @staticmethod
    def hash_mpv_db_entry(dict):
        hash_list=[dict['CAMB_run_ID'], dict['minClusterMass'], dict['high_res_TF'], dict['high_res_mulitplier'], dict['use_jenkins_mass_function_TF'], dict['window_function_ID']]
        for z in np.round(dict['z_vals'], decimals=3):
            hash_list.append(z)
        return hash(tuple(hash_list))
