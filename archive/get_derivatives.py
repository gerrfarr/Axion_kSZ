import time
from mean_pairwise_velocity import *
from mpi4py import MPI
from helpers import MyProcessPool, execute
import dill
from generate_parameters import ParameterGenerator
import os
from save_output import SaveOutput

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
name = MPI.Get_processor_name()


class Derivatives:

    def __init__(self, run_code, axion_masses, axion_frac_vals, omegaMh2_vals, omegaBh2_vals, h_vals, ns_vals, logAs_1010_vals, fiducial_model, r_vals, z_vals, high_res=True, high_res_multiplier=5, run_camb=True, do_unbiased=False, recover=True, jenkins_mass_function=False, window_function=mean_pairwise_velocity.GAUSSIAN):
        """

        :type fiducial_model: Physics
        """
        self.__run_code=run_code
        self.__r_vals=r_vals
        self.__z_vals=z_vals

        self.__high_res=high_res
        self.__high_res_multiplier=high_res_multiplier
        self.__run_camb=run_camb
        self.__do_unbiased = do_unbiased
        self.__recover=recover
        self.__jenkins_mass_function=jenkins_mass_function
        self.__window_function=window_function

        self.__paramGen = ParameterGenerator(axion_masses, axion_frac_vals, fiducial_model.f_axion, omegaMh2_vals, fiducial_model.Omega0*fiducial_model.h**2, omegaBh2_vals, fiducial_model.OmegaB*fiducial_model.h**2, h_vals, fiducial_model.h, ns_vals, fiducial_model.n, logAs_1010_vals, fiducial_model.logAs_1010)
        self.__paramGen.save_to_file("params_list_{}.dat".format(run_code))

    @staticmethod
    def run_axionCAMB(fileroot, phys):

        start_time=time.time()
        try:
            subprocess.call("/home/gerrit/kSZ/axionCAMB/camb /home/gerrit/kSZ/axionCAMB/inifiles/params.ini 1 {} 2 {} 3 {} 4 {} 5 {} 6 {} 7 {} 8 T 9 {}".format(phys.Omega_Bh2, max([(phys.Omega0-phys.OmegaB)*phys.h**2*(1-phys.f_axion), 1.0e-6]), max([(phys.Omega0-phys.OmegaB)*phys.h**2*phys.f_axion, 1.0e-6]), phys.m_axion, 100*phys.h, phys.n, np.exp(phys.logAs_1010)/1.0e10, fileroot), shell=True)
        except Exception as ex:
            print(str(ex))
            print("AxionCAMB failed!")

        return time.time()-start_time

    @staticmethod
    def do_the_work(ID, run_code, z_vals, high_res=True, high_res_multiplier=1, run_camb=True, do_unbiased=False, recover=True, jenkins_mass_function=False, window_function=mean_pairwise_velocity.GAUSSIAN):
        phys_file="params_list_{}.dat".format(run_code)

        with open(os.getcwd()+"/log_{}/log_{}.txt".format(run_code,ID), "a+", buffering=1) as output_file:
            if recover:
                try:
                    data= np.load("recovery_storage/results_{}_{}.npy".format(run_code, ID), allow_pickle=True)
                    output_file.write("Data recovered!\n")
                    return data
                except Exception as ex:
                    output_file.write(str(ex)+"\n")
            phys=None
            with open(phys_file, 'rb') as file:
                phys=dill.load(file)[ID]
            filename="camb_{}_{}".format(run_code, ID)

            if run_camb:
                with SaveOutput(output_file):
                    if not os.path.exists(filename+'_matterpower_out.dat'):
                        print("Running AxionCAMB...\n")
                        print("Time taken by AxionCAMB: {:.2f} s\n".format(Derivatives.run_axionCAMB(filename, phys)))
                    else:
                        print("CAMB data recovered!")

            try:
                with SaveOutput(output_file):
                    startMPV=time.time()
                    phys.print_params()

                    P_CAMB = np.loadtxt(filename + '_matterpower_out.dat')
                    p_interp = interpolate(P_CAMB[:, 0] * phys.h, P_CAMB[:, 1] / phys.h ** 3, phys.P_cdm, phys.P_cdm)
                    gInterpolator=GrowthInterpolation(filename, physics=phys)


                    mpv=mean_pairwise_velocity(p_interp, gInterpolator, kmin=min(P_CAMB[:,0]*phys.h), kmax=max(P_CAMB[:,0]*phys.h), mMin=1e14, mMax=1e16, physics=phys, AXIONS=True, jenkins_mass=jenkins_mass_function, window_function=window_function)

                    if not do_unbiased:
                        v_vals_array=[]
                        r_vals=[]
                        for z in z_vals:
                            rs, vs=mpv.compute(z, do_unbiased=False, high_res=high_res, high_res_multi=high_res_multiplier)
                            v_vals_array.append(vs)
                            r_vals=rs
                        np.save("recovery_storage/results_{}_{}".format(run_code, ID), (r_vals, v_vals_array))
                        output_file.write("Time taken for mean pairwise velocity computation: {:.2f} s\n".format(time.time()-startMPV))
                        output_file.write("___")
                        return r_vals, v_vals_array
                    else:
                        v_vals_array=[]
                        v_vals_unbiased_array=[]
                        r_vals=[]
                        for z in z_vals:
                            rs, vs, vs_unbiased=mpv.compute(z, do_unbiased=True, high_res=high_res, high_res_multi=high_res_multiplier)
                            v_vals_array.append(vs)
                            v_vals_unbiased_array.append(vs_unbiased)
                            r_vals=rs
                        np.save("recovery_storage/results_{}_{}".format(run_code, ID), (r_vals, v_vals_array,v_vals_unbiased_array))
                        output_file.write("Time taken for mean pairwise velocity computation: {:.2f} s\n".format(time.time()-startMPV))
                        output_file.write("___\n")

                    return r_vals, v_vals_array, v_vals_unbiased_array
            except OSError as ex:
                output_file.write(str(ex) + "\n")
                print("Velocity Spectra with ID {} did not compute because CAMB failed!".format(ID))
                np.save("recovery_storage/results_{}_{}".format(run_code, ID), None)
                return None

            except Exception as ex:
                output_file.write(str(ex)+"\n")
                print ("Something went wrong in process {}!".format(ID))
                return None

    def run(self):

        assert(rank==0)

        start=time.time()

        wd = os.getcwd()
        path=wd+"/log_{}".format(self.__run_code)
        try:
            os.mkdir(path)
        except OSError:
            print("Creation of the directory %s failed" % path)
        else:
            print("Successfully created the directory %s " % path)

        len_params=len(self.__paramGen.get_params())

        print("Scattering {} tasks to {} nodes...".format(len_params, size))

        id_vals_chunked=np.array_split(range(0, len_params), size)

        run_information=(self.__run_code, self.__z_vals, self.__high_res, self.__high_res_multiplier, self.__run_camb, self.__do_unbiased, self.__recover, self.__jenkins_mass_function, self.__window_function)

        run_information=comm.bcast(run_information, root=0)
        id_vals=comm.scatter(id_vals_chunked, root=0)

        with MyProcessPool(min([12,len(id_vals)])) as p:
            results=list(p.imap(lambda ID: self.do_the_work(ID, *run_information), id_vals))

        results = comm.gather(results, root=0)

        flatten = lambda l: [item for sublist in l for item in sublist]
        results=flatten(results)
        print ("Gathered {} results".format(len(results)))

        results=self.__paramGen.regroup_results(results)

        derivatives=self.__paramGen.get_derivatives(results, r_vals_eval=self.__r_vals, deg=3, z_vals=self.__z_vals)
        np.save("ds_out_{}".format(self.__run_code), np.array(derivatives))
        print("Total time take: {:.3f}".format(time.time() - start))
        return derivatives

    @staticmethod
    def run_child_nodes():
        assert(rank!=0)
        id_vals_chunked = None
        run_information = None

        run_information = comm.bcast(run_information, root=0)
        id_vals = comm.scatter(id_vals_chunked, root=0)

        with MyProcessPool(6) as p:
            results = list(p.imap(lambda ID: Derivatives.do_the_work(ID, *run_information), id_vals))

        results = comm.gather(results, root=0)

        assert results is None
