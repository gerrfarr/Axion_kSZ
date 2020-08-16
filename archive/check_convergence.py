import time
import dill
import os
import subprocess
from mpi4py import MPI
from generate_parameters import ParameterGenerator
from mean_pairwise_velocity import *

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
name = MPI.Get_processor_name()


def run_axionCAMB(fileroot, phys, output_file):
    """
    start_time=time.time()
    args=shlex.split("/home/gerrit/kSZ/axionCAMB_mod/camb \
        /home/gerrit/kSZ/axionCAMB_mod/inifiles/params.ini 1 {} 2 {} 3 {} 4 {} 5 {} 6 {} 7 T 8 {}"\
        .format(phys.Omega_Bh2, max([(phys.Omega0*phys.h**2-phys.Omega_Bh2)*(1-phys.f_axion), 1.0e-6]), max([(phys.Omega0*phys.h**2-phys.Omega_Bh2)*(phys.f_axion), 1.0e-6]), phys.m_axion, 100*phys.h, phys.n, fileroot))

    proc = subprocess.call(args)
    return time.time()-start_time"""
    start_time=time.time()
    subprocess.call("/home/gerrit/kSZ/axionCAMB_mod/camb /home/gerrit/kSZ/axionCAMB_mod/inifiles/params.ini 1 {} 2 {} 3 {} 4 {} 5 {} 6 {} 7 T 8 {}".format(phys.Omega_Bh2, max([(phys.Omega0*phys.h**2-phys.Omega_Bh2)*(1-phys.f_axion), 1.0e-6]), max([(phys.Omega0*phys.h**2-phys.Omega_Bh2)*(phys.f_axion), 1.0e-6]), phys.m_axion, 100*phys.h, phys.n, fileroot), shell = True)

    return time.time()-start_time


def do_the_work(ID, run_code, phys_file, ms=[1,2,4,6], run_camb=True, do_unbiased=False, recover=True):

    with open(os.getcwd()+"/log_{}/log_{}.txt".format(run_code,ID), "a+") as output_file:
        if recover:
            try:
                data= np.load("recovery_storage/results_{}_{}.npy".format(run_code, ID))
                output_file.write("Data recovered!\n")
                return data
            except Exception as ex:
                output_file.write(str(ex)+"\n")
        phys=None
        with open(phys_file, 'rb') as file:
            phys=dill.load(file)[ID]
        filename="camb_{}_{}".format(run_code, ID)

        if run_camb:
            output_file.write("Running AxionCAMB...\n")
            output_file.write("Time taken by AxionCAMB: {:.2f} s\n".format(run_axionCAMB(filename, phys, output_file)))

        try:
            startMPV=time.time()

            gInterpolator=GrowthInterpolation(filename+'_evolution.dat', physics=phys)
            P_CAMB=np.loadtxt(filename+'_matterpower_out.dat')
            p_interp=interpolate(P_CAMB[:,0]*phys.h, P_CAMB[:,1]/phys.h**3, phys.P_cdm, phys.P_cdm)


            mpv=mean_pairwise_velocity(p_interp, gInterpolator, kmin=min(P_CAMB[:,0]*phys.h), kmax=max(P_CAMB[:,0]*phys.h), mMin=1e14, mMax=1e16, physics=phys, AXIONS=True)

            if not do_unbiased:
                v_vals_array=[]
                r_vals=[]
                for m in ms:
                    rs, vs=mpv.compute(1.15, high_res=True, do_unbiased=False, high_res_multi=m)
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
                for m in ms:
                    rs, vs, vs_unbiased=mpv.compute(z, high_res=True, do_unbiased=True, high_res_multi=m)
                    v_vals_array.append(vs)
                    v_vals_unbiased_array.append(vs_unbiased)
                    r_vals=rs
                np.save("recovery_storage/results_{}_{}".format(run_code, ID), (r_vals, v_vals_array,v_vals_unbiased_array))
                output_file.write("Time taken for mean pairwise velocity computation: {:.2f} s\n".format(time.time()-startMPV))
                output_file.write("___")

                return r_vals, v_vals_array, v_vals_unbiased_array
        except Exception as ex:
            output_file.write(str(ex)+"\n")
            print("Something went wrong in process {}!".format(ID))


ax_masses=[5.0e-25,1.0e-25,1.0e-26]#np.logspace(-25, -20, 10)
AxionFrac_Vals=[0.1]

m_vals=[1, 2, 4, 6]

run_code="run_7_2019-10-11_convergence_check"


if rank==0:
    start=time.time()

    wd = os.getcwd()
    path=wd+"/log_{}".format(run_code)
    try:
        os.mkdir(path)
    except OSError:
        print ("Creation of the directory %s failed" % path)
    else:
        print ("Successfully created the directory %s " % path)

    paramGen=ParameterGenerator(ax_masses, AxionFrac_Vals)
    paramGen.save_to_file("params_list_{}.dat".format(run_code))
    len_params=len(paramGen.get_params())

    print("Scattering {} tasks to {} nodes...".format(len_params, size))

    chunk_size=len_params//size
    id_vals_chunked=np.array_split(range(0, len_params), size)#[range(i,i+chunk_size) for i in xrange(0, len_params, chunk_size)]

else:
    id_vals_chunked=None
    data=None

id_vals=comm.scatter(id_vals_chunked, root=0)

with MyProcessPool(6) as p:
    results=list(p.imap(lambda ID: do_the_work(ID, run_code, "params_list_{}.dat".format(run_code), m_vals, recover=True), id_vals))

results = comm.gather(results, root=0)

if rank==0:
    flatten = lambda l: [item for sublist in l for item in sublist]
    results=flatten(results)
    print("Gathered {} results".format(len(results)))

    #results=paramGen.regroup_results(results)

    #derivatives=paramGen.get_derivatives(results, x=0, deg=3, z_vals=z_vals)
    #np.savetxt("ds_out_{}.txt".format(run_code), np.array(derivatives)[:,0,0,:])
    print("Total time take: {:.3f}".format(time.time()-start))
else:
    assert results is None

