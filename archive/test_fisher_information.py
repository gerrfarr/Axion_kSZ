import numpy as np
from covariance import covariance_matrix as Cov
from archive.get_derivatives import Derivatives
from physics import Physics
from numerics import interpolate
from mpi4py import MPI
from mean_pairwise_velocity import mean_pairwise_velocity

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
name = MPI.Get_processor_name()


if rank==0:
    run_code = "fisher_sharp-k_lin_2020-06-18"

    axion_masses=np.logspace(-27, -20, 8)
    differentiation_step_size=np.logspace(np.log10(7e-3), -1, 25)
    differentiation_step_num=10
    """
    convergence:    Sharp-k filter  d_step=0.02
                    Gaussian filter d_step=0.04
                    Top-Hat filter  d_step=0.04
                    No filter       d_step=
    """

    window=mean_pairwise_velocity.SHARP_K

    phys = Physics(True, True)

    OmegaMh2 = phys.Omega0*phys.h**2
    OmegaBh2 = phys.OmegaB * phys.h ** 2

    omegaMh2_vals = np.linspace(OmegaMh2-2*OmegaMh2*0.05, OmegaMh2+2*OmegaMh2*0.05, 5)
    omegaBh2_vals = np.linspace(OmegaBh2-2*OmegaBh2*0.05, OmegaBh2+2*OmegaBh2*0.05, 5)
    h_vals = np.linspace(phys.h-2*phys.h*0.05, phys.h+2*phys.h*0.05, 5)
    ns_vals = np.linspace(phys.n-2*0.005, phys.n+2*0.005, 5)
    logAs_1010_vals = np.linspace(phys.logAs_1010-2*phys.logAs_1010*0.05, phys.logAs_1010+2*phys.logAs_1010*0.05, 5)

    delta_r=2.0
    r_vals=np.arange(20.0, 180.0, delta_r)
    zmin=0.1
    zmax=0.6
    Nz=5
    z_step=(zmax-zmin)/Nz
    z_vals=np.linspace(zmin+z_step/2, zmax-z_step/2, Nz)

    filename = "../../data/axion_frac=0.000_matterpower_out.dat"

    P_CAMB = np.loadtxt(filename)
    p_interp = interpolate(P_CAMB[:, 0] * phys.h, P_CAMB[:, 1] / phys.h ** 3, phys.P_cdm, phys.P_cdm)

    try:
        cov=Cov.load_covariance("covariance_matrix_{}.dat".format(run_code))
    except (IOError, OSError) as e:
        print("Covariance matrix not found. Recomputing...")
        cov = Cov(p_interp, None, zmin, zmax, Nz, r_vals, delta_r, 10000/129600*np.pi, 120.0, kmin=min(P_CAMB[:, 0] * phys.h), kmax=max(P_CAMB[:, 0] * phys.h), mMin=0.6e14, mMax=1e16, physics=phys, window_function=window)
        cov.save_covariance("covariance_matrix_{}.dat".format(run_code))

    fisher_matrices=[]
    differentiation_step_size = comm.bcast(differentiation_step_size, root=0)
    for i in range(len(differentiation_step_size)):
        axion_frac_vals=np.linspace(0.0, differentiation_step_size[i],differentiation_step_num+1)
        ds= Derivatives("{}_d_step={}".format(run_code, str(differentiation_step_size[i])), axion_masses, axion_frac_vals, omegaMh2_vals, omegaBh2_vals, h_vals, ns_vals, logAs_1010_vals, phys, r_vals, z_vals, window_function=window)
        derivatives=ds.run()
        fisher_matrices_tmp=[]
        for j in range(len(axion_masses)):
            fisher_matrix = np.empty((6, 6))
            for p in range(0,6):
                for q in range(p,6):
                    fisher=0.0
                    for k in range(Nz):
                        for m in range(len(r_vals)):
                            for n in range(len(r_vals)):
                                if p==0:
                                    d1 = derivatives[p][j][k][m]
                                else:
                                    d1 = derivatives[p][k][m]

                                if q==0:
                                    d2 = derivatives[q][j][k][n]
                                else:
                                    d2 = derivatives[q][k][n]

                                fisher+=d1*cov.get_inverted_covariance_interpolation(k, r_vals[m], r_vals[n])*d2
                    fisher_matrix[p,q],fisher_matrix[q,p]=fisher,fisher
            fisher_matrices_tmp.append(fisher_matrix)
        fisher_matrices.append(fisher_matrices_tmp)

    np.save("fisher_matrices_{}".format(run_code), fisher_matrices)

else:
    differentiation_step_size=None
    differentiation_step_size=comm.bcast(differentiation_step_size, root=0)
    for i in range(len(differentiation_step_size)):
        Derivatives.run_child_nodes()
