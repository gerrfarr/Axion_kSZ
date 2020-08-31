import numpy as np
from ..physics import Physics
from ..generate_parameters import ParameterGenerator

STAGE_II=0
STAGE_III=1
STAGE_IV=2

data_dir="/home/gerrit/kSZ/output/fisher_final/"
run_code="2020-07-29_sharp_k_lin_n=10_stageIV"
stage = STAGE_IV

derivatives = np.load(data_dir+"derivatives_results_{}.npy".format(run_code), allow_pickle=True)
velocities = np.load(data_dir+"velocities_results_{}.npy".format(run_code), allow_pickle=True)

plotting_derivatives=[None for i in range(6)]


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

fiducial_params=[0.0, phys.f_axion, OmegaMh2, OmegaBh2, phys.h, phys.n, phys.logAs_1010]

for i in range(len(step_sizes)):
    axion_frac_vals = np.arange(0.0, min([step_sizes[i] * n_steps + 1e-5, 1.0 + 1e-5]), step_sizes[i])
    # axion_frac_vals = np.linspace(0.0, step_sizes[i]*n_steps, n_steps + 1)



for i in range(1, 6):
