from physics import Physics
import time
from mean_pairwise_velocity import mean_pairwise_velocity
from covariance import covariance_matrix as Cov
from numerics import interpolate
from save_output import SaveOutput
from helpers import *
from read_mode_evolution import GrowthInterpolation



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

def run_mpv(log_path, result_path, file_root, physics_params, window, minMass, z_vals, doUnbiased=True, high_res=True, high_res_multi=5):
  with open(log_path, "a+", buffering=1) as log_file:
        with SaveOutput(log_file):
            try:
                startMPV = time.time()

                P_CAMB = np.loadtxt(file_root + '_matterpower_out.dat')
                p_interp = interpolate(P_CAMB[:, 0] * physics_params.h, P_CAMB[:, 1] / physics_params.h ** 3, physics_params.P_cdm, physics_params.P_cdm)
                gInterpolator = GrowthInterpolation(file_root, physics=physics_params)

                mpv = mean_pairwise_velocity(p_interp, gInterpolator, kmin=min(P_CAMB[:, 0] * physics_params.h), kmax=max(P_CAMB[:, 0] * physics_params.h), mMin=minMass, mMax=1e16, physics=physics_params, AXIONS=True, jenkins_mass=True, window_function=window)

                v_vals_array = []
                r_vals = []
                if doUnbiased:
                    v_vals_unbiased_array=[]
                for z in z_vals:
                    if doUnbiased:
                        rs, vs, vs_unbiased = mpv.compute(z, do_unbiased=doUnbiased, high_res=high_res, high_res_multi=high_res_multi)
                        v_vals_unbiased_array.append(vs_unbiased)
                    else:
                        rs, vs = mpv.compute(z, do_unbiased=doUnbiased, high_res=high_res, high_res_multi=high_res_multi)

                    v_vals_array.append(vs)
                    r_vals = rs
                if doUnbiased:
                    np.save(result_path, (r_vals, v_vals_array, v_vals_unbiased_array))
                    return (r_vals, v_vals_array, v_vals_unbiased_array)
                else:
                    np.save(result_path, (r_vals, v_vals_array))
                    return (r_vals, v_vals_array)

                print("Time taken for mean pairwise velocity computation: {:.2f} s\n".format(time.time() - startMPV))
                print("___\n")

            except OSError as ex:
                print(str(ex) + "\n")
                print("Velocity Spectra did not compute because CAMB failed!\n")
                return False

            except Exception as ex:
                print(str(ex) + "\n")
                print("Something went wrong in process {}!\n")
                return False



STAGE_II=0
STAGE_III=1
STAGE_IV=2

run_code = "2020-09-07_sharp_k_stageIII_plotting"
window = mean_pairwise_velocity.SHARP_K
stage = STAGE_III

axion_masses = [1.0e-27, 5.0e-27, 1.0e-26, 5.0e-26, 1.0e-25, 5.0e-25, 1.0e-24, 5.0e-24]
axion_frac_vals = [0.0, 0.25, 0.5, 0.75, 1.0]

phys = Physics(True, True, READ_H=False)

OmegaMh2 = phys.Omega0 * phys.h ** 2
OmegaBh2 = phys.OmegaB * phys.h ** 2

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

print("Total time take to get covariance matrices: {:.2}s".format(time.time()-start_cov))

params=[]
for m in axion_masses:
    for f in axion_frac_vals:
        phys = Physics.create_parameter_set(axion_mass=m, axion_frac=f)
        params.append(phys)

#with MyProcessPool(12) as p:
#    p.map(lambda i: run_axionCAMB("CAMB_run4plots_{}".format(i), params[i]), range(0, len(params)))

with MyProcessPool(12) as p:
    outputs = p.map(lambda i:  run_mpv("MPV_run4plots_log_{}".format(i), "MPV_run4plots_{}".format(i), "CAMB_run4plots_{}".format(i), params[i], window, minClusterMass[stage], z_vals), range(0, len(params)))

final=[]
for i in range(len(axion_masses)):
    final_tmp=[]
    for j in range(len(axion_frac_vals)):
        final_tmp.append(outputs[i*len(axion_frac_vals)+j])
    final.append(final_tmp)

np.save("{}_data".format(run_code), final)







