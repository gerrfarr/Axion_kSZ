import numpy as np

data_dir="/home/gerrit/kSZ/output/fisher_final/"
run_code="2020-07-29_sharp_k_lin_n=10_stageIV"

derivatives = np.load(data_dir+"derivatives_results_{}.npy".format(run_code), allow_pickle=True)
velocities = np.load(data_dir+"derivatives_results_{}.npy".format(run_code), allow_pickle=True)
