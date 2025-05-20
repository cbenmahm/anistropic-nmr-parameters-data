import ase.io as ase_io
import numpy as np
from e3nn import io
import json 
import torch


# load frames: test or train
frames = ase_io.read("/path/to/data", ":")

# load the mean and std data
with open("data_mean_std.json") as f:
	data = json.load(f)


# get transformations
symm_transform = io.CartesianTensor("ij=ji")
antisymm_transform = io.CartesianTensor("ij=-ji")


for x in frames:

    si_idx = x.numbers == 14
    o_idx = x.numbers == 8

    ms = torch.from_numpy(x.arrays["ms"])
    ms = ms.reshape(-1, 3, 3)

    symm = symm_transform.from_cartesian(ms)
    antisymm = antisymm_transform.from_cartesian(ms)

    s = symm.cpu().numpy().copy()
    a = antisymm.cpu().numpy().copy()

    s_mean = np.ones_like(s)
    s_std = np.ones_like(s)
    s_mean[si_idx] *= data["symm_si_mean"]
    s_std[si_idx] *= data["symm_si_std"]
    s_mean[o_idx] *= data["symm_o_mean"]
    s_std[o_idx] *= data["symm_o_std"]

    a_mean = np.ones_like(a)
    a_std = np.ones_like(a)
    a_mean[si_idx] *= data["antisymm_si_mean"]
    a_std[si_idx] *= data["antisymm_si_std"]
    a_mean[o_idx] *= data["antisymm_o_mean"]
    a_std[o_idx] *= data["antisymm_o_std"]

    all_mean = np.hstack((s_mean[:, 0][:, np.newaxis], a_mean, s_mean[:, 1:]))
    all_std = np.hstack((s_std[:, 0][:, np.newaxis], a_std, s_std[:, 1:]))

    x.arrays["ms_all"] = np.hstack((s[:, 0][:, np.newaxis], a, s[:, 1:]))

    x.arrays["ms_all_center_stand"] = x.arrays["ms_all"].copy()
    x.arrays["ms_all_center_stand"] -= all_mean
    x.arrays["ms_all_center_stand"] /= all_std

ase_io.write("/path/to/output/data", train)	
