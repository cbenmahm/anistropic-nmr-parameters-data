import ase.io as ase_io
import torch
from e3nn import io
import numpy as np
import json

frames = ase_io.read("/path/to/predicted/ml/frames", ":")

with open("data_mean_std.json") as f:
	data = json.load(f)
for key in data:
    data[key] = np.array(data[key])

symm_transform = io.CartesianTensor("ij=ji")
antisymm_transform = io.CartesianTensor("ij=-ji")

symm_cols = [0, 4, 5, 6, 7, 8]
antisymm_cols = [1, 2, 3]

for x in frames:
	si_idx = x.numbers == 14
	o_idx = x.numbers == 8

	magres = x.arrays["magres"]
	
	magres[:, 0][si_idx] *= data["symm_si_std"][0]
        magres[:, 0][si_idx] += data["symm_si_mean"][0]

        magres[:, 0][o_idx] *= data["symm_o_std"][0]
        magres[:, 0][o_idx] += data["symm_o_mean"][0]

        magres[:, 4:][si_idx] *= data["symm_si_std"][1:]
        magres[:, 4:][si_idx] += data["symm_si_mean"][1:]

        magres[:, 4:][o_idx] *= data["symm_o_std"][1:]
        magres[:, 4:][o_idx] += data["symm_o_mean"][1:]

        magres[:, 1:4][si_idx] *= data["antisymm_si_std"]
        magres[:, 1:4][si_idx] += data["antisymm_si_mean"]

        magres[:, 1:4][o_idx] *= data["antisymm_o_std"]
        magres[:, 1:4][o_idx] += data["antisymm_o_mean"]

	ms = ct_symm.to_cartesian(torch.from_numpy(magres[..., symm_cols]))
        ms += ct_antisymm.to_cartesian(torch.from_numpy(magres[..., antisymm_cols]))

	x.arrays["ms"] = ms.numpy()

ase_io.write("/path/to/xyz/with/tensors", frames)
