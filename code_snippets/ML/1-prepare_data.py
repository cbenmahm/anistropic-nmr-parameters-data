import numpy as np
import ase.io as ase_io
import json
from e3nn import io
import torch

train = ase_io.read("../../data/train_test/test_ml.xyz", ":")


# define the transformations
symm_transform = io.CartesianTensor("ij=ji")
antisymm_transform = io.CartesianTensor("ij=-ji")


# get the mean and std of L=0,1,2 contributions across the training set

ms_symm_si = torch.tensor([])
ms_antisymm_si = torch.tensor([])
ms_symm_o = torch.tensor([])
ms_antisymm_o = torch.tensor([])
for i in range(len(train[:10])):
        numbers = train[i].numbers
        si_idx = numbers == 14
        o_idx = numbers == 8
        ms = torch.from_numpy(train[i].arrays["ms"])
        ms = ms.reshape(-1, 3, 3)
        symm = symm_transform.from_cartesian(ms)
        antisymm = antisymm_transform.from_cartesian(ms)

        ms_symm_si = torch.cat((ms_symm_si, symm[si_idx]), dim=0)
        ms_symm_o = torch.cat((ms_symm_o, symm[o_idx]), dim=0)
        ms_antisymm_si = torch.cat((ms_antisymm_si, antisymm[si_idx]), dim=0)
        ms_antisymm_o = torch.cat((ms_antisymm_o, antisymm[o_idx]), dim=0)

symm_si_mean = ms_symm_si.mean(axis=0).cpu().numpy()
symm_si_std = ms_symm_si.std(axis=0).cpu().numpy()

symm_o_mean = ms_symm_o.mean(axis=0).cpu().numpy()
symm_o_std = ms_symm_o.std(axis=0).cpu().numpy()

antisymm_si_mean = ms_antisymm_si.mean(axis=0).cpu().numpy()
antisymm_si_std = ms_antisymm_si.std(axis=0).cpu().numpy()

antisymm_o_mean = ms_antisymm_o.mean(axis=0).cpu().numpy()
antisymm_o_std = ms_antisymm_o.std(axis=0).cpu().numpy()

# collect data in a dictionary and dump as json for later use
data = {}

data["symm_si_mean"] = symm_si_mean.tolist()
data["symm_si_std"] = symm_si_std.tolist()

data["symm_o_mean"] = symm_o_mean.tolist()
data["symm_o_std"] = symm_o_std.tolist()

data["antisymm_si_mean"] = antisymm_si_mean.tolist()
data["antisymm_si_std"] = antisymm_si_std.tolist()

data["antisymm_o_mean"] = antisymm_o_mean.tolist()
data["antisymm_o_std"] = antisymm_o_std.tolist()

with open("data_mean_std.json", "w") as f:
    json.dump(data, f)
