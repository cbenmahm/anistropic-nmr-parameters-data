# Supporting Data for "Graph-neural-network predictions of solid-state NMR parameters from spherical tensor decomposition"

---

This repository contains the supporting data of:

<div align="center">

> **[Graph-neural-network predictions of solid-state NMR parameters from spherical tensor decomposition](https://arxiv.org/abs/2412.15063)**\
> _Chiheb Ben Mahmoud, Louise A. M. Rosset, Jonathan R. Yates, and Volker L. Deringer_

</div>

## Repo Overview

- **[data/](data/)** contains the data necessary to generate the figures and the tables and organized as follows:
  - train_test: contains the `extxyz` files of the training and test structures
  - aSiO2_models: contains the 300-atom structural models used in Figure 5
  - hypothetical_zeolites: contains the hypothetical zeolites presented in Figure 6
  - cristobalite: contains the cristobalite trajectory of Figure 7
- **[notebooks/](notebooks/)** contains the jupyter notebooks to generate the figures
- **[models/](models/)** contains the tneosr NMR ML models  used in this work
- **[code_snippets/](code_snippets/)** contains examples of the DFT input files, how to generate the training data using melt-quench-anneal simulations, and snippets to train the NequIP tensor models
## Dependencies

To run the notebooks, you need the usual dependencies (`numpy`, `matplotlib`, `jupyter`, `ase`, `torch`).

To use the trained models, you can use our modifications to the `NequIP` architecture, found at this [GitHub repository](https://github.com/cbenmahm/nequip-nmr-parameters).
