# Python Idealized SpecTral Ocean Modeling

[Doc](https://pistol-doc.readthedocs.io/en/latest/)

Other relevant packages: 
- [pyqg](https://github.com/crocha700/pyqg/blob/master/pyqg/qg_model.py)
- [spectralDNS](https://github.com/spectralDNS/spectralDNS)

Several cores: QuasiGeostrophic (QG), HYdrostatic (HY), Non-Hydrostatic (NH)

Heavily relies shenfun library that may be found [here](https://github.com/spectralDNS/shenfun).

## Installation

```
conda create --name pistol -c conda-forge -c spectralDNS python=3.6 shenfun h5py-parallel matplotlib
source activate pistol
conda install -c conda-forge jupyter 
```

## Run

```
mpirun -np 4  python code.py
```

