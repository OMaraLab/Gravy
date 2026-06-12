## Graph neural network to assign bond vibrational frequency

Neural network model discussed in "A graph-based machine learning framework to assign empirical interaction parameters for novel molecules" (doi)

It is recommeded that you use a new conda environment to install this package and its dependencies.
```
conda create --name gravy python=3.11.8
```

### Dependencies
- python 3.11.8
- CUDA 12.1 (optional, see point about DGL)
- [chemical_equivalence](https://github.com/ATB-UQ/chemical_equivalence)
- [atb_output](https://github.com/ATB-UQ/atb_outputs)
- [NXMol](https://github.com/OMaraLab/NXMol)
- **[DGL](https://www.dgl.ai/pages/start.html)** 
**NOTE** that the specific DGL version specified in `pyproject.toml` is tested and compliant with CUDA 12.1 on Linux/WSL. DGL has dropped support for Windows and MacOS, so if you're on those platforms, you will need to comment out the DGL line in `pyproject.toml` and manually install DGL with an older version of torch.
### Installation
```
# clone this repo
cd Gravy
pip install .
```
### Inference
Example PDB files are in `src/gravy/examples`. To execute the dexverapamil example, simply run ```python query.py```.

To use your own PDB file (geometry should be optimised), edit ```query.py``` so that
```
PDB_PATH
MOL_NAME
NET_CHARGE
```
reflect your molecule of interest.

### TODO
- ~~Manual passing of fractional bond orders during PDB~~
