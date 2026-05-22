## Graph neural network to assign bond vibrational frequency

Neural network model discussed in "A graph-based machine learning framework to assign empirical interaction parameters for novel molecules" (doi)

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
- Manual passing of fractional bond orders during PDB
