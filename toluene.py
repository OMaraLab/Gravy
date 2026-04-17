import torch
import pprint
import numpy as np
from chemistry_data_structure.parsing.input_parsers import ATB_QMData_to_Molecule3D
from chemistry_data_structure.helpers.ir_conversion import wavenumber_to_gromacs_fc
from featurize import load_qm_data
from train import init_model, init_process_group, graphDataset, edgeFeatureSAGEConv
from utils import load_charges

charges = load_charges("../netcharges_40000.csv")
qm_data = load_qm_data("21", "../hessian_data_40000")
mol3D = ATB_QMData_to_Molecule3D(qm_data, charges['21'])

dataset = graphDataset("original_hessian_40000", "../graphs/original_hessian_40000_complete_graphs.bin")
device = torch.device('cpu')
init_process_group("gloo", init_method="tcp://127.0.0.1:12345", world_size=1, rank=0)
checkpoins = ["../checkpoints/original_hessian_40000_best_fold_2_of_10.pt", "../checkpoints/original_hessian_40000_best_fold_3_of_10.pt", "../checkpoints/original_hessian_40000_best_fold_4_of_10.pt", "../checkpoints/original_hessian_40000_best_fold_5_of_10.pt", "../checkpoints/original_hessian_40000_best_fold_6_of_10.pt"]

tol_freq = np.zeros((len(checkpoins), len(dataset[29518].edges()[0])))
for idy, point in enumerate(checkpoins):
    model, _, _ = init_model(dataset[0], 0, device, point)
    model.eval()
    with torch.no_grad():
        tol_fc = model(dataset[29518], dataset[29518].ndata['h'], dataset[29518].edata['e'])
        for idx, (x, y) in enumerate(zip(tol_fc, zip(dataset[29518].edges()[0], dataset[29518].edges()[1]))):
            freq = wavenumber_to_gromacs_fc(None, mol3D.atoms[str(y[0].item())].element, mol3D.atoms[str(y[1].item())].element, backward=True, gmx_fc=x.item())
            print(idx)
            tol_freq[idy][idx]+= freq

std = np.std(tol_freq, axis=0).tolist()
for x, y in zip(std, zip(dataset[29518].edges()[0], dataset[29518].edges()[1])):
    print(y, x)


seminario_freq = []
for x in mol3D.bonds:
    seminario_freq.append(wavenumber_to_gromacs_fc(None, mol3D.atoms[x[0]].element, mol3D.atoms[x[1]].element, backward=True, gmx_fc=mol3D.bonds[x].get("force_constant")))

pprint.pprint(tol_freq)
mol3D.draw_graph(node_label_mode='id', node_size=15)

