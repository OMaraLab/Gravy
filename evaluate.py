# import torch
import pickle
# import re
from collections import defaultdict

def nested_defaultdict():
    return defaultdict(list)
#
# a = pickle.load(open('../gathered_neighbours.pickle', 'rb'))
# pairs_to_evaluate = [('C', 'H'), ('C', 'O'), ('H', 'O')]
# bonds_to_evaluate = {}
# iter = 0
# for pair in a.keys():
#     if pair in pairs_to_evaluate:
#         sorted_dict = {k : v for k, v in sorted(a[pair].items(), key=lambda item: len(item[1]), reverse=True)}
#         for k, v in sorted_dict.items():
#             if k == ('C3', 'C3C3H1') or k == ('C3', 'C4O1O2') or k == ('C4H1', 'O2'):
#                 bonds_to_evaluate[k] = v
#                 print(f"Pair: {pair}, Neighbourhood: {k}, Bonds: {len(v)}")
#                 break
# #
# mol_to_evaluate = []
# for _, v in bonds_to_evaluate.items():
#     for a_b_molID in v:
#         a_b_ids = re.findall(r'\d+', a_b_molID)
#         a_b_ele = re.findall(r'[A-Z][a-z]?', a_b_molID)
#         mol_to_evaluate.append([a_b_ids, a_b_ele])
#
# unique_mol_to_evaluate = set([y[0][2] for y in mol_to_evaluate])
# to_evaluate = defaultdict(nested_defaultdict)
# for molID in unique_mol_to_evaluate:
#     for ids, pair in mol_to_evaluate:
#         if ids[2] == molID:
#             to_evaluate[molID][tuple(pair)].append(ids)
# for x in to_evaluate.items():
#     print(x)
#
# pickle.dump(to_evaluate, open('to_evaluate.pickle', 'wb'))
to_evaluate = pickle.load(open('to_evaluate.pickle', 'rb'))
# print(to_evaluate.keys())
#
import dgl
import torch
from utils import progress_bar
from train import edgeFeatureSAGEConv
from chemistry_data_structure.helpers.ir_conversion import wavenumber_to_gromacs_fc

graphs, molIDs = dgl.load_graphs("./graphs/original_hessian_complete_graphs.bin")
# model = edgeFeatureSAGEConv(graphs[0].ndata["h"].shape[1], graphs[0].edata["e"].shape[1], 512, 64, 2, "pool")
# model.load_state_dict(torch.load('./checkpoints/1st_degree_best.pt', map_location=torch.device('cpu'))["model_state_dict"])
CH_loss = []
CO_loss = []
OH_loss = []
# model.eval()
bond_types = set()
for molID in progress_bar(to_evaluate.keys(), prefix="Evaluating molecules"):
    for index, id in enumerate(molIDs['names']):
        if int(molID) == id.item():
            graph = graphs[index]
            # predicted_scores = model(graph, graph.ndata["h"], graph.edata["e"])
            predicted_scores = graph.edata["score"]
            for key in to_evaluate[molID].keys():
                bond_types.add(key)
            if ('C', 'H') in to_evaluate[molID].keys() or ('H', 'C') in to_evaluate[molID].keys():
                for bond in to_evaluate[molID][('C', 'H')] + to_evaluate[molID][('H', 'C')]:
                    atom1, atom2, _ = bond
                    atom1 = int(atom1)
                    atom2 = int(atom2)
                    for z, (src, dst) in enumerate(zip(graph.edges()[0], graph.edges()[1])):
                        if (src.item() == atom1 and dst.item() == atom2) or (src.item() == atom2 and dst.item() == atom1):
                            CH_loss.append(wavenumber_to_gromacs_fc(None, 'C', 'H', True, predicted_scores[z].item()))
            elif ('O', 'H') in to_evaluate[molID].keys() or ('H', 'O') in to_evaluate[molID].keys():
                for bond in to_evaluate[molID][('O', 'H')] + to_evaluate[molID][('H', 'O')]:
                    atom1, atom2, _ = bond
                    atom1 = int(atom1)
                    atom2 = int(atom2)
                    for z, (src, dst) in enumerate(zip(graph.edges()[0], graph.edges()[1])):
                        if (src.item() == atom1 and dst.item() == atom2) or (src.item() == atom2 and dst.item() == atom1):
                            OH_loss.append(wavenumber_to_gromacs_fc(None, 'O', 'H', True, predicted_scores[z].item()))
            elif ('C', 'O') in to_evaluate[molID].keys() or ('O', 'C') in to_evaluate[molID].keys():
                for bond in to_evaluate[molID][('C', 'O')] + to_evaluate[molID][('O', 'C')]:
                    atom1, atom2, _ = bond
                    atom1 = int(atom1)
                    atom2 = int(atom2)
                    for z, (src, dst) in enumerate(zip(graph.edges()[0], graph.edges()[1])):
                        if (src.item() == atom1 and dst.item() == atom2) or (src.item() == atom2 and dst.item() == atom1):
                            CO_loss.append(wavenumber_to_gromacs_fc(None, 'C', 'O', True, predicted_scores[z].item()))
print(f"bond types: {bond_types}")
pickle.dump({"CH": CH_loss, "CO": CO_loss, "OH": OH_loss}, open("evaluate/unprocessed_CH_CO_OH.pickle", "wb"))
