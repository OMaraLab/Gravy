import pickle
from collections import defaultdict

import dgl
import torch
import pandas as pd
from progress_bar
from sklearn.preprocessing import MinMaxScaler

from constants import nodeFeatures as nf


class preprocessDataset:
    def __init__(self, dataset_prefix, dataset_path, overwrite_with=None):
        self.ndatas = pickle.load(
            open(f"{dataset_path}/{dataset_prefix}_graph_ndatas.pickle", "rb")
        )
        self.edatas = pickle.load(
            open(f"{dataset_path}/{dataset_prefix}_graph_edatas.pickle", "rb")
        )
        self.graphs = pickle.load(
            open(f"{dataset_path}/{dataset_prefix}_graphs.pickle", "rb")
        )
        self.overwrite_with = overwrite_with
        self.dataset_prefix = dataset_prefix
        self.dataset_path = dataset_path

    def _overwrite_feature_vectors(
        self, new_dataset_prefix, new_dataset_path, update_all=False
    ):
        """
        Overwrite entries in the current dataset with commensurate entries from a new dataset. Used to update force constants on bonds that are considered equivalent under the various aggregation schemes. It is not usually necessary to update the node features and graph structure.
        """
        new_edatas = pickle.load(
            open(f"{new_dataset_path}/{new_dataset_prefix}_graph_edatas.pickle", "rb")
        )
        if update_all:
            new_ndatas = pickle.load(
                open(
                    f"{new_dataset_path}/{new_dataset_prefix}_graph_ndatas.pickle", "rb"
                )
            )
            new_graphs = pickle.load(
                open(f"{new_dataset_path}/{new_dataset_prefix}_graphs.pickle", "rb")
            )

        for molID in self.edatas:
            if molID in new_edatas:
                assert len(self.edatas[molID]) == len(
                    new_edatas[molID]
                ), f"Number of edges in edatas for {molID} does not match between current and new dataset: {len(self.edatas[molID])} vs {len(new_edatas[molID])}"
                self.edatas[molID] = new_edatas[molID]
            if update_all and molID in new_ndatas and molID in new_graphs:
                self.ndatas[molID] = new_ndatas[molID]
                self.graphs[molID] = new_graphs[molID]

    def _concatenate_feature_vecctors(self, molID, check=True):
        """
        For a certain molecule, concatenate its node vectors from a per-pair dictionary format (from get_bond_features()) to a per-node list format. Also concatenate its edge vectors from a per-pair dictionary format to a per-edge list format.
        """
        if check:
            for id, pair in self.edatas.items():
                assert (
                    tuple(reversed(pair)) not in self.edatas[id]
                ), f"Pair {pair} in edatas of {id} is reversed in the dictionary, which is not allowed."

        unpaired_ndatas = []
        for i in range(self.graphs[molID].num_nodes()):
            for x in self.ndatas[molID]:
                atom1_nei, atom2_nei = x[nf.FIRST_DEGREE_NEIGHBOURS]
                if x[nf.ATOM1_ID] == i:
                    # see features in constants.nodeFeatures
                    unpaired_ndatas.append(
                        x[nf.ATOM1_ELEMENT : nf.ATOM2_ID] + [atom1_nei] + [molID]
                    )
                    break
                elif x[nf.ATOM2_ID] == i:
                    unpaired_ndatas.append(
                        x[nf.ATOM2_ELEMENT : nf.BOND_LENGTH] + [atom2_nei] + [molID]
                    )
                    break

        assert (
            len(unpaired_ndatas) == self.graphs[molID].num_nodes()
        ), f"Number of unpaired ndatas {len(unpaired_ndatas)} does not match number of nodes {self.graphs[molID].num_nodes()} in graph {molID}"

        edatas = []
        # NOTE: self.graphs[molID].edges() returns two tensors containing outgoing and incoming nodes in the bidirectional molecular graph.
        for atom1, atom2 in zip(
            self.graphs[molID].edges()[0].tolist(),
            self.graphs[molID].edges()[1].tolist(),
        ):
            for ndata in self.ndatas[molID]:
                if (atom1, atom2) == (ndata[nf.ATOM1_ID], ndata[nf.ATOM2_ID]) or (
                    atom1,
                    atom2,
                ) == (ndata[nf.ATOM2_ID], ndata[nf.ATOM1_ID]):
                    edatas.append(ndata[nf.BOND_LENGTH : nf.FIRST_DEGREE_NEIGHBOURS])
                    break
            for pair in self.edatas[molID]:
                if (str(atom1), str(atom2)) == pair:
                    edatas[-1].append([molID, self.edatas[molID][pair], pair])
                elif (str(atom2), str(atom1)) == pair:
                    edatas[-1].append([molID, self.edatas[molID][pair], pair[::-1]])
                    break
        assert (
            len(edatas) == self.graphs[molID].num_edges()
        ), f"Number of edatas {len(edatas)} does not match number of edges {self.graphs[molID].num_edges()} in graph {molID}"

        return unpaired_ndatas, edatas

    def _one_hot_encode_features(self, features_list: list):
        ndatas_dataframe = pd.DataFrame(
            features_list,
            columns=[
                "element",
                "atomic_number",
                "radius",
                "mass",
                "electronegativity",
                "hybridisation",
                "nei",
                "molID",
            ],
        )
        ohe_elements = pd.get_dummies(ndatas_dataframe["element"], prefix="ele_")
        ohe_neighbours = pd.get_dummies(ndatas_dataframe["nei"], prefix="nei_")
        ndatas_dataframe.drop(columns=["element", "nei"], inplace=True)
        return ndatas_dataframe.join(ohe_elements).join(ohe_neighbours)

    def _parse_feature_dataframe(self, features_dataframe):
        """
        Parse the feature dataframe to get a list of [molID, start_idx, num] where start_idx is the first feature of molID in the feature dataframe, and num is the number of features belonging to that molID inlucding the one at start_idx (this assumes the features for each molID are contiguous in features_dataframe).
        """
        index_list = []
        current_mol = (features_dataframe.iloc[0], 0, 1)
        for i in range(1, len(features_dataframe)):
            if features_dataframe.iloc[i] == current_mol[0]:
                current_mol = (current_mol[0], current_mol[1], current_mol[2] + 1)
            else:
                index_list.append(current_mol)
                current_mol = (features_dataframe.iloc[i], i, 1)
        index_list.append(current_mol)

        return index_list

    def _normalize_features(self, features):
        return MinMaxScaler().fit_transform(features)

    def _save_graphs(self):
        graph_list = []
        molID_list = []
        for molID in self.graphs:
            graph_list.append(self.graphs[molID])
            molID_list.append(int(molID))
        fn = (
            f"{self.dataset_path}/graphs/{self.dataset_prefix}_complete_graphs.bin"
            if not self.overwrite_with
            else f"{self.dataset_path}/graphs/{self.dataset_prefix}_overwrite_with_{self.overwrite_with}_complete_graphs.bin"
        )
        dgl.save_graphs(fn, graph_list, {"names": torch.tensor(molID_list)})

    def process(
        self,
        update_all=False,
    ):
        """
        Create dgl graphs with node and edge features from the dataset.
        """
        if self.overwrite_with:
            self._overwrite_feature_vectors(
                self.overwrite_with, self.dataset_path, update_all=update_all
            )
        all_ndatas = []
        all_edatas = []
        for molID in progress_bar(
            self.graphs, prefix="creating node and edge feature vectors"
        ):
            unpaired_ndatas, edatas = self._concatenate_feature_vecctors(molID)
            all_ndatas.extend(unpaired_ndatas)
            all_edatas.extend(edatas)

        ndatas_df = self._one_hot_encode_features(all_ndatas)
        ndatas_index_list = self._parse_feature_dataframe(ndatas_df["molID"])
        ndatas_array = self._normalize_features(
            ndatas_df
        )  # fit_transform() returns ndarray, so this line needs to come after the previous
        for molID in progress_bar(self.graphs, prefix="assigning node features"):
            for ndata_molID in ndatas_index_list:
                if molID == ndata_molID[0]:
                    self.graphs[molID].ndata["h"] = torch.from_numpy(
                        ndatas_array[ndata_molID[1] : (ndata_molID[1] + ndata_molID[2])]
                    ).to(torch.float32)

        edatas_df = pd.DataFrame(
            all_edatas, columns=["bond_length", "bond_order", "molID_fc_pair"]
        )
        edatas_index_list = self._parse_feature_dataframe(
            edatas_df["molID_fc_pair"].apply(lambda x: x[0])
        )
        edatas_score_array = edatas_df["molID_fc_pair"].apply(lambda x: x[1]).to_numpy()
        edatas_feat_array = self._normalize_features(
            edatas_df.drop(columns=["molID_fc_pair"])
        )
        for molID in progress_bar(self.graphs, prefix="assigning edge features"):
            for edata_molID in edatas_index_list:
                if molID == str(edata_molID[0]):
                    self.graphs[molID].edata["e"] = torch.from_numpy(
                        edatas_feat_array[
                            edata_molID[1] : (edata_molID[1] + edata_molID[2])
                        ]
                    ).to(torch.float32)
                    self.graphs[molID].edata["score"] = torch.from_numpy(
                        edatas_score_array[
                            edata_molID[1] : (edata_molID[1] + edata_molID[2])
                        ]
                    ).to(torch.float32)
                    break
        self._save_graphs()
