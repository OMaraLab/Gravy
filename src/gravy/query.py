import pprint
import time
from collections import defaultdict

import numpy as np
import torch
from chemistry_data_structure.helpers.ir_conversion import wavenumber_to_gromacs_fc
from chemistry_data_structure.parsing.input_parsers import pdb_to_Molecule3D
from gravy.featurize import write_full_mol_graph
from gravy.preprocess import preprocessDataset
from gravy.train import init_model, init_process_group

# Global state to prevent double initialization of DGL backend
_inference_env_initialized = False


def init_inference_env(port: int = 12345):
    """
    Initialize the process group for distributed computation (required by DGL).
    Ensures it is only initialized once per process to prevent errors.
    """
    global _inference_env_initialized
    if not _inference_env_initialized:
        try:
            init_process_group(
                "gloo",
                init_method=f"tcp://127.0.0.1:{port}",
                world_size=1,
                rank=0,
            )
            _inference_env_initialized = True
        except Exception as e:
            print(f"Warning during init_process_group: {e}")


def load_inference_model(sample_graph, checkpoint_path, device=torch.device("cpu")):
    """
    Initialize and load model state dict from a single checkpoint weight file.
    Returns an evaluation-ready PyTorch model instance.
    """
    model, _, _ = init_model(sample_graph, 0, device, checkpoint_path)
    model.eval()
    return model


def predict_force_constants(
    pdb_content: str,
    model,
    node_scaler_path: str,
    edge_scaler_path: str,
    net_charge: int = 0,
    mol_name: str = "query",
    amber_units: bool = False,  # True if you'd like kcal/mol/Å^2 instead of Gromacs units (kJ/mol/nm^2)
):
    """
    Run complete end-to-end GNN inference from a PDB structure completely in-memory.

    Args:
        pdb_content: String contents of a PDB file.
        model: A single evaluation-ready PyTorch GNN model instance.
        node_scaler_path: Path to the fitted node feature MinMaxScaler (.joblib).
        edge_scaler_path: Path to the fitted edge feature MinMaxScaler (.joblib).
        net_charge: Net formal charge of the molecule.
        mol_name: Name identifier for the molecule.

    Returns:
        results: Dict mapping undirected bond pairs (sorted atom name tuples) to:
            - "gmx_fc": Predicted Gromacs force constant
            - "wavenumber": Predicted wavenumber in cm^-1
            - "atom_elements": Tuple of element symbols (e.g. ('C', 'H'))
        mol3D: The Molecule3D molecular structure instance.
    """
    # 1. Instantiate Molecule3D from PDB string
    mol3D = pdb_to_Molecule3D(
        pdb_content,
        mol_name=mol_name,
        net_charge=net_charge,
        assign_bond_orders_and_charges=True,
    )

    # 2. Featurize in-memory
    graph_ndatas = {}
    graph_edatas = {}
    graphs = {}
    write_full_mol_graph(
        mol3D=mol3D,
        graph_edatas=graph_edatas,
        graph_ndatas=graph_ndatas,
        graphs=graphs,
    )

    # 3. Preprocess and scale completely in-memory
    preprocessor = preprocessDataset(
        ndatas=graph_ndatas,
        edatas=graph_edatas,
        graphs=graphs,
    )
    preprocessor.process(
        load_scaler_path=[node_scaler_path, edge_scaler_path],
        save_graphs=False,
    )

    graph = preprocessor.graphs[mol_name]
    num_edges = graph.num_edges()

    # 4. GNN Inference (single model)
    with torch.no_grad():
        pred_fcs = model(graph, graph.ndata["h"], graph.edata["e"])
        preds = pred_fcs.cpu().numpy().flatten()

    # 5. Aggregate predictions (average bidirectional directed edges into undirected bonds)
    src_nodes, dst_nodes = graph.edges()
    src_nodes = src_nodes.tolist()
    dst_nodes = dst_nodes.tolist()

    bond_predictions = defaultdict(list)

    for edge_idx in range(num_edges):
        u_idx = str(src_nodes[edge_idx])
        v_idx = str(dst_nodes[edge_idx])

        # Physical bond key is undirected (sorted tuple of atom name strings)
        bond_key = tuple(sorted([u_idx, v_idx]))

        u_elem = mol3D.atoms[u_idx].element
        v_elem = mol3D.atoms[v_idx].element

        amber_fc_val = None
        gmx_fc_val = preds[edge_idx]
        if amber_units:
            amber_fc_val = gmx_fc_val / 418.4

        # Convert to wavenumber (cm^-1)
        wavenumber_val = wavenumber_to_gromacs_fc(
            None,
            u_elem,
            v_elem,
            backward=True,
            gmx_fc=float(gmx_fc_val),
        )
        bond_predictions[bond_key].append(
            {
                "fc": gmx_fc_val if not amber_units else amber_fc_val,
                "wavenumber": wavenumber_val,
                "elements": (u_elem, v_elem),
            }
        )

    results = {}
    for bond_key, preds_list in bond_predictions.items():
        assert (len(preds_list) == 2) and preds_list[0]["fc"] == preds_list[1][
            "fc"
        ], f"Expected exactly 2 identical predictions for bond {bond_key}, but got: { preds_list}"
        fc = preds_list[0]["fc"]
        wavenumber = preds_list[0]["wavenumber"]

        results[bond_key] = {
            "fc": float(fc),
            "wavenumber": float(wavenumber),
            "atom_elements": preds_list[0]["elements"],
        }

    return results, mol3D


if __name__ == "__main__":
    # --- Demo: Single-model inference on a PDB file ---

    NODE_SCALER = "./scalers/ndatas.z"
    EDGE_SCALER = "./scalers/edatas.z"
    CHECKPOINT = "./checkpoints/discrete_bond_order_weights.pt"

    # edit here
    PDB_PATH = "./examples/dexverapamil.pdb"
    MOL_NAME = "dexverapamil"
    NET_CHARGE = 0

    # Step 1: Initialize DGL environment
    init_inference_env()

    # Step 2: Build a dummy graph for model architecture initialisation
    pdb_content = open(PDB_PATH, "r").read()

    # We need a sample graph to initialise the model architecture,
    # so we run the featurization + preprocessing pipeline first.
    mol3D_tmp = pdb_to_Molecule3D(
        pdb_content,
        mol_name=MOL_NAME,
        net_charge=NET_CHARGE,
        assign_bond_orders_and_charges=True,
    )

    tmp_ndatas, tmp_edatas, tmp_graphs = {}, {}, {}
    write_full_mol_graph(
        mol3D=mol3D_tmp,
        graph_edatas=tmp_edatas,
        graph_ndatas=tmp_ndatas,
        graphs=tmp_graphs,
    )
    tmp_preprocessor = preprocessDataset(
        ndatas=tmp_ndatas, edatas=tmp_edatas, graphs=tmp_graphs
    )
    tmp_preprocessor.process(
        load_scaler_path=[NODE_SCALER, EDGE_SCALER], save_graphs=False
    )
    sample_graph = tmp_preprocessor.graphs[MOL_NAME]

    # Step 3: Load the single model
    model = load_inference_model(sample_graph, CHECKPOINT)

    # Step 4: Run inference
    start_time = time.process_time()
    results, mol3D = predict_force_constants(
        pdb_content=pdb_content,
        model=model,
        node_scaler_path=NODE_SCALER,
        edge_scaler_path=EDGE_SCALER,
        net_charge=NET_CHARGE,
        mol_name=MOL_NAME,
        amber_units=False,  # Set to True if you'd like kcal/mol/Å^2 instead of Gromacs units (kJ/mol/nm^2)
    )
    end_time = time.process_time()

    # Step 5: Print results
    print(f"\nInference time: {end_time - start_time:.4f} seconds (CPU)")
    print(f"\nPredicted force constants for {len(results)} bonds:\n")
    print(f"{'Bond':<15} {'Elements':<12} {'GMX FC':>10} {'Wavenumber':>12}")
    print("-" * 52)
    for bond_key, data in results.items():
        bond_order = mol3D.bond_orders.get(
            (bond_key[0], bond_key[1])
        ) or mol3D.bond_orders.get((bond_key[1], bond_key[0]))
        # If bond orders are fractional, use get() and not the bond_orders attribute
        # bond_order = mol3D.bonds[bond_key[0], bond_key[1]].get("bond_order")
        bond_order_mapping = ["-", "=", "≡"]
        bond_str = f"({int(bond_key[0])+1}, {int(bond_key[1])+1})"  # 1-based indices because PDB files are 1-indexed
        bond_symbol = "-"
        for symbol, order in zip(bond_order_mapping, [1, 2, 3]):
            if float(bond_order) < (order + 0.3):
                bond_symbol = symbol
                break
        elem_str = f"{data['atom_elements'][0]}{bond_symbol}{data['atom_elements'][1]}"
        print(
            f"{bond_str:<15} {elem_str:<12} {data['fc']:>10.2f} {data['wavenumber']:>12.2f}"
        )
