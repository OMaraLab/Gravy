import dgl
import os
import pickle
import random
import re
import statistics
from sys import exit

from chemistry_data_structure.helpers.graphs import calc_equal_bonds, cull_equal_bonds
from pulp import PulpSolverError
from utils import load_charges, progress_bar


def load_qm_data(
    molid: str, data_dir: str = "/home/yaofu/data/atb_fc/NXMol/src/hessian_data"
):
    """
    Load pickled atb hessian data by molid.
    NOTE: os.walk removes training slashes. DO NOT add training slashes when calling!!!

    @params:
        molid    - Required  : molecule id (Str)
    """
    for dirpath, _, filename in os.walk(data_dir):
        if dirpath == f"{data_dir}/{molid}":
            if len(filename) == 1:
                with open(f"{dirpath}/{filename[0]}", "rb") as fh:
                    return pickle.load(fh)
            else:
                continue
    raise FileNotFoundError(f"Could not find {molid} in {data_dir}")


def load_mol3D(qm_data, net_charge, molID, discretise_bond_order=False):
    """
    Load a mol3Decule object from qm_data.

    @params:
        qm_data    - Required  : qm data returned from load_qm_data (Dict)
        net_charge - Optional  : net charge of the molecule (Int)
        molID       - Optional  : molID of the molecule (Str)
    """
    from chemistry_data_structure.parsing.input_parsers import ATB_QMData_to_Molecule3D

    return ATB_QMData_to_Molecule3D(qm_data, net_charge=net_charge, name=molID, discretise_bond_order=discretise_bond_order)


def get_bond_features(mol3D, i: int, j: int) -> list:
    """
    Encode the features of a bond in a mol3Decule into a list.

    @params:
        mol3D    - Required  : mol3Decule object (mol3D)
    """
    return [
        int(i),
        mol3D.atoms[i].element,
        mol3D.atoms[i].atomic_number,
        mol3D.atoms[i].radius,
        mol3D.atoms[i].mass,
        mol3D.atoms[i].electronegativity,
        mol3D.calcNumBonds(i),
        int(j),
        mol3D.atoms[j].element,
        mol3D.atoms[j].atomic_number,
        mol3D.atoms[j].radius,
        mol3D.atoms[j].mass,
        mol3D.atoms[j].electronegativity,
        mol3D.calcNumBonds(j),
        mol3D.bonds[i, j].get("bond_length"),
        mol3D.bonds[i, j].get("bond_order"),
        mol3D.BFS_edge(i, j, 1),
    ]


def calc_mol3D_vectors(mol3D, bond_list=None, calc_X=True):
    X = []
    y = {}
    if bond_list:
        update_list = bond_list
    else:
        update_list = mol3D.bonds

    for i, j in update_list:
        y[i, j] = mol3D.bonds[i, j].get("force_constant")

        if calc_X == False:
            continue

        if random.random() < 0.5:
            X.append(get_bond_features(mol3D, i, j))
        else:
            X.append(get_bond_features(mol3D, j, i))
    if calc_X:
        return X, y
    else:
        return y


def make_graph(mol3D):
    """
    Make a DGL graph from a mol3D object.
    """
    u = []
    v = []
    for i, j in mol3D.bonds:
        u.append(int(i))
        v.append(int(j))
    tmp = u
    u = u + v
    v = v + tmp
    return dgl.graph((u, v))


def merge_edatas(edata1, edata2):
    for k, v in edata2.items():
        if k not in edata1 and k[::-1] not in edata1:
            # this condition happens when fdb thinks there should be a bond but that bond does not exist after ATB_QMData_to_Molecule3D
            continue
        else:
            assert bool(k in edata1) != bool(
                k[::-1] in edata1
            ), f"Bond {k} is already in edata1, but its reverse {k[::-1]} is also present. edatas1: {edata1}"
            if k in edata1:
                edata1[k] = v
            elif k[::-1] in edata1:
                edata1[k[::-1]] = v
            assert bool(k in edata1) != bool(k[::-1] in edata1)
    return edata1


def write_full_mol_graph(
    mol3D,
    graph_edatas,
    graph_ndatas,
    graphs,
    bond=None,
    fc=None,
    rotational_equivalence=False,
):
    if rotational_equivalence:
        calc_equal_bonds(mol3D)
        cull_equal_bonds(mol3D, "mean", True)

    graph_ndatas[mol3D.name], graph_edatas[mol3D.name] = calc_mol3D_vectors(
        mol3D, None, True
    )
    if bond is not None and fc is not None:
        graph_edatas[mol3D.name] = merge_edatas(graph_edatas[mol3D.name], {bond: fc})
    graphs[mol3D.name] = make_graph(mol3D)


def write_partial_mol_graph(mol3D, bond, graph_edatas, fc):
    """
    When writing partial molecular graphs, we only write the features of the bond specified by the bond tuple. The node data remain unchanged.
    """
    graph_edatas[mol3D.name] = merge_edatas(graph_edatas[mol3D.name], {bond: fc})


def write_graph_features(mol3D, atom1, atom2, fc, graph_edatas, graph_ndatas, graphs):
    """
    Reassign a bond force constant, and then write either a full or partial molecular graph and features depending on whether the molID is already in the provided graph/feature sets. If reassignment is not needed, the function will simply write a full molecular graph.
    """
    if atom1 == None and atom2 == None and fc == None:
        write_full_mol_graph(mol3D, graph_edatas, graph_ndatas, graphs)

    elif atom1 != None and atom2 != None and fc != None:
        if mol3D.name in graph_ndatas:
            write_partial_mol_graph(mol3D, (atom1, atom2), graph_edatas, fc)
            value = graph_edatas[mol3D.name].get((atom1, atom2)) or graph_edatas[
                mol3D.name
            ].get((atom2, atom1))
            assert (
                value == fc or value == None # None handles the case where the bond does not exist due to fdb error in merge_edatas
            ), f"Bond ({atom1}, {atom2}) in {mol3D.name} has force constant {value}, expected {fc}."
        else:
            write_full_mol_graph(
                mol3D, graph_edatas, graph_ndatas, graphs, (atom1, atom2), fc
            )
    else:
        raise ValueError(
            f"atom1, atom2, and fc must all be None, or all be defined. Got atom1: {atom1}, atom2: {atom2}, fc: {fc}."
        )


def create_graph_dataset(
    hessian_data_path, charges_fn, gathered_neighbours=None, fdb_path=None, check=False, output_prefix="",
    discretise_bond_order=False
):
    """
    Generate molecular graphs and features from gathered neighbours. Will generate as many graphs as there are unique molIDs in gathered_neighbours.

    @params:
        gathered_neighbours - Optional  : returned from fragment_search.gathered_neighbours
        fdb_path            - Optional  : path to the FDB fragments
        check               - Optional  : checks if all gathered neighbours have the same force constant
    """
    from fragment_search import calc_mean_fc
    from chemistry_data_structure.helpers.ir_conversion import (
        get_distr_from_hessian_FDB,
        get_molecules_in_FDB_fragment,
    )

    assert (bool(gathered_neighbours) != bool(fdb_path)) or (
        gathered_neighbours == fdb_path == None
    ), "Either gathered_neighbours or fdb_path must be provided, or neither, but not both."

    charges = load_charges(charges_fn)
    graph_ndatas = {}
    graph_edatas = {}
    graphs = {}

    if gathered_neighbours:
        for pair_type, nei_dict in progress_bar(
            gathered_neighbours.items(),
            prefix="Writing graphs from gathered neighbours",
        ):
            sorted_neis = sorted(
                nei_dict.items(), key=lambda x: len(x[1]), reverse=True
            )
            bonds_processed = []
            for nei, bonds in sorted_neis:
                mean_fc = calc_mean_fc(bonds)
                for bond in bonds:
                    atom1, atom2, molID = re.findall(r"\d+", bond)
                    try:
                        mol3D = load_mol3D(
                            load_qm_data(molID), net_charge=charges[molID], molID=molID
                        )
                        write_graph_features(
                            mol3D,
                            atom1,
                            atom2,
                            mean_fc,
                            graph_edatas,
                            graph_ndatas,
                            graphs,
                        )
                        bonds_processed.append(bond)
                        if check and (atom1 is not None and atom2 is not None):
                            value = graph_edatas[molID].get(
                                (atom1, atom2)
                            ) or graph_edatas[molID].get((atom2, atom1))
                            assert (
                                value == mean_fc
                            ), f"Bond {bond} in {molID} has force constant {value}, expected {mean_fc}."
                    except (KeyError, PulpSolverError):
                        continue
            if check:
                for _ in range(5):
                    nei, bonds_to_check = random.choice(sorted_neis)
                    print(f"Checking bonds for nei: {nei}, pair_type: {pair_type}")
                    print("bonds_to_check:", bonds_to_check)
                    assert set(bonds_to_check).issubset(set(bonds_processed)), (
                        f"Not all bonds in nei {nei} pair_type ({pair_type}) were processed. "
                        f"Processed bonds: {bonds_processed}, "
                        f"Bonds to check: {bonds_to_check}"
                    )
                    mean_fc = calc_mean_fc(bonds_to_check)
                    for bond in bonds_to_check:
                        try:
                            atom1, atom2, molID = re.findall(r"\d+", bond)
                            value = graph_edatas[molID].get(
                                (atom1, atom2)
                            ) or graph_edatas[molID].get((atom2, atom1))
                            assert (
                                value == mean_fc
                            ), f"Bond {bond} in {molID} has force constant {value}, expected {mean_fc}."
                        except KeyError:
                            print(f"KeyError for bond {bond} in {molID}.")
                            continue
    elif fdb_path:
        for fdb_file in progress_bar(
            os.listdir(fdb_path), prefix="Writing graphs from FDB fragments"
        ):
            fdb_id = fdb_file.split(".")[0] if fdb_file.endswith(".json") else None
            if not fdb_id:
                continue
            fdb_frag_fcs = get_distr_from_hessian_FDB(fdb_id, fdb_path=fdb_path)[:-1]
            if not fdb_frag_fcs:
                with open(f"excl_fdb_frags_{output_prefix}_feat", "a") as fh:
                    fh.write(f"excluded fdb_id is {fdb_id}\n")
                continue
            mean_fc = statistics.mean(fdb_frag_fcs)
            for mol3D, bond_list in get_molecules_in_FDB_fragment(
                fdb_id, fdb_path, charges
            ):
                for a, b in bond_list:
                    atom1, atom2 = str(a - 1), str(b - 1)
                    try:
                        write_graph_features(
                            mol3D,
                            atom1,
                            atom2,
                            mean_fc,
                            graph_edatas,
                            graph_ndatas,
                            graphs,
                        )
                    except KeyError:
                        continue

    else:
        for molID in progress_bar(
            os.listdir(hessian_data_path),
            prefix="Writing graphs from hessian data (not gathering neighbours)",
        ):
            try:
                mol3D = load_mol3D(
                    load_qm_data(molID, hessian_data_path), net_charge=charges[molID], molID=molID, discretise_bond_order=discretise_bond_order
                )
                write_graph_features(
                    mol3D, None, None, None, graph_edatas, graph_ndatas, graphs
                )
            except Exception as e:
                print(f"Error loading {molID}. {type(e).__name__}: {e}")
                continue

    pickle.dump(graph_ndatas, open(f"{output_prefix}_graph_ndatas.pickle", "wb"))
    pickle.dump(graph_edatas, open(f"{output_prefix}_graph_edatas.pickle", "wb"))
    pickle.dump(graphs, open(f"{output_prefix}_graphs.pickle", "wb"))
