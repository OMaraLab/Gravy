from collections import defaultdict
import os
import pickle
import random
import re
from statistics import mean

from chemistry_data_structure.parsing import hessian_analysis, input_parsers
import featurize, utils


def nested_defaultdict_list():
    return defaultdict(list)


def gather_neighbours(
    csv_path: str,
    data_dir: str = "hessian_data",
    depth: int = 1,
    write: bool = True,
    output_path: str = "gathered_neighbours.pickle",
    molIDs_list: list[str] = None,
):
    """
    Gather the {depth}-degree neighbours of every bond in a dataset into a defaultdict(defaultdict(list)), where k : v of the outer dict is bond type (e.g., C:C) : neighbourhood type, and k : v of the inner dict is neighbourhood type : list of bonds with that neighbourhood type. In the list of bonds, each bond is in the format of the return value of Molecule3D.BFS_edge(hybridisation = True).
    NOTE: CURRENTLY ONLY WORKS FOR FIRST DEGREE NEIGHBOURS as BFS_edge is only capable of returning the first degree neighbours.
    """
    gathered_neighbours = defaultdict(nested_defaultdict_list)
    net_charges = utils.load_charges(csv_path)

    for x in utils.progress_bar(
        os.listdir(data_dir), prefix="Gathering neighbours from molecules"
    ):
        if molIDs_list and x not in molIDs_list:
            continue
        qm = featurize.load_qm_data(x, data_dir=data_dir)
        try:
            mol3D = input_parsers.ATB_QMData_to_Molecule3D(
                qm, net_charge=net_charges[x], name=x
            )
        except Exception:
            continue

        nei_a = nei_b = bond_tag = None
        for i, j in mol3D.bonds:
            nei_a, nei_b, bond_tag = mol3D.BFS_edge(i, j, depth, hybridisation=True)
            ele_a_id, ele_b_id, _ = tuple(bond_tag.split("_"))
            ele_a = re.search(r'[A-Z]+', ele_a_id).group()
            ele_b = re.search(r'[A-Z]+', ele_b_id).group()

            ele_key = tuple(sorted((ele_a, ele_b)))
            nei_key = tuple(sorted((nei_a, nei_b)))

            gathered_neighbours[ele_key][nei_key].append(bond_tag)

    print(
        f"Gathered {len(gathered_neighbours)} element pairs, {len([nei for bond_type in gathered_neighbours for nei in gathered_neighbours[bond_type]])} bond environments, and {sum([len(gathered_neighbours[bond_type][nei]) for bond_type in gathered_neighbours for nei in gathered_neighbours[bond_type]])} bonds from {len(os.listdir(data_dir))} molecules. Written to {output_path}."
    )
    if write:
        pickle.dump(gathered_neighbours, open(output_path, "wb"))
    return gathered_neighbours


def calc_mean_fc(bond_list: list, output_fc_list: bool = False):
    """
    Calculates the mean force constant for a list of bonds. Each bond in the list should be in the format of the return value of Molecule3D.BFS_edge(hybridisation=True), i.e., C1_C2_2001 ({elementa}{atomID}_{element}{atomID}_{molID})
    """
    fc_list = []
    for x in bond_list:
        IDs = re.findall(r"(\d+)", x)
        atoms = re.findall(r"[a-zA-Z]+", x)
        if len(IDs) != 3 or len(atoms) != 2:
            raise ValueError(f"Invalid bond format: {x}")
        qm_data = featurize.load_qm_data(IDs[2])
        qm_pairs = [(a, b) for a, b, _ in qm_data["bond_order"]]
        umatrix, eigmatrix = hessian_analysis.cal_eigen_matrix(
            qm_data["primary_axis_coords"], qm_data["hessian"]
        )
        if (int(IDs[0]) + 1, int(IDs[1]) + 1) in qm_pairs:
            fc_list.append(
                hessian_analysis.cal_stretching(
                    (int(IDs[0]) + 1, int(IDs[1]) + 1), umatrix, eigmatrix
                )
            )

        elif (int(IDs[1]) + 1, int(IDs[0]) + 1) in qm_pairs:
            fc_list.append(
                hessian_analysis.cal_stretching(
                    (int(IDs[1]) + 1, int(IDs[0]) + 1), umatrix, eigmatrix
                )
            )
        else:
            print(f"Bond not found in QM data: {IDs}")
            continue
    mean_fc = mean(fc_list)

    if output_fc_list:
        return mean_fc, fc_list
    return mean_fc


def get_popular_fragmnets(gathered_neighbours_path: str, top_n: int = 5):
    """
    Find the top n most prevalent fragments in a set of gathered neighbours (written by gather_neighbours()).
    NOTE: I should probably do the sorting in gather_neighbours() itself, but I would probably lose backward compatibility somewhere down the pipeline.
    """
    gathered_neighbours = pickle.load(open(gathered_neighbours_path, "rb"))
    sorted_gathered_neighbours = defaultdict(dict)
    for pair in gathered_neighbours:
        sorted_neighbourhoods_within_pair = sorted(
            gathered_neighbours[pair],
            key=lambda nei: len(gathered_neighbours[pair][nei]),
            reverse=True,
        )
        sorted_gathered_neighbours[pair] = {
            nei: gathered_neighbours[pair][nei]
            for nei in sorted_neighbourhoods_within_pair
        }
    n_top_fragments_in_each_pair = []
    for pair in sorted_gathered_neighbours:
        n = 0
        for nei in sorted_gathered_neighbours[pair]:
            n_top_fragments_in_each_pair.append(
                (
                    pair,
                    nei,
                    len(sorted_gathered_neighbours[pair][nei]),
                    sorted_gathered_neighbours[pair][nei],
                )
            )
            n += 1
            if n > top_n:
                break
    return sorted(n_top_fragments_in_each_pair, key=lambda x: x[2], reverse=True)


def draw_and_select_top_fragments(
    gathered_neighbours_path: str,
    data_dir: str,
    net_charge_path: str,
    top_fragments: list,
    n: int = 5,
):
    """
    Interactively draw and select the top n most prevalent fragments in the list of popular fragments returned by get_popular_fragments using Molecule3D method.
    """
    fragments_by_prevalence = defaultdict(list)
    gathered_neighbours = pickle.load(open(gathered_neighbours_path, "rb"))
    net_charges = utils.load_charges(net_charge_path)
    draw_more = True
    current_index = 0
    while draw_more:
        for i in range(current_index, n):
            pair, nei, count, bonds = top_fragments[i]
            print(f"Drawing fragment: {pair}/{nei}, which appears {count} times in the dataset.")
            draw_more_of_this_pair = True
            while draw_more_of_this_pair:
                bond = random.choice(gathered_neighbours[pair][nei])
                atom1, atom2, molID = re.findall(r"(\d+)", bond)
                print(f"Drawing molecule {molID}")
                qm_data = featurize.load_qm_data(molID, data_dir=data_dir)
                mol3D = input_parsers.ATB_QMData_to_Molecule3D(
                    qm_data, net_charge=net_charges[molID], name=molID
                )
                mol3D.draw_graph(show=True, node_size=15, mark_atoms=[atom1, atom2])
                draw_more_of_this_pair = input("Draw another of this pair? (y/n): ").lower() == "y"
            if input("Add this fragment to the list? (y/n): ").lower() == "y":
                fragments_by_prevalence[pair].append((nei, count, bonds))
            current_index = i
        current_index += 1
        n += 5
        draw_more = input("Draw more? (y/n): ").lower() == "y"
    return fragments_by_prevalence

