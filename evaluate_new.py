from chemistry_data_structure.helpers.ir_conversion import wavenumber_to_gromacs_fc
from collections import defaultdict
import dgl.data
import numpy as np
import re
from constants import ir_intervals, fg_map
from train import edgeFeatureSAGEConv, graphDataset
from utils import progress_bar
from scipy.spatial import distance
import torch
from scipy.stats import gaussian_kde
from math import sqrt
import statistics
import matplotlib.pyplot as plt


def hsv_transform_cmap(cmap, factor):
    """
    makes colormap darker or lighter by transforming the V channel in HSV color space
    """
    from matplotlib.colors import ListedColormap, rgb_to_hsv, hsv_to_rgb
    from matplotlib import colormaps as cm

    rgb = (cm[cmap](np.linspace(0, 1, cm[cmap].N)) ** 0.8)[:, :3]
    hsv = rgb_to_hsv(rgb)
    hsv[:, 2] = hsv[:, 2] * factor
    new_rgb = hsv_to_rgb(hsv)
    return ListedColormap(new_rgb)


def get_test_set_molIDs(dataset_path: str, seed: int):
    """
    Reads the dataset file and extracts the test set IDs.
    """
    dataset = graphDataset("dataset", dataset_path)
    _, val_set, test_set = dgl.data.split_dataset(
        dataset, frac_list=[0.8, 0.1, 0.1], shuffle=True, random_state=seed
    )
    test_id_array = np.concatenate((test_set.indices, val_set.indices))
    test_molIDs = [str(dataset.molIDs["names"][i].item()) for i in test_id_array]
    return test_molIDs


def adib(value, lower_bound, upper_bound):
    """
    Calculates the mean squared distance of a single wavenumber from a defined IR band. 
    If the value is within the band, the distance is 0. Otherwise, it's the shortest 
    distance to either bound.
    """
    if lower_bound <= value <= upper_bound:
        return 0.0
    elif value < lower_bound:
        return (lower_bound - value) ** 2
    else:
        return (value - upper_bound) ** 2


def calculate_adib_for_bonds(
    bonds: list,
    dataset_path: str,
    model_path: str,
    lower_bound: float,
    upper_bound: float,
    plot: bool = True,
    title: str = "Wavenumber Distribution",
    plot_bounds: tuple = None,
):
    """
    Calculate the average distance from IR bands for a given set of bonds, using 
    both the GNN predictions and the Seminario force constants (both converted to 
    wavenumbers). Also draws two histograms of the wavenumber distributions with 
    the IR boundaries labelled.
    """
    to_evaluate = defaultdict(list)
    for bond in bonds:
        atom1_id, atom2_id, molID = re.findall(r"(\d+)", bond)
        atom1_ele, atom2_ele = re.findall(r"([A-Z]+)", bond)
        to_evaluate[molID].append(
            ((atom1_ele, atom2_ele), (int(atom1_id), int(atom2_id)))
        )

    gnn_wavenumbers = []
    seminario_wavenumbers = []
    model_params = torch.load(model_path, map_location="cpu")
    dataset = graphDataset("dataset", dataset_path)
    model = edgeFeatureSAGEConv(
        dataset[0].ndata["h"].shape[1],
        dataset[0].edata["e"].shape[1],
        512,
        64,
        2,
        "pool",
    )
    model.load_state_dict(model_params["model_state_dict"])
    model.eval()
    with torch.no_grad():
        for a in progress_bar(to_evaluate, prefix="Calculating wavenumbers"):
            for idx, b in enumerate(dataset.molIDs["names"]):
                if int(a) == b.item():
                    graph = dataset[idx]
                    predicted_scores = model(graph, graph.ndata["h"], graph.edata["e"])
                    for eles, ids in to_evaluate[a]:
                        for z, (src, dst) in enumerate(
                            zip(graph.edges()[0], graph.edges()[1])
                        ):
                            if (src.item() == ids[0] and dst.item() == ids[1]) or (
                                src.item() == ids[1] and dst.item() == ids[0]
                            ):
                                gnn_wavenumbers.append(
                                    wavenumber_to_gromacs_fc(
                                        None,
                                        eles[0],
                                        eles[1],
                                        True,
                                        predicted_scores[z].item(),
                                    )
                                )
                                seminario_wavenumbers.append(
                                    wavenumber_to_gromacs_fc(
                                        None,
                                        eles[0],
                                        eles[1],
                                        True,
                                        graph.edata["score"][z].item(),
                                    )
                                )
    gnn_adib = np.array([adib(x, lower_bound, upper_bound) for x in gnn_wavenumbers])
    seminario_adib = np.array(
        [adib(x, lower_bound, upper_bound) for x in seminario_wavenumbers]
    )
    if plot:

        _, axes = plt.subplots(1, 2, figsize=(18, 5))
        for idx, data in enumerate([seminario_wavenumbers, gnn_wavenumbers]):
            ax = axes[idx]
            ax.set_xlabel("Wavenumber (cm$^{-1}$)")
            ax.set_ylabel("Frequency")

            line1_y = lower_bound
            line2_y = upper_bound

            ax.axvline(
                line1_y,
                color="red",
                linestyle=":",
                linewidth=2,
                label=f"Line 1 at {line1_y:.2f}",
            )
            ax.axvline(
                line2_y,
                color="red",
                linestyle=":",
                linewidth=2,
                label=f"Line 2 at {line2_y:.2f}",
            )

            # ax.axvline(
            #     np.mean(data),
            #     color="blue",
            #     linestyle=":",
            #     linewidth=2,
            #     label="Mean",
            # )
            #
            ax.legend()
            colormap = hsv_transform_cmap("inferno_r", 0.95)
            kde = gaussian_kde(data)
            x_values = np.linspace(
                min(seminario_wavenumbers) - 100, max(seminario_wavenumbers) + 100, 1000
            )
            distances = [sqrt(adib(x, lower_bound, upper_bound)) for x in x_values]
            print(min(distances), max(distances))
            y_values = kde(x_values)
            norm = plt.Normalize(vmin=0, vmax=2300)
            ax.set_xlim(min(x_values - 100), max(x_values) + 100)
            ax.set_ylim(0, 0.0055)
            ax.tick_params(direction="in")
            for i in range(len(x_values) - 1):
                average_distance = (distances[i] + distances[i + 1]) / 2
                color = colormap(norm(average_distance))
                # Plot the filled area for this small segment (x[i] to x[i+1])
                ax.fill_between(
                    x_values[i : i + 2],  # x-coordinates of the trapezoid (two points)
                    y_values[i : i + 2],  # y-coordinates of the top curve
                    y2=0,  # y-coordinate of the base
                    color=color,  # Individual color for the segment
                )
            distance_ticks = np.linspace(0, 2300, 5)
            sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, orientation="vertical")

            # Set the colorbar ticks to show the distance values
            cbar.set_ticks(distance_ticks)
            # normalized_distance_ticks = norm(distance_ticks) ** 0.8
            # real_distances = (normalized_distance_ticks ** (1/0.8)) * (max(distances) - min(distances)) + min(distances)
            cbar.set_ticklabels([f"{t:.2f}" for t in distance_ticks])

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f"{title}_wavenumber_distribution.svg")
        plt.show()
        plt.close()

        for data, name in zip(
            [gnn_wavenumbers, seminario_wavenumbers], ["GNN", "Seminario_method"]
        ):
            with open(f"{title}_{name}_wavenumber_distribution.xvg", "w") as f:
                f.write(f'@title "{title} {name} method Wavenumber Distribution"\n')
                f.write(f'@xaxis  label "Wavenumber (cm^-1)"\n')
                f.write(f'@yaxis  label "Frequency"\n')
                for i, x in enumerate(data):
                    f.write(f"{i} {x}\n")
                f.flush()
                f.close()

    return gnn_adib, seminario_adib, gnn_wavenumbers, seminario_wavenumbers


def get_gnn_loss_on_fragments(
    pair: tuple,
    nei_list: list,
    fragment_dict: dict,
    fg_key: str,
    title: str,
    plot_bounds: tuple = None,
):
    bonds = []
    for nei in nei_list:
        if nei in fragment_dict[pair]:
            bonds.append(fragment_dict[pair][nei])
            assert (
                nei in fg_map[fg_key]
            ), f"{nei} are not {fg_map[fg_key]}, or you need to add it"

    bonds = [x for y in bonds for x in y]
    if fg_key in ir_intervals:
        return calculate_adib_for_bonds(
            bonds,
            "./graphs/original_hessian_40000_complete_graphs.bin",
            "./checkpoints/original_hessian_40000_best_fold_4_of_10.pt",
            ir_intervals[fg_key][0],
            ir_intervals[fg_key][1],
            title=title,
            plot_bounds=plot_bounds,
        )
    else:
        print(f"{fg_key} not in ir_intervals, skipping...")


def cal_std_for_k_folds(fold_list: list):
    for fold in fold_list:
        pass
