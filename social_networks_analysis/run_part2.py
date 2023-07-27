from pathlib import Path
import utils
import logging
import pandas as pd
from pandas import DataFrame
import numpy as np
from numpy import int64, float64
from numpy.typing import NDArray
import networkx as nx
from matplotlib import pyplot as plt
import math

from run_intro import RunIntroOutput
import constants

# TODO


class RunPart2Input:
    def __init__(self, sx_df: pd.DataFrame, t_min: int64, t_max: int64, DT: int64, dt: float64, time_spans: list[int64]):
        self.sx_df = sx_df
        self.t_min = t_min
        self.t_max = t_max
        self.DT = DT
        self.dt = dt
        self.time_spans = time_spans

    def __str__(self):
        return f"sx_df: {self.sx_df}\nt_min: {self.t_min}\nt_max: {self.t_max}\nDT={self.DT}\ndt={self.dt}"


def create_networks(t_low: int64, t_mid: int64, t_upper: int64, sx_df: DataFrame, index: int) -> tuple[nx.Graph, nx.Graph]:
    logging.debug(
        f"creating Graph{index} [t_low,t_mid]=[{t_low}-{t_mid}] and Graph{index+1} [t_mid,t_upper]=[{t_mid},{t_upper}]")
    sx_in_timespan = sx_df[(sx_df[constants.DFCOL_UNIXTS]
                            >= t_low) & (sx_df[constants.DFCOL_UNIXTS] < t_upper)]

    sx_in_tlow = sx_in_timespan[(sx_in_timespan[constants.DFCOL_UNIXTS]
                                 >= t_low) & (sx_in_timespan[constants.DFCOL_UNIXTS] < t_mid)]
    sx_in_tupper = sx_in_timespan[(sx_in_timespan[constants.DFCOL_UNIXTS]
                                   >= t_mid) & (sx_in_timespan[constants.DFCOL_UNIXTS] < t_upper)]

    nodes = utils.nodes_from_df(sx_in_timespan)
    # graph tlow
    edges_tlow = utils.edges_from_df(sx_in_tlow)
    graph_tlow = nx.Graph()
    graph_tlow.add_nodes_from(nodes)
    graph_tlow.add_edges_from(edges_tlow)
    # graph_tlow.remove_edges_from(nx.selfloop_edges(graph_tlow))
    # graph tupper
    edges_tupper = utils.edges_from_df(sx_in_tupper)
    graph_upper = nx.Graph()
    graph_upper.add_nodes_from(nodes)
    graph_upper.add_edges_from(edges_tupper)
    # graph_upper.remove_edges_from(nx.selfloop_edges(graph_upper))
    return graph_tlow, graph_upper


def plot_nodes(plots_dir: Path, indexes: list[int], all_nodes_or_edges: list[int], name: str, block: bool = False):
    plt.clf()
    logging.debug(f"plotting {name}")
    plt.suptitle(f"{name}")
    x = [index for index in range(0, len(all_nodes_or_edges), 1)]
    plt.bar(x, all_nodes_or_edges, 1, color="blue", edgecolor="yellow")
    plt.xlabel("graph index")
    plt.ylabel(f"{name}")
    indexes_labels = list(map(lambda index: f"T{index}-T{index+1}", indexes))
    plt.xticks(x, indexes_labels, fontsize=8)
    figure_file: Path = Path(plots_dir).joinpath(f"{name}.png")
    plt.savefig(figure_file)
    plt.show(block=block)


def plot_edges(plots_dir: Path, indexes: list[int], all_edges_tlow: list[int], all_edges_tupper: list[int], name: str, block: bool = False):
    plt.clf()
    logging.debug(f"plotting {name}")
    plt.suptitle(f"{name}")
    all_edges = []
    for i, _ in enumerate(all_edges_tlow):
        all_edges.append(all_edges_tlow[i])
        all_edges.append(all_edges_tupper[i])
    x = [index for index in range(0, len(all_edges), 1)]
    plt.bar(x, all_edges, 1, color="blue", edgecolor="yellow")
    plt.xlabel("graph index")
    plt.ylabel(f"{name}")
    all_indexes = []
    for index in indexes:
        all_indexes.append(index)
        all_indexes.append(index+1)
    indexes_labels = list(map(lambda index: f"T{index}", all_indexes))
    plt.xticks(x, indexes_labels, fontsize=8)
    figure_file: Path = Path(plots_dir).joinpath(f"{name}.png")
    plt.savefig(figure_file)
    plt.show(block=block)


def get_id_index_dicts(nodes: NDArray[int64]) -> tuple[dict[int64, int64], dict[int64, int64]]:
    nodes.sort()
    index_to_id_dict: dict[int64, int64] = {}
    id_to_index_dict: dict[int64, int64] = {}
    for index, node in enumerate(nodes):
        index_to_id_dict[int64(index)] = node
        id_to_index_dict[node] = int64(index)

    return index_to_id_dict, id_to_index_dict


def distance_dict_to_ndarray(nodes_len: int, id_to_index_dict: dict[int64, int64], distance_dict: dict[int64, dict[int64, int64]]) -> NDArray[float64]:
    distance_array: NDArray[float64] = np.zeros(
        shape=(nodes_len, nodes_len), dtype=float64)
    for source, target_dict in distance_dict.items():
        source_index = id_to_index_dict[source]
        for target, value in target_dict.items():
            target_index = id_to_index_dict[target]
            distance_array[source_index, target_index] = float64(value)

    return distance_array


def calculate_sdg_array(graph: nx.Graph, nodes_len: int, id_to_index_dict: dict[int64, int64]) -> NDArray[float64]:
    dgeodesic = nx.shortest_path_length(graph)
    dgeodesic_dict: dict[int64, dict[int64, int64]] = dict(dgeodesic)
    dgeodesic_array = distance_dict_to_ndarray(
        nodes_len, id_to_index_dict, dgeodesic_dict)
    sdg_array: NDArray[float64] = np.negative(dgeodesic_array)
    return sdg_array


def calculate_scn_array(graph: nx.Graph, nodes: NDArray[int64]) -> NDArray[float64]:
    nodes_len: int = len(nodes)
    scn_array: NDArray[float64] = np.zeros(
        shape=(nodes_len, nodes_len), dtype=float64)
    for i, source in enumerate(nodes):
        for j, target in enumerate(nodes):
            common_neighbors_len = len(list(
                nx.common_neighbors(graph, source, target)))
            scn_array[i, j] = common_neighbors_len

    return scn_array


def calculate_sjc_array(graph: nx.Graph, nodes_len: int, id_to_index_dict: dict[int64, int64]) -> NDArray[float64]:
    sjc_array: NDArray[float64] = np.zeros(
        shape=(nodes_len, nodes_len), dtype=float64)
    jaccard_coefficient = nx.jaccard_coefficient(graph)
    for u, v, p in jaccard_coefficient:
        i = id_to_index_dict[u]
        j = id_to_index_dict[v]
        sjc_array[i, j] = p
        sjc_array[j, i] = p

    return sjc_array


def calculate_sa_array(graph: nx.Graph, nodes_len: int, id_to_index_dict: dict[int64, int64]) -> NDArray[float64]:
    sa_array: NDArray[float64] = np.zeros(
        shape=(nodes_len, nodes_len), dtype=float64)
    adamic_adar_index = nx.adamic_adar_index(graph)
    for u, v, p in adamic_adar_index:
        i = id_to_index_dict[u]
        j = id_to_index_dict[v]
        sa_array[i, j] = p
        sa_array[j, i] = p

    return sa_array


def calculate_spa_array(graph: nx.Graph, nodes_len: int, id_to_index_dict: dict[int64, int64]) -> NDArray[float64]:
    spa_array: NDArray[float64] = np.zeros(
        shape=(nodes_len, nodes_len), dtype=float64)
    adamic_adar_index = nx.preferential_attachment(graph)
    for u, v, p in adamic_adar_index:
        i = id_to_index_dict[u]
        j = id_to_index_dict[v]
        spa_array[i, j] = p
        spa_array[j, i] = p

    return spa_array


def networks_calculations(graph_tlow: nx.Graph, graph_tupper: nx.Graph):
    nodes: NDArray[int64] = np.array(graph_tlow.nodes)
    nodes.sort()
    nodes_len: int = len(nodes)
    index_to_id_dict, id_to_index_dict = get_id_index_dicts(nodes)
    # tlow
    adjacency_matrix_tlow = nx.to_numpy_array(
        graph_tlow, nodelist=sorted(graph_tlow.nodes()))
    sdg_array_tlow = calculate_sdg_array(
        graph_tlow, nodes_len, id_to_index_dict)
    scn_array_tlow = calculate_scn_array(graph_tlow, nodes)
    sjc_array_tlow = calculate_sjc_array(
        graph_tlow, nodes_len, id_to_index_dict)
    sa_array_tlow = calculate_sa_array(graph_tlow, nodes_len, id_to_index_dict)
    spa_array_tlow = calculate_spa_array(
        graph_tlow, nodes_len, id_to_index_dict)
    # tupper
    adjacency_matrix_tupper = nx.to_numpy_array(
        graph_tupper, nodelist=sorted(graph_tupper.nodes()))
    sdg_array_tupper = calculate_sdg_array(
        graph_tupper, nodes_len, id_to_index_dict)
    scn_array_tupper = calculate_scn_array(graph_tupper, nodes)
    sjc_array_tupper = calculate_sjc_array(
        graph_tupper, nodes_len, id_to_index_dict)
    sa_array_tupper = calculate_sa_array(
        graph_tupper, nodes_len, id_to_index_dict)
    spa_array_tupper = calculate_spa_array(
        graph_tupper, nodes_len, id_to_index_dict)

    if constants.ENABLE_PART2_VALIDATIONS:
        # assertions for logic consistency
        assert (adjacency_matrix_tlow == adjacency_matrix_tlow.T).all(
        ), "part2->networks_calculations: adjacency_matrix_tlow should be symetrical"
        assert (adjacency_matrix_tupper == adjacency_matrix_tupper.T).all(
        ), "part2->networks_calculations: adjacency_matrix_tupper should be symetrical"

        assert (sdg_array_tlow == sdg_array_tlow.T).all(
        ), "part2->networks_calculations: sdg_array_tlow should be symetrical"
        assert (sdg_array_tupper == sdg_array_tupper.T).all(
        ), "part2->networks_calculations: sdg_array_tupper should be symetrical"

        assert (scn_array_tlow == scn_array_tlow.T).all(
        ), "part2->networks_calculations: scn_array_tlow should be symetrical"
        assert (scn_array_tupper == scn_array_tupper.T).all(
        ), "part2->networks_calculations: scn_array_tupper should be symetrical"

        assert (sjc_array_tlow == sjc_array_tlow.T).all(
        ), "part2->networks_calculations: sjc_array_tlow should be symetrical"
        assert (sjc_array_tupper == sjc_array_tupper.T).all(
        ), "part2->networks_calculations: sjc_array_tupper should be symetrical"

        assert (sa_array_tlow == sa_array_tlow.T).all(
        ), "part2->networks_calculations: sa_array_tlow should be symetrical"
        assert (sa_array_tupper == sa_array_tupper.T).all(
        ), "part2->networks_calculations: sa_array_tupper should be symetrical"

        assert (spa_array_tlow == spa_array_tlow.T).all(
        ), "part2->networks_calculations: spa_array_tlow should be symetrical"
        assert (spa_array_tupper == spa_array_tupper.T).all(
        ), "part2->networks_calculations: spa_array_tupper should be symetrical"


def run_part2(run_part2_input: RunPart2Input):
    logging.debug("start part2")
    part2_output_dir = Path(constants.OUTPUT_DIR).joinpath("part2")
    if not Path.exists(part2_output_dir):
        Path.mkdir(part2_output_dir, exist_ok=True, parents=True)
    part2_cache_dir = Path(constants.CACHE_DIR).joinpath("part2")
    if not Path.exists(part2_cache_dir):
        Path.mkdir(part2_cache_dir, exist_ok=True, parents=True)

    if constants.USE_CACHE_PART2:
        return

    time_spans = run_part2_input.time_spans
    print('last-previous', time_spans[-1] - time_spans[-2])
    print('previous-preprevious', time_spans[-2] - time_spans[-3])
    all_nodes: list[int] = []
    all_edges_tlow: list[int] = []
    all_edges_tupper: list[int] = []
    indexes: list[int] = []
    # calculate which indexes to show. The last element is the upper limit of the last graph so it is excluded.
    show_part2_indexes = utils.parts_indexes_from_list(
        time_spans[:-1], constants.N_SHOW_PART2)
    # last index is previous index in part2
    show_part2_indexes[-1] = show_part2_indexes[-1] - 1
    show_part2_indexes_set = set(show_part2_indexes)
    for index, t_low in enumerate(time_spans):
        if index < len(time_spans) - 2:
            if index in show_part2_indexes_set:
                t_mid = time_spans[index+1]
                t_upper = time_spans[index+2]
                graph_tlow, graph_tupper = create_networks(t_low, t_mid, t_upper, run_part2_input.sx_df,
                                                           index)
                indexes.append(index)
                all_nodes.append(len(graph_tlow.nodes))
                all_edges_tlow.append(len(graph_tlow.edges))
                all_edges_tupper.append(len(graph_tupper.edges))

                # if index == show_part2_indexes[-1]:
                if index == 0:
                    networks_calculations(graph_tlow, graph_tupper)

    print('indexes', indexes)
    print('all_nodes', all_nodes)
    print('all_edges', all_edges_tlow)
    plot_nodes(part2_output_dir, indexes, all_nodes, "Nodes")
    plot_edges(part2_output_dir, indexes,
               all_edges_tlow, all_edges_tupper, "Edges")
    # plot_nodes(part2_output_dir, indexes, all_edges_tupper, "Edges Tj+")

    logging.debug("end part2")
