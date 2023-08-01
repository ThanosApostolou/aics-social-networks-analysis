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


class RunPart2Output:
    def __init__(self, graph_tlow: nx.Graph, graph_tupper: nx.Graph):
        self.graph_tlow = graph_tlow
        self.graph_tupper = graph_tupper


def create_networks(t_low: int64, t_mid: int64, t_upper: int64, sx_df: DataFrame, index: int) -> tuple[nx.Graph, nx.Graph, dict[int64, int64], dict[int64, int64]]:
    logging.debug(
        f"creating Graph{index} [t_low,t_mid]=[{t_low}-{t_mid}] and Graph{index+1} [t_mid,t_upper]=[{t_mid},{t_upper}]")
    sx_in_timespan = sx_df[(sx_df[constants.DFCOL_UNIXTS]
                            >= t_low) & (sx_df[constants.DFCOL_UNIXTS] < t_upper)]

    sx_in_tlow = sx_in_timespan[(sx_in_timespan[constants.DFCOL_UNIXTS]
                                 >= t_low) & (sx_in_timespan[constants.DFCOL_UNIXTS] < t_mid)]
    sx_in_tupper = sx_in_timespan[(sx_in_timespan[constants.DFCOL_UNIXTS]
                                   >= t_mid) & (sx_in_timespan[constants.DFCOL_UNIXTS] < t_upper)]

    nodes, index_to_id_dict, id_to_index_dict = utils.nodes_from_df(
        sx_in_timespan)
    # graph tlow
    edges_tlow = utils.edges_from_df(sx_in_tlow, id_to_index_dict)
    graph_tlow = nx.Graph()
    graph_tlow.add_nodes_from(nodes)
    graph_tlow.add_edges_from(edges_tlow)
    # graph_tlow.remove_edges_from(nx.selfloop_edges(graph_tlow))
    # graph tupper
    edges_tupper = utils.edges_from_df(sx_in_tupper, id_to_index_dict)
    graph_upper = nx.Graph()
    graph_upper.add_nodes_from(nodes)
    graph_upper.add_edges_from(edges_tupper)
    # graph_upper.remove_edges_from(nx.selfloop_edges(graph_upper))
    return graph_tlow, graph_upper, index_to_id_dict, id_to_index_dict


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


def calculate_adjacency_matrix(graph: nx.Graph, nodes_len: int, path: Path):
    logging.debug("start calculate_adjacency_matrix")
    adjacency_column_cached: NDArray[float64] | None = utils.load_from_cache(
        path, constants.USE_CACHE_PART2)
    if adjacency_column_cached is not None:
        return

    adjacency_matrix: NDArray[float64] = nx.to_numpy_array(
        graph, nodelist=sorted(graph.nodes()))
    if constants.ENABLE_PART2_VALIDATIONS:
        # assertions for logic consistency
        assert (adjacency_matrix == adjacency_matrix.T).all(
        ), "part2->networks_calculations: adjacency_matrix should be symetrical"

    adjacency_column = utils.convert_symetrical_array_to_column(
        adjacency_matrix, nodes_len)
    utils.dump_to_cache(path, adjacency_column)


def distance_dict_to_ndarray(nodes_len: int, distance_dict: dict[int64, dict[int64, int64]]) -> NDArray[float64]:
    logging.debug("start distance_dict_to_ndarray")
    distance_array: NDArray[float64] = np.zeros(
        shape=(nodes_len, nodes_len), dtype=float64)
    for source, target_dict in distance_dict.items():
        for target, value in target_dict.items():
            distance_array[source, target] = float64(value)

    return distance_array


def calculate_dgeodesic_array(graph: nx.Graph, nodes_len: int):
    logging.debug("start calculate_dgeodesic_array")

    dgeodesic = nx.shortest_path_length(graph)
    dgeodesic = dict(dgeodesic)
    dgeodesic = distance_dict_to_ndarray(
        nodes_len, dgeodesic)
    return dgeodesic


def calculate_sdg_array(graph: nx.Graph, nodes_len: int, path: Path):
    logging.debug("start calculate_sdg_array")
    sdg_column_cached: NDArray[float64] | None = utils.load_from_cache(
        path, constants.USE_CACHE_PART2)
    if sdg_column_cached is not None:
        return

    sdg_array: NDArray[float64] = np.negative(
        calculate_dgeodesic_array(graph, nodes_len))
    if constants.ENABLE_PART2_VALIDATIONS:
        # assertions for logic consistency
        assert (sdg_array == sdg_array.T).all(
        ), "part2->networks_calculations: sdg_array should be symetrical"
    sdg_column = utils.convert_symetrical_array_to_column(sdg_array, nodes_len)
    utils.dump_to_cache(path, sdg_column)


# def calculate_scn_array(graph: nx.Graph, nodes: NDArray[int64]) -> NDArray[float64]:
#     logging.debug("start calculate_scn_array")
#     nodes_len: int = len(nodes)
#     scn_array: NDArray[float64] = np.zeros(
#         shape=(nodes_len, nodes_len), dtype=float64)
#     for i, source in enumerate(nodes):
#         for j, target in enumerate(nodes):
#             common_neighbors_len = len(list(
#                 nx.common_neighbors(graph, source, target)))
#             scn_array[i, j] = common_neighbors_len

#     return scn_array

# def calculate_sdg_array(graph: nx.Graph, nodes_len: int, path: Path):
#     logging.debug("start calculate_sdg_array")
#     sdg_column_cached: NDArray[float64] | None = utils.load_from_cache(
#         path, constants.USE_CACHE_PART2)
#     if sdg_column_cached is not None:
#         return

#     shortest_paths_array: NDArray[float64] = nx.floyd_warshall_numpy(graph)
#     sdg_array: NDArray[float64] = np.negative(shortest_paths_array)
#     print('sdg_array[0]', sdg_array[0])
#     if constants.ENABLE_PART2_VALIDATIONS:
#         # assertions for logic consistency
#         assert (sdg_array == sdg_array.T).all(
#         ), "part2->networks_calculations: sdg_array should be symetrical"

#     sdg_column = utils.convert_symetrical_array_to_column(
#         sdg_array, nodes_len)
#     utils.dump_to_cache(path, sdg_column)


def calculate_scn_array(graph: nx.Graph, nodes_len: int, path: Path):
    logging.debug("start calculate_scn_array")
    scn_column_cached: NDArray[float64] | None = utils.load_from_cache(
        path, constants.USE_CACHE_PART2)
    if scn_column_cached is not None:
        return

    scn_array: NDArray[float64] = np.zeros(
        shape=(nodes_len, nodes_len), dtype=float64)
    common_neighbor_centrality = nx.common_neighbor_centrality(graph, alpha=1)
    for u, v, p in common_neighbor_centrality:
        scn_array[u, v] = p
        scn_array[v, u] = p

    if constants.ENABLE_PART2_VALIDATIONS:
        # assertions for logic consistency
        assert (scn_array == scn_array.T).all(
        ), "part2->networks_calculations: scn_array should be symetrical"

    scn_column = utils.convert_symetrical_array_to_column(scn_array, nodes_len)
    utils.dump_to_cache(path, scn_column)


def calculate_sjc_array(graph: nx.Graph, nodes_len: int, path: Path):
    logging.debug("start calculate_sjc_array")
    sjc_column_cached: NDArray[float64] | None = utils.load_from_cache(
        path, constants.USE_CACHE_PART2)
    if sjc_column_cached is not None:
        return

    sjc_array: NDArray[float64] = np.zeros(
        shape=(nodes_len, nodes_len), dtype=float64)
    jaccard_coefficient = nx.jaccard_coefficient(graph)
    for u, v, p in jaccard_coefficient:
        sjc_array[u, v] = p
        sjc_array[v, u] = p

    if constants.ENABLE_PART2_VALIDATIONS:
        # assertions for logic consistency
        assert (sjc_array == sjc_array.T).all(
        ), "part2->networks_calculations: sjc_array should be symetrical"

    sjc_column = utils.convert_symetrical_array_to_column(sjc_array, nodes_len)
    utils.dump_to_cache(path, sjc_column)


def calculate_sa_array(graph: nx.Graph, nodes_len: int, path: Path):
    logging.debug("start calculate_sa_array")
    sa_column_cached: NDArray[float64] | None = utils.load_from_cache(
        path, constants.USE_CACHE_PART2)
    if sa_column_cached is not None:
        return

    sa_array: NDArray[float64] = np.zeros(
        shape=(nodes_len, nodes_len), dtype=float64)
    adamic_adar_index = nx.adamic_adar_index(graph)
    for u, v, p in adamic_adar_index:
        sa_array[u, v] = p
        sa_array[v, u] = p

    if constants.ENABLE_PART2_VALIDATIONS:
        # assertions for logic consistency
        assert (sa_array == sa_array.T).all(
        ), "part2->networks_calculations: sa_array should be symetrical"

    sa_column = utils.convert_symetrical_array_to_column(sa_array, nodes_len)
    utils.dump_to_cache(path, sa_column)


def calculate_spa_array(graph: nx.Graph, nodes_len: int, path: Path):
    logging.debug("start calculate_spa_array")
    spa_column_cached: NDArray[float64] | None = utils.load_from_cache(
        path, constants.USE_CACHE_PART2)
    if spa_column_cached is not None:
        return

    spa_array: NDArray[float64] = np.zeros(
        shape=(nodes_len, nodes_len), dtype=float64)
    adamic_adar_index = nx.preferential_attachment(graph)
    for u, v, p in adamic_adar_index:
        spa_array[u, v] = p
        spa_array[v, u] = p

    if constants.ENABLE_PART2_VALIDATIONS:
        # assertions for logic consistency
        assert (spa_array == spa_array.T).all(
        ), "part2->networks_calculations: spa_array should be symetrical"

    spa_column = utils.convert_symetrical_array_to_column(spa_array, nodes_len)
    utils.dump_to_cache(path, spa_column)


def networks_calculations(graph_tlow: nx.Graph, graph_tupper: nx.Graph, id_to_index_dict: dict[int64, int64], part2_cache_dir: Path):
    logging.debug("start part2 networks_calculations")
    nodes: NDArray[int64] = np.array(graph_tlow.nodes)
    nodes.sort()
    nodes_len: int = len(nodes)
    # tlow
    calculate_adjacency_matrix(
        graph_tlow, nodes_len, part2_cache_dir.joinpath("adjacency_column_tlow"))
    calculate_sdg_array(graph_tlow, nodes_len,
                        part2_cache_dir.joinpath("sdg_column_tlow"))
    calculate_scn_array(
        graph_tlow, nodes_len, part2_cache_dir.joinpath("scn_column_tlow"))
    calculate_sjc_array(
        graph_tlow, nodes_len, part2_cache_dir.joinpath("sjc_column_tlow"))
    calculate_sa_array(graph_tlow, nodes_len,
                       part2_cache_dir.joinpath("sa_column_tlow"))
    calculate_spa_array(
        graph_tlow, nodes_len, part2_cache_dir.joinpath("spa_column_tlow"))
    # tupper
    calculate_adjacency_matrix(
        graph_tupper, nodes_len, part2_cache_dir.joinpath("adjacency_column_tupper"))
    calculate_sdg_array(graph_tupper, nodes_len,
                        part2_cache_dir.joinpath("sdg_column_tupper"))
    calculate_scn_array(
        graph_tupper, nodes_len, part2_cache_dir.joinpath("scn_column_tuper"))
    calculate_sjc_array(
        graph_tupper, nodes_len, part2_cache_dir.joinpath("sjc_column_tupper"))
    calculate_sa_array(
        graph_tupper, nodes_len, part2_cache_dir.joinpath("sa_column_tupper"))
    calculate_spa_array(
        graph_tupper, nodes_len, part2_cache_dir.joinpath("spa_column_tupper"))

    logging.debug("end part2 networks_calculations")


def run_part2(run_part2_input: RunPart2Input) -> RunPart2Output:
    logging.debug("start part2")
    part2_output_dir = Path(constants.OUTPUT_DIR).joinpath("part2")
    if not Path.exists(part2_output_dir):
        Path.mkdir(part2_output_dir, exist_ok=True, parents=True)
    part2_cache_dir = Path(constants.CACHE_DIR).joinpath("part2")
    if not Path.exists(part2_cache_dir):
        Path.mkdir(part2_cache_dir, exist_ok=True, parents=True)

    cache_file = part2_cache_dir.joinpath("run_part2_output.pkl")
    run_part2_output: RunPart2Output | None = utils.load_from_cache(
        cache_file, constants.USE_CACHE_PART2)
    if run_part2_output is not None:
        return run_part2_output

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
    nc_output: tuple[NDArray[float64], NDArray[float64], NDArray[float64], NDArray[float64], NDArray[float64], NDArray[float64],
                     NDArray[float64], NDArray[float64], NDArray[float64], NDArray[float64], NDArray[float64], NDArray[float64]]

    graph_tlow: nx.Graph | None = None
    graph_tupper: nx.Graph | None = None
    for index, t_low in enumerate(time_spans):
        if index < len(time_spans) - 2:
            if index in show_part2_indexes_set:
                t_mid = time_spans[index+1]
                t_upper = time_spans[index+2]
                graph_tlow, graph_tupper, _, id_to_index_dict = create_networks(t_low, t_mid, t_upper, run_part2_input.sx_df,
                                                                                index)
                indexes.append(index)
                all_nodes.append(len(graph_tlow.nodes))
                all_edges_tlow.append(len(graph_tlow.edges))
                all_edges_tupper.append(len(graph_tupper.edges))

                if index == show_part2_indexes[-1]:
                    # if index == 0:
                    networks_calculations_output = networks_calculations(
                        graph_tlow, graph_tupper, id_to_index_dict, part2_cache_dir)

    print('indexes', indexes)
    print('all_nodes', all_nodes)
    print('all_edges', all_edges_tlow)
    plot_nodes(part2_output_dir, indexes, all_nodes, "Nodes")
    plot_edges(part2_output_dir, indexes,
               all_edges_tlow, all_edges_tupper, "Edges")
    # plot_nodes(part2_output_dir, indexes, all_edges_tupper, "Edges Tj+")

    logging.debug("end part2")
    assert graph_tlow is not None and graph_tupper is not None
    run_part2_output = RunPart2Output(graph_tlow, graph_tupper)
    utils.dump_to_cache(cache_file, run_part2_output)
    return run_part2_output
