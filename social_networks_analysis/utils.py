from pathlib import Path
import pickle
import networkx as nx
import matplotlib.pyplot as plt
import math
from pandas import DataFrame
import numpy as np
from numpy import int64
from numpy.typing import NDArray
import logging
from typing import Any
from social_networks_analysis import constants


def load_from_cache(cache_file_path: Path, use_cache: bool) -> Any | None:
    if use_cache and Path.exists(cache_file_path):
        with open(cache_file_path, "rb") as cache_file:
            cache_run_intro_output = pickle.load(cache_file)
            cache_file.close()
            return cache_run_intro_output
    else:
        return None


def dump_to_cache(cache_file_path: Path, data: Any):
    with open(cache_file_path, "wb") as cache_file:
        pickle.dump(data, cache_file)
        cache_file.close()


def plot_graph(G: nx.classes.Graph, name: str = "Graph", with_labels: bool = False, block: bool = False, font_size: int = 6, plots_dir: Path = Path(constants.OUTPUT_DIR)):
    """Plots a Graph both as a whole
    and saves it to plots_dir
    """
    plt.figure(1)
    plt.suptitle(f"{name}")
    nx.drawing.draw_networkx(G, with_labels=with_labels,
                             node_size=8, font_size=font_size)
    figure_file: Path = Path(plots_dir).joinpath(f"{name}_plot.png")
    plt.savefig(figure_file)
    plt.show(block=block)


def plot_graph_largest_components(G: nx.classes.Graph, max_largest_components: int = 64, name: str = "Graph", with_labels: bool = False, block: bool = False, font_size: int = 6, plots_dir: Path = Path(constants.OUTPUT_DIR)):
    """Plots a Graph's max_largest_components
    and saves it to plots_dir
    """
    largest_components = sorted(
        nx.connected_components(G), key=len, reverse=True)[:max_largest_components]
    plt.figure(1)
    plt.suptitle(
        f"{name}: {max_largest_components} largest Connected Components")
    root = math.ceil(math.sqrt(len(largest_components)))
    for i, component in enumerate(largest_components):
        plt.subplot(root, root, i+1)
        H = G.subgraph(component)
        nx.drawing.draw_networkx(H, with_labels=with_labels,
                                 node_size=10, font_size=font_size)

    figure_file: Path = Path(plots_dir).joinpath(
        f"{name}_connected_components_plot.png")
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


def nodes_from_df(sx_df: DataFrame) -> tuple[NDArray[int64], dict[int64, int64], dict[int64, int64]]:
    srcs: NDArray[int64] = sx_df[constants.DFCOL_SRC].unique()
    dsts: NDArray[int64] = sx_df[constants.DFCOL_DST].unique()

    nodes_ids = np.sort(np.unique(np.concatenate((srcs, dsts), axis=0)))
    index_to_id_dict, id_to_index_dict = get_id_index_dicts(nodes_ids)
    nodes = np.array(range(0, len(nodes_ids), 1))
    return nodes, index_to_id_dict, id_to_index_dict


def edges_from_df(sx_df: DataFrame, id_to_index_dict: dict[int64, int64]) -> list[tuple[int64, int64]]:
    # test = list(set([(int64(1), int64(2)), (2, 3)]))
    # logging.debug(test)
    # edges = list(set(list(
    #     sx_df[[constants.DFCOL_SRC, constants.DFCOL_DST]].to_records(index=False))))
    edges_set: set[tuple[int64, int64]] = set()
    for _, row in sx_df.iterrows():
        source_id = row[constants.DFCOL_SRC]
        target_id = row[constants.DFCOL_DST]
        source = id_to_index_dict[source_id]
        target = id_to_index_dict[target_id]
        edges_set.add((source, target))
    edges = list(edges_set)
    # logging.debug(edges)
    return edges


def graph_dict_from_df(sx_df: DataFrame) -> dict[int64, set[int64]]:
    graph_dict: dict[int64, set[int64]] = {}
    for _, row in sx_df.iterrows():
        src = row[constants.DFCOL_SRC]
        dst = row[constants.DFCOL_DST]
        if not src in graph_dict:
            graph_dict[src] = set()
        if (src != dst):
            graph_dict[src].add(dst)
    return graph_dict


def parts_indexes_from_list(mylist: list, n_indexes: int) -> list[int]:
    step = math.ceil(len(mylist) / (n_indexes - 1))
    indexes = [i for i in range(0, len(mylist) - 1, step)]
    if indexes[-1] < len(mylist) - 1:
        indexes.append(len(mylist) - 1)

    return indexes


def convert_symetrical_array_to_column(array: NDArray, length: int) -> NDArray:
    logging.debug("start convert_symetrical_array_to_column")
    # column_len = length * (length - 1) // 2 + length
    # column: NDArray = np.zeros(column_len)
    # counter = 0
    # for i in range(0, length, 1):
    #     for j in range(i, length, 1):
    #         column[counter] = array[i, j]
    #         counter += 1
    # # combination n choose 2 plust diagonal
    # assert len(column) == column_len == counter

    column = array[np.triu_indices(length)]
    assert len(column) == length * (length - 1) // 2 + length

    return column
