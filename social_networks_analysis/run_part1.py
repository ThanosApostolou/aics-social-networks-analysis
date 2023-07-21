from pathlib import Path
import utils
import logging
import pandas as pd
from pandas import DataFrame
import numpy as np
from numpy import int64, float64
import networkx as nx
from matplotlib import pyplot as plt
import math

from run_intro import RunIntroOutput
import constants


class RunPart1Input:
    def __init__(self, sx_df: pd.DataFrame, t_min: int64, t_max: int64, DT: int64, dt: float64, time_spans: list[int64]):
        self.sx_df = sx_df
        self.t_min = t_min
        self.t_max = t_max
        self.DT = DT
        self.dt = dt
        self.time_spans = time_spans

    def __str__(self):
        return f"sx_df: {self.sx_df}\nt_min: {self.t_min}\nt_max: {self.t_max}\nDT={self.DT}\ndt={self.dt}"


def plot_histogram(data: list, index: int, type: str, plots_dir: Path, block: bool = False):
    plt.figure(1)
    name = f"Graph{index}_{type}"
    logging.debug(
        f"plotting {name}")
    plt.suptitle(f"{name}")
    # n = 10.0
    # min_d = min(data)
    # max_d = max(data)
    # dn = (max_d - min_d) / n
    # print(f"min{min_d}, max{max_d}, dn{dn}")
    # bins = [abin for abin in np.arange(min_d, max_d, dn)]
    bins = [abin for abin in np.arange(0, 1.0, 0.05)]
    bins.append(1.0)
    plt.hist(data, bins=bins, color="blue")
    plt.xlabel("centrality")
    plt.ylabel("nodes")
    xticks = [axtick for axtick in np.arange(0, 1.0, 0.1)]
    xticks.append(1.0)
    plt.xticks(xticks)
    # counts, bins = np.histogram(data, bins=10)
    # plt.hist(counts[:-1], bins=list(bins), weights=counts)
    figure_file: Path = Path(plots_dir).joinpath(f"{name}.png")
    plt.savefig(figure_file)
    plt.show(block=block)


def create_network(t_low: int64, t_upper: int64, sx_df: DataFrame, index: int) -> nx.Graph:
    logging.debug(
        f"creating Graph{index} between t_low {t_low} and t_upper {t_upper}")
    sx_in_timespan = sx_df[(sx_df[constants.DFCOL_UNIXTS]
                            >= t_low) & (sx_df[constants.DFCOL_UNIXTS] < t_upper)]

    # nodes = utils.nodes_from_df(sx_in_timespan)
    # edges = utils.edges_from_df(sx_in_timespan)
    # graph = nx.Graph()
    # graph.add_nodes_from(nodes)
    # graph.add_edges_from(edges)
    graph_dict = utils.graph_dict_from_df(sx_in_timespan)
    graph = nx.Graph(graph_dict)
    return graph


def calculate_centralities(graph: nx.Graph, t_low: int64, t_upper: int64, index: int, part1_output_dir: Path, part1_cache_dir: Path):
    if constants.SHOULD_PLOT_GRAPH:
        logging.debug(
            f"plotting Graph{index} t_low {t_low}, t_upper {t_upper}")
        utils.plot_graph(
            graph, name=f"Graph{index}", plots_dir=part1_output_dir)

    # degree_centrality
    cache_file = part1_cache_dir.joinpath(f"Graph{index}_degree_centrality")
    degree_centrality_list: list | None = utils.load_from_cache(
        cache_file, constants.USE_CACHE_PART1_DATA)
    if degree_centrality_list is None:
        logging.debug(
            f"calculating Graph{index} t_low {t_low}, t_upper {t_upper} degree_centrality")
        degree_centrality_dict = nx.degree_centrality(graph)
        degree_centrality_list = [
            val for val in degree_centrality_dict.values()]
        utils.dump_to_cache(cache_file, degree_centrality_list)

    plot_histogram(degree_centrality_list, index,
                   "DegreeCentrality", part1_output_dir)

    # closeness_centrality
    cache_file = part1_cache_dir.joinpath(f"Graph{index}_closeness_centrality")
    closeness_centrality_list: list | None = utils.load_from_cache(
        cache_file, constants.USE_CACHE_PART1_DATA)
    if closeness_centrality_list is None:
        logging.debug(
            f"calculating Graph{index} t_low {t_low}, t_upper {t_upper} closeness_centrality")
        closeness_centrality_dict = nx.closeness_centrality(graph)
        closeness_centrality_list = [
            val for val in closeness_centrality_dict.values()]
        utils.dump_to_cache(cache_file, closeness_centrality_list)

    plot_histogram(closeness_centrality_list, index,
                   "ClosenessCentrality", part1_output_dir)

    # betweenness_centrality
    cache_file = part1_cache_dir.joinpath(
        f"Graph{index}_betweenness_centrality")
    betweenness_centrality_list: list | None = utils.load_from_cache(
        cache_file, constants.USE_CACHE_PART1_DATA)
    if betweenness_centrality_list is None:
        logging.debug(
            f"calculating Graph{index} t_low {t_low}, t_upper {t_upper} betweenness_centrality")
        betweenness_centrality_dict = nx.betweenness_centrality(graph)
        betweenness_centrality_list = [
            val for val in betweenness_centrality_dict.values()]
        utils.dump_to_cache(cache_file, betweenness_centrality_list)

    plot_histogram(betweenness_centrality_list, index,
                   "BetweenessCentrality", part1_output_dir)

    # eigenvector_centrality
    cache_file = part1_cache_dir.joinpath(
        f"Graph{index}_eigenvector_centrality")
    eigenvector_centrality_list: list | None = utils.load_from_cache(
        cache_file, constants.USE_CACHE_PART1_DATA)
    if eigenvector_centrality_list is None:
        logging.debug(
            f"calculating Graph{index} t_low {t_low}, t_upper {t_upper} eigenvector_centrality")
        eigenvector_centrality_dict = nx.eigenvector_centrality(
            graph, tol=0.00001)
        eigenvector_centrality_list = [
            val for val in eigenvector_centrality_dict.values()]
        utils.dump_to_cache(cache_file, eigenvector_centrality_list)

    plot_histogram(eigenvector_centrality_list, index,
                   "EigenvectorCentrality", part1_output_dir)

    # katz_centrality
    cache_file = part1_cache_dir.joinpath(f"Graph{index}_katz_centrality")
    katz_centrality_list: list | None = utils.load_from_cache(
        cache_file, constants.USE_CACHE_PART1_DATA)
    if katz_centrality_list is None:
        logging.debug(
            f"calculating Graph{index} t_low {t_low}, t_upper {t_upper} katz_centrality")
        katz_centrality_dict = nx.katz_centrality(
            graph, max_iter=100000, tol=1.0)
        katz_centrality_list = [
            val for val in katz_centrality_dict.values()]
        utils.dump_to_cache(cache_file, katz_centrality_list)

    plot_histogram(katz_centrality_list, index,
                   "KatzCentrality", part1_output_dir)


def plot_nodes_or_edges(plots_dir: Path, indexes: list[int], all_nodes_or_edges: list[int], name: str, block: bool = False):
    plt.clf()
    logging.debug(f"plotting {name}")
    plt.suptitle(f"{name}")
    x = [index for index in range(0, len(all_nodes_or_edges), 1)]
    plt.bar(x, all_nodes_or_edges, 1, color="blue")
    plt.xlabel("graph index")
    plt.ylabel(f"{name}")
    plt.xticks(x, indexes)
    figure_file: Path = Path(plots_dir).joinpath(f"{name}.png")
    plt.savefig(figure_file)
    plt.show(block=block)


def run_part1(run_part1_input: RunPart1Input):
    logging.debug("start part1")
    part1_output_dir = Path(constants.OUTPUT_DIR).joinpath("part1")
    if not Path.exists(part1_output_dir):
        Path.mkdir(part1_output_dir, exist_ok=True, parents=True)
    part1_cache_dir = Path(constants.CACHE_DIR).joinpath("part1")
    if not Path.exists(part1_cache_dir):
        Path.mkdir(part1_cache_dir, exist_ok=True, parents=True)

    if constants.USE_CACHE_PART1:
        return

    time_spans = run_part1_input.time_spans
    mid = len(time_spans) // 2 - \
        1 if len(time_spans) % 2 == 0 else len(time_spans) // 2
    all_nodes: list[int] = []
    all_edges: list[int] = []
    indexes: list[int] = []
    # calculate which indexes to show. The last element is the upper limit of the last graph so it is excluded.
    show_part1_indexes = utils.parts_indexes_from_list(
        time_spans[:-1], constants.N_SHOW_PART1)
    show_part1_indexes_set = set(show_part1_indexes)
    for index, t_low in enumerate(time_spans):
        if index < len(time_spans) - 1:
            if index in show_part1_indexes_set:
                t_upper = time_spans[index+1]
                graph = create_network(t_low, t_upper, run_part1_input.sx_df,
                                       index)
                indexes.append(index)
                all_nodes.append(len(graph.nodes))
                all_edges.append(len(graph.edges))
                calculate_centralities(graph, t_low, t_upper,
                                       index, part1_output_dir, part1_cache_dir)

    print('indexes', indexes)
    print('all_nodes', all_nodes)
    print('all_edges', all_edges)
    plot_nodes_or_edges(part1_output_dir, indexes, all_nodes, "Nodes")
    plot_nodes_or_edges(part1_output_dir, indexes, all_edges, "Edges")

    logging.debug("end part1")
