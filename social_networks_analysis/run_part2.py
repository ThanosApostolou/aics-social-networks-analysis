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


def create_networks(t_low: int64, t_mid: int64, t_upper: int64, sx_df: DataFrame, index: int, part1_output_dir: Path, part1_cache_dir: Path):
    logging.debug(
        f"creating Graph{index} between t_low {t_low} and t_upper {t_mid}")
    sx_in_timespan = sx_df[(sx_df[constants.DFCOL_UNIXTS]
                            >= t_low) & (sx_df[constants.DFCOL_UNIXTS] < t_mid)]

    # nodes = utils.nodes_from_df(sx_in_timespan)
    # edges = utils.edges_from_df(sx_in_timespan)
    # graph = nx.Graph()
    # graph.add_nodes_from(nodes)
    # graph.add_edges_from(edges)
    graph_dict = utils.graph_dict_from_df(sx_in_timespan)
    graph = nx.Graph(graph_dict)
    if constants.SHOULD_PLOT_GRAPH:
        logging.debug(
            f"plotting Graph{index} t_low {t_low}, t_upper {t_mid}")
        utils.plot_graph(
            graph, name=f"Graph{index}", plots_dir=part1_output_dir)

    # degree_centrality
    cache_file = part1_cache_dir.joinpath(f"Graph{index}_degree_centrality")
    degree_centrality_list: list | None = utils.load_from_cache(
        cache_file, constants.USE_CACHE_PART1_DATA)
    if degree_centrality_list is None:
        logging.debug(
            f"calculating Graph{index} t_low {t_low}, t_upper {t_mid} degree_centrality")
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
            f"calculating Graph{index} t_low {t_low}, t_upper {t_mid} closeness_centrality")
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
            f"calculating Graph{index} t_low {t_low}, t_upper {t_mid} betweenness_centrality")
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
            f"calculating Graph{index} t_low {t_low}, t_upper {t_mid} eigenvector_centrality")
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
            f"calculating Graph{index} t_low {t_low}, t_upper {t_mid} katz_centrality")
        katz_centrality_dict = nx.katz_centrality(
            graph, max_iter=100000, tol=1.0)
        katz_centrality_list = [
            val for val in katz_centrality_dict.values()]
        utils.dump_to_cache(cache_file, katz_centrality_list)

    plot_histogram(katz_centrality_list, index,
                   "KatzCentrality", part1_output_dir)


def run_part2(run_part2_input: RunPart2Input):
    logging.debug("start part1")
    part1_output_dir = Path(constants.OUTPUT_DIR).joinpath("part1")
    if not Path.exists(part1_output_dir):
        Path.mkdir(part1_output_dir, exist_ok=True, parents=True)
    part1_cache_dir = Path(constants.CACHE_DIR).joinpath("part1")
    if not Path.exists(part1_cache_dir):
        Path.mkdir(part1_cache_dir, exist_ok=True, parents=True)

    if constants.USE_CACHE_PART1:
        return

    time_spans = run_part2_input.time_spans
    mid = len(time_spans) // 2 - \
        1 if len(time_spans) % 2 == 0 else len(time_spans) // 2
    for index, t_low in enumerate(time_spans):
        if index < len(time_spans) - 2 and (index < constants.N_SHOW_HISTOGRAMS or (index >= mid and index < mid + constants.N_SHOW_HISTOGRAMS) or index >= len(time_spans) - 1 - constants.N_SHOW_HISTOGRAMS):
            t_mid = time_spans[index+1]
            t_upper = time_spans[index+2]
            create_networks(t_low, t_mid, t_upper, run_part2_input.sx_df,
                            index, part1_output_dir, part1_cache_dir)

    logging.debug("end part1")
