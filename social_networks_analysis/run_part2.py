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

    print('indexes', indexes)
    print('all_nodes', all_nodes)
    print('all_edges', all_edges_tlow)
    plot_nodes(part2_output_dir, indexes, all_nodes, "Nodes")
    plot_edges(part2_output_dir, indexes, all_edges_tlow, all_edges_tupper, "Edges")
    # plot_nodes(part2_output_dir, indexes, all_edges_tupper, "Edges Tj+")

    logging.debug("end part2")
