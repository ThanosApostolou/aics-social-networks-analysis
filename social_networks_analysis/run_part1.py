from pathlib import Path
import utils
import logging
import pandas as pd
from pandas import DataFrame
import numpy as np
from numpy import int64, float64
import networkx as nx

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


def create_network(t_low: int64, t_upper: int64, sx_df: DataFrame, index: int, part1_output_dir: Path):
    logging.debug(
        f"creating Graph between t_low {t_low} and t_upper {t_upper}")
    sx_in_timespan = sx_df[(sx_df[constants.DFCOL_UNIXTS]
                            >= t_low) & (sx_df[constants.DFCOL_UNIXTS] < t_upper)]

    # nodes = utils.nodes_from_df(sx_in_timespan)
    # edges = utils.edges_from_df(sx_in_timespan)
    # graph = nx.Graph()
    # graph.add_nodes_from(nodes)
    # graph.add_edges_from(edges)
    graph_dict = utils.graph_dict_from_df(sx_in_timespan)
    graph = nx.Graph(graph_dict)
    logging.debug(
        f"ploting Graph between t_low {t_low} and t_upper {t_upper}")
    utils.plot_graph(graph, name=f"Graph{index}", plots_dir=part1_output_dir)
    # logging.debug('nodes')
    # logging.debug(nodes)
    # logging.debug('edges')
    # logging.debug(edges)


def run_part1(run_part1_input: RunPart1Input):
    logging.debug("start part1")
    part1_output_dir = Path(constants.OUTPUT_DIR).joinpath("part1")
    if not Path.exists(part1_output_dir):
        Path.mkdir(part1_output_dir, exist_ok=True, parents=True)

    # neighbors = {
    #     1: set([2]),
    #     2: set([1, 2]),
    #     3: set([2]),
    #     4: set()
    # }
    # graph = nx.Graph(neighbors)
    # utils.plot_graph(graph)
    # return
    show_time_spans = run_part1_input.time_spans[-constants.SHOW_N:]

    for index, t_low in enumerate(show_time_spans):
        if index < len(show_time_spans) - 1:
            t_upper = show_time_spans[index+1]
            create_network(t_low, t_upper, run_part1_input.sx_df,
                           index, part1_output_dir)

    logging.debug("end part1")
