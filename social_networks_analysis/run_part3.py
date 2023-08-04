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


class RunPart3Input:
    def __init__(self, sx_df: pd.DataFrame, t_min: int64, t_max: int64, DT: int64, dt: float64, time_spans: list[int64]):
        self.sx_df = sx_df
        self.t_min = t_min
        self.t_max = t_max
        self.DT = DT
        self.dt = dt
        self.time_spans = time_spans

    def __str__(self):
        return f"sx_df: {self.sx_df}\nt_min: {self.t_min}\nt_max: {self.t_max}\nDT={self.DT}\ndt={self.dt}"


def run_part3(run_part3_input: RunPart3Input) -> None:
    logging.debug("start part3")
    part3_output_dir = Path(constants.OUTPUT_DIR).joinpath("part3")
    if not Path.exists(part3_output_dir):
        Path.mkdir(part3_output_dir, exist_ok=True, parents=True)
    part3_cache_dir = Path(constants.CACHE_DIR).joinpath("part3")
    if not Path.exists(part3_cache_dir):
        Path.mkdir(part3_cache_dir, exist_ok=True, parents=True)
