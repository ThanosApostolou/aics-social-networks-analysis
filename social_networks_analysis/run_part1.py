import logging
import pandas as pd
from pandas import DataFrame
import numpy as np
from numpy import int64, float64

from run_intro import RunIntroOutput


class RunPart1Input:
    def __init__(self, sx_df: pd.DataFrame, t_min: int64, t_max: int64, DT: int64, dt: float64):
        self.sx_df = sx_df
        self.t_min = t_min
        self.t_max = t_max
        self.DT = DT
        self.dt = dt

    def __str__(self):
        return f"sx_df: {self.sx_df}\nt_min: {self.t_min}\nt_max: {self.t_max}\nDT={self.DT}\ndt={self.dt}"



def create_network(t_low: int64, t_upper: int64, sx_df: DataFrame):
    logging.debug(f"creating Network between t_low {t_low} and t_upper {t_upper}")


def run_part1(run_part1_input: RunPart1Input):
    logging.debug("start part1")
    logging.debug(run_part1_input)
    time_low_spans: list[int64] = [int64(np.ceil(t)) for t in np.arange(run_part1_input.t_min, run_part1_input.t_max, run_part1_input.dt, dtype=float64)]
    logging.debug(f"time_spans:\n{time_low_spans}")
    for t_low in time_low_spans:
        t_upper = t_low + int64(np.ceil(run_part1_input.dt))
        create_network(t_low, t_upper, run_part1_input.sx_df)



    logging.debug("end part1")
