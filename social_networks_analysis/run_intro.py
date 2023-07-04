import logging
from urllib import request
import logging
import gzip
import shutil
import pandas as pd
import numpy as np
from numpy import int64, float64
from numpy.typing import NDArray
from pathlib import Path
import pickle

import constants


class RunIntroOutput:
    def __init__(self, sx_df: pd.DataFrame, t_min: int64, t_max: int64, DT: int64, dt: float64):
        self.sx_df = sx_df
        self.t_min = t_min
        self.t_max = t_max
        self.DT = DT
        self.dt = dt


def download_and_extract_dataset(dataset_path: Path):
    if not Path.exists(Path(constants.DATASETS_DIR)):
        Path.mkdir(Path(constants.DATASETS_DIR), parents=True, exist_ok=True)

    dataset_gz_path = Path(constants.DATASETS_DIR).joinpath(
        constants.DATASET_GZ_NAME)
    # download dataset if not exists
    if not Path.exists(dataset_gz_path):
        logging.info(f"could not find {dataset_gz_path}, downloading it")
        request.urlretrieve(constants.DATASET_URL, Path(
            constants.DATASETS_DIR).joinpath(constants.DATASET_GZ_NAME))
    else:
        logging.info(f"{dataset_gz_path} is already downloaded")

    # extract dataset if not extracted
    if not Path.exists(dataset_path):
        logging.info(f"could not find {dataset_path}, extracting dataset")
        with gzip.open(dataset_gz_path, "rb") as infile:
            with open(dataset_path, "wb") as outfile:
                shutil.copyfileobj(infile, outfile)
    else:
        logging.info(f"{dataset_path}, is already extracted")


def read_df(dataset_path: Path) -> pd.DataFrame:
    logging.info(f"reading {dataset_path} as DataFrame")
    sx_df = pd.read_table(dataset_path, delim_whitespace=True,
                          index_col=None, header=None, names=constants.DFCOLS)
    logging.debug(sx_df.describe())
    logging.debug(sx_df.head())
    logging.debug("%s", sx_df.dtypes)
    return sx_df


def run_intro() -> RunIntroOutput:
    logging.debug("start intro")
    cache_dir_path = Path(constants.CACHE_DIR)
    if not Path.exists(cache_dir_path):
        Path.mkdir(cache_dir_path, parents=True, exist_ok=True)

    cache_run_intro_output_path = cache_dir_path.joinpath(
        "cache_run_intro_output")
    if constants.USE_CACHE_INTRO and Path.exists(cache_run_intro_output_path):
        with open(cache_run_intro_output_path, "rb") as cache_run_intro_output_file:
            cache_run_intro_output: RunIntroOutput = pickle.load(
                cache_run_intro_output_file)
            cache_run_intro_output_file.close()
            return cache_run_intro_output

    dataset_path = Path(constants.DATASETS_DIR).joinpath(
        constants.DATASET_NAME)
    download_and_extract_dataset(dataset_path)

    sx_df = read_df(dataset_path)
    sx_df = sx_df.sort_values(by=constants.DFCOL_UNIXTS, ascending=True)

    srcs: NDArray[int64] = sx_df[constants.DFCOL_SRC].unique()
    dsts: NDArray[int64] = sx_df[constants.DFCOL_DST].unique()

    all_users = np.sort(np.unique(np.concatenate((srcs, dsts), axis=0)))
    logging.debug("all_users=\n%s", all_users)

    unixts: NDArray[int64] = sx_df[constants.DFCOL_UNIXTS].to_numpy()
    t_min: int64 = unixts.min()
    t_max: int64 = unixts.max()
    DT: int64 = t_max - t_min
    # dt: int64 = np.ceil(DT / constants.N)
    dt: float64 = DT / constants.N
    logging.info(f"t_min={t_min}, t_max={t_max}, DT={DT}, dt={dt}")

    logging.debug("end intro")
    run_intro_output = RunIntroOutput(sx_df, t_min, t_max, DT, dt)
    with open(cache_run_intro_output_path, "wb") as cache_run_intro_output_file:
        pickle.dump(run_intro_output, cache_run_intro_output_file)
        cache_run_intro_output_file.close()

    return run_intro_output