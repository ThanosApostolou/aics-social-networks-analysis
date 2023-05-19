import logging
from urllib import request
import os
import logging
import gzip
import shutil
import pandas as pd
import numpy as np
from numpy import int64
from numpy.typing import NDArray

import constants


def download_and_extract_dataset(dataset_path: str):
    if not os.path.exists(constants.DATASETS_DIR):
        os.makedirs(constants.DATASETS_DIR, exist_ok=True)

    dataset_gz_path = os.path.join(
        constants.DATASETS_DIR, constants.DATASET_GZ_NAME)
    # download dataset if not exists
    if not os.path.exists(dataset_gz_path):
        logging.info(f"could not find {dataset_gz_path}, downloading it")
        request.urlretrieve(constants.DATASET_URL, os.path.join(
            constants.DATASETS_DIR, constants.DATASET_GZ_NAME))
    else:
        logging.info(f"{dataset_gz_path} is already downloaded")

    # extract dataset if not extracted
    if not os.path.exists(dataset_path):
        logging.info(f"could not find {dataset_path}, extracting dataset")
        with gzip.open(dataset_gz_path, "rb") as infile:
            with open(dataset_path, "wb") as outfile:
                shutil.copyfileobj(infile, outfile)
    else:
        logging.info(f"{dataset_path}, is already extracted")


def read_df(dataset_path: str) -> pd.DataFrame:
    logging.info(f"reading {dataset_path} as DataFrame")
    sx_df = pd.read_table(dataset_path, delim_whitespace=True,
                          index_col=None, header=None, names=constants.DFCOLS)
    
    logging.debug(sx_df.describe())
    logging.debug(sx_df.head())
    logging.debug("%s", sx_df.dtypes)
    return sx_df


def run_intro() -> pd.DataFrame:
    logging.debug("start intro")

    dataset_path = os.path.join(constants.DATASETS_DIR, constants.DATASET_NAME)
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
    dt = DT / constants.N
    logging.info(f"t_min={t_min}, t_max={t_max}, DT={DT}, dt={dt}")

    logging.debug("end intro")
    return sx_df
