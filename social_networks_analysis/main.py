from urllib import request
import os
import logging
import gzip
import shutil
import pandas as pd

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
    return sx_df


def main():
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"),
                        format='%(asctime)s %(levelname)s:\n%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    logging.info("start main")

    dataset_path = os.path.join(constants.DATASETS_DIR, constants.DATASET_NAME)
    download_and_extract_dataset(dataset_path)

    sx_df = read_df(dataset_path)
    logging.info(sx_df.describe())
    logging.info(sx_df.head())

    logging.info('end main')


if __name__ == "__main__":
    main()
