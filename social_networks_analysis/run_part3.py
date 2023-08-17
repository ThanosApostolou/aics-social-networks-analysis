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
import tensorflow as tf
from sklearn import preprocessing

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
        return f"sx_df: {self.sx_df}\nt_min: {self.t_min}\nt_max: {self.t_max}\nDT={self.DT}\ndt={self.dt}"\


def plot_curves(epochs, hist, metrics_names, part3_output_dir: Path):
    """Plot a curve of one or more classification metrics vs. epoch."""
    # metrics should be one of the names shown in:
    # https://www.tensorflow.org/tutorials/structured_data/imbalanced_data#define_the_model_and_metrics
    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    for m in metrics_names:
        x = hist[m]
        plt.plot(epochs[1:], x[1:], label=m)
    plt.legend()
    plt.savefig(part3_output_dir.joinpath("train_metrics"))
    plt.show(block=False)
    logging.debug("Defined the plot_curve function.")


def create_model(metrics: list, learning_rate: float) -> tf.keras.Sequential:
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(5,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid),
        # tf.keras.layers.Dense(2)
    ])
    #   loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                #   loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                  metrics=metrics)
    return model


def create_train_sets() -> tuple[NDArray, NDArray]:
    logging.debug("start create_train_sets")
    part2_cache_dir = Path(constants.CACHE_DIR).joinpath("part2")
    # sources_column = utils.load_from_cache(
    #     part2_cache_dir.joinpath("sources_column"), True)
    # targets_column = utils.load_from_cache(
    #     part2_cache_dir.joinpath("targets_column"), True)
    sdg_column_tlow: NDArray | None = utils.load_from_cache(
        part2_cache_dir.joinpath("sdg_column_tlow"), True)
    scn_column_tlow: NDArray | None = utils.load_from_cache(
        part2_cache_dir.joinpath("scn_column_tlow"), True)
    sjc_column_tlow: NDArray | None = utils.load_from_cache(
        part2_cache_dir.joinpath("sjc_column_tlow"), True)
    sa_column_tlow: NDArray | None = utils.load_from_cache(
        part2_cache_dir.joinpath("sa_column_tlow"), True)
    spa_column_tlow: NDArray | None = utils.load_from_cache(
        part2_cache_dir.joinpath("spa_column_tlow"), True)
    assert sdg_column_tlow is not None
    assert scn_column_tlow is not None
    assert sjc_column_tlow is not None
    assert sa_column_tlow is not None
    assert spa_column_tlow is not None

    train_features = np.column_stack(
        (sdg_column_tlow, scn_column_tlow, sjc_column_tlow, sa_column_tlow, spa_column_tlow))
    train_features_normalized: NDArray = preprocessing.normalize(train_features, axis=0)
    # training_orig_df = pd.DataFrame(train_features, columns=[
    #     constants.NNDFCOL_SRC, constants.NNDFCOL_DST, constants.NNDFCOL_SDG, constants.NNDFCOL_SCN, constants.NNDFCOL_SJC, constants.NNDFCOL_SA, constants.NNDFCOL_SPA, constants.NNDFCOL_ADJACENCY])
    # logging.debug("write to_csv")
    # training_orig_df.head(100).to_csv(part3_output_dir.joinpath(
    #     "training_orig_df"), index=False, header=True)

    train_labels = utils.load_from_cache(
        part2_cache_dir.joinpath("adjacency_column_tlow"), True)
    assert train_labels is not None
    logging.debug("end create_train_sets")
    return train_features_normalized, train_labels



def train_model(part3_cache_dir: Path, part3_output_dir: Path, mymodel: tf.keras.Sequential, metrics_names: list[str], epochs: int, batch_size: int):
    train_features, train_labels = create_train_sets()
    history = mymodel.fit(train_features, train_labels,
                          epochs=epochs, batch_size=batch_size)

    # The list of epochs is stored separately from the rest of history.
    epochs = history.epoch

    # To track the progression of training, gather a snapshot
    # of the model's mean squared error at each epoch.
    hist = pd.DataFrame(history.history)
    plot_curves(epochs, hist, metrics_names, part3_output_dir)


def create_test_sets() -> tuple[NDArray, NDArray]:
    logging.debug("start create_test_sets")
    part2_cache_dir = Path(constants.CACHE_DIR).joinpath("part2")
    sdg_column_tupper: NDArray | None = utils.load_from_cache(
        part2_cache_dir.joinpath("sdg_column_tupper"), True)
    scn_column_tupper: NDArray | None = utils.load_from_cache(
        part2_cache_dir.joinpath("scn_column_tupper"), True)
    sjc_column_tupper: NDArray | None = utils.load_from_cache(
        part2_cache_dir.joinpath("sjc_column_tupper"), True)
    sa_column_tupper: NDArray | None = utils.load_from_cache(
        part2_cache_dir.joinpath("sa_column_tupper"), True)
    spa_column_tupper: NDArray | None = utils.load_from_cache(
        part2_cache_dir.joinpath("spa_column_tupper"), True)
    assert sdg_column_tupper is not None
    assert scn_column_tupper is not None
    assert sjc_column_tupper is not None
    assert sa_column_tupper is not None
    assert spa_column_tupper is not None

    test_features = np.column_stack(
        (sdg_column_tupper, scn_column_tupper, sjc_column_tupper, sa_column_tupper, spa_column_tupper))
    test_features_normalized: NDArray = preprocessing.normalize(test_features, axis=0)
    test_labels = utils.load_from_cache(
        part2_cache_dir.joinpath("adjacency_column_tupper"), True)
    assert test_labels is not None
    logging.debug("end create_test_sets")
    return test_features_normalized, test_labels

def evaluate_model(mymodel: tf.keras.Sequential, batch_size: int, test_features: NDArray, test_labels: NDArray):
    evaluation = mymodel.evaluate(test_features, test_labels, batch_size=batch_size)
    logging.info('evaluation')
    print(mymodel.metrics_names)
    print(evaluation)
    return evaluation


def predict_model(mymodel: tf.keras.Sequential, batch_size: int, test_features: NDArray, test_labels: NDArray):
    logging.debug("start predict_model")
    # probability_model = tf.keras.Sequential([mymodel, tf.keras.layers.Softmax()])
    predictions = mymodel.predict(test_features, batch_size=batch_size)
    predictions[predictions > 0.5] = 1
    predictions[predictions <= 0.5] = 0
    non_zero_indices = np.where(predictions > 0)[0]
    logging.info('test_features[non_zero_indices]')
    logging.info(test_features[non_zero_indices])
    logging.info('predictions[non_zero_indices]')
    logging.info(predictions[non_zero_indices])
    logging.debug("end predict_model")
    return predictions


def run_part3(run_part3_input: RunPart3Input) -> None:
    logging.debug("start part3")
    part3_output_dir = Path(constants.OUTPUT_DIR).joinpath("part3")
    if not Path.exists(part3_output_dir):
        Path.mkdir(part3_output_dir, exist_ok=True, parents=True)
    part3_cache_dir = Path(constants.CACHE_DIR).joinpath("part3")
    if not Path.exists(part3_cache_dir):
        Path.mkdir(part3_cache_dir, exist_ok=True, parents=True)


    learning_rate=0.001
    epochs = 10
    batch_size = 10000
    metrics = [tf.keras.metrics.BinaryAccuracy(name='binary_accuracy'), tf.keras.metrics.Precision(
        name='precision'), tf.keras.metrics.Recall(name='recall')]
    metrics_names: list[str] = ['loss']
    metrics_names.extend(list(map(lambda metric: metric.name, metrics)))

    mymodel = create_model(metrics, learning_rate)
    train_model(part3_cache_dir, part3_output_dir, mymodel, metrics_names, epochs, batch_size)
    test_features, test_labels = create_test_sets()
    evaluation = evaluate_model(mymodel, batch_size, test_features, test_labels)
    predictions = predict_model(mymodel, batch_size, test_features, test_labels)
    logging.debug("end part3")
