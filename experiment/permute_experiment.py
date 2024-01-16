#!/usr/bin/python
# -*- coding: utf-8 -*-

import argparse
import os
import sys
import pickle

import tensorflow as tf
import numpy as np
from tqdm import tqdm

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
tf.config.experimental.enable_op_determinism()

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, ".."))

from abc import ABCMeta, abstractmethod
from concurrent.futures import ProcessPoolExecutor

from source.new_grid import create_cov_matrix
from source.vit import load_vit_visualizer


class PararellExperiment(metaclass=ABCMeta):
    def __init__(self, num_iter: int, num_results: int, num_worker: int):
        self.num_iter = num_iter
        self.num_results = num_results
        self.num_worker = num_worker

    @abstractmethod
    def iter_experiment(self, args) -> tuple:
        pass

    def experiment(self, dataset: list) -> list:
        with ProcessPoolExecutor(max_workers=self.num_worker) as executor:
            results = list(
                tqdm(executor.map(self.iter_experiment, dataset), total=self.num_iter)
            )
        results = [result for result in results if result is not None]
        return results[: self.num_results]

    @abstractmethod
    def run_experiment(self):
        pass


class ExperimentViT(PararellExperiment):
    def __init__(
        self,
        num_results: int,
        num_worker: int,
        seed: int,
        d: int,
        architecture: str,
        noise: str,
    ):
        super().__init__(int(num_results * 1.02), num_results, num_worker)
        self.num_results = num_results
        self.seed = seed

        self.d = d
        self.architecture = architecture

        self.rng = np.random.default_rng(seed=self.seed)
        self.shuffler = np.random.default_rng(seed=self.seed + 10000)

        self.noise = noise

        if self.noise == "iid":
            self.cov = 1.0
        elif self.noise == "corr":
            self.cov = create_cov_matrix(self.d * self.d)
        else:
            raise ValueError("noise must be iid or corr")

    def iter_experiment(self, args) -> tuple:
        tf.config.experimental.enable_op_determinism()
        data = args
        if self.architecture == "medium":
            vit_visualizer = load_vit_visualizer(
                self.d,
                max(2, self.d // 8),
                model_path=f"model/model{self.d}.h5",
            )
        elif self.architecture == "small":
            vit_visualizer = load_vit_visualizer(
                self.d,
                max(2, self.d // 8),
                model_path=f"model/model{self.d}_small.h5",
                num_blocks=4,
                num_heads=2,
                embedding_dim=32,
            )
        elif self.architecture == "large":
            vit_visualizer = load_vit_visualizer(
                self.d,
                max(2, self.d // 8),
                model_path=f"model/model{self.d}_large.h5",
                num_blocks=12,
                num_heads=8,
                embedding_dim=128,
            )
        elif self.architecture == "huge":
            vit_visualizer = load_vit_visualizer(
                self.d,
                max(2, self.d // 8),
                model_path=f"model/model{self.d}_huge.h5",
                num_blocks=16,
                num_heads=16,
                embedding_dim=256,
            )

        flatten_data = np.reshape(data, [-1])
        obs_data = tf.reshape(flatten_data, [1, self.d, self.d, 1])
        attention_region = vit_visualizer(obs_data) >= 0.6
        mask = tf.cast(tf.reshape(attention_region, [-1]), dtype=tf.float64)
        eta = (mask / tf.reduce_sum(mask)) - ((1.0 - mask) / tf.reduce_sum(1.0 - mask))
        obs_data = tf.reshape(obs_data, [-1])
        obs_stat = tf.tensordot(obs_data, eta, axes=1)

        count = 0
        for _ in range(1000):
            data = self.shuffler.permutation(flatten_data)
            data = tf.reshape(data, [1, self.d, self.d, 1])
            attention_region = vit_visualizer(data) >= 0.6
            mask = tf.cast(tf.reshape(attention_region, [-1]), dtype=tf.float64)
            eta = (mask / tf.reduce_sum(mask)) - (
                (1.0 - mask) / tf.reduce_sum(1.0 - mask)
            )
            data = tf.reshape(data, [-1])
            data_stat = tf.tensordot(data, eta, axes=1)
            if tf.abs(data_stat) > tf.abs(obs_stat):
                count += 1

        return count / 1000

    def run_experiment(self):
        if self.noise == "iid":
            dataset = self.rng.normal(0.0, self.cov, (self.num_iter, self.d, self.d, 1))
        elif self.noise == "corr":
            dataset = self.rng.multivariate_normal(
                np.zeros(self.d * self.d), self.cov, self.num_iter
            )
        else:
            raise ValueError("noise must be iid or corr")
        dataset = [data.reshape(self.d, self.d, 1) for data in dataset]

        self.results = self.experiment(dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_results", type=int, default=100)
    parser.add_argument("--num_worker", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--d", type=int, default=16)
    parser.add_argument("--architecture", type=str, default="medium")
    parser.add_argument("--noise", type=str, default="iid")
    args = parser.parse_args()

    print("permute", args.d, args.noise)

    experiment = ExperimentViT(
        args.num_results,
        args.num_worker,
        args.seed,
        args.d,
        args.architecture,
        args.noise,
    )

    print("start experiment")
    experiment.run_experiment()
    print("end experiment")

    result_path = f"results_permute"
    if not os.path.exists(result_path):
        os.mkdir(result_path)

    if args.architecture == "medium":
        architecture = ""
    else:
        architecture = args.architecture

    file_name = f"{args.noise}_seed{args.seed}d{args.d}{architecture}.pkl"
    print(file_name)
    print()
    file_path = os.path.join(result_path, file_name)

    with open(file_path, "wb") as f:
        pickle.dump(experiment.results, f)
