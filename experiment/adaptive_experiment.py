#!/usr/bin/python
# -*- coding: utf-8 -*-

import argparse
import os
import sys
import pickle

import tensorflow as tf
import numpy as np
from time import time
from tqdm import tqdm

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
tf.config.experimental.enable_op_determinism()

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, ".."))

from abc import ABCMeta, abstractmethod
from concurrent.futures import ProcessPoolExecutor

from source.new_grid import make_image, create_cov_matrix, AdaptiveGridBasedSIforViT
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
        signal: float,
        noise: str,
    ):
        super().__init__(int(num_results * 1.02), num_results, num_worker)
        self.num_results = num_results
        self.seed = seed

        self.d = d
        self.architecture = architecture
        self.signal = signal

        self.rng = np.random.default_rng(seed=self.seed)

        self.noise = noise
        self.eps_min = 10 ** (-4.0)
        self.eps_max = 0.2

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

        si = AdaptiveGridBasedSIforViT(
            self.d,
            vit_visualizer,
            cov=self.cov,
            eps_min=self.eps_min,
            eps_max=self.eps_max,
        )

        si.construct_hypothesis(data)
        start = time()
        try:
            result = si.inference()
        except Exception as e:
            print(e)
            return None

        if result is None:
            return None

        return (result, time() - start)

    def run_experiment(self):
        if self.signal == 0.0:
            if self.noise == "iid":
                dataset = self.rng.normal(
                    0.0, self.cov, (self.num_iter, self.d, self.d, 1)
                )
            elif self.noise == "corr":
                dataset = self.rng.multivariate_normal(
                    np.zeros(self.d * self.d), self.cov, self.num_iter
                )
            else:
                raise ValueError("noise must be iid or corr")
            dataset = [data.reshape(self.d, self.d, 1) for data in dataset]
        else:
            dataset = []
            for _ in range(self.num_iter):
                dataset.append(make_image(self.rng, self.d, self.signal, self.cov))

        self.results = self.experiment(dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_results", type=int, default=100)
    parser.add_argument("--num_worker", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--d", type=int, default=16)
    parser.add_argument("--architecture", type=str, default="medium")
    parser.add_argument("--signal", type=float, default=0.0)
    parser.add_argument("--noise", type=str, default="iid")
    args = parser.parse_args()

    print(args.noise, args.d, args.signal)

    experiment = ExperimentViT(
        args.num_results,
        args.num_worker,
        args.seed,
        args.d,
        args.architecture,
        args.signal,
        args.noise,
    )

    print("start experiment")
    experiment.run_experiment()
    print("end experiment")

    result_path = f"results_{args.noise}"
    if not os.path.exists(result_path):
        os.mkdir(result_path)

    if args.architecture == "medium":
        architecture = ""
    else:
        architecture = args.architecture

    file_name = (
        f"adaptive_seed{args.seed}d{args.d}signal{args.signal}{architecture}.pkl"
    )
    print(args.noise, file_name)
    file_path = os.path.join(result_path, file_name)

    with open(file_path, "wb") as f:
        pickle.dump(experiment.results, f)
