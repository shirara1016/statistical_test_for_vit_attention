#!/usr/bin/python
# -*- coding: utf-8 -*-

import argparse
import pickle

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, ".."))


def summarize_result(type_of_result="adaptive", is_null=True):
    num_seeds = 2
    if type_of_result == "adaptive":
        num_seeds = 10
        if is_null:
            num_seeds = 100

    result_dict = {}
    if is_null:
        for noise in ["iid", "corr"]:
            result_dict[noise] = {}
            for d in [8, 16, 32, 64]:
                result_dict[noise][f"d{d}"] = []
                for seed in range(num_seeds):
                    with open(
                        f"results_{noise}/{type_of_result}_seed{seed}d{d}signal0.0.pkl",
                        "rb",
                    ) as f:
                        result_dict[noise][f"d{d}"].append(pickle.load(f))
            for architecture in ["small", "large", "huge"]:
                result_dict[noise][f"d16{architecture}"] = []
                for seed in range(num_seeds):
                    with open(
                        f"results_{noise}/{type_of_result}_seed{seed}d16signal0.0{architecture}.pkl",
                        "rb",
                    ) as f:
                        result_dict[noise][f"d16{architecture}"].append(pickle.load(f))

    else:
        for noise in ["iid", "corr"]:
            result_dict[noise] = {}
            for signal in [1.0, 2.0, 3.0, 4.0]:
                result_dict[noise][f"{signal}"] = []
                for seed in range(num_seeds):
                    with open(
                        f"results_{noise}/{type_of_result}_seed{seed}d16signal{signal}.pkl",
                        "rb",
                    ) as f:
                        result_dict[noise][f"{signal}"].append(pickle.load(f))

    with open(
        f"results/{type_of_result}_{'null' if is_null else 'alter'}_dict.pkl", "wb"
    ) as f:
        pickle.dump(result_dict, f)


def summarize_result_permute():
    num_seeds = 10

    result_dict = {}
    for noise in ["iid", "corr"]:
        result_dict[noise] = {}
        for d in [8, 16, 32, 64]:
            result_dict[noise][f"d{d}"] = []
            for seed in range(num_seeds):
                with open(f"results_permute/{noise}_seed{seed}d{d}.pkl", "rb") as f:
                    result_dict[noise][f"d{d}"].append(pickle.load(f))
        for architecture in ["small", "large", "huge"]:
            result_dict[noise][f"d16{architecture}"] = []
            for seed in range(num_seeds):
                with open(
                    f"results_permute/{noise}_seed{seed}d16{architecture}.pkl", "rb"
                ) as f:
                    result_dict[noise][f"d16{architecture}"].append(pickle.load(f))

    with open(f"results/permute_dict.pkl", "wb") as f:
        pickle.dump(result_dict, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=1000)
    args = parser.parse_args()

    if 0 <= args.num < 4:
        keys = [
            ["adaptive", True],
            ["adaptive", False],
            ["fine", True],
            ["combi", True],
        ]
        summarize_result(*keys[args.num])

    if args.num == 4:
        summarize_result_permute()
