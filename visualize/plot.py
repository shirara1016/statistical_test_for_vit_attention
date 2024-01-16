#!/usr/bin/python
# -*- coding: utf-8 -*-

import pickle
import argparse

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, ".."))

plt.rcParams["font.size"] = 16


def make_key_list(mode):
    if mode == "image":
        return ["d8", "d16", "d32", "d64"]
    elif mode == "architecture":
        return ["d16small", "d16", "d16large", "d16huge"]


def convert_to_plist_at_seed(result_dict, noise, key, seed):
    return np.array([result.p_value for result, _ in result_dict[noise][key][seed]])


def convert_to_reject_rate(result_dict, noise, key, num_seeds=10, alpha=0.05):
    p_list = []
    for seed in range(num_seeds):
        p_list += convert_to_plist_at_seed(result_dict, noise, key, seed).tolist()
    return np.mean(np.array(p_list) < alpha)


def convert_to_time(result_dict, noise, key, num_seeds=2):
    times = []
    for seed in range(num_seeds):
        times += [time for _, time in result_dict[noise][key][seed]]
    return np.mean(times)


def convert_to_bonf_reject_rate(result_dict, noise, key, num_seeds=10):
    stats = []
    for seed in range(num_seeds):
        stats += [result.stat for result, _ in result_dict[noise][key][seed]]
    th_dict = {"d8": 9.473, "d16": 18.82, "d32": 37.65, "d64": 75.33}
    keys = [item for item in th_dict.keys() if item in key]
    if len(keys) > 0:
        th = th_dict[keys[0]]
    else:
        th = 18.82
    return np.mean(np.abs(stats) > th)


def convert_to_naive_reject_rate(result_dict, noise, key, num_seeds=10):
    stats = []
    for seed in range(num_seeds):
        stats += [result.stat for result, _ in result_dict[noise][key][seed]]
    return np.mean(2 * norm.cdf(-np.abs(stats)) < 0.05)


def convert_to_permute_reject_rate(result_dict, noise, key, num_seeds=10):
    p_lsit = []
    for seed in range(num_seeds):
        p_lsit += [p_value for p_value in result_dict[noise][key][seed]]
    return np.mean(np.array(p_lsit) < 0.05)


def null_fpr_plot(
    adaptive_dict,
    permute_dict,
    noise,
    mode,
    is_save=True,
    is_title=False,
):
    plt.figure()
    adaptive_fprs = []
    bonf_fprs, permute_fprs, naive_fprs = [], [], []

    key_list = make_key_list(mode)

    for key in key_list:
        adaptive_fprs.append(convert_to_reject_rate(adaptive_dict, noise, key))
        bonf_fprs.append(convert_to_bonf_reject_rate(adaptive_dict, noise, key))
        naive_fprs.append(convert_to_naive_reject_rate(adaptive_dict, noise, key))
        permute_fprs.append(convert_to_permute_reject_rate(permute_dict, noise, key))

    labels = [1, 2, 3, 4]

    plt.plot(labels, adaptive_fprs, label="adaptive", marker="x")
    plt.plot(labels, bonf_fprs, label="bonferroni", marker="x")
    plt.plot(labels, permute_fprs, label="permutation", marker="x")
    plt.plot(labels, naive_fprs, label="naive", marker="x")

    plt.plot(labels, 0.05 * np.ones(len(labels)), linestyle="--", color="red", lw=0.5)
    plt.xticks(
        labels,
        [64, 256, 1024, 4096]
        if mode == "image"
        else ["small", "base", "large", "huge"],
    )
    plt.ylim(-0.03, 1.03)

    plt.xlabel("Image Size" if mode == "image" else "Architecture")
    plt.ylabel("Type I Error Rate")
    plt.legend(frameon=False, loc="upper left" if mode == "image" else "center left")
    if is_title:
        title = "Correlation" if noise == "corr" else "Independence"
        plt.title(title)
    if is_save:
        plt.savefig(
            f"images/null_fpr_{mode}_{noise}.pdf",
            transparent=True,
            bbox_inches="tight",
            pad_inches=0,
        )
    plt.show()


def null_time_plot(
    adaptive_dict,
    fine_dict,
    combi_dict,
    noise,
    mode,
    is_save=True,
    is_title=False,
):
    plt.figure()
    adaptive_times = []
    fine_times, combi_times = [], []

    key_list = make_key_list(mode)

    for key in key_list:
        adaptive_times.append(convert_to_time(adaptive_dict, noise, key))
        fine_times.append(convert_to_time(fine_dict, noise, key))
        combi_times.append(convert_to_time(combi_dict, noise, key))

    labels = [1, 2, 3, 4]

    plt.plot(labels, adaptive_times, label="adaptive", marker="x")

    plt.plot(labels, fine_times, label="fixed", marker="x")
    plt.plot(labels, combi_times, label="combination", marker="x")

    plt.xticks(
        labels,
        [64, 256, 1024, 4096]
        if mode == "image"
        else ["small", "base", "large", "huge"],
    )

    plt.xlabel("Image Size" if mode == "image" else "Architecture")
    plt.ylabel("Computation Time (s)")
    if mode == "image":
        plt.ylim(-100, 3200)
        loc = "center right"
    else:
        plt.ylim(-200, 8500)
        loc = "upper left"
    plt.legend(frameon=False, loc=loc)
    if is_title:
        title = "Correlation" if noise == "corr" else "Independence"
        plt.title(title)
    if is_save:
        plt.savefig(
            f"images/null_time_{mode}_{noise}.pdf",
            transparent=True,
            bbox_inches="tight",
            pad_inches=0,
        )
    plt.show()


def alter_tpr_plot(
    adaptive_dict,
    noise,
    is_save=True,
    is_title=False,
):
    plt.figure()
    adaptive_tprs = []
    bonf_tprs = []

    labels = [1, 2, 3, 4]
    key_list = ["1.0", "2.0", "3.0", "4.0"]

    for key in key_list:
        adaptive_tprs.append(convert_to_reject_rate(adaptive_dict, noise, key))
        bonf_tprs.append(convert_to_bonf_reject_rate(adaptive_dict, noise, key))

    plt.plot(labels, adaptive_tprs, label="adaptive", marker="x")
    plt.plot(labels, bonf_tprs, label="bonferroni", marker="x")

    # plt.plot(labels, 0.05 * np.ones(len(labels)), linestyle="--", color="red", lw=0.5)
    plt.xticks(labels, key_list)
    plt.ylim(-0.03, 1.03)

    plt.xlabel("signal")
    plt.ylabel("Power")
    plt.legend(frameon=False, loc="upper left")
    if is_title:
        title = "Correlation" if noise == "corr" else "Independence"
        plt.title(title)
    if is_save:
        plt.savefig(
            f"images/alter_tpr_{noise}.pdf",
            transparent=True,
            bbox_inches="tight",
            pad_inches=0,
        )
    plt.show()


def alter_time_plot(
    adaptive_dict,
    fine_dict,
    combi_dict,
    noise,
    is_save=True,
    is_title=False,
):
    plt.figure()
    adaptive_times = []
    fine_times, combi_times = [], []
    labels = [1, 2, 3, 4]
    key_list = ["1.0", "2.0", "3.0", "4.0"]

    for key in key_list:
        adaptive_times.append(convert_to_time(adaptive_dict, noise, key))
        fine_times.append(convert_to_time(fine_dict, noise, key))
        combi_times.append(convert_to_time(combi_dict, noise, key))

    plt.plot(labels, adaptive_times, label="adaptive", marker="x")

    plt.plot(labels, fine_times, label="fine grid", marker="x")
    plt.plot(labels, combi_times, label="combi grid", marker="x")

    plt.xticks(labels, key_list)

    plt.xlabel("signal")
    plt.ylabel("Computation Time (s)")
    plt.legend(frameon=False, loc="upper left")
    if is_title:
        title = "Correlation" if noise == "corr" else "Independence"
        plt.title(title)
    if is_save:
        plt.savefig(
            f"images/alter_time_{noise}.pdf",
            transparent=True,
            bbox_inches="tight",
            pad_inches=0,
        )
    plt.show()


def multi_fpr_plot(result_dict, noise, mode, is_save=True, is_title=False):
    plt.figure()
    fpr_001, fpr_005, fpr_010 = [], [], []
    std_001, std_005, std_010 = [], [], []
    num = 100

    key_list = make_key_list(mode)

    for key in key_list:
        temp_001, temp_005, temp_010 = [], [], []
        for seed in range(num):
            p_list = convert_to_plist_at_seed(result_dict, noise, key, seed)
            temp_001.append(np.mean(p_list < 0.01))
            temp_005.append(np.mean(p_list < 0.05))
            temp_010.append(np.mean(p_list < 0.10))
        fpr_001.append(np.mean(temp_001))
        fpr_005.append(np.mean(temp_005))
        fpr_010.append(np.mean(temp_010))
        std_001.append(np.std(temp_001))
        std_005.append(np.std(temp_005))
        std_010.append(np.std(temp_010))

    std_001, std_005, std_010 = np.array(std_001), np.array(std_005), np.array(std_010)
    labels = [1, 2, 3, 4]
    plt.errorbar(
        labels,
        fpr_005,
        yerr=1.96 * std_005 / np.sqrt(num),
        label="alpha=0.05",
        fmt="o-",
        capsize=2,
        markersize=2,
        lw=0.7,
    )
    plt.errorbar(
        labels,
        fpr_001,
        yerr=1.96 * std_001 / np.sqrt(num),
        label="alpha=0.01",
        fmt="o-",
        capsize=2,
        markersize=2,
        lw=0.7,
    )
    plt.errorbar(
        labels,
        fpr_010,
        yerr=1.96 * std_010 / np.sqrt(num),
        label="alpha=0.10",
        fmt="o-",
        capsize=2,
        markersize=2,
        lw=0.7,
    )
    plt.plot(labels, 0.01 * np.ones(len(labels)), linestyle="--", color="red", lw=0.5)
    plt.plot(labels, 0.05 * np.ones(len(labels)), linestyle="--", color="red", lw=0.5)
    plt.plot(labels, 0.10 * np.ones(len(labels)), linestyle="--", color="red", lw=0.5)
    plt.xticks(
        labels,
        [64, 256, 1024, 4096]
        if mode == "image"
        else ["small", "base", "large", "huge"],
    )
    plt.ylim(0.0, 0.2)
    plt.yticks([0.0, 0.01, 0.05, 0.1, 0.15, 0.2])

    plt.xlabel("Image Size" if mode == "image" else "Architecture")
    plt.ylabel("Type I Error Rate")
    plt.legend(frameon=False, loc="upper left")
    if is_title:
        title = "Correlation" if noise == "corr" else "Independence"
        plt.title(title)
    if is_save:
        plt.savefig(
            f"images/multi_fpr_{mode}_{noise}.pdf",
            transparent=True,
            bbox_inches="tight",
            pad_inches=0,
        )
    plt.show()


if __name__ == "__main__":
    is_save, is_title = True, False

    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, default="null")
    parser.add_argument("--noise", type=str, default="noise")
    parser.add_argument("--mode", type=str, default="image")
    parser.add_argument("--timer", type=int, default=0)
    parser.add_argument("--grid", type=str, default="adaptive")
    parser.add_argument("--alpha", type=float, default=0.05)

    args = parser.parse_args()

    if args.experiment == "null":
        with open("results/fine_null_dict.pkl", "rb") as f:
            fine_null_dict = pickle.load(f)
        with open("results/combi_null_dict.pkl", "rb") as f:
            combi_null_dict = pickle.load(f)
        with open("results/adaptive_null_dict.pkl", "rb") as f:
            adaptive_null_dict = pickle.load(f)
        with open("results/permute_dict.pkl", "rb") as f:
            permute_dict = pickle.load(f)

    if args.experiment == "alter":
        with open("results/adaptive_alter_dict.pkl", "rb") as f:
            adaptive_alter_dict = pickle.load(f)

    if args.experiment == "null":
        if args.timer == 1:
            null_time_plot(
                adaptive_null_dict,
                fine_null_dict,
                combi_null_dict,
                args.noise,
                args.mode,
                is_save,
                is_title,
            )
        else:
            null_fpr_plot(
                adaptive_null_dict,
                permute_dict,
                args.noise,
                args.mode,
                is_save,
                is_title,
            )

    if args.experiment == "alter":
        if args.timer == 1:
            pass
            # alter_time_plot(
            #     adaptive_alter_dict,
            #     fine_alter_dict,
            #     combi_alter_dict,
            #     args.noise,
            #     is_save,
            #     is_title,
            # )
        else:
            alter_tpr_plot(
                adaptive_alter_dict,
                args.noise,
                is_save,
                is_title,
            )

    if args.experiment == "multi":
        with open(f"results/adaptive_null_dict.pkl", "rb") as f:
            null_dict = pickle.load(f)
        multi_fpr_plot(null_dict, args.noise, args.mode, is_save, is_title)
