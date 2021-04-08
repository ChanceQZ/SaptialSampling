# -*- coding: utf-8 -*-

"""
@File: SpatialSimulatedAnnealing.py
@Author: Chance (Qian Zhen)
@Description: 
@Date: 4/6/21
"""

import os
import math
import random
import tqdm
import numpy as np
import shapefile
import matplotlib.pyplot as plt
from utils import nearest_distance_vector, nearest_neighbor_index



class SpatialSimulatedAnnealing:
    def __init__(self, pnts, n_sample, cost_func, max_iter=1000, threshold=None, max_reject=None):
        self._origin_pnts = pnts
        self.n_sample = n_sample
        self.cost_func = cost_func
        self.max_iter = max_iter
        self.threshold = threshold
        self.max_reject = max_reject

        # Initialize
        self._best_sample_idx, self.best_sample = self.initial_sample()
        # different idx set
        self._dif_idx = list(set(range(len(self._origin_pnts))) - set(self._best_sample_idx))
        # cost function value list
        self.scores = [self.cost_func(self.best_sample, self.n_sample)]
        # optimized cost function value list
        self.best_scores = [[0, self.cost_func(self.best_sample, self.n_sample)]]

    def initial_sample(self):
        sample_idx = random.sample(range(len(self._origin_pnts)), self.n_sample)
        sample = np.array(self._origin_pnts)[sample_idx].tolist()
        return sample_idx, sample

    def replace(self, sample_idx):
        """
        Replace the worst sample in sampling set with random sample in difference set.
        :param sample_idx:
        :return:
        """
        idx_s = np.argmin(nearest_distance_vector(np.array(self._origin_pnts)[sample_idx].tolist()))
        rand_d = random.randint(0, len(self._dif_idx) - 1)
        sample_idx[idx_s] = self._dif_idx[rand_d]
        return sample_idx

    def simulated_annealing_sampling(self):
        p = 0 # convertion probability
        reject_cnt = 0
        for iter_idx in tqdm.tqdm(range(self.max_iter)):
            # Termination conditions
            if self.threshold and self.cost_func(self.best_sample, self.n_sample) <= self.threshold:
                break

            if self.max_reject and reject_cnt >= self.max_reject:
                break

            temp_sample_idx = self.replace(self._best_sample_idx.copy())

            temp_sample = np.array(self._origin_pnts)[temp_sample_idx].tolist()
            score = self.cost_func(temp_sample, self.n_sample)
            self.scores.append(score)

            if score < self.best_scores[-1][1]:
                p = 1
            else:
                p = int(1 / (1 + math.exp(score - self.best_scores[-1][1])) > random.uniform(0, 1))

            if p:
                reject_cnt = 0
                self._best_sample_idx = temp_sample_idx
                self.best_sample = np.array(self._origin_pnts)[self._best_sample_idx].tolist()
                self._dif_idx = list(set(range(len(self._origin_pnts))) - set(self._best_sample_idx))
                self.best_scores.append([iter_idx, score])
            else:
                reject_cnt += 1

    def write_file(self, path):
        with open(os.path.join(path, "best_sample.txt"), "w") as f:
            for sample in self.best_sample:
                f.write(str(sample[0]))
                f.write(",")
                f.write(str(sample[1]))
                f.write("\n")

    def visualize(self):
        fig, axes = plt.subplots(1, 3, figsize=(16, 4))
        axes[0].plot(range(len(self.scores)), self.scores)
        axes[0].scatter(np.array(self.best_scores)[:, 0], np.array(self.best_scores)[:, 1], c="r")
        axes[0].set_xlabel("Iteration")
        axes[0].set_ylabel("Cost")
        axes[0].set_title("Cost curve")

        axes[1].scatter(np.array(self.best_sample)[:, 0], np.array(self.best_sample)[:, 1])
        axes[1].set_xlabel("Longitude")
        axes[1].set_ylabel("Latitude")
        axes[1].set_title("Best sampling local view")

        axes[2].scatter(np.array(self._origin_pnts)[:, 0], np.array(self._origin_pnts)[:, 1])
        axes[2].scatter(np.array(self.best_sample)[:, 0], np.array(self.best_sample)[:, 1], c="r")
        axes[2].set_xlabel("Longitude")
        axes[2].set_ylabel("Latitude")
        axes[2].set_title("Best sampling global view")
        plt.show()


if __name__ == "__main__":
    pnts_path = "points/all_pt.shp"
    pnts_shp = shapefile.Reader(pnts_path)
    pnts = [pnt.points[0] for pnt in pnts_shp.shapes()]

    A = 1000  # area
    n_sample = 50
    max_iter = 1000
    SSA = SpatialSimulatedAnnealing(pnts, n_sample, nearest_neighbor_index, max_iter)
    SSA.simulated_annealing_sampling()

    SSA.visualize()
    SSA.write_file("./")