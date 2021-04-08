# -*- coding: utf-8 -*-

"""
@File: utils.py
@Author: Chance (Qian Zhen)
@Description: 
@Date: 4/8/21
"""

import math
import numpy as np
from scipy.spatial import distance


def nearest_distance_vector(xy_list):
    """

    :param xy_list: Coordinates (X and Y) list.
    :return: The minimum distance from each point to anther.
    """
    # distance matrix
    dis_mat = distance.cdist(xy_list, xy_list, 'euclidean')
    # delete diag from matrix
    del_diag_dis_mat = dis_mat[~np.eye(dis_mat.shape[0], dtype=bool)]
    min_dist = del_diag_dis_mat.reshape(dis_mat.shape[0], -1).min(axis=1)
    return min_dist


def nearest_neighbor_index(xy_list, n, A=1):
    """

    :param n: Number of samples.
    :param A: Study area.
    :return: Nearest neighbor index.
    """
    min_dist_sum = nearest_distance_vector(xy_list).sum()
    nni = min_dist_sum / (0.5 * n * math.sqrt(A / n))
    return 1 / nni
