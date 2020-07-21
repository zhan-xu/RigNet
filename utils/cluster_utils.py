#-------------------------------------------------------------------------------
# Name:        cluster_utils.py
# Purpose:     utilize functions for clustering
# RigNet Copyright 2020 University of Massachusetts
# RigNet is made available under General Public License Version 3 (GPLv3), or under a Commercial License.
# Please see the LICENSE README.txt file in the main directory for more information and instruction on using and licensing RigNet.
#-------------------------------------------------------------------------------

import sys
sys.path.append("./")
import numpy as np


def meanshift_cluster(pts_in, bandwidth, weights=None, max_iter=20):
    """
    Meanshift clustering
    :param pts_in: input points
    :param bandwidth: bandwidth
    :param weights: weights per pts indicting its importance in the clustering
    :return: points after clustering
    """
    diff = 1e10
    num_iter = 1
    while diff > 1e-3 and num_iter < max_iter:
        Y = np.sum(((pts_in[np.newaxis, ...] - pts_in[:, np.newaxis, :]) ** 2), axis=2)
        K = np.maximum(bandwidth**2 - Y, np.zeros(Y.shape))
        if weights is not None:
            K = K * weights
        row_sums = K.sum(axis=0, keepdims=True)
        P = K / (row_sums + 1e-10)
        P = P.transpose()
        pts_in_prim = 0.3 * (np.matmul(P, pts_in) - pts_in) + pts_in
        diff = np.sqrt(np.sum((pts_in_prim - pts_in)**2))
        pts_in = pts_in_prim
        num_iter += 1
    return pts_in


def nms_meanshift(pts_in, density, bandwidth):
    """
    NMS to extract modes after meanshift. Code refers to sci-kit-learn.
    :param pts_in: input points
    :param density: density at each point
    :param bandwidth: bandwidth used in meanshift. Used here as neighbor region for NMS
    :return: extracted clusters.
    """
    Y = np.sum(((pts_in[np.newaxis, ...] - pts_in[:, np.newaxis, :]) ** 2), axis=2)
    sorted_ids = np.argsort(density)[::-1]
    unique = np.ones(len(sorted_ids), dtype=np.bool)
    dist = np.sqrt(Y)
    for i in sorted_ids:
        if unique[i]:
            neighbor_idxs = np.argwhere(dist[:, i] <= bandwidth)
            unique[neighbor_idxs.squeeze()] = 0
            unique[i] = 1  # leave the current point as unique
    pts_in = pts_in[unique]
    return pts_in


