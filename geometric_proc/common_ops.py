#-------------------------------------------------------------------------------
# Name:        common_ops.py
# Purpose:     common functions for geometry processing
# RigNet Copyright 2020 University of Massachusetts
# RigNet is made available under General Public License Version 3 (GPLv3), or under a Commercial License.
# Please see the LICENSE README.txt file in the main directory for more information and instruction on using and licensing RigNet.
#-------------------------------------------------------------------------------

import numpy as np
import time
import open3d as o3d
from scipy.sparse import lil_matrix
from scipy.sparse.csgraph import dijkstra


def get_bones(skel):
    """
    extract bones from skeleton struction
    :param skel: input skeleton
    :return: bones are B*6 array where each row consists starting and ending points of a bone
             bone_name are a list of B elements, where each element consists starting and ending joint name
             leaf_bones indicate if this bone is a virtual "leaf" bone.
             We add virtual "leaf" bones to the leaf joints since they always have skinning weights as well
    """
    bones = []
    bone_name = []
    leaf_bones = []
    this_level = [skel.root]
    while this_level:
        next_level = []
        for p_node in this_level:
            p_pos = np.array(p_node.pos)
            next_level += p_node.children
            for c_node in p_node.children:
                c_pos = np.array(c_node.pos)
                bones.append(np.concatenate((p_pos, c_pos))[np.newaxis, :])
                bone_name.append([p_node.name, c_node.name])
                leaf_bones.append(False)
                if len(c_node.children) == 0:
                    bones.append(np.concatenate((c_pos, c_pos))[np.newaxis, :])
                    bone_name.append([c_node.name, c_node.name+'_leaf'])
                    leaf_bones.append(True)
        this_level = next_level
    bones = np.concatenate(bones, axis=0)
    return bones, bone_name, leaf_bones


def calc_surface_geodesic(mesh):
    # We denselu sample 4000 points to be more accuracy.
    samples = mesh.sample_points_poisson_disk(number_of_points=4000)
    pts = np.asarray(samples.points)
    pts_normal = np.asarray(samples.normals)

    time1 = time.time()
    N = len(pts)
    verts_dist = np.sqrt(np.sum((pts[np.newaxis, ...] - pts[:, np.newaxis, :]) ** 2, axis=2))
    verts_nn = np.argsort(verts_dist, axis=1)
    conn_matrix = lil_matrix((N, N), dtype=np.float32)

    for p in range(N):
        nn_p = verts_nn[p, 1:6]
        norm_nn_p = np.linalg.norm(pts_normal[nn_p], axis=1)
        norm_p = np.linalg.norm(pts_normal[p])
        cos_similar = np.dot(pts_normal[nn_p], pts_normal[p]) / (norm_nn_p * norm_p + 1e-10)
        nn_p = nn_p[cos_similar > -0.5]
        conn_matrix[p, nn_p] = verts_dist[p, nn_p]
    [dist, predecessors] = dijkstra(conn_matrix, directed=False, indices=range(N),
                                    return_predecessors=True, unweighted=False)

    # replace inf distance with euclidean distance + 8
    # 6.12 is the maximal geodesic distance without considering inf, I add 8 to be safer.
    inf_pos = np.argwhere(np.isinf(dist))
    if len(inf_pos) > 0:
        euc_distance = np.sqrt(np.sum((pts[np.newaxis, ...] - pts[:, np.newaxis, :]) ** 2, axis=2))
        dist[inf_pos[:, 0], inf_pos[:, 1]] = 8.0 + euc_distance[inf_pos[:, 0], inf_pos[:, 1]]

    verts = np.array(mesh.vertices)
    vert_pts_distance = np.sqrt(np.sum((verts[np.newaxis, ...] - pts[:, np.newaxis, :]) ** 2, axis=2))
    vert_pts_nn = np.argmin(vert_pts_distance, axis=0)
    surface_geodesic = dist[vert_pts_nn, :][:, vert_pts_nn]
    time2 = time.time()
    print('surface geodesic calculation: {} seconds'.format((time2 - time1)))
    return surface_geodesic
