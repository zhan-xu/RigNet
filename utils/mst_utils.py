#-------------------------------------------------------------------------------
# Name:        mst_utils.py
# Purpose:     utilize functions for skeleton generation
# RigNet Copyright 2020 University of Massachusetts
# RigNet is made available under General Public License Version 3 (GPLv3), or under a Commercial License.
# Please see the LICENSE README.txt file in the main directory for more information and instruction on using and licensing RigNet.
#-------------------------------------------------------------------------------

import sys
import numpy as np
from utils.tree_utils import TreeNode
from utils.rig_parser import Skel


def inside_check(pts, vox):
    """
    Check where points are inside or outside the mesh based on its voxelization.
    :param pts: points to be checked
    :param vox: voxelized mesh
    :return: internal points, and index of them in the input array.
    """
    vc = (pts - vox.translate) / vox.scale * vox.dims[0]
    vc = np.round(vc).astype(int)
    ind1 = np.logical_and(np.all(vc >= 0, axis=1), np.all(vc < 88, axis=1))
    vc = np.clip(vc, 0, 87)
    ind2 = vox.data[vc[:, 0], vc[:, 1], vc[:, 2]]
    ind = np.logical_and(ind1, ind2)
    pts = pts[ind]
    return pts, np.argwhere(ind).squeeze()


def sample_on_bone(p_pos, ch_pos):
    """
    sample points on a bone
    :param p_pos: parent joint position
    :param ch_pos: child joint position
    :return: a array of samples on this bone.
    """
    ray = ch_pos - p_pos
    bone_length = np.sqrt(np.sum((p_pos - ch_pos) ** 2))
    num_step = np.round(bone_length / 0.01)
    i_step = np.arange(1, num_step + 1)
    unit_step = ray / (num_step + 1e-30)
    unit_step = np.repeat(unit_step[np.newaxis, :], num_step, axis=0)
    res = p_pos + unit_step * i_step[:, np.newaxis]
    return res


def minKey(key, mstSet, nV):
    # Initilaize min value
    min = sys.maxsize
    for v in range(nV):
        if key[v] < min and mstSet[v] == False:
            min = key[v]
            min_index = v
    return min_index


def primMST(graph, init_id):
    """
    Original prim MST algorithm https://www.geeksforgeeks.org/prims-minimum-spanning-tree-mst-greedy-algo-5/
    """
    nV = graph.shape[0]
    # Key values used to pick minimum weight edge in cut
    key = [sys.maxsize] * nV
    parent = [None] * nV  # Array to store constructed MST
    mstSet = [False] * nV
    # Make key init_id so that this vertex is picked as first vertex
    key[init_id] = 0
    parent[init_id] = -1  # First node is always the root of

    for cout in range(nV):
        # Pick the minimum distance vertex from
        # the set of vertices not yet processed.
        # u is always equal to src in first iteration
        u = minKey(key, mstSet, nV)

        # Put the minimum distance vertex in
        # the shortest path tree
        mstSet[u] = True

        # Update dist value of the adjacent vertices
        # of the picked vertex only if the current
        # distance is greater than new distance and
        # the vertex in not in the shotest path tree
        for v in range(nV):
            # graph[u][v] is non zero only for adjacent vertices of m
            # mstSet[v] is false for vertices not yet included in MST
            # Update the key only if graph[u][v] is smaller than key[v]
            if graph[u,v] > 0 and mstSet[v] == False and key[v] > graph[u,v]:
                key[v] = graph[u,v]
                parent[v] = u

    return parent, key


def primMST_symmetry(graph, init_id, joints):
    """
    my modified prim algorithm to generate a tree as symmetric as possible.
    Not guaranteed to be symmetric. All heuristics.
    :param graph: pairwise cost matrix
    :param init_id: init node ID as root
    :param joints: joint positions J*3
    :return:
    """
    joint_mapping = {}
    left_joint_ids = np.argwhere(joints[:, 0] < -2e-2).squeeze(1).tolist()
    middle_joint_ids = np.argwhere(np.abs(joints[:, 0]) <= 2e-2).squeeze(1).tolist()
    right_joint_ids = np.argwhere(joints[:, 0] > 2e-2).squeeze(1).tolist()
    for i in range(len(left_joint_ids)):
        joint_mapping[left_joint_ids[i]] = right_joint_ids[i]
    for i in range(len(right_joint_ids)):
        joint_mapping[right_joint_ids[i]] = left_joint_ids[i]

    if init_id not in middle_joint_ids:
        #find nearest joint in the middle to be root
        if len(middle_joint_ids) > 0:
            nearest_id = np.argmin(np.linalg.norm(joints[middle_joint_ids, :] - joints[init_id, :][np.newaxis, :], axis=1))
            init_id = middle_joint_ids[nearest_id]

    nV = graph.shape[0]
    # Key values used to pick minimum weight edge in cut
    key = [sys.maxsize] * nV
    parent = [None] * nV  # Array to store constructed MST
    mstSet = [False] * nV
    # Make key init_id so that this vertex is picked as first vertex
    key[init_id] = 0
    parent[init_id] = -1  # First node is always the root of

    while not all(mstSet):
        # Pick the minimum distance vertex from
        # the set of vertices not yet processed.
        # u is always equal to src in first iteration
        u = minKey(key, mstSet, nV)
        # left cases
        if u in left_joint_ids and parent[u] in middle_joint_ids:
            u2 = joint_mapping[u]
            if mstSet[u2] is False:
                mstSet[u2] = True
                parent[u2] = parent[u]
                key[u2] = graph[u2, parent[u2]]
        elif u in left_joint_ids and parent[u] in left_joint_ids:
            u2 = joint_mapping[u]
            if mstSet[u2] is False:
                mstSet[u2] = True
                parent[u2] = joint_mapping[parent[u]]
                key[u2] = graph[u2, parent[u2]]
        elif u in middle_joint_ids and parent[u] in left_joint_ids:
            # form loop
            u2 = None
        # right cases
        elif u in right_joint_ids and parent[u] in middle_joint_ids:
            u2 = joint_mapping[u]
            if mstSet[u2] is False:
                mstSet[u2] = True
                parent[u2] = parent[u]
                key[u2] = graph[u2, parent[u2]]
        elif u in right_joint_ids and parent[u] in right_joint_ids:
            u2 = joint_mapping[u]
            if mstSet[u2] is False:
                mstSet[u2] = True
                parent[u2] = joint_mapping[parent[u]]
                key[u2] = graph[u2, parent[u2]]
        elif u in middle_joint_ids and parent[u] in right_joint_ids:
            # form loop
            u2 = None
        # middle case
        else:
            u2 = None

        mstSet[u] = True

        # Update dist value of the adjacent vertices
        # of the picked vertex only if the current
        # distance is greater than new distance and
        # the vertex in not in the shotest path tree
        for v in range(nV):
            # graph[u][v] is non zero only for adjacent vertices of m
            # mstSet[v] is false for vertices not yet included in MST
            # Update the key only if graph[u][v] is smaller than key[v]
            if graph[u,v] > 0 and mstSet[v] == False and key[v] > graph[u,v]:
                key[v] = graph[u,v]
                parent[v] = u
            if u2 is not None and graph[u2,v] > 0 and mstSet[v] == False and key[v] > graph[u2,v]:
                key[v] = graph[u2, v]
                parent[v] = u2

    return parent, key


def loadSkel_recur(p_node, parent_id, joint_name, joint_pos, parent):
    """
    Converst prim algorithm result to our skel/info format recursively
    :param p_node: Root node
    :param parent_id: parent name of current step of recursion.
    :param joint_name: list of joint names
    :param joint_pos: joint positions
    :param parent: parent index returned by prim alg.
    :return: p_node (root) will be expanded to linked with all joints
    """
    for i in range(len(parent)):
        if parent[i] == parent_id:
            if joint_name is not None:
                ch_node = TreeNode(joint_name[i], tuple(joint_pos[i]))
            else:
                ch_node = TreeNode('joint_{}'.format(i), tuple(joint_pos[i]))
            p_node.children.append(ch_node)
            ch_node.parent = p_node
            loadSkel_recur(ch_node, i, joint_name, joint_pos, parent)


def unique_rows(a):
    """
    remove repeat rows from a numpy array
    """
    a = np.ascontiguousarray(a)
    unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))


def increase_cost_for_outside_bone(cost_matrix, joint_pos, vox):
    """
    increase connectivity cost for bones outside the meshs
    """
    for i in range(len(joint_pos)):
        for j in range(i+1, len(joint_pos)):
            bone_samples = sample_on_bone(joint_pos[i], joint_pos[j])
            bone_samples_vox = (bone_samples - vox.translate) / vox.scale * vox.dims[0]
            bone_samples_vox = np.round(bone_samples_vox).astype(int)

            ind1 = np.logical_and(np.all(bone_samples_vox >= 0, axis=1), np.all(bone_samples_vox < vox.dims[0], axis=1))
            bone_samples_vox = np.clip(bone_samples_vox, 0, vox.dims[0]-1)
            ind2 = vox.data[bone_samples_vox[:, 0], bone_samples_vox[:, 1], bone_samples_vox[:, 2]]
            in_flags = np.logical_and(ind1, ind2)
            outside_bone_sample = np.sum(in_flags == False)

            if outside_bone_sample > 1:
                cost_matrix[i, j] = 2 * outside_bone_sample
                cost_matrix[j, i] = 2 * outside_bone_sample
            if np.abs(joint_pos[i, 0]) < 2e-2 and np.abs(joint_pos[j, 0]) < 2e-2:
                cost_matrix[i, j] *= 0.5
                cost_matrix[j, i] *= 0.5
    return cost_matrix


def flip(pred_joints):
    """
    symmetrize the predicted joints by reflecting joints on the left half space to the right
    :param pred_joints: raw predicted joints
    :return: symmetrized predicted joints
    """
    pred_joints_left = pred_joints[np.argwhere(pred_joints[:, 0] < -2e-2).squeeze(), :]
    pred_joints_middle = pred_joints[np.argwhere(np.abs(pred_joints[:, 0]) <= 2e-2).squeeze(), :]

    if pred_joints_left.ndim == 1:
        pred_joints_left = pred_joints_left[np.newaxis, :]
    if pred_joints_middle.ndim == 1:
        pred_joints_middle = pred_joints_middle[np.newaxis, :]

    pred_joints_middle[:, 0] = 0.0
    pred_joints_right = np.copy(pred_joints_left)
    pred_joints_right[:, 0] = -pred_joints_right[:, 0]
    pred_joints_res = np.concatenate((pred_joints_left, pred_joints_middle, pred_joints_right), axis=0)
    side_indicator = np.concatenate((-np.ones(len(pred_joints_left)), np.zeros(len(pred_joints_middle)), np.ones(len(pred_joints_right))), axis=0)
    return pred_joints_res, side_indicator

