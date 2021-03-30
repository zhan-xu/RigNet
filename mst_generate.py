#-------------------------------------------------------------------------------
# Name:        mst_generate.py
# Purpose:     Generate skeleton as a tree based on predicted joints.
# RigNet Copyright 2020 University of Massachusetts
# RigNet is made available under General Public License Version 3 (GPLv3), or under a Commercial License.
# Please see the LICENSE README.txt file in the main directory for more information and instruction on using and licensing RigNet.
#-------------------------------------------------------------------------------

import os
import cv2
import argparse
import numpy as np
import open3d as o3d
from utils import binvox_rw
from utils.tree_utils import TreeNode
from utils.rig_parser import Skel
from utils.vis_utils import show_obj_skel, draw_shifted_pts
from utils.io_utils import readPly
from utils.cluster_utils import meanshift_cluster, nms_meanshift
from utils.mst_utils import primMST_symmetry, loadSkel_recur, increase_cost_for_outside_bone, flip, inside_check, sample_on_bone
from gen_dataset import get_geo_edges, get_tpl_edges
from geometric_proc.common_ops import calc_surface_geodesic

import torch
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops

from models.ROOT_GCN import ROOTNET
from models.PairCls_GCN import PairCls

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def predict_joints(model_id, args):
    """
    predict joints for a specified model
    :param model_id: processed model ID number
    :param args:
    :return: predicted joints, and voxelized mesh
    """
    vox_folder = os.path.join(args.dataset_folder, 'vox/')
    mesh_folder = os.path.join(args.dataset_folder, 'obj_remesh/')
    raw_pred = os.path.join(args.res_folder, '{:d}.ply'.format(model_id))
    vox_file = os.path.join(vox_folder, '{:d}.binvox'.format(model_id))
    mesh_file = os.path.join(mesh_folder, '{:d}.obj'.format(model_id))
    pred_attn = np.load(os.path.join(args.res_folder, '{:d}_attn.npy'.format(model_id)))

    with open(vox_file, 'rb') as fvox:
        vox = binvox_rw.read_as_3d_array(fvox)
    pred_joints = readPly(raw_pred)
    pred_joints, index_inside = inside_check(pred_joints, vox)
    pred_attn = pred_attn[index_inside, :]
    # img = draw_shifted_pts(mesh_file, pred_joints, weights=pred_attn)

    bandwidth = np.load(os.path.join(args.res_folder, '{:d}_bandwidth.npy'.format(model_id)))
    bandwidth = bandwidth[0]
    pred_joints = pred_joints[pred_attn.squeeze() > 1e-3]
    pred_attn = pred_attn[pred_attn.squeeze() > 1e-3]

    # reflect raw points
    pred_joints_reflect = pred_joints * np.array([[-1, 1, 1]])
    pred_joints = np.concatenate((pred_joints, pred_joints_reflect), axis=0)
    pred_attn = np.tile(pred_attn, (2, 1))
    # img = draw_shifted_pts(mesh_file, pred_joints, weights=pred_attn)
    # cv2.imwrite(os.path.join(res_folder, '{:s}_raw.jpg'.format(model_id)), img[:, :, ::-1])

    pred_joints = meanshift_cluster(pred_joints, bandwidth, pred_attn, max_iter=20)
    Y_dist = np.sum(((pred_joints[np.newaxis, ...] - pred_joints[:, np.newaxis, :]) ** 2), axis=2)
    density = np.maximum(bandwidth ** 2 - Y_dist, np.zeros(Y_dist.shape))
    # density = density * pred_attn
    density = np.sum(density, axis=0)
    density_sum = np.sum(density)
    pred_joints_ = pred_joints[density / density_sum > args.threshold_best]
    density_ = density[density / density_sum > args.threshold_best]
    pred_joints_ = nms_meanshift(pred_joints_, density_, bandwidth)
    pred_joints_, _ = flip(pred_joints_)

    reduce_threshold = args.threshold_best
    while len(pred_joints_) < 2 and reduce_threshold > 1e-7:
        # print('reducing')
        reduce_threshold = reduce_threshold / 1.3
        pred_joints_ = pred_joints[density / density_sum >= reduce_threshold]
        density_ = density[density / density_sum > reduce_threshold]
        pred_joints_ = nms_meanshift(pred_joints_, density_, bandwidth)
        pred_joints_, _ = flip(pred_joints_)
    if reduce_threshold <= 1e-7:
        pred_joints_ = nms_meanshift(pred_joints_, density, bandwidth)
        pred_joints_, _ = flip(pred_joints_)

    pred_joints = pred_joints_
    # img = draw_shifted_pts(mesh_file, pred_joints)
    # cv2.imwrite(os.path.join(res_folder, '{:d}_joint.jpg'.format(model_id)), img)
    # np.save(os.path.join(res_folder, '{:d}_joint.npy'.format(model_id)), pred_joints)
    return pred_joints, vox


def getInitId(data, model):
    """
    predict root joint ID via rootnet
    :param data:
    :param model:
    :return:
    """
    with torch.no_grad():
        root_prob, _ = model(data, shuffle=False)
        root_prob = torch.sigmoid(root_prob).data.cpu().numpy()
    root_id = np.argmax(root_prob)
    return root_id


def create_single_data(mesh, vox, surface_geodesic, pred_joints):
    """
    create data used as input to networks, wrapped by Data structure in pytorch-gemetric library
    :param mesh: input mesh loaded by open3d
    :param vox: voxelized mesh
    :param surface_geodesic: geodesic distance matrix of all vertices
    :param pred_joints: predicted joints
    :return: wrapped data structure
    """
    mesh_v = np.asarray(mesh.vertices)
    mesh_vn = np.asarray(mesh.vertex_normals)
    mesh_f = np.asarray(mesh.triangles)

    # vertices
    v = np.concatenate((mesh_v, mesh_vn), axis=1)
    v = torch.from_numpy(v).float()

    # topology edges
    print("     gathering topological edges.")
    tpl_e = get_tpl_edges(mesh_v, mesh_f).T
    tpl_e = torch.from_numpy(tpl_e).long()
    tpl_e, _ = add_self_loops(tpl_e, num_nodes=v.size(0))

    # geodesic edges
    print("     gathering geodesic edges.")
    geo_e = get_geo_edges(surface_geodesic, mesh_v).T
    geo_e = torch.from_numpy(geo_e).long()
    geo_e, _ = add_self_loops(geo_e, num_nodes=v.size(0))

    batch = np.zeros(len(v))
    batch = torch.from_numpy(batch).long()

    pair_all = []
    for joint1_id in range(len(pred_joints)):
        for joint2_id in range(joint1_id + 1, len(pred_joints)):
            dist = np.linalg.norm(pred_joints[joint1_id] - pred_joints[joint2_id])
            bone_samples = sample_on_bone(pred_joints[joint1_id], pred_joints[joint2_id])
            bone_samples_inside, _ = inside_check(bone_samples, vox)
            outside_proportion = len(bone_samples_inside) / (len(bone_samples) + 1e-10)
            pair = np.array([joint1_id, joint2_id, dist, outside_proportion, 1])
            pair_all.append(pair)
    pair_all = np.array(pair_all)
    pair_all = torch.from_numpy(pair_all).float()
    num_pair = len(pair_all)
    num_joint = len(pred_joints)
    if len(pred_joints) < len(mesh_v):
        pred_joints = np.tile(pred_joints, (round(1.0 * len(mesh_v) / len(pred_joints) + 0.5), 1))
        pred_joints = pred_joints[:len(mesh_v), :]
    elif len(pred_joints) > len(mesh_v):
        pred_joints = pred_joints[:len(mesh_v), :]
    pred_joints = torch.from_numpy(pred_joints).float()

    data = Data(x=torch.from_numpy(mesh_vn), pos=torch.from_numpy(mesh_v).float(), batch=batch, y=pred_joints,
                pairs=pair_all, num_pair=[num_pair], tpl_edge_index=tpl_e, geo_edge_index=geo_e, num_joint=[num_joint]).to(device)
    return data


def run_mst_generate(args):
    """
    generate skeleton in batch
    :param args: input folder path and data folder path
    """
    test_list = np.loadtxt(os.path.join(args.dataset_folder, 'test_final.txt'), dtype=np.int)
    root_select_model = ROOTNET()
    root_select_model.to(device)
    root_select_model.eval()
    root_checkpoint = torch.load(args.rootnet)
    root_select_model.load_state_dict(root_checkpoint['state_dict'])
    connectivity_model = PairCls()
    connectivity_model.to(device)
    connectivity_model.eval()
    conn_checkpoint = torch.load(args.bonenet)
    connectivity_model.load_state_dict(conn_checkpoint['state_dict'])

    for model_id in test_list:
        print(model_id)
        pred_joints, vox = predict_joints(model_id, args)
        mesh_filename = os.path.join(args.dataset_folder, 'obj_remesh/{:d}.obj'.format(model_id))
        mesh = o3d.io.read_triangle_mesh(mesh_filename)
        surface_geodesic = calc_surface_geodesic(mesh)
        data = create_single_data(mesh, vox, surface_geodesic, pred_joints)
        root_id = getInitId(data, root_select_model)
        with torch.no_grad():
            cost_matrix, _ = connectivity_model.forward(data)
            connect_prob = torch.sigmoid(cost_matrix)
        pair_idx = data.pairs.long().data.cpu().numpy()
        cost_matrix = np.zeros((data.num_joint[0], data.num_joint[0]))
        cost_matrix[pair_idx[:, 0], pair_idx[:, 1]] = connect_prob.data.cpu().numpy().squeeze()
        cost_matrix = cost_matrix + cost_matrix.transpose()
        cost_matrix = -np.log(cost_matrix+1e-10)
        #cost_matrix = flip_cost_matrix(pred_joints, cost_matrix)
        cost_matrix = increase_cost_for_outside_bone(cost_matrix, pred_joints, vox)

        skel = Skel()
        parent, key, root_id = primMST_symmetry(cost_matrix, root_id, pred_joints)
        for i in range(len(parent)):
            if parent[i] == -1:
                skel.root = TreeNode('root', tuple(pred_joints[i]))
                break
        loadSkel_recur(skel.root, i, None, pred_joints, parent)
        img = show_obj_skel(mesh_filename, skel.root)
        cv2.imwrite(os.path.join(args.res_folder, '{:d}_skel.jpg'.format(model_id)), img[:,:,::-1])
        skel.save(os.path.join(args.res_folder, '{:d}_skel.txt'.format(model_id)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset_folder', default='/media/zhanxu/4T1/ModelResource_RigNetv1_preproccessed/', type=str)
    parser.add_argument('--res_folder', default='results/gcn_meanshift/best_25/', type=str)
    parser.add_argument('--rootnet', default='checkpoints/rootnet/model_best.pth.tar', type=str)
    parser.add_argument('--bonenet', default='checkpoints/bonenet/model_best.pth.tar', type=str)
    parser.add_argument('--threshold_best', default=1e-5, type=float)
    args = parser.parse_args()
    print(args)
    run_mst_generate(args)
