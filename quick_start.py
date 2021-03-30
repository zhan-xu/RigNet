# ---------------------------------------------------------------------------------------------------------
# Name:        quick_start.py
# Purpose:     An easy-to-use demo. Also serves as an interface of the pipeline.
# RigNet Copyright 2020 University of Massachusetts
# RigNet is made available under General Public License Version 3 (GPLv3), or under a Commercial License.
# Please see the LICENSE README.txt file in the main directory for more information and instruction on using and licensing RigNet.
# ---------------------------------------------------------------------------------------------------------

import os
from sys import platform
import trimesh
import numpy as np
import open3d as o3d
import itertools as it

import torch
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops

from utils import binvox_rw
from utils.rig_parser import Skel, Info
from utils.tree_utils import TreeNode
from utils.io_utils import assemble_skel_skin
from utils.vis_utils import draw_shifted_pts, show_obj_skel, show_mesh_vox
from utils.cluster_utils import meanshift_cluster, nms_meanshift
from utils.mst_utils import increase_cost_for_outside_bone, primMST_symmetry, loadSkel_recur, inside_check, flip

from geometric_proc.common_ops import get_bones, calc_surface_geodesic
from geometric_proc.compute_volumetric_geodesic import pts2line, calc_pts2bone_visible_mat

from gen_dataset import get_tpl_edges, get_geo_edges
from mst_generate import sample_on_bone, getInitId
from run_skinning import post_filter

from models.GCN import JOINTNET_MASKNET_MEANSHIFT as JOINTNET
from models.ROOT_GCN import ROOTNET
from models.PairCls_GCN import PairCls as BONENET
from models.SKINNING import SKINNET


def normalize_obj(mesh_v):
    dims = [max(mesh_v[:, 0]) - min(mesh_v[:, 0]),
            max(mesh_v[:, 1]) - min(mesh_v[:, 1]),
            max(mesh_v[:, 2]) - min(mesh_v[:, 2])]
    scale = 1.0 / max(dims)
    pivot = np.array([(min(mesh_v[:, 0]) + max(mesh_v[:, 0])) / 2, min(mesh_v[:, 1]),
                      (min(mesh_v[:, 2]) + max(mesh_v[:, 2])) / 2])
    mesh_v[:, 0] -= pivot[0]
    mesh_v[:, 1] -= pivot[1]
    mesh_v[:, 2] -= pivot[2]
    mesh_v *= scale
    return mesh_v, pivot, scale


def create_single_data(mesh_filaname):
    """
    create input data for the network. The data is wrapped by Data structure in pytorch-geometric library
    :param mesh_filaname: name of the input mesh
    :return: wrapped data, voxelized mesh, and geodesic distance matrix of all vertices
    """
    mesh = o3d.io.read_triangle_mesh(mesh_filaname)
    mesh.compute_vertex_normals()
    mesh_v = np.asarray(mesh.vertices)
    mesh_vn = np.asarray(mesh.vertex_normals)
    mesh_f = np.asarray(mesh.triangles)

    mesh_v, translation_normalize, scale_normalize = normalize_obj(mesh_v)
    mesh_normalized = o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(mesh_v), triangles=o3d.utility.Vector3iVector(mesh_f))
    o3d.io.write_triangle_mesh(mesh_filename.replace("_remesh.obj", "_normalized.obj"), mesh_normalized)

    # vertices
    v = np.concatenate((mesh_v, mesh_vn), axis=1)
    v = torch.from_numpy(v).float()

    # topology edges
    print("     gathering topological edges.")
    tpl_e = get_tpl_edges(mesh_v, mesh_f).T
    tpl_e = torch.from_numpy(tpl_e).long()
    tpl_e, _ = add_self_loops(tpl_e, num_nodes=v.size(0))

    # surface geodesic distance matrix
    print("     calculating surface geodesic matrix.")
    surface_geodesic = calc_surface_geodesic(mesh)

    # geodesic edges
    print("     gathering geodesic edges.")
    geo_e = get_geo_edges(surface_geodesic, mesh_v).T
    geo_e = torch.from_numpy(geo_e).long()
    geo_e, _ = add_self_loops(geo_e, num_nodes=v.size(0))

    # batch
    batch = torch.zeros(len(v), dtype=torch.long)

    # voxel
    if not os.path.exists(mesh_filaname.replace('_remesh.obj', '_normalized.binvox')):
        if platform == "linux" or platform == "linux2":
            os.system("./binvox -d 88 -pb " + mesh_filaname.replace("_remesh.obj", "_normalized.obj"))
        elif platform == "win32":
            os.system("binvox.exe -d 88 " + mesh_filaname.replace("_remesh.obj", "_normalized.obj"))
        else:
            raise Exception('Sorry, we currently only support windows and linux.')

    with open(mesh_filaname.replace('_remesh.obj', '_normalized.binvox'), 'rb') as fvox:
        vox = binvox_rw.read_as_3d_array(fvox)

    data = Data(x=v[:, 3:6], pos=v[:, 0:3], tpl_edge_index=tpl_e, geo_edge_index=geo_e, batch=batch)
    return data, vox, surface_geodesic, translation_normalize, scale_normalize


def predict_joints(input_data, vox, joint_pred_net, threshold, bandwidth=None, mesh_filename=None):
    """
    Predict joints
    :param input_data: wrapped input data
    :param vox: voxelized mesh
    :param joint_pred_net: network for predicting joints
    :param threshold: density threshold to filter out shifted points
    :param bandwidth: bandwidth for meanshift clustering
    :param mesh_filename: mesh filename for visualization
    :return: wrapped data with predicted joints, pair-wise bone representation added.
    """
    data_displacement, _, attn_pred, bandwidth_pred = joint_pred_net(input_data)
    y_pred = data_displacement + input_data.pos
    y_pred_np = y_pred.data.cpu().numpy()
    attn_pred_np = attn_pred.data.cpu().numpy()
    y_pred_np, index_inside = inside_check(y_pred_np, vox)
    attn_pred_np = attn_pred_np[index_inside, :]
    y_pred_np = y_pred_np[attn_pred_np.squeeze() > 1e-3]
    attn_pred_np = attn_pred_np[attn_pred_np.squeeze() > 1e-3]

    # symmetrize points by reflecting
    y_pred_np_reflect = y_pred_np * np.array([[-1, 1, 1]])
    y_pred_np = np.concatenate((y_pred_np, y_pred_np_reflect), axis=0)
    attn_pred_np = np.tile(attn_pred_np, (2, 1))

    #img = draw_shifted_pts(mesh_filename, y_pred_np, weights=attn_pred_np)
    if bandwidth is None:
        bandwidth = bandwidth_pred.item()
    y_pred_np = meanshift_cluster(y_pred_np, bandwidth, attn_pred_np, max_iter=40)
    #img = draw_shifted_pts(mesh_filename, y_pred_np, weights=attn_pred_np)

    Y_dist = np.sum(((y_pred_np[np.newaxis, ...] - y_pred_np[:, np.newaxis, :]) ** 2), axis=2)
    density = np.maximum(bandwidth ** 2 - Y_dist, np.zeros(Y_dist.shape))
    density = np.sum(density, axis=0)
    density_sum = np.sum(density)
    y_pred_np = y_pred_np[density / density_sum > threshold]
    attn_pred_np = attn_pred_np[density / density_sum > threshold][:, 0]
    density = density[density / density_sum > threshold]

    #img = draw_shifted_pts(mesh_filename, y_pred_np, weights=attn_pred_np)
    pred_joints = nms_meanshift(y_pred_np, density, bandwidth)
    pred_joints, _ = flip(pred_joints)
    #img = draw_shifted_pts(mesh_filename, pred_joints)

    # prepare and add new data members
    pairs = list(it.combinations(range(pred_joints.shape[0]), 2))
    pair_attr = []
    for pr in pairs:
        dist = np.linalg.norm(pred_joints[pr[0]] - pred_joints[pr[1]])
        bone_samples = sample_on_bone(pred_joints[pr[0]], pred_joints[pr[1]])
        bone_samples_inside, _ = inside_check(bone_samples, vox)
        outside_proportion = len(bone_samples_inside) / (len(bone_samples) + 1e-10)
        attr = np.array([dist, outside_proportion, 1])
        pair_attr.append(attr)
    pairs = np.array(pairs)
    pair_attr = np.array(pair_attr)
    pairs = torch.from_numpy(pairs).float()
    pair_attr = torch.from_numpy(pair_attr).float()
    pred_joints = torch.from_numpy(pred_joints).float()
    joints_batch = torch.zeros(len(pred_joints), dtype=torch.long)
    pairs_batch = torch.zeros(len(pairs), dtype=torch.long)

    input_data.joints = pred_joints
    input_data.pairs = pairs
    input_data.pair_attr = pair_attr
    input_data.joints_batch = joints_batch
    input_data.pairs_batch = pairs_batch
    return input_data


def predict_skeleton(input_data, vox, root_pred_net, bone_pred_net, mesh_filename):
    """
    Predict skeleton structure based on joints
    :param input_data: wrapped data
    :param vox: voxelized mesh
    :param root_pred_net: network to predict root
    :param bone_pred_net: network to predict pairwise connectivity cost
    :param mesh_filename: meshfilename for debugging
    :return: predicted skeleton structure
    """
    root_id = getInitId(input_data, root_pred_net)
    pred_joints = input_data.joints.data.cpu().numpy()

    with torch.no_grad():
        connect_prob, _ = bone_pred_net(input_data, permute_joints=False)
        connect_prob = torch.sigmoid(connect_prob)
    pair_idx = input_data.pairs.long().data.cpu().numpy()
    prob_matrix = np.zeros((len(input_data.joints), len(input_data.joints)))
    prob_matrix[pair_idx[:, 0], pair_idx[:, 1]] = connect_prob.data.cpu().numpy().squeeze()
    prob_matrix = prob_matrix + prob_matrix.transpose()
    cost_matrix = -np.log(prob_matrix + 1e-10)
    cost_matrix = increase_cost_for_outside_bone(cost_matrix, pred_joints, vox)

    pred_skel = Info()
    parent, key, root_id = primMST_symmetry(cost_matrix, root_id, pred_joints)
    for i in range(len(parent)):
        if parent[i] == -1:
            pred_skel.root = TreeNode('root', tuple(pred_joints[i]))
            break
    loadSkel_recur(pred_skel.root, i, None, pred_joints, parent)
    pred_skel.joint_pos = pred_skel.get_joint_dict()
    #show_mesh_vox(mesh_filename, vox, pred_skel.root)
    try:
        img = show_obj_skel(mesh_filename, pred_skel.root)
    except:
        print("Visualization is not supported on headless servers. Please consider other headless rendering methods.")
    return pred_skel


def calc_geodesic_matrix(bones, mesh_v, surface_geodesic, mesh_filename, subsampling=False):
    """
    calculate volumetric geodesic distance from vertices to each bones
    :param bones: B*6 numpy array where each row stores the starting and ending joint position of a bone
    :param mesh_v: V*3 mesh vertices
    :param surface_geodesic: geodesic distance matrix of all vertices
    :param mesh_filename: mesh filename
    :return: an approaximate volumetric geodesic distance matrix V*B, were (v,b) is the distance from vertex v to bone b
    """

    if subsampling:
        mesh0 = o3d.io.read_triangle_mesh(mesh_filename)
        mesh0 = mesh0.simplify_quadric_decimation(3000)
        o3d.io.write_triangle_mesh(mesh_filename.replace(".obj", "_simplified.obj"), mesh0)
        mesh_trimesh = trimesh.load(mesh_filename.replace(".obj", "_simplified.obj"))
        subsamples_ids = np.random.choice(len(mesh_v), np.min((len(mesh_v), 1500)), replace=False)
        subsamples = mesh_v[subsamples_ids, :]
        surface_geodesic = surface_geodesic[subsamples_ids, :][:, subsamples_ids]
    else:
        mesh_trimesh = trimesh.load(mesh_filename)
        subsamples = mesh_v
    origins, ends, pts_bone_dist = pts2line(subsamples, bones)
    pts_bone_visibility = calc_pts2bone_visible_mat(mesh_trimesh, origins, ends)
    pts_bone_visibility = pts_bone_visibility.reshape(len(bones), len(subsamples)).transpose()
    pts_bone_dist = pts_bone_dist.reshape(len(bones), len(subsamples)).transpose()
    # remove visible points which are too far
    for b in range(pts_bone_visibility.shape[1]):
        visible_pts = np.argwhere(pts_bone_visibility[:, b] == 1).squeeze(1)
        if len(visible_pts) == 0:
            continue
        threshold_b = np.percentile(pts_bone_dist[visible_pts, b], 15)
        pts_bone_visibility[pts_bone_dist[:, b] > 1.3 * threshold_b, b] = False

    visible_matrix = np.zeros(pts_bone_visibility.shape)
    visible_matrix[np.where(pts_bone_visibility == 1)] = pts_bone_dist[np.where(pts_bone_visibility == 1)]
    for c in range(visible_matrix.shape[1]):
        unvisible_pts = np.argwhere(pts_bone_visibility[:, c] == 0).squeeze(1)
        visible_pts = np.argwhere(pts_bone_visibility[:, c] == 1).squeeze(1)
        if len(visible_pts) == 0:
            visible_matrix[:, c] = pts_bone_dist[:, c]
            continue
        for r in unvisible_pts:
            dist1 = np.min(surface_geodesic[r, visible_pts])
            nn_visible = visible_pts[np.argmin(surface_geodesic[r, visible_pts])]
            if np.isinf(dist1):
                visible_matrix[r, c] = 8.0 + pts_bone_dist[r, c]
            else:
                visible_matrix[r, c] = dist1 + visible_matrix[nn_visible, c]
    if subsampling:
        nn_dist = np.sum((mesh_v[:, np.newaxis, :] - subsamples[np.newaxis, ...])**2, axis=2)
        nn_ind = np.argmin(nn_dist, axis=1)
        visible_matrix = visible_matrix[nn_ind, :]
        os.remove(mesh_filename.replace(".obj", "_simplified.obj"))
    return visible_matrix


def predict_skinning(input_data, pred_skel, skin_pred_net, surface_geodesic, mesh_filename, subsampling=False):
    """
    predict skinning
    :param input_data: wrapped input data
    :param pred_skel: predicted skeleton
    :param skin_pred_net: network to predict skinning weights
    :param surface_geodesic: geodesic distance matrix of all vertices
    :param mesh_filename: mesh filename
    :return: predicted rig with skinning weights information
    """
    global device, output_folder
    num_nearest_bone = 5
    bones, bone_names, bone_isleaf = get_bones(pred_skel)
    mesh_v = input_data.pos.data.cpu().numpy()
    print("     calculating volumetric geodesic distance from vertices to bone. This step takes some time...")
    geo_dist = calc_geodesic_matrix(bones, mesh_v, surface_geodesic, mesh_filename, subsampling=subsampling)
    input_samples = []  # joint_pos (x, y, z), (bone_id, 1/D)*5
    loss_mask = []
    skin_nn = []
    for v_id in range(len(mesh_v)):
        geo_dist_v = geo_dist[v_id]
        bone_id_near_to_far = np.argsort(geo_dist_v)
        this_sample = []
        this_nn = []
        this_mask = []
        for i in range(num_nearest_bone):
            if i >= len(bones):
                this_sample += bones[bone_id_near_to_far[0]].tolist()
                this_sample.append(1.0 / (geo_dist_v[bone_id_near_to_far[0]] + 1e-10))
                this_sample.append(bone_isleaf[bone_id_near_to_far[0]])
                this_nn.append(0)
                this_mask.append(0)
            else:
                skel_bone_id = bone_id_near_to_far[i]
                this_sample += bones[skel_bone_id].tolist()
                this_sample.append(1.0 / (geo_dist_v[skel_bone_id] + 1e-10))
                this_sample.append(bone_isleaf[skel_bone_id])
                this_nn.append(skel_bone_id)
                this_mask.append(1)
        input_samples.append(np.array(this_sample)[np.newaxis, :])
        skin_nn.append(np.array(this_nn)[np.newaxis, :])
        loss_mask.append(np.array(this_mask)[np.newaxis, :])

    skin_input = np.concatenate(input_samples, axis=0)
    loss_mask = np.concatenate(loss_mask, axis=0)
    skin_nn = np.concatenate(skin_nn, axis=0)
    skin_input = torch.from_numpy(skin_input).float()
    input_data.skin_input = skin_input
    input_data.to(device)

    skin_pred = skin_pred_net(data)
    skin_pred = torch.softmax(skin_pred, dim=1)
    skin_pred = skin_pred.data.cpu().numpy()
    skin_pred = skin_pred * loss_mask

    skin_nn = skin_nn[:, 0:num_nearest_bone]
    skin_pred_full = np.zeros((len(skin_pred), len(bone_names)))
    for v in range(len(skin_pred)):
        for nn_id in range(len(skin_nn[v, :])):
            skin_pred_full[v, skin_nn[v, nn_id]] = skin_pred[v, nn_id]
    print("     filtering skinning prediction")
    tpl_e = input_data.tpl_edge_index.data.cpu().numpy()
    skin_pred_full = post_filter(skin_pred_full, tpl_e, num_ring=1)
    skin_pred_full[skin_pred_full < np.max(skin_pred_full, axis=1, keepdims=True) * 0.35] = 0.0
    skin_pred_full = skin_pred_full / (skin_pred_full.sum(axis=1, keepdims=True) + 1e-10)
    skel_res = assemble_skel_skin(pred_skel, skin_pred_full)
    return skel_res


def tranfer_to_ori_mesh(filename_ori, filename_remesh, pred_rig):
    """
    convert the predicted rig of remeshed model to the rig of the original model.
    Just assign skinning weight based on nearest neighbor
    :param filename_ori: original mesh filename
    :param filename_remesh: remeshed mesh filename
    :param pred_rig: predicted rig
    :return: predicted rig for original mesh
    """
    mesh_remesh = o3d.io.read_triangle_mesh(filename_remesh)
    mesh_ori = o3d.io.read_triangle_mesh(filename_ori)
    tranfer_rig = Info()

    vert_remesh = np.asarray(mesh_remesh.vertices)
    vert_ori = np.asarray(mesh_ori.vertices)

    vertice_distance = np.sqrt(np.sum((vert_ori[np.newaxis, ...] - vert_remesh[:, np.newaxis, :]) ** 2, axis=2))
    vertice_raw_id = np.argmin(vertice_distance, axis=0)  # nearest vertex id on the fixed mesh for each vertex on the remeshed mesh

    tranfer_rig.root = pred_rig.root
    tranfer_rig.joint_pos = pred_rig.joint_pos
    new_skin = []
    for v in range(len(vert_ori)):
        skin_v = [v]
        v_nn = vertice_raw_id[v]
        skin_v += pred_rig.joint_skin[v_nn][1:]
        new_skin.append(skin_v)
    tranfer_rig.joint_skin = new_skin
    return tranfer_rig


if __name__ == '__main__':
    input_folder = "quick_start/"

    # downsample_skinning is used to speed up the calculation of volumetric geodesic distance
    # and to save cpu memory in skinning calculation.
    # Change to False to be more accurate but less efficient.
    downsample_skinning = True

    # load all weights
    print("loading all networks...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    jointNet = JOINTNET()
    jointNet.to(device)
    jointNet.eval()
    jointNet_checkpoint = torch.load('checkpoints/gcn_meanshift/model_best.pth.tar')
    jointNet.load_state_dict(jointNet_checkpoint['state_dict'])
    print("     joint prediction network loaded.")

    rootNet = ROOTNET()
    rootNet.to(device)
    rootNet.eval()
    rootNet_checkpoint = torch.load('checkpoints/rootnet/model_best.pth.tar')
    rootNet.load_state_dict(rootNet_checkpoint['state_dict'])
    print("     root prediction network loaded.")

    boneNet = BONENET()
    boneNet.to(device)
    boneNet.eval()
    boneNet_checkpoint = torch.load('checkpoints/bonenet/model_best.pth.tar')
    boneNet.load_state_dict(boneNet_checkpoint['state_dict'])
    print("     connection prediction network loaded.")

    skinNet = SKINNET(nearest_bone=5, use_Dg=True, use_Lf=True)
    skinNet_checkpoint = torch.load('checkpoints/skinnet/model_best.pth.tar')
    skinNet.load_state_dict(skinNet_checkpoint['state_dict'])
    skinNet.to(device)
    skinNet.eval()
    print("     skinning prediction network loaded.")

    # Here we provide 16~17 examples. For best results, we will need to override the learned bandwidth and its associated threshold
    # To process other input characters, please first try the learned bandwidth (0.0429 in the provided model), and the default threshold 1e-5.
    # We also use these two default parameters for processing all test models in batch.

    #model_id, bandwidth, threshold = "smith", None, 1e-5
    model_id, bandwidth, threshold = "17872", 0.045, 0.75e-5
    #model_id, bandwidth, threshold = "8210", 0.05, 1e-5
    #model_id, bandwidth, threshold = "8330", 0.05, 0.8e-5
    #model_id, bandwidth, threshold = "9477", 0.043, 2.5e-5
    #model_id, bandwidth, threshold = "17364", 0.058, 0.3e-5
    #model_id, bandwidth, threshold = "15930", 0.055, 0.4e-5
    #model_id, bandwidth, threshold = "8333", 0.04, 2e-5
    #model_id, bandwidth, threshold = "8338", 0.052, 0.9e-5
    #model_id, bandwidth, threshold = "3318", 0.03, 0.92e-5
    #model_id, bandwidth, threshold = "15446", 0.032, 0.58e-5
    #model_id, bandwidth, threshold = "1347", 0.062, 3e-5
    #model_id, bandwidth, threshold = "11814", 0.06, 0.6e-5
    #model_id, bandwidth, threshold = "2982", 0.045, 0.3e-5
    #model_id, bandwidth, threshold = "2586", 0.05, 0.6e-5
    #model_id, bandwidth, threshold = "8184", 0.05, 0.4e-5
    #model_id, bandwidth, threshold = "9000", 0.035, 0.16e-5

    # create data used for inferece
    print("creating data for model ID {:s}".format(model_id))
    mesh_filename = os.path.join(input_folder, '{:s}_remesh.obj'.format(model_id))
    data, vox, surface_geodesic, translation_normalize, scale_normalize = create_single_data(mesh_filename)
    data.to(device)

    print("predicting joints")
    data = predict_joints(data, vox, jointNet, threshold, bandwidth=bandwidth,
                          mesh_filename=mesh_filename.replace("_remesh.obj", "_normalized.obj"))
    data.to(device)
    print("predicting connectivity")
    pred_skeleton = predict_skeleton(data, vox, rootNet, boneNet,
                                     mesh_filename=mesh_filename.replace("_remesh.obj", "_normalized.obj"))
    print("predicting skinning")
    pred_rig = predict_skinning(data, pred_skeleton, skinNet, surface_geodesic,
                                mesh_filename.replace("_remesh.obj", "_normalized.obj"),
                                subsampling=downsample_skinning)

    # here we reverse the normalization to the original scale and position
    pred_rig.normalize(scale_normalize, -translation_normalize)

    print("Saving result")
    if True:
        # here we use original mesh tesselation (without remeshing)
        mesh_filename_ori = os.path.join(input_folder, '{:s}_ori.obj'.format(model_id))
        pred_rig = tranfer_to_ori_mesh(mesh_filename_ori, mesh_filename, pred_rig)
        pred_rig.save(mesh_filename_ori.replace('.obj', '_rig.txt'))
    else:
        # here we use remeshed mesh
        pred_rig.save(mesh_filename.replace('.obj', '_rig.txt'))
    print("Done!")
