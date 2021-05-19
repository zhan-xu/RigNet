#-------------------------------------------------------------------------------
# Name:        gen_dataset.py
# Purpose:     Script to generate data for skeleton and connectivity predition stage
#              Change dataset_folder to the folder where you put the downloaded pre-processed data
# RigNet Copyright 2020 University of Massachusetts
# RigNet is made available under General Public License Version 3 (GPLv3), or under a Commercial License.
# Please see the LICENSE README.txt file in the main directory for more information and instruction on using and licensing RigNet.
#-------------------------------------------------------------------------------

import os
import shutil
import numpy as np
import open3d as o3d
from multiprocessing import Pool
from utils.io_utils import mkdir_p
from utils.rig_parser import Info
from geometric_proc.common_ops import calc_surface_geodesic, get_bones


def get_tpl_edges(remesh_obj_v, remesh_obj_f):
    edge_index = []
    for v in range(len(remesh_obj_v)):
        face_ids = np.argwhere(remesh_obj_f == v)[:, 0]
        neighbor_ids = []
        for face_id in face_ids:
            for v_id in range(3):
                if remesh_obj_f[face_id, v_id] != v:
                    neighbor_ids.append(remesh_obj_f[face_id, v_id])
        neighbor_ids = list(set(neighbor_ids))
        neighbor_ids = [np.array([v, n])[np.newaxis, :] for n in neighbor_ids]
        neighbor_ids = np.concatenate(neighbor_ids, axis=0)
        edge_index.append(neighbor_ids)
    edge_index = np.concatenate(edge_index, axis=0)
    return edge_index


def get_geo_edges(surface_geodesic, remesh_obj_v):
    edge_index = []
    surface_geodesic += 1.0 * np.eye(len(surface_geodesic))  # remove self-loop edge here
    for i in range(len(remesh_obj_v)):
        geodesic_ball_samples = np.argwhere(surface_geodesic[i, :] <= 0.06).squeeze(1)
        if len(geodesic_ball_samples) > 10:
            geodesic_ball_samples = np.random.choice(geodesic_ball_samples, 10, replace=False)
        edge_index.append(np.concatenate((np.repeat(i, len(geodesic_ball_samples))[:, np.newaxis],
                                          geodesic_ball_samples[:, np.newaxis]), axis=1))
    edge_index = np.concatenate(edge_index, axis=0)
    return edge_index


def genDataset(process_id):
    global dataset_folder
    print("process ID {:d}".format(process_id))
    if process_id < 6:
        model_list = np.loadtxt(os.path.join(dataset_folder, 'train_final.txt'), dtype=int)
        model_list = model_list[365*process_id: 365*(process_id+1)]
        split_name = 'train'
    elif process_id == 6:
        model_list = np.loadtxt(os.path.join(dataset_folder, 'val_final.txt'), dtype=int)
        split_name = 'val'
    elif process_id == 7:
        model_list = np.loadtxt(os.path.join(dataset_folder, 'test_final.txt'), dtype=int)
        split_name = 'test'

    mkdir_p(os.path.join(dataset_folder, split_name))
    for model_id in model_list:
        remeshed_obj_filename = os.path.join(dataset_folder, 'obj_remesh/{:d}.obj'.format(model_id))
        info_filename = os.path.join(dataset_folder, 'rig_info_remesh/{:d}.txt'.format(model_id))
        remeshed_obj = o3d.io.read_triangle_mesh(remeshed_obj_filename)
        remesh_obj_v = np.asarray(remeshed_obj.vertices)
        if not remeshed_obj.has_vertex_normals():
            remeshed_obj.compute_vertex_normals()
        remesh_obj_vn = np.asarray(remeshed_obj.vertex_normals)
        remesh_obj_f = np.asarray(remeshed_obj.triangles)
        rig_info = Info(info_filename)

        #vertices
        vert_filename = os.path.join(dataset_folder, '{:s}/{:d}_v.txt'.format(split_name, model_id))
        input_feature = np.concatenate((remesh_obj_v, remesh_obj_vn), axis=1)
        np.savetxt(vert_filename, input_feature, fmt='%.6f')

        #topology edges
        edge_index = get_tpl_edges(remesh_obj_v, remesh_obj_f)
        graph_filename = os.path.join(dataset_folder, '{:s}/{:d}_tpl_e.txt'.format(split_name, model_id))
        np.savetxt(graph_filename, edge_index, fmt='%d')

        # geodesic_edges
        surface_geodesic = calc_surface_geodesic(remeshed_obj)
        edge_index = get_geo_edges(surface_geodesic, remesh_obj_v)
        graph_filename = os.path.join(dataset_folder, '{:s}/{:d}_geo_e.txt'.format(split_name, model_id))
        np.savetxt(graph_filename, edge_index, fmt='%d')

        # joints
        joint_pos = rig_info.get_joint_dict()
        joint_name_list = list(joint_pos.keys())
        joint_pos_list = list(joint_pos.values())
        joint_pos_list = [np.array(i) for i in joint_pos_list]
        adjacent_matrix = rig_info.adjacent_matrix()
        joint_filename = os.path.join(dataset_folder, '{:s}/{:d}_j.txt'.format(split_name, model_id))
        adj_filename = os.path.join(dataset_folder, '{:s}/{:d}_adj.txt'.format(split_name, model_id))
        np.savetxt(adj_filename, adjacent_matrix, fmt='%d')
        np.savetxt(joint_filename, np.array(joint_pos_list), fmt='%.6f')

        # pre_trained attn
        shutil.copyfile(os.path.join(dataset_folder, 'pretrain_attention/{:d}.txt'.format(model_id)), 
                        os.path.join(dataset_folder, '{:s}/{:d}_attn.txt'.format(split_name, model_id)))
        
        # voxel
        shutil.copyfile(os.path.join(dataset_folder, 'vox/{:d}.binvox'.format(model_id)), 
                        os.path.join(dataset_folder, '{:s}/{:d}.binvox'.format(split_name, model_id)))

        #skinning information
        num_nearest_bone = 5
        geo_dist = np.load(os.path.join(dataset_folder, "volumetric_geodesic/{:d}_volumetric_geo.npy".format(model_id)))
        bone_pos, bone_names, bone_isleaf = get_bones(rig_info)

        input_samples = []  # mesh_vertex_id, (bone_id, 1 / D_g, is_leaf) * N
        ground_truth_labels = []  # w_1, w_2, ..., w_N
        for vert_remesh_id in range(len(remesh_obj_v)):
            this_sample = [vert_remesh_id]
            this_label = []
            skin = rig_info.joint_skin[vert_remesh_id]
            skin_w = {}
            for i in np.arange(1, len(skin), 2):
                skin_w[skin[i]] = float(skin[i + 1])
            bone_id_near_to_far = np.argsort(geo_dist[vert_remesh_id, :])
            for i in range(num_nearest_bone):
                if i >= len(bone_id_near_to_far):
                    this_sample += [-1, 0, 0]
                    this_label.append(0.0)
                    continue
                bone_id = bone_id_near_to_far[i]
                this_sample.append(bone_id)
                this_sample.append(1.0 / (geo_dist[vert_remesh_id, bone_id] + 1e-10))
                this_sample.append(bone_isleaf[bone_id])
                start_joint_name = bone_names[bone_id][0]
                if start_joint_name in skin_w:
                    this_label.append(skin_w[start_joint_name])
                    del skin_w[start_joint_name]
                else:
                    this_label.append(0.0)

            input_samples.append(this_sample)
            ground_truth_labels.append(this_label)

        with open(os.path.join(dataset_folder, '{:s}/{:d}_skin.txt'.format(split_name, model_id)), 'w') as fout:
            for i in range(len(bone_pos)):
                fout.write('bones {:s} {:s} {:.6f} {:.6f} {:.6f} '
                           '{:.6f} {:.6f} {:.6f}\n'.format(bone_names[i][0], bone_names[i][1],
                                                           bone_pos[i, 0], bone_pos[i, 1], bone_pos[i, 2],
                                                           bone_pos[i, 3], bone_pos[i, 4], bone_pos[i, 5]))
            for i in range(len(input_samples)):
                fout.write('bind {:d} '.format(input_samples[i][0]))
                for j in np.arange(1, len(input_samples[i]), 3):
                    fout.write('{:d} {:.6f} {:d} '.format(input_samples[i][j], input_samples[i][j + 1], input_samples[i][j + 2]))
                fout.write('\n')
            for i in range(len(ground_truth_labels)):
                fout.write('influence ')
                for j in range(len(ground_truth_labels[i])):
                    fout.write('{:.3f} '.format(ground_truth_labels[i][j]))
                fout.write('\n')


if __name__ == '__main__':
    dataset_folder = "/media/zhanxu/4T/ModelResource_RigNetv1_preproccessed/"
    p = Pool(8)
    p.map(genDataset, [0, 1, 2, 3, 4, 5, 6, 7])
    #genDataset(0)
