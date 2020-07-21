#-------------------------------------------------------------------------------
# Name:        compute_volumetric_geodesic.py
# Purpose:     his script calculates volumetric geodesic distance between vertices and bones.
#              The shortest paths start from bones, then hit all "visible" vertices w.r.t each bone, i.e. the first hit on the surface is the vertex itself.
#              For "invisible" vertices, find the nearest "visible" vertices along surface by surface geodesic distance, and then go interior to the bone.
# RigNet Copyright 2020 University of Massachusetts
# RigNet is made available under General Public License Version 3 (GPLv3), or under a Commercial License.
# Please see the LICENSE README.txt file in the main directory for more information and instruction on using and licensing RigNet.
#-------------------------------------------------------------------------------

import sys
sys.path.append("./")
import os
import trimesh
import numpy as np
import open3d as o3d
from utils.os_utils import mkdir_p
from utils.rig_parser import Info
from geometric_proc.common_ops import get_bones, calc_surface_geodesic


def pts2line(pts, lines):
    '''
    Calculate points-to-bone distance. Point to line segment distance refer to
    https://stackoverflow.com/questions/849211/shortest-distance-between-a-point-and-a-line-segment
    :param pts: N*3
    :param lines: N*6, where [N,0:3] is the starting position and [N, 3:6] is the ending position
    :return: origins are the neatest projected position of the point on the line.
             ends are the points themselves.
             dist is the distance in between, which is the distance from points to lines.
             Origins and ends will be used for generate rays.
    '''
    l2 = np.sum((lines[:, 3:6] - lines[:, 0:3]) ** 2, axis=1)
    origins = np.zeros((len(pts) * len(lines), 3))
    ends = np.zeros((len(pts) * len(lines), 3))
    dist = np.zeros((len(pts) * len(lines)))
    for l in range(len(lines)):
        if np.abs(l2[l]) < 1e-8:  # for zero-length edges
            origins[l * len(pts):(l + 1) * len(pts)] = lines[l][0:3]
        else:  # for other edges
            t = np.sum((pts - lines[l][0:3][np.newaxis, :]) * (lines[l][3:6] - lines[l][0:3])[np.newaxis, :], axis=1) / \
                l2[l]
            t = np.clip(t, 0, 1)
            t_pos = lines[l][0:3][np.newaxis, :] + t[:, np.newaxis] * (lines[l][3:6] - lines[l][0:3])[np.newaxis, :]
            origins[l * len(pts):(l + 1) * len(pts)] = t_pos
        ends[l * len(pts):(l + 1) * len(pts)] = pts
        dist[l * len(pts):(l + 1) * len(pts)] = np.linalg.norm(
            origins[l * len(pts):(l + 1) * len(pts)] - ends[l * len(pts):(l + 1) * len(pts)], axis=1)
    return origins, ends, dist


def calc_pts2bone_visible_mat(mesh, origins, ends):
    '''
    Check whether the surface point is visible by the internal bone.
    Visible is defined as no occlusion on the path between.
    :param mesh:
    :param surface_pts: points on the surface (n*3)
    :param origins: origins of rays
    :param ends: ends of the rays, together with origins, we can decide the direction of the ray.
    :return: binary visibility matrix (n*m), where 1 indicate the n-th surface point is visible to the m-th ray
    '''
    ray_dir = ends - origins
    RayMeshIntersector = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)
    locations, index_ray, index_tri = RayMeshIntersector.intersects_location(origins, ray_dir + 1e-15)
    locations_per_ray = [locations[index_ray == i] for i in range(len(ray_dir))]
    min_hit_distance = []
    for i in range(len(locations_per_ray)):
        if len(locations_per_ray[i]) == 0:
            min_hit_distance.append(np.linalg.norm(ray_dir[i]))
        else:
            min_hit_distance.append(np.min(np.linalg.norm(locations_per_ray[i] - origins[i], axis=1)))
    min_hit_distance = np.array(min_hit_distance)
    distance = np.linalg.norm(ray_dir, axis=1)
    vis_mat = (np.abs(min_hit_distance - distance) < 1e-4)
    return vis_mat


def show_visible_mat(mesh_filename, joint_pos, vis_mat, joint_id):
    from utils.vis_utils import drawSphere

    mesh_o3d = o3d.io.read_triangle_mesh(mesh_filename)
    mesh_trimesh = trimesh.load(mesh_filename)
    visible = vis_mat[:, joint_id]

    mesh_ls = o3d.geometry.LineSet.create_from_triangle_mesh(mesh_o3d)
    mesh_ls.colors = o3d.utility.Vector3dVector([[0.8, 0.8, 0.8] for i in range(len(mesh_ls.lines))])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(mesh_trimesh.vertices)[visible])
    pcd.colors = o3d.utility.Vector3dVector(np.repeat(np.array([[0.0, 0.0, 1.0]]), int(np.sum(visible)), axis=0))
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(mesh_ls)
    vis.add_geometry(drawSphere(joint_pos[joint_id], 0.005, color=[1.0, 0.0, 0.0]))
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()


def one_process(dataset_folder, start_id, end_id):
    model_list = np.loadtxt(os.path.join(dataset_folder, 'model_list.txt'), dtype=int)
    model_list = model_list[start_id: end_id]
    remesh_obj_folder = os.path.join(dataset_folder, "obj_remesh")
    mkdir_p(os.path.join(dataset_folder, "volumetric_geodesic/"))
    
    for model_id in model_list:
        print(model_id)
        if os.path.exists(os.path.join(dataset_folder, "volumetric_geodesic/{:d}_volumetric_geo.npy".format(model_id))):
            continue
        remeshed_obj_filename = os.path.join(dataset_folder, 'obj_remesh/{:d}.obj'.format(model_id))
        ori_obj_filename = os.path.join(dataset_folder, 'obj/{:d}.obj'.format(model_id))
        info_filename = os.path.join(dataset_folder, 'rig_info/{:d}.txt'.format(model_id))
        
        pts = np.array(o3d.io.read_triangle_mesh(os.path.join(remesh_obj_folder, '{:d}.obj'.format(model_id))).vertices)

        mesh_remesh = trimesh.load(remeshed_obj_filename)
        mesh_ori = trimesh.load(ori_obj_filename)
        rig_info = Info(info_filename)
        bones, bone_name, _ = get_bones(rig_info)
        origins, ends, pts_bone_dist = pts2line(pts, bones)

        if os.path.exists(os.path.join(dataset_folder, "volumetric_geodesic/{:d}_visibility_raw.npy".format(model_id))):
            pts_bone_visibility = np.load(os.path.join(dataset_folder, "volumetric_geodesic/{:d}_visibility_raw.npy".format(model_id)))
        else:
            # pick one mesh with fewer faces to speed up
            if len(mesh_remesh.faces) < len(mesh_ori.faces):
                trimesh.repair.fix_normals(mesh_remesh)
                pts_bone_visibility = calc_pts2bone_visible_mat(mesh_remesh, origins, ends)
            else:
                trimesh.repair.fix_normals(mesh_ori)
                pts_bone_visibility = calc_pts2bone_visible_mat(mesh_ori, origins, ends)
            pts_bone_visibility = pts_bone_visibility.reshape(len(bones), len(pts)).transpose()
            #np.save(os.path.join(dataset_folder, "volumetric_geodesic/{:d}_visibility_raw.npy".format(model_id)), pts_bone_visibility)
        pts_bone_dist = pts_bone_dist.reshape(len(bones), len(pts)).transpose()

        # remove visible points which are too far
        if os.path.exists(os.path.join(dataset_folder, "volumetric_geodesic/{:d}_visibility_filtered.npy".format(model_id))):
            pts_bone_visibility = np.load(os.path.join(dataset_folder, "volumetric_geodesic/{:d}_visibility_filtered.npy".format(model_id)))
        else:
            for b in range(pts_bone_visibility.shape[1]):
                visible_pts = np.argwhere(pts_bone_visibility[:, b] == 1).squeeze(1)
                if len(visible_pts) == 0:
                   continue
                threshold_b = np.percentile(pts_bone_dist[visible_pts, b], 15)
                pts_bone_visibility[pts_bone_dist[:, b] > 1.3 * threshold_b, b] = False
            #np.save(os.path.join(dataset_folder, "volumetric_geodesic/{:d}_visibility_filtered.npy".format(model_id)), pts_bone_visibility)

        mesh = o3d.io.read_triangle_mesh(os.path.join(remesh_obj_folder, '{:d}.obj'.format(model_id)))
        surface_geodesic = calc_surface_geodesic(mesh)

        visible_matrix = np.zeros(pts_bone_visibility.shape)
        visible_matrix[np.where(pts_bone_visibility == 1)] = pts_bone_dist[np.where(pts_bone_visibility == 1)]
        euc_dist = np.sqrt(np.sum((pts[np.newaxis, ...] - pts[:, np.newaxis, :]) ** 2, axis=2))
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
        np.save(os.path.join(dataset_folder, "volumetric_geodesic/{:d}_volumetric_geo.npy".format(model_id)), visible_matrix)


if __name__ == '__main__':
    start_id = int(sys.argv[1])
    end_id = int(sys.argv[2])
    dataset_folder = "/media/zhanxu/4T1/ModelResource_Dataset/"
    one_process(dataset_folder, start_id, end_id)
