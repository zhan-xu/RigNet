#-------------------------------------------------------------------------------
# Name:        vis_utils.py
# Purpose:     utilize functions for visualization, highly relied on Open3D (0.9.0)
# RigNet Copyright 2020 University of Massachusetts
# RigNet is made available under General Public License Version 3 (GPLv3), or under a Commercial License.
# Please see the LICENSE README.txt file in the main directory for more information and instruction on using and licensing RigNet.
#-------------------------------------------------------------------------------

import os
import cv2
import glob
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt


def drawSphere(center, radius, color=[0.0,0.0,0.0]):
    mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    transform_mat = np.eye(4)
    transform_mat[0:3, -1] = center
    mesh_sphere.transform(transform_mat)
    mesh_sphere.paint_uniform_color(color)
    return mesh_sphere


def drawCone(bottom_center, top_position, color=[0.6, 0.6, 0.9]):
    cone = o3d.geometry.TriangleMesh.create_cone(radius=0.007, height=np.linalg.norm(top_position - bottom_center)+1e-6)
    line1 = np.array([0.0, 0.0, 1.0])
    line2 = (top_position - bottom_center) / (np.linalg.norm(top_position - bottom_center)+1e-6)
    v = np.cross(line1, line2)
    c = np.dot(line1, line2) + 1e-8
    k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    R = np.eye(3) + k + np.matmul(k, k) * (1 / (1 + c))
    if np.abs(c + 1.0) < 1e-4: # the above formula doesn't apply when cos(∠(𝑎,𝑏))=−1
        R = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
    T = bottom_center + 5e-3 * line2
    #print(R)
    cone.transform(np.concatenate((np.concatenate((R, T[:, np.newaxis]), axis=1), np.array([[0.0, 0.0, 0.0, 1.0]])), axis=0))
    cone.paint_uniform_color(color)
    return cone


def show_obj_skel(mesh_name, root):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    ctr = vis.get_view_control()

    # draw mesh
    mesh = o3d.io.read_triangle_mesh(mesh_name)
    mesh_ls = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
    mesh_ls.colors = o3d.utility.Vector3dVector([[0.8, 0.8, 0.8] for i in range(len(mesh_ls.lines))])
    vis.add_geometry(mesh_ls)

    vis.add_geometry(drawSphere(root.pos, 0.01, color=[0.1, 0.1, 0.1]))
    this_level = root.children
    while this_level:
        next_level = []
        for p_node in this_level:
            vis.add_geometry(drawSphere(p_node.pos, 0.008, color=[1.0, 0.0, 0.0])) # [0.3, 0.1, 0.1]
            vis.add_geometry(drawCone(np.array(p_node.parent.pos), np.array(p_node.pos)))
            next_level+=p_node.children
        this_level = next_level

    #param = o3d.io.read_pinhole_camera_parameters('sideview.json')
    #ctr.convert_from_pinhole_camera_parameters(param)

    vis.run()
    #vis.update_geometry()
    #vis.poll_events()
    #vis.update_renderer()

    #param = ctr.convert_to_pinhole_camera_parameters()
    #o3d.io.write_pinhole_camera_parameters('sideview.json', param)

    image = vis.capture_screen_float_buffer()
    vis.destroy_window()
    image = np.asarray(image) * 255
    image = image.astype(np.uint8)
    return image


def draw_shifted_pts(mesh_name, pts, weights=None):
    mesh = o3d.io.read_triangle_mesh(mesh_name)
    mesh_ls = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
    mesh_ls.colors = o3d.utility.Vector3dVector([[0.8, 0.8, 0.8] for i in range(len(mesh_ls.lines))])
    pred_joints = o3d.geometry.PointCloud()
    pred_joints.points = o3d.utility.Vector3dVector(pts)
    if weights is None:
        color_joints = [[1.0, 0.0, 0.0] for i in range(len(pts))]
    else:
        import matplotlib.pyplot as plt
        cmap = plt.cm.get_cmap('YlOrRd')
        #weights = (weights - np.min(weights)) / (np.max(weights) - np.min(weights))
        #weights = 1 / (1 + np.exp(-weights))
        color_joints = cmap(weights.squeeze())
        color_joints = color_joints[:, :-1]
    pred_joints.colors = o3d.utility.Vector3dVector(color_joints)

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    ctr = vis.get_view_control()
    vis.add_geometry(mesh_ls)
    vis.add_geometry(pred_joints)

    param = o3d.io.read_pinhole_camera_parameters('sideview.json')
    ctr.convert_from_pinhole_camera_parameters(param)

    #vis.run()
    vis.update_geometry()
    vis.poll_events()
    vis.update_renderer()

    #param = ctr.convert_to_pinhole_camera_parameters()
    #o3d.io.write_pinhole_camera_parameters('sideview.json', param)

    image = vis.capture_screen_float_buffer()
    vis.destroy_window()
    image = np.asarray(image) * 255
    image = image.astype(np.uint8)
    return image


def draw_joints(mesh_name, pts):
    mesh = o3d.io.read_triangle_mesh(mesh_name)
    mesh_ls = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
    mesh_ls.colors = o3d.utility.Vector3dVector([[0.8, 0.8, 0.8] for i in range(len(mesh_ls.lines))])
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    ctr = vis.get_view_control()
    vis.add_geometry(mesh_ls)
    for joint_pos in pts:
        vis.add_geometry(drawSphere(joint_pos, 0.006, color=[1.0, 0.0, 0.0]))

    param = o3d.io.read_pinhole_camera_parameters('sideview.json')
    ctr.convert_from_pinhole_camera_parameters(param)

    #vis.run()
    vis.update_geometry()
    vis.poll_events()
    vis.update_renderer()

    image = vis.capture_screen_float_buffer()
    vis.destroy_window()
    image = np.asarray(image) * 255
    image = image.astype(np.uint8)
    return image


def volume_to_cubes(volume, threshold=0, dim=[1., 1., 1.]):
    #o = np.array([-dim[0]/2., -dim[1]/2., -dim[2]/2.])
    o = np.array([0, 0, 0])
    step = np.array([dim[0]/volume.shape[0], dim[1]/volume.shape[1], dim[2]/volume.shape[2]])
    points = []
    lines = []
    for x in range(1, volume.shape[0]-1):
        for y in range(1, volume.shape[1]-1):
            for z in range(1, volume.shape[2]-1):
                pos = o + np.array([x, y, z]) * step
                if volume[x, y, z] > threshold:
                    vidx = len(points)
                    POS = pos + step*0.95
                    xx = pos[0]
                    yy = pos[1]
                    zz = pos[2]
                    XX = POS[0]
                    YY = POS[1]
                    ZZ = POS[2]

                    points.append(np.array([xx, yy, zz])[np.newaxis, :])
                    points.append(np.array([xx, YY, zz])[np.newaxis, :])
                    points.append(np.array([XX, YY, zz])[np.newaxis, :])
                    points.append(np.array([XX, yy, zz])[np.newaxis, :])
                    points.append(np.array([xx, yy, ZZ])[np.newaxis, :])
                    points.append(np.array([xx, YY, ZZ])[np.newaxis, :])
                    points.append(np.array([XX, YY, ZZ])[np.newaxis, :])
                    points.append(np.array([XX, yy, ZZ])[np.newaxis, :])

                    lines.append(np.array([vidx + 1, vidx + 2]))
                    lines.append(np.array([vidx + 2, vidx + 6]))
                    lines.append(np.array([vidx + 6, vidx + 5]))
                    lines.append(np.array([vidx + 1, vidx + 5]))

                    lines.append(np.array([vidx + 1, vidx + 1]))
                    lines.append(np.array([vidx + 3, vidx + 3]))
                    lines.append(np.array([vidx + 7, vidx + 7]))
                    lines.append(np.array([vidx + 5, vidx + 5]))

                    lines.append(np.array([vidx + 0, vidx + 3]))
                    lines.append(np.array([vidx + 0, vidx + 4]))
                    lines.append(np.array([vidx + 4, vidx + 7]))
                    lines.append(np.array([vidx + 7, vidx + 3]))

    return points, lines


def show_mesh_vox(mesh_filename, vox):
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    vox_pts, vox_lines = volume_to_cubes(vox.data)
    vox_pts = np.concatenate(vox_pts, axis=0)
    line_set_vox = o3d.geometry.LineSet()
    line_set_vox.points = o3d.utility.Vector3dVector(vox_pts+np.array(vox.translate)[np.newaxis, :])
    line_set_vox.lines = o3d.utility.Vector2iVector(vox_lines)
    colors = [[0.0, 0.0, 1.0] for i in range(len(vox_lines))]
    line_set_vox.colors = o3d.utility.Vector3dVector(colors)
    vis.add_geometry(line_set_vox)

    mesh = o3d.io.read_triangle_mesh(mesh_filename)
    mesh_ls = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
    mesh_ls.colors = o3d.utility.Vector3dVector([[1.0, 0.0, 0.0] for i in range(len(mesh_ls.lines))])
    vis.add_geometry(mesh_ls)

    vis.run()
    vis.destroy_window()

    return
