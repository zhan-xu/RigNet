#-------------------------------------------------------------------------------
# Name:        compute_surface_geodesic.py
# Purpose:     This script calculates surface geodesic distance between all pair of vertices.
#              It doesn't rely on mesh topology. Surface is densely sampled to form a graph and get shortest path between samples.
#              Geodesic distance between pair of vertices are the geodesic distance between nearest samples to both vertices.
# RigNet Copyright 2020 University of Massachusetts
# RigNet is made available under General Public License Version 3 (GPLv3), or under a Commercial License.
# Please see the LICENSE README.txt file in the main directory for more information and instruction on using and licensing RigNet.
#-------------------------------------------------------------------------------

import sys
sys.path.append("./")
import os
import glob
import numpy as np
import open3d as o3d
from multiprocessing import Pool
from geometric_proc.common_ops import calc_surface_geodesic


def one_process(process_id):
    start_id = process_id * 340
    end_id = (process_id + 1) * 340
    print("processing {:d} to {:d}\n".format(start_id, end_id))

    remesh_obj_folder = "/media/zhanxu/4T1/ModelResource_Dataset/obj_remesh/"
    res_folder = "/media/zhanxu/4T1/ModelResource_Dataset/surface_geodesic/"

    remesh_obj_folder = glob.glob(remesh_obj_folder + '*.obj')
    for remesh_obj_filename in remesh_obj_folder[start_id: end_id]:
        model_id = remesh_obj_filename.split('/')[-1].split('.')[0]
        print(model_id)
        surface_geodesic = calc_surface_geodesic(o3d.io.read_triangle_mesh(remesh_obj_filename))
        np.save(os.path.join(res_folder, "{:s}_surface_geo.npy".format(model_id)), surface_geodesic.astype(np.float16))


if __name__ == '__main__':
    #one_process(0)
    p = Pool(4)
    p.map(one_process, [0, 1, 2, 3])
