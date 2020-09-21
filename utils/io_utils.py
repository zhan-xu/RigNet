#-------------------------------------------------------------------------------
# Name:        io_utils.py
# Purpose:     utilize functions for file IO
# RigNet Copyright 2020 University of Massachusetts
# RigNet is made available under General Public License Version 3 (GPLv3), or under a Commercial License.
# Please see the LICENSE README.txt file in the main directory for more information and instruction on using and licensing RigNet.
#-------------------------------------------------------------------------------

import os
import numpy as np
from utils.os_utils import mkdir_p
from utils.tree_utils import TreeNode
from utils.rig_parser import Info
from geometric_proc.compute_volumetric_geodesic import get_bones


def readPly(filename):
    with open(filename, 'r') as fin:
        lines = fin.readlines()
    pts = []
    for li in lines[7:]:
        words = li.split()
        pts.append(np.array([[float(words[0]), float(words[1]), float(words[2])]]))
    pts = np.concatenate(pts, axis=0)
    return pts

def writePly(pts, filename):
    with open(filename, 'w') as f:
        pn = pts.shape[0]
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write('element vertex %d\n' % (pn) )
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        f.write('end_header\n')
        for i in range(pn):
            f.write('%f %f %f\n' % (pts[i, 0],  pts[i, 1],  pts[i, 2]) )


def output_point_cloud_ply(xyzs, name, output_folder):
    if not os.path.exists( output_folder ):
        mkdir_p(  output_folder  )
    print('write: ' + os.path.join(output_folder, name + '.ply'))
    with open(os.path.join(output_folder, name + '.ply'), 'w') as f:
        pn = xyzs.shape[0]
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write('element vertex %d\n' % (pn) )
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        f.write('end_header\n')
        for i in range(pn):
            f.write('%f %f %f\n' % (xyzs[i][0],  xyzs[i][1],  xyzs[i][2]) )


def add_duplicate_joints(skel):
    this_level = [skel.root]
    while this_level:
        next_level = []
        for p_node in this_level:
            if len(p_node.children) > 1:
                new_children = []
                for dup_id in range(len(p_node.children)):
                    p_node_new = TreeNode(p_node.name + '_dup_{:d}'.format(dup_id), p_node.pos)
                    p_node_new.overlap=True
                    p_node_new.parent = p_node
                    p_node_new.children = [p_node.children[dup_id]]
                    # for user interaction, we move overlapping joints a bit to its children
                    p_node_new.pos = np.array(p_node_new.pos) + 0.03 * np.linalg.norm(np.array(p_node.children[dup_id].pos) - np.array(p_node_new.pos))
                    p_node_new.pos = (p_node_new.pos[0], p_node_new.pos[1], p_node_new.pos[2])
                    p_node.children[dup_id].parent = p_node_new
                    new_children.append(p_node_new)
                p_node.children = new_children
            p_node.overlap = False
            next_level += p_node.children
        this_level = next_level
    return skel


def mapping_bone_index(bones_old, bones_new):
    bone_map = {}
    for i in range(len(bones_old)):
        bone_old = bones_old[i][np.newaxis, :]
        dist = np.linalg.norm(bones_new - bone_old, axis=1)
        ni = np.argmin(dist)
        bone_map[i] = ni
    return bone_map


def assemble_skel_skin(skel, attachment):
    bones_old, bone_names_old, _ = get_bones(skel)
    skel_new = add_duplicate_joints(skel)
    bones_new, bone_names_new, _ = get_bones(skel_new)
    bone_map = mapping_bone_index(bones_old, bones_new)
    skel_new.joint_pos = skel_new.get_joint_dict()
    skel_new.joint_skin = []

    for v in range(len(attachment)):
        vi_skin = [str(v)]
        skw = attachment[v]
        skw = skw / (np.sum(skw) + 1e-10)
        for i in range(len(skw)):
            if i == len(bones_old):
                break
            if skw[i] > 1e-5:
                bind_joint_name = bone_names_new[bone_map[i]][0]
                bind_weight = skw[i]
                vi_skin.append(bind_joint_name)
                vi_skin.append(str(bind_weight))
        skel_new.joint_skin.append(vi_skin)
    return skel_new


def output_rigging(skel_name, attachment, output_folder, name):
    skel = Info(skel_name)
    skel_new = assemble_skel_skin(skel, attachment)
    skel_new.save(os.path.join(output_folder, str(name) + '_rig.txt'))
