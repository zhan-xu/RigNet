#-------------------------------------------------------------------------------
# Name:        skeleton_dataset.py
# Purpose:     torch_geometric dataset wrapper for skeleton training and inference
# RigNet Copyright 2020 University of Massachusetts
# RigNet is made available under General Public License Version 3 (GPLv3), or under a Commercial License.
# Please see the LICENSE README.txt file in the main directory for more information and instruction on using and licensing RigNet.
#-------------------------------------------------------------------------------

import os
import torch
import numpy as np
import glob
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import add_self_loops
from utils import binvox_rw


class GraphDataset(InMemoryDataset):
    def __init__(self, root):
        super(GraphDataset, self).__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        raw_v_filelist = glob.glob(os.path.join(self.root, '*_v.txt'))
        return raw_v_filelist

    @property
    def processed_file_names(self):
        return '{:s}_skeleton_data.pt'.format(self.root.split('/')[-1])

    def __len__(self):
        return len(self.raw_paths)

    def download(self):
        pass

    def sample_on_bone(self, p_pos, ch_pos):
        ray = ch_pos - p_pos
        bone_length = np.sqrt(np.sum((p_pos - ch_pos) ** 2))
        num_step = np.round(bone_length / 0.01)
        i_step = np.arange(1, num_step + 1)
        unit_step = ray / (num_step + 1e-30)
        unit_step = np.repeat(unit_step[np.newaxis, :], num_step, axis=0)
        res = p_pos + unit_step * i_step[:, np.newaxis]
        return res

    def inside_check(self, pts, vox):
        vc = (pts - vox.translate) / vox.scale * vox.dims[0]
        vc = np.round(vc).astype(int)
        ind1 = np.logical_and(np.all(vc >= 0, axis=1), np.all(vc < vox.dims[0], axis=1))
        vc = np.clip(vc, 0, vox.dims[0]-1)
        ind2 = vox.data[vc[:, 0], vc[:, 1], vc[:, 2]]
        ind = np.logical_and(ind1, ind2)
        pts = pts[ind]
        return pts, np.argwhere(ind).squeeze()

    def process(self):
        data_list = []
        i = 0.0
        for v_filename in self.raw_paths:
            print('preprecessing data complete: {:.4f}%'.format(100 * i / len(self.raw_paths)))
            i += 1.0
            v = np.loadtxt(v_filename)
            m = np.loadtxt(v_filename.replace('_v.txt', '_attn.txt'))
            v = torch.from_numpy(v).float()
            m = torch.from_numpy(m).long()
            tpl_e = np.loadtxt(v_filename.replace('_v.txt', '_tpl_e.txt')).T
            geo_e = np.loadtxt(v_filename.replace('_v.txt', '_geo_e.txt')).T
            tpl_e = torch.from_numpy(tpl_e).long()
            geo_e = torch.from_numpy(geo_e).long()
            tpl_e, _ = add_self_loops(tpl_e, num_nodes=v.size(0))
            geo_e, _ = add_self_loops(geo_e, num_nodes=v.size(0))
            y = np.loadtxt(v_filename.replace('_v.txt', '_j.txt'))
            num_joint = len(y)
            joint_pos = y
            if len(y) < len(v):
                y = np.tile(y, (round(1.0 * len(v) / len(y) + 0.5), 1))
                y = y[:len(v), :]
            elif len(y) > len(v):
                y = y[:len(v), :]
            y = torch.from_numpy(y).float()

            adj = np.loadtxt(v_filename.replace('_v.txt', '_adj.txt'), dtype=np.uint8)

            vox_file = v_filename.replace('_v.txt', '.binvox')
            with open(vox_file, 'rb') as fvox:
                vox = binvox_rw.read_as_3d_array(fvox)
            pair_all = []
            for joint1_id in range(adj.shape[0]):
                for joint2_id in range(joint1_id + 1, adj.shape[1]):
                    dist = np.linalg.norm(joint_pos[joint1_id] - joint_pos[joint2_id])
                    bone_samples = self.sample_on_bone(joint_pos[joint1_id], joint_pos[joint2_id])
                    bone_samples_inside, _ = self.inside_check(bone_samples, vox)
                    outside_proportion = len(bone_samples_inside) / (len(bone_samples) + 1e-10)
                    pair = np.array([joint1_id, joint2_id, dist, outside_proportion, adj[joint1_id, joint2_id]])
                    pair_all.append(pair)
            pair_all = np.array(pair_all)
            pair_all = torch.from_numpy(pair_all).float()
            num_pair = len(pair_all)

            name = int(v_filename.split('/')[-1].split('_')[0])
            data_list.append(Data(x=v[:, 3:6], pos=v[:, 0:3], name=name, mask=m, y=y, num_joint=num_joint,
                                  tpl_edge_index=tpl_e, geo_edge_index=geo_e, pairs=pair_all, num_pair=num_pair))
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
