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
import itertools as it
from utils import binvox_rw
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import add_self_loops


class SkeletonData(Data):
    def __init__(self, x=None, pos=None, name=None, mask=None, joints=None,
                 tpl_edge_index=None, geo_edge_index=None, pairs=None, pair_attr=None):
        super(SkeletonData, self).__init__()
        self.x = x
        self.pos = pos
        self.name = name
        self.mask = mask
        self.joints = joints
        self.tpl_edge_index = tpl_edge_index
        self.geo_edge_index = geo_edge_index
        self.pairs = pairs
        self.pair_attr = pair_attr

    def __inc__(self, key, value):
        if key == 'pairs':
            return self.joints.size(0)
        else:
            return super(SkeletonData, self).__inc__(key, value)


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
            tpl_e = np.loadtxt(v_filename.replace('_v.txt', '_tpl_e.txt')).T
            geo_e = np.loadtxt(v_filename.replace('_v.txt', '_geo_e.txt')).T
            joints = np.loadtxt(v_filename.replace('_v.txt', '_j.txt'))
            adj = np.loadtxt(v_filename.replace('_v.txt', '_adj.txt'), dtype=np.uint8)

            vox_file = v_filename.replace('_v.txt', '.binvox')
            with open(vox_file, 'rb') as fvox:
                vox = binvox_rw.read_as_3d_array(fvox)
            pairs = list(it.combinations(range(adj.shape[0]), 2))
            pair_attr = []
            for pr in pairs:
                dist = np.linalg.norm(joints[pr[0]] - joints[pr[1]])
                bone_samples = self.sample_on_bone(joints[pr[0]], joints[pr[1]])
                bone_samples_inside, _ = self.inside_check(bone_samples, vox)
                outside_proportion = len(bone_samples_inside) / (len(bone_samples) + 1e-10)
                attr = np.array([dist, outside_proportion, adj[pr[0], pr[1]]])
                pair_attr.append(attr)
            pairs = np.array(pairs)
            pair_attr = np.array(pair_attr)
            name = int(v_filename.split('/')[-1].split('_')[0])

            v = torch.from_numpy(v).float()
            m = torch.from_numpy(m).long()
            tpl_e = torch.from_numpy(tpl_e).long()
            geo_e = torch.from_numpy(geo_e).long()
            tpl_e, _ = add_self_loops(tpl_e, num_nodes=v.size(0))
            geo_e, _ = add_self_loops(geo_e, num_nodes=v.size(0))
            joints = torch.from_numpy(joints).float()
            pairs = torch.from_numpy(pairs).float()
            pair_attr = torch.from_numpy(pair_attr).float()
            data_list.append(SkeletonData(x=v[:, 3:6], pos=v[:, 0:3], name=name, mask=m, joints=joints,
                                          tpl_edge_index=tpl_e, geo_edge_index=geo_e, pairs=pairs, pair_attr=pair_attr))
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
