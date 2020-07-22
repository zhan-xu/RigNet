#-------------------------------------------------------------------------------
# Name:        skin_dataset.py
# Purpose:     torch_geometric dataset wrapper for skinning training and inference
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


class SkinDataset(InMemoryDataset):
    def __init__(self, root):
        super(SkinDataset, self).__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        raw_v_filelist = glob.glob(os.path.join(self.root, '*_v.txt'))
        return raw_v_filelist

    @property
    def processed_file_names(self):
        return '{:s}_skinning_data.pt'.format(self.root.split('/')[-1])

    def __len__(self):
        return len(self.raw_paths)

    def download(self):
        pass

    def load_skin(self, filename):
        with open(filename, 'r') as fin:
            lines = fin.readlines()
        bones = []
        input = []
        label = []
        nearest_bone_ids = []
        loss_mask_all = []
        for li in lines:
            words = li.strip().split()
            if words[0] == 'bones':
                bones.append([float(w) for w in words[3:]])
            elif words[0] == 'bind':
                words = [float(w) for w in words[1:]]
                sample_input = []
                sample_nearest_bone_ids = []
                loss_mask = []
                for i in range(self.num_nearest_bone):
                    if int(words[3 * i + 1]) == -1:
                        ## go around. words[3] may be also invalid.
                        sample_nearest_bone_ids.append(int(words[1]))
                        sample_input += bones[int(words[1])]
                        sample_input.append(words[2])
                        sample_input.append(int(words[3]))
                        loss_mask.append(0)
                    else:
                        sample_nearest_bone_ids.append(int(words[3 * i + 1]))
                        sample_input += bones[int(words[3 * i + 1])]
                        sample_input.append(words[3 * i + 2])
                        sample_input.append(int(words[3 * i + 3]))
                        loss_mask.append(1)
                input.append(np.array(sample_input)[np.newaxis, :])
                nearest_bone_ids.append(np.array(sample_nearest_bone_ids)[np.newaxis, :])
                loss_mask_all.append(np.array(loss_mask)[np.newaxis, :])
            elif words[0] == 'influence':
                sample_label = np.array([float(w) for w in words[1:]])[np.newaxis, :]
                label.append(sample_label)

        input = np.concatenate(input, axis=0)
        nearest_bone_ids = np.concatenate(nearest_bone_ids, axis=0)
        label = np.concatenate(label, axis=0)
        loss_mask_all = np.concatenate(loss_mask_all, axis=0)

        return input, nearest_bone_ids, label, loss_mask_all

    def process(self):
        data_list = []
        self.num_nearest_bone = 5
        i = 0.0
        for v_filename in self.raw_paths:
            print('preprecessing data complete: {:.4f}%'.format(100 * i / len(self.raw_paths)))
            i += 1.0
            name = int(v_filename.split('/')[-1].split('_')[0])
            v = np.loadtxt(v_filename)
            v = torch.from_numpy(v).float()
            tpl_e = np.loadtxt(v_filename.replace('_v.txt', '_tpl_e.txt')).T
            geo_e = np.loadtxt(v_filename.replace('_v.txt', '_geo_e.txt')).T
            tpl_e = torch.from_numpy(tpl_e).long()
            geo_e = torch.from_numpy(geo_e).long()
            tpl_e, _ = add_self_loops(tpl_e, num_nodes=v.size(0))
            geo_e, _ = add_self_loops(geo_e, num_nodes=v.size(0))
            skin_input, skin_nn, skin_label, loss_mask = self.load_skin(v_filename.replace('_v.txt', '_skin.txt'))
            
            skin_input = torch.from_numpy(skin_input).float()
            skin_label = torch.from_numpy(skin_label).float()
            skin_nn = torch.from_numpy(skin_nn).long()
            loss_mask = torch.from_numpy(loss_mask).long()
            num_skin = len(skin_input)

            name = int(v_filename.split('/')[-1].split('_')[0])
            data_list.append(Data(x=v[:, 3:6], pos=v[:, 0:3], skin_input=skin_input, skin_label=skin_label,
                                  skin_nn=skin_nn, loss_mask=loss_mask, num_skin=num_skin, name=name,
                                  tpl_edge_index=tpl_e, geo_edge_index=geo_e))
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
