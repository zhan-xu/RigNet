#-------------------------------------------------------------------------------
# Name:        PairCls_GCN.py
# Purpose:     definition of connectivity prediction module.
# RigNet Copyright 2020 University of Massachusetts
# RigNet is made available under General Public License Version 3 (GPLv3), or under a Commercial License.
# Please see the LICENSE README.txt file in the main directory for more information and instruction on using and licensing RigNet.
#-------------------------------------------------------------------------------
import numpy as np
import torch
from models.gcn_basic_modules import MLP, GCU
from torch.nn import Sequential, Dropout, Linear
from torch_scatter import scatter_max
from torch_geometric.nn import PointConv, fps, radius, global_max_pool, knn_interpolate


class SAModule(torch.nn.Module):
    def __init__(self, ratio, r, nn):
        super(SAModule, self).__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointConv(nn)

    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.ratio)
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx],
                          max_num_neighbors=64)
        edge_index = torch.stack([col, row], dim=0)
        x = self.conv(x, (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch


class GlobalSAModule(torch.nn.Module):
    def __init__(self, nn):
        super(GlobalSAModule, self).__init__()
        self.nn = nn

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch


class FPModule(torch.nn.Module):
    def __init__(self, k, nn):
        super(FPModule, self).__init__()
        self.k = k
        self.nn = nn

    def forward(self, x, pos, batch, x_skip, pos_skip, batch_skip):
        x = knn_interpolate(x, pos, pos_skip, batch, batch_skip, k=self.k)
        if x_skip is not None:
            x = torch.cat([x, x_skip], dim=1)
        x = self.nn(x)
        return x, pos_skip, batch_skip


class ShapeEncoder(torch.nn.Module):
    def __init__(self, aggr='max'):
        super(ShapeEncoder, self).__init__()
        self.gcu_1 = GCU(in_channels=3, out_channels=64, aggr=aggr)
        self.gcu_2 = GCU(in_channels=64, out_channels=128, aggr=aggr)
        self.gcu_3 = GCU(in_channels=128, out_channels=256, aggr=aggr)
        self.mlp_glb = MLP([(64 + 128 + 256), 256, 64])

    def forward(self, data):
        x_1 = self.gcu_1(data.pos, data.tpl_edge_index, data.geo_edge_index)
        x_2 = self.gcu_2(x_1, data.tpl_edge_index, data.geo_edge_index)
        x_3 = self.gcu_3(x_2, data.tpl_edge_index, data.geo_edge_index)
        x_4 = self.mlp_glb(torch.cat([x_1, x_2, x_3], dim=1))
        x_global_shape, _ = scatter_max(x_4, data.batch, dim=0)
        return x_global_shape


class JointEncoder(torch.nn.Module):
    def __init__(self):
        super(JointEncoder, self).__init__()
        #self.mlp_1 = MLP([3, 64, 128, 1024])
        #self.mlp_2 = MLP([1024, 256, 128])

        self.sa1_module_joints = SAModule(0.999, 0.4, MLP([3, 64, 64, 128]))
        self.sa2_module_joints = SAModule(0.33, 0.6, MLP([128 + 3, 128, 128, 256]))
        self.sa3_module_joints = GlobalSAModule(MLP([256 + 3, 256, 256, 512, 256, 128]))

    def forward(self, joints, joints_batch):
        '''x1 = self.mlp_1(joints_norepeat)
        x_glb, _ = scatter_max(x1, joints_batch, dim=0)
        x_glb = self.mlp_2(x_glb)
        return x_glb'''

        sa0_joints = (None, joints, joints_batch)
        sa1_joints = self.sa1_module_joints(*sa0_joints)
        sa2_joints = self.sa2_module_joints(*sa1_joints)
        sa3_joints = self.sa3_module_joints(*sa2_joints)
        x_glb_joint = sa3_joints[0]
        return x_glb_joint


class PairCls(torch.nn.Module):
    def __init__(self):
        super(PairCls, self).__init__()
        self.expand_joint_feature = Sequential(MLP([8, 32, 64, 128, 256])) 
        self.shape_encoder = ShapeEncoder()
        self.joint_encoder = JointEncoder()
        input_concat_dim = 448
        self.mix_transform = Sequential(MLP([input_concat_dim, 128, 64]), Dropout(0.7), Linear(64, 1))

    def forward(self, data, permute_joints=True):
        joint_feature = self.joint_encoder(data.joints, data.joints_batch)
        joint_feature = torch.repeat_interleave(joint_feature, torch.bincount(data.pairs_batch), dim=0)
        shape_feature = self.shape_encoder(data)
        shape_feature = torch.repeat_interleave(shape_feature, torch.bincount(data.pairs_batch), dim=0)

        if permute_joints:
            rand_permute = (torch.rand(len(data.pairs))>=0.5).long().to(data.pairs.device)
            joints_pair = torch.cat((data.joints[torch.gather(data.pairs, dim=1, index=rand_permute.unsqueeze(dim=1)).squeeze(dim=1).long()],
                                     data.joints[torch.gather(data.pairs, dim=1, index=1-rand_permute.unsqueeze(dim=1)).squeeze(dim=1).long()],
                                     data.pair_attr[:, :-1]), dim=1)
        else:
            joints_pair = torch.cat((data.joints[data.pairs[:,0].long()], data.joints[data.pairs[:,1].long()], data.pair_attr[:, :-1]), dim=1)
        pair_feature = self.expand_joint_feature(joints_pair)
        pair_feature = torch.cat((shape_feature, joint_feature, pair_feature), dim=1)
        pre_label = self.mix_transform(pair_feature)
        gt_label = data.pair_attr[:, -1].unsqueeze(1)
        return pre_label, gt_label
