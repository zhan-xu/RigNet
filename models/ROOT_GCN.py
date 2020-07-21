#-------------------------------------------------------------------------------
# Name:        Root_GCN.py
# Purpose:     definition of root prediction module.
# RigNet Copyright 2020 University of Massachusetts
# RigNet is made available under General Public License Version 3 (GPLv3), or under a Commercial License.
# Please see the LICENSE README.txt file in the main directory for more information and instruction on using and licensing RigNet.
#-------------------------------------------------------------------------------
import torch
from models.gcn_basic_modules import MLP, GCU
from torch_scatter import scatter_max
from torch.nn import Sequential, Linear
from torch_geometric.nn import PointConv, fps, radius, global_max_pool, knn_interpolate
__all__ = ['ROOTNET']

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
        self.mlp_glb = MLP([(64 + 128 + 256), 128])
        #self.mlp_glb = MLP([(64 + 128 + 256), 512])

    def forward(self, data):
        x_1 = self.gcu_1(data.pos, data.tpl_edge_index, data.geo_edge_index)
        x_2 = self.gcu_2(x_1, data.tpl_edge_index, data.geo_edge_index)
        x_3 = self.gcu_3(x_2, data.tpl_edge_index, data.geo_edge_index)
        x_4 = self.mlp_glb(torch.cat([x_1, x_2, x_3], dim=1))
        x_global, _ = scatter_max(x_4, data.batch, dim=0)
        return x_global


class JointEncoder(torch.nn.Module):
    def __init__(self):
        super(JointEncoder, self).__init__()
        '''self.mlp_1 = MLP([4, 64])
        self.mlp_2 = MLP([64, 128, 1024])
        self.mlp_3 = MLP([1088, 512, 256, 128, 64])'''
        self.sa1_joint = SAModule(0.999, 0.4, MLP([4, 64, 64, 128]))
        self.sa2_joint = SAModule(0.33, 0.6, MLP([128 + 3, 128, 128, 256]))
        self.sa3_joint = GlobalSAModule(MLP([256 + 3, 256, 256, 512]))
        self.fp3_joint = FPModule(1, MLP([512 + 256, 256, 256]))
        self.fp2_joint = FPModule(3, MLP([256 + 128, 128, 128]))
        self.fp1_joint = FPModule(3, MLP([128 + 1, 128, 128]))

    def forward(self, x, pos, batch):
        '''x1= self.mlp_1(torch.cat((pos, x), dim=1))
        x2 = self.mlp_2(x1)
        x_glb, _ = scatter_max(x2, batch, dim=0)
        x_glb = torch.repeat_interleave(x_glb, torch.bincount(batch), dim=0)
        x3 = self.mlp_3(torch.cat((x_glb, x1), dim=1))
        return x3'''
        sa0_joint = (x, pos, batch)
        sa1_joint = self.sa1_joint(*sa0_joint)
        sa2_joint = self.sa2_joint(*sa1_joint)
        sa3_joint = self.sa3_joint(*sa2_joint)
        fp3_joint = self.fp3_joint(*sa3_joint, *sa2_joint)
        fp2_joint = self.fp2_joint(*fp3_joint, *sa1_joint)
        x_joint, _, _ = self.fp1_joint(*fp2_joint, *sa0_joint)
        return x_joint


class ROOTNET(torch.nn.Module):
    def __init__(self):
        super(ROOTNET, self).__init__()
        self.shape_encoder = ShapeEncoder()
        self.joint_encoder = JointEncoder()
        self.back_layers = Sequential(MLP([128 + 128, 200, 64]), Linear(64, 1))

    def forward(self, data, shuffle=True):
        joints_norepeat = []
        joints_batch = []
        joints_label = []
        joints = data.y
        for i in range(len(torch.unique(data.batch))):
            joint_sample = joints[data.batch == i, :]
            joint_sample = joint_sample[:data.num_joint[i], :]
            # create ground-truth label. The first one is always the root
            joint_label = joint_sample.new(torch.Size((joint_sample.shape[0], 1))).zero_()
            joint_label[0, 0] = 1
            # random shuffle
            if shuffle:
                idx = torch.randperm(joint_label.nelement())
                joint_label = joint_label[idx]
                joint_sample = joint_sample[idx]
            # add to the batch
            joints_norepeat.append(joint_sample)
            joints_label.append(joint_label)
            joints_batch.append(data.batch.new_full((data.num_joint[i],), i))
        joints_norepeat = torch.cat(joints_norepeat, dim=0)
        joints_batch = torch.cat(joints_batch)
        joints_label = torch.cat(joints_label)

        x_glb_shape = self.shape_encoder(data)
        shape_feature = torch.repeat_interleave(x_glb_shape, torch.bincount(joints_batch), dim=0)
        joint_feature = self.joint_encoder(torch.abs(joints_norepeat[:, 0:1]), joints_norepeat, joints_batch)
        x_joint = torch.cat([shape_feature, joint_feature], dim=1)
        x_joint = self.back_layers(x_joint)
        return x_joint, joints_label
