#-------------------------------------------------------------------------------
# Name:        SKINNING.py
# Purpose:     definition of skinning prediction module.
# RigNet Copyright 2020 University of Massachusetts
# RigNet is made available under General Public License Version 3 (GPLv3), or under a Commercial License.
# Please see the LICENSE README.txt file in the main directory for more information and instruction on using and licensing RigNet.
#-------------------------------------------------------------------------------
import numpy as np
import torch
from torch_scatter import scatter_max
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN, Dropout
from models.gcn_basic_modules import GCU, MLP

__all__ = ['SKINNET', 'skinnet']


class SKINNET(torch.nn.Module):
    def __init__(self, nearest_bone, use_Dg, use_Lf, aggr='max'):
        super(SKINNET, self).__init__()
        self.num_nearest_bone = nearest_bone
        self.use_Dg = use_Dg
        self.use_Lf = use_Lf
        if self.use_Dg and self.use_Lf:
            input_dim = 3 + self.num_nearest_bone * 8
        elif self.use_Dg and not self.use_Lf:
            input_dim = 3 + self.num_nearest_bone * 7
        elif self.use_Lf and not self.use_Dg:
            input_dim = 3 + self.num_nearest_bone * 7
        else:
            input_dim = 3 + self.num_nearest_bone * 6
        self.multi_layer_tranform1 = MLP([input_dim, 128, 64])
        self.gcu1 = GCU(in_channels=64, out_channels=512, aggr=aggr)
        self.gcu2 = GCU(in_channels=512, out_channels=256, aggr=aggr)
        self.gcu3 = GCU(in_channels=256, out_channels=256, aggr=aggr)
        self.multi_layer_tranform2 = MLP([512, 512, 1024])

        self.cls_branch = Seq(Lin(1024 + 256, 1024), ReLU(), BN(1024), Lin(1024, 512), ReLU(), BN(512),
                              Lin(512, self.num_nearest_bone))

    def forward(self, data):
        samples = data.skin_input
        if self.use_Dg and self.use_Lf:
            samples = samples[:, 0: 8 * self.num_nearest_bone]
        elif self.use_Dg and not self.use_Lf:
            samples = samples[:, np.arange(samples.shape[1]) % 8 != 7]
            samples = samples[:, 0: 7 * self.num_nearest_bone]
        elif self.use_Lf and not self.use_Dg:
            samples = samples[:, np.arange(samples.shape[1]) % 8 != 6]
            samples = samples[:, 0: 7 * self.num_nearest_bone]
        else:
            samples = samples[:, np.arange(samples.shape[1]) % 8 != 7]
            samples = samples[:, np.arange(samples.shape[1]) % 7 != 6]
            samples = samples[:, 0: 6 * self.num_nearest_bone]

        raw_input = torch.cat([data.pos, samples], dim=1)

        x_0 = self.multi_layer_tranform1(raw_input)
        x_1 = self.gcu1(x_0, data.tpl_edge_index, data.geo_edge_index)

        x_global = self.multi_layer_tranform2(x_1)
        x_global, _ = scatter_max(x_global, data.batch, dim=0)

        x_2 = self.gcu2(x_1, data.tpl_edge_index, data.geo_edge_index)
        x_3 = self.gcu3(x_2, data.tpl_edge_index, data.geo_edge_index)
        x_global = torch.repeat_interleave(x_global, torch.bincount(data.batch), dim=0)
        x_4 = torch.cat([x_3, x_global], dim=1)

        skin_cls_pred = self.cls_branch(x_4)
        return skin_cls_pred


def skinnet(nearest_bone, use_Dg, use_Lf):
    model = SKINNET(nearest_bone=nearest_bone, use_Dg=use_Dg, use_Lf=use_Lf)
    return model
