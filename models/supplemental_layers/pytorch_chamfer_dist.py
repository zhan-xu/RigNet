import torch
from torch_scatter import scatter_mean


def chamfer_distance_with_average(p1, p2):

    '''
    Calculate Chamfer Distance between two point sets
    :param p1: size[1, N, D]
    :param p2: size[1, M, D]
    :param debug: whether need to output debug info
    :return: sum of Chamfer Distance of two point sets
    '''

    assert p1.size(0) == 1 and p2.size(0) == 1
    assert p1.size(2) == p2.size(2)
    p1 = p1.repeat(p2.size(1), 1, 1)
    p1 = p1.transpose(0, 1)
    p2 = p2.repeat(p1.size(0), 1, 1)
    dist = torch.add(p1, torch.neg(p2))
    dist_norm = torch.norm(dist, 2, dim=2)
    dist1 = torch.min(dist_norm, dim=1)[0]
    dist2 = torch.min(dist_norm, dim=0)[0]
    loss = 0.5 * ((torch.mean(dist1)) + (torch.mean(dist2)))
    return loss

