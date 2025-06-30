import torch

from .utils import unmask

def chamfer_loss(output_set, output_mask, target_set, target_mask):
    sizes = (~output_mask).long().sum(dim=1).tolist()
    out = output_set.flatten(0, 1)  # [B * N, C]
    out_mask = output_mask.flatten()  # [B * N]
    tgt = target_set.flatten(0, 1)  # [B * N, C]
    tgt_mask = target_mask.flatten()  # [B * N]
    out = out[~out_mask, :]  # [M, C]
    tgt = tgt[~tgt_mask, :]  # [M, C]
    outs = out.split(sizes, 0)
    tgts = tgt.split(sizes, 0)
    cd = list()
    for o, t in zip(outs, tgts):  # [m, C]
        o_ = o.unsqueeze(1).repeat(1, t.size(0), 1)  # [m, m, C]
        t_ = t.unsqueeze(0).repeat(o.size(0), 1, 1)  # [m, m, C]
        l2 = (o_ - t_).pow(2).sum(dim=-1)  # [m, m]
        tdist = l2.min(0)[0].sum()  # min over outputs
        odist = l2.min(1)[0].sum()  # min over targets
        cd.append(odist + tdist)
    loss = sum(cd) / float(len(cd))  # batch average
    return loss
