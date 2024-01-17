import torch
from torch import nn
import math
from typing import Optional


def pairwise_distance(Xa, Xb):
    """
    Reference: https://www.maximerobeyns.com/fragments/matdist
    """
    assert Xa.size(1) == Xb.size(1)  # ensure D matches
    return Xa.pow(2).sum(1)[:, None] + Xb.pow(2).sum(1) - 2*Xa @ Xb.T


def pairwise_inner(Xa, Xb):
    assert Xa.size(1) == Xb.size(1)  # ensure D matches
    Xa_exp = Xa.unsqueeze(1)
    Xb_exp = Xb.unsqueeze(0)
    inner = (Xa_exp * Xb_exp).sum(-1)
    return inner


def pairwise_distance_weight(Xa, Xb, bw):
    """
    Reference: https://www.maximerobeyns.com/fragments/matdist
    """
    assert Xa.size(1) == Xb.size(1)  # ensure D matches

    Xa_exp = Xa.unsqueeze(1)
    Xb_exp = Xb.unsqueeze(0)
    dist = (Xa_exp - Xb_exp) * (Xa_exp - Xb_exp) / bw.view(1, 1, -1)
    return dist.sum(-1)


def median(tensor):
    tensor = tensor.flatten().sort()[0]
    length = tensor.shape[0]

    if length % 2 == 0:
        szh = length // 2
        kth = [szh - 1, szh]
    else:
        kth = [(length - 1) // 2]
    return tensor[kth].mean()


class Kernel(nn.Module):
    """docstring for Kernel."""
    def __init__(self):
        super(Kernel, self).__init__()


class RBF(Kernel):
    """docstring for RBF."""
    def __init__(self, band_width: Optional[float] = None):
        super(RBF, self).__init__()
        if band_width is None:
            self.register_buffer('band_width', None)
        else:
            self.band_width = nn.Parameter(
                torch.tensor(band_width), requires_grad=False)

    def forward(self, x, y=None):
        if y is None:
            pdist = pairwise_distance(x, x)
        else:
            pdist = pairwise_distance(x, y)
        if self.band_width is None:  # use median trick
            sigma = median(pdist.detach()) / (
                2 * torch.tensor(math.log(x.size(0) + 1)))
        else:
            sigma = self.band_width ** 2
        kxy = torch.exp(- pdist / sigma / 2.)
        return kxy


class ARD(Kernel):
    """docstring for ARD."""
    def __init__(self, feature_len: int):
        super(ARD, self).__init__()
        self.feture_len = feature_len
        self.log_band_width = nn.Parameter(
            torch.empty(feature_len), requires_grad=True)
        torch.nn.init.normal_(self.log_band_width.data, 0., .1)

    def forward(self, x, y=None):

        bw = torch.exp(self.log_band_width)

        if y is None:
            pdist = pairwise_distance_weight(x, x, bw)
        else:
            pdist = pairwise_distance_weight(x, y, bw)
        kxy = torch.exp(- 0.5 * pdist)
        return kxy


class Mix(Kernel):
    """docstring for Mix."""
    def __init__(
        self,
        band_width: Optional[float] = None,
        r: float = 0.1,
        c: float = 0.0
    ):
        super(Mix, self).__init__()
        if band_width is None:
            self.register_buffer('band_width', None)
        else:
            self.band_width = nn.Parameter(
                torch.tensor(band_width), requires_grad=False)

        self.r = r
        self.c = c

    def forward(self, x, y=None):
        if y is None:
            pdist = pairwise_distance(x, x)
            inner = pairwise_inner(x, x)
        else:
            pdist = pairwise_distance(x, y)
            inner = pairwise_inner(x, y)

        if self.band_width is None:  # use median trick
            sigma = median(pdist.detach()) / (
                2 * torch.tensor(math.log(x.size(0) + 1)))
        else:
            sigma = self.band_width ** 2

        kxy = torch.exp(- pdist / sigma / 2.) + self.r * (inner + self.c) ** 2
        return kxy


if __name__ == "__main__":
    rbf = RBF(0.1)
    x = torch.randn(128, 5)
    y = torch.randn(64, 5)
    kxy = rbf(x, y)
