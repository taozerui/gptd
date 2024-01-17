import math
import torch
from numbers import Real


def safe_log(x, eps=1e-20):
    return torch.log(x + eps)


def logit_link(x):
    return 1 / (1 + torch.exp(- x))


def gaussian_repar(mu, sigma):
    noise = torch.randn_like(mu) * sigma
    return mu + noise


def multivariate_gaussian_repar(mu, low_rank_l, sigma):
    assert mu.ndim == 2
    noise1 = torch.einsum('ni, i-> ni', torch.randn_like(mu), sigma)
    noise2 = torch.einsum('ni, ik-> nk', torch.randn_like(mu), low_rank_l)
    return mu + noise1 + noise2


def gaussian_log_prob(x, x_mu, x_sigma):
    if isinstance(x_sigma, Real):
        log_x_sigma = math.log(x_sigma)
    else:
        log_x_sigma = torch.log(x_sigma)

    log_prob = - 0.5 * math.log(2 * math.pi) - log_x_sigma \
        - 0.5 * (torch.pow(x - x_mu, 2) / x_sigma ** 2)
    return log_prob


def nb_log_prob(x, p, zeta):
    pass


def probit_function(x):
    return 0.5 * (1. + torch.special.erf(x / math.sqrt(2)))
