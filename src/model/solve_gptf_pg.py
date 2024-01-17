import math
import torch
from torch import nn, einsum
from torch.nn import init
from torch.distributions import NegativeBinomial, Bernoulli
from typing import List, Optional

from .kernels import Kernel
from .utils import logit_link, safe_log

INIT_MEAN_MEAN = 0.
INIT_MEAN_SIGMA = 0.5


def torch_solve(x, y):
    return torch.linalg.solve_triangular(x, y, upper=False)


class SolveGPTFPolyaGamma(nn.Module):
    def __init__(
        self,
        tensor_shape: List[int],
        data_num: int,
        data_type: str,
        kernel: Kernel,
        rank: int,
        num_inducing: int,
        prior_precision: float = 1.0,
        zeta: Optional[float] = None,
        init_lr: float = 1.0,
        adapt_lr: bool = True,
        n_mc: int = 10,
    ):
        super(SolveGPTFPolyaGamma, self).__init__()
        self.tensor_shape = list(tensor_shape)
        self.dim = len(tensor_shape)
        self.data_num = data_num
        self.data_type = data_type
        self.kernel = kernel
        self.rank = rank
        self.num_inducing = num_inducing
        self.prior_precision = prior_precision
        self.lr = init_lr
        self.adapt_lr = adapt_lr
        self.n_mc = n_mc
        self.mc_count = 0

        self._init_latent_factor()
        self._init_inducing_points()
        self._init_inducing_inputs()
        if adapt_lr:
            self._init_ng_state()
        else:
            self.register_parameter('ng_state', None)

        if data_type == 'count':
            self.zeta = nn.Parameter(torch.tensor(zeta), requires_grad=False)
        else:
            self.register_parameter('zeta', None)

    def _init_latent_factor(self):
        latent = []
        for _, s in enumerate(self.tensor_shape):
            mean_cache = nn.Parameter(torch.empty(s, self.rank))
            init.normal_(mean_cache.data, INIT_MEAN_MEAN, INIT_MEAN_SIGMA)
            latent.append(mean_cache)

        self.latent = nn.ParameterList(latent)

    def _init_inducing_points(self):
        inducing_points = {}
        for points in ['u', 'v']:
            mean = nn.Parameter(torch.empty(self.num_inducing), requires_grad=False)
            init.normal_(mean.data, INIT_MEAN_MEAN, INIT_MEAN_SIGMA)

            sigma = nn.Parameter(torch.empty(self.num_inducing, self.num_inducing), requires_grad=False)
            l = torch.randn(self.num_inducing, self.num_inducing)
            # sigma.data = torch.eye(self.num_inducing) + l.tril() @ l.tril().T * 1.
            # sigma.data = l.tril() @ l.tril().T * 0.2
            sigma.data = torch.eye(self.num_inducing)

            inducing_points[f'mean_{points}'] = mean
            inducing_points[f'sigma_{points}'] = sigma
        self.inducing_points = nn.ParameterDict(inducing_points)

    def _init_inducing_inputs(self):
        inducing_inputs = {}
        dim = self.dim * self.rank
        for points in ['u', 'v']:
            mean = nn.Parameter(torch.empty(self.num_inducing, dim))
            init.normal_(mean.data, INIT_MEAN_MEAN, INIT_MEAN_SIGMA)
            inducing_inputs[points] = mean

        self.inducing_inputs = nn.ParameterDict(inducing_inputs)
    
    def _init_ng_state(self):
        eta_g = nn.Parameter(torch.empty(2 * self.num_inducing + 2 * self.num_inducing ** 2), requires_grad=False)
        init.zeros_(eta_g.data)
        eta_h = nn.Parameter(torch.zeros(1), requires_grad=False)
        eta_tau = nn.Parameter(torch.ones(1) * self.n_mc, requires_grad=False)

        ng_state = {
            'eta_g': eta_g, 'eta_h': eta_h, 'eta_tau': eta_tau,
        }
        self.ng_state = nn.ParameterDict(ng_state)

    def adaptive_learning_rate(self, grad_u_eta1, grad_u_eta2, grad_v_eta1, grad_v_eta2):
        grad_eta = torch.cat([
            grad_u_eta1.reshape(-1), grad_u_eta2.reshape(-1),
            grad_v_eta1.reshape(-1), grad_v_eta2.reshape(-1)
        ])
        if self.n_mc > 0:
            self.ng_state['eta_g'].data += grad_eta
            self.ng_state['eta_h'].data += grad_eta.pow(2).sum()
            self.mc_count += 1
        else:
            if self.mc_count > 0:
                self.ng_state['eta_g'].data /= self.mc_count
                self.ng_state['eta_h'].data /= self.mc_count
                self.mc_count = 0
            self.ng_state['eta_g'].data = \
                (1 - 1 / self.ng_state['eta_tau']) * self.ng_state['eta_g'] + 1 / self.ng_state['eta_tau'] * grad_eta
            self.ng_state['eta_h'].data = \
                (1 - 1 / self.ng_state['eta_tau']) * self.ng_state['eta_h'] + 1 / self.ng_state['eta_tau'] * grad_eta.pow(2).sum()
            rho = self.ng_state['eta_g'].pow(2).sum() / self.ng_state['eta_h']
            self.lr = rho.item()
            self.ng_state['eta_tau'].data = self.ng_state['eta_tau'] * (1 - rho) + 1
        self.n_mc -= 1

    def forward(self, idx, x=None, predict=False):
        z = []
        for d in range(self.dim):
            z.append(self.latent[d][idx[:, d]])
        z_ten = torch.cat(z, -1)

        knn = self.kernel(z_ten)

        # do not need to be updated
        kuu = self.kernel(self.inducing_inputs['u'])
        kvv = self.kernel(self.inducing_inputs['v'])
        kuv = self.kernel(self.inducing_inputs['u'], self.inducing_inputs['v'])
        kvf = self.kernel(self.inducing_inputs['v'], z_ten)
        kuf = self.kernel(self.inducing_inputs['u'], z_ten)
        Lu = torch.linalg.cholesky(kuu)
        Lv_k = torch.linalg.cholesky(kvv)
        kuu_inv = torch.cholesky_inverse(Lu)
        kvv_inv = torch.cholesky_inverse(Lv_k)
        Lu_inv_kuv = torch_solve(Lu, kuv)
        Lu_inv_kuf = torch_solve(Lu, kuf)
        Cvv = kvv - torch.matmul(Lu_inv_kuv.T, Lu_inv_kuv)
        cvf = kvf - torch.matmul(Lu_inv_kuv.T, Lu_inv_kuf)
        Lv = torch.linalg.cholesky(Cvv)
        Lv_inv_cvf = torch_solve(Lv, cvf)
        Cvv_inv = torch.cholesky_inverse(Lv)

        kappa_u = torch_solve(Lu, Lu_inv_kuf)
        kappa_v = torch_solve(Lv, Lv_inv_cvf)

        mu_fv = einsum('in, i-> n', kappa_v, self.inducing_points['mean_v'])
        mu_fu = einsum('in, i-> n', kappa_u, self.inducing_points['mean_u'])
        f = mu_fv + mu_fu
        y_hat = 1. / (1 + torch.exp(- f))
        if predict:
            if x is None:
                logp = None
            else:
                x = x.to(dtype=y_hat.dtype)
                if self.data_type == 'binary':
                    logp = Bernoulli(probs=y_hat).log_prob(x)
                else:
                    logp = NegativeBinomial(
                        total_count=self.zeta, probs=y_hat).log_prob(x)
                    y_hat = self.zeta * torch.exp(f)
            return y_hat, logp
        assert x is not None
        x = x.to(dtype=f.dtype)
        if self.data_type == 'binary':
            chi = x - 0.5
            pg_b = torch.ones_like(x[0])
        else:
            assert self.data_type == 'count'
            chi = (x - self.zeta) / 2
            pg_b = x + self.zeta
        batch_size = x.shape[0]

        k_tilde = knn - torch.matmul(Lu_inv_kuf.T, Lu_inv_kuf)
        cvv_inv_cvf = torch_solve(Lv, Lv_inv_cvf)
        sigma_fv = k_tilde + einsum(
            'ni, ij, jm-> nm', cvv_inv_cvf.T, self.inducing_points['sigma_v'], cvv_inv_cvf
        ) - torch.matmul(Lv_inv_cvf.T, Lv_inv_cvf)

        # update pg variable
        c_ = (mu_fv + mu_fu) ** 2 + torch.diag(sigma_fv) + \
            einsum('in, ij, jn-> n', kappa_u, self.inducing_points['sigma_u'], kappa_u)
        c = torch.sqrt(c_)
        theta = (pg_b / (2 * c)) * torch.tanh(c / 2)

        # natural gradient update of \mu and \Sigma
        scale = self.data_num / batch_size
        grad_cache = []
        for points in ['u', 'v']:
            if points == 'u':
                sigma_inv = torch.cholesky_inverse(Lu)
                mu = self.inducing_points['mean_u']
                kappa = kappa_u
                kinv = kuu_inv
            else:
                sigma_inv = torch.cholesky_inverse(Lv)
                mu = self.inducing_points['mean_v']
                kappa = kappa_v
                kinv = Cvv_inv
            eta1 = einsum('ij, j-> i', sigma_inv, mu)
            eta2 = - 0.5 * sigma_inv

            if points == 'u':
                sub_term = chi - theta * mu_fv
            else:
                sub_term = chi - theta * mu_fu
            delta_eta1 = scale * einsum('in, n-> i', kappa, sub_term) - eta1
            delta_eta2 = - 0.5 * (
                kinv + scale * einsum('in, n, jn-> ij', kappa, theta, kappa)) - eta2
            eta1 += self.lr * delta_eta1 / scale
            eta2 += self.lr * delta_eta2 / scale

            self.inducing_points[f'sigma_{points}'].data = torch.linalg.inv(-2 * eta2)
            self.inducing_points[f'mean_{points}'].data = einsum(
                'ni, i-> n', self.inducing_points[f'sigma_{points}'].data, eta1
            )
            grad_cache.append(eta1)
            grad_cache.append(eta2)

        if self.adapt_lr:
            self.adaptive_learning_rate(*grad_cache)

        # compute elbo
        mu_fv = einsum('in, i-> n', kappa_v, self.inducing_points['mean_v'])
        mu_fu = einsum('in, i-> n', kappa_u, self.inducing_points['mean_u'])
        sigma_fv = k_tilde + einsum(
            'ni, ij, jm-> nm', cvv_inv_cvf.T, self.inducing_points['sigma_v'], cvv_inv_cvf
        ) - torch.matmul(Lv_inv_cvf.T, Lv_inv_cvf)
        c_ = (mu_fv + mu_fu) ** 2 + torch.diag(sigma_fv) + \
            einsum('in, ij, jn-> n', kappa_u, self.inducing_points['sigma_u'], kappa_u)
        c = torch.sqrt(c_)
        theta = (pg_b / (2 * c)) * torch.tanh(c / 2)

        log_p_y = 2 * chi * (mu_fv + mu_fu) - \
            theta *  (mu_fv + mu_fu) ** 2 - \
            theta * torch.diag(sigma_fv) - \
            theta * einsum('in, ij, jn-> n', kappa_u, self.inducing_points['sigma_u'], kappa_u) + \
            c ** 2 * theta - 2 * safe_log(torch.cosh(c / 2)) * pg_b

        kl_div = {}
        for points in ['u', 'v']:
            if points == 'u':
                kmm = kuu
                kmm_inv = kuu_inv
                sigma = self.inducing_points['sigma_u']
                mu = self.inducing_points['mean_u']
            else:
                kmm = kvv
                kmm_inv = kvv_inv
                sigma = self.inducing_points['sigma_v']
                mu = self.inducing_points['mean_v']
            kl_div[points] = torch.logdet(kmm) - torch.logdet(sigma) + \
                torch.trace(torch.matmul(sigma, kmm_inv)) + \
                torch.einsum('i, ik, k->', mu, kmm_inv, mu)
        latent_prior = 0
        for d in range(self.dim):
            latent_prior = latent_prior + self.latent[d].pow(2).sum()

        elbo = log_p_y.mean() - (kl_div['u'] + kl_div['v'] + self.prior_precision * latent_prior) / self.data_num
        return elbo
