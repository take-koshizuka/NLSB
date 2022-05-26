import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint_adjoint as odeint

class ConcatSquashLinear(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(ConcatSquashLinear, self).__init__()
        self.layer = nn.Linear(dim_in, dim_out)
        self.hyper_bias = nn.Linear(1, dim_out, bias=False)
        self.hyper_gate = nn.Linear(1, dim_out)

    def forward(self, t, x):
        out = self.layer(x) * torch.sigmoid(self.hyper_gate(t.view(1, 1))) \
            + self.hyper_bias(t.view(1, 1))
        return out

class ODEnet(nn.Module):
    """
    Helper class to make neural nets for use in continuous normalizing flows
    """
    def __init__(self, input_dim, hidden_dims, num_squeeze=0):
        super(ODEnet, self).__init__()
        self.num_squeeze = num_squeeze
        # build layers and add them
        layers = []
        activation_fns = []

        for i, out_dim in enumerate(hidden_dims):
            if i == 0:
                layer = ConcatSquashLinear(input_dim, out_dim)
                layers.append(layer)
            else:
                layer = ConcatSquashLinear(hidden_dims[i-1], out_dim)
                layers.append(layer)
            activation_fns.append(nn.Tanh())
        layers.append(ConcatSquashLinear(hidden_dims[-1], input_dim))

        self.layers = nn.ModuleList(layers)
        self.activation_fns = nn.ModuleList(activation_fns)

    def forward(self, t, dx):
        for l, layer in enumerate(self.layers):
            dx = layer(t, dx)
            # if not last layer, use nonlinearity
            if l < len(self.layers) - 1:
                dx = self.activation_fns[l](dx)
        return dx

def divergence_bf(dx, y, **unused_kwargs):
    sum_diag = 0.
#     print(dx.shape, dx.requires_grad, y.shape, y.requires_grad)
    for i in range(y.shape[1]):
        sum_diag += torch.autograd.grad(dx[:, i].sum(), y, create_graph=True)[0].contiguous()[:, i].contiguous()
    return sum_diag.contiguous()

def divergence_approx(f, y, e=None):
    e_dzdx = torch.autograd.grad(f, y, e, create_graph=True)
    e_dzdx_e = e_dzdx[0] * e
    approx_tr_dzdx = e_dzdx_e.view(y.shape[0], -1).sum(dim=1)
    return approx_tr_dzdx

def sample_rademacher_like(y):
    return torch.randint(low=0, high=2, size=y.shape).to(y) * 2 - 1

def sample_gaussian_like(y):
    return torch.randn_like(y)

class ODEfunc(nn.Module):
    def __init__(self, diffeq, input_dim, residual=False, rademacher=False):
        super(ODEfunc, self).__init__()
        self.diffeq = diffeq
        self.residual = residual
        self.rademacher = rademacher
        self.input_dim = input_dim
        self._e = None
        self.register_buffer("_num_evals", torch.tensor(0.))

    def initialize(self, e=None):
        self._e = e
        self._num_evals.fill_(0)

    def forward(self, t, x_aug):
        x = x_aug[:, :self.input_dim]
        self._num_evals += 1
        # convert to tensor
        # t = torch.tensor(t).type_as(y)
        batchsize = x.shape[0]
        # Sample and fix the noise.
        if self._e is None:
            if self.rademacher:
                self._e = sample_rademacher_like(x)
            else:
                self._e = sample_gaussian_like(x)
        
        with torch.set_grad_enabled(True):
            x.requires_grad_(True)
            t.requires_grad_(True)
            dx = self.diffeq(t, x)
            dl = divergence_bf(dx, x, e=self._e).view(batchsize, 1)
        
        dv = 0.5 * torch.sum(torch.pow(dx, 2), 1, keepdims=True)
        return torch.cat((dx, dl, dv) , 1)

class TrajectoryNet(nn.Module):
    def __init__(self, input_dim, int_t_to_noise, time_scale, diffeq_cfg, odefunc_cfg, criterion_cfg, solver_cfg) -> None:
        super(TrajectoryNet, self).__init__()
        self.int_t_to_noise = int_t_to_noise
        self.time_scale = time_scale
        self.diffeq = ODEnet(input_dim, **diffeq_cfg)
        self.odefunc = ODEfunc(self.diffeq, input_dim, **odefunc_cfg)
        self.criterion_cfg = criterion_cfg
        self.solver_cfg = solver_cfg

    def forward(self, ts, x0):
        # xT -> x0 -> z(t=0)
        z = F.pad(x0, (0, 2, 0, 0), value=0)

        self.odefunc.initialize()

        atol = self.solver_cfg['atol']
        rtol = self.solver_cfg['rtol']

        zs = odeint(
            self.odefunc,
            z,
            torch.tensor(ts, device=z.device),
            atol=[atol, atol] + [1e20]  if self.solver_cfg['method'] == 'dopri5' else atol,
            rtol=[rtol, rtol] + [1e20]  if self.solver_cfg['method'] == 'dopri5' else rtol,
            method=self.solver_cfg['method']
        )

        # ASSUME all examples are equally weighted
        xs = zs[:, :, :-2]
        log_det = zs[-1, :, -2]
        loss_L  = torch.abs(zs[-1, :, -1])
        # shape(xs) = (t_size, batch_size, dim)
        return dict(xs=xs, log_det=log_det, loss_L=loss_L, loss_R=torch.zeros_like(loss_L))
        
    def criterion(self, batch_size, base_nll, log_det, losses_L, losses_R, alpha_L, alpha_R):
        nll = base_nll - log_det
        base_nll = nll[:-batch_size]
        loss_D = torch.mean(nll[-batch_size:])
        loss_L = torch.mean(losses_L)
        loss = (loss_D + alpha_L * loss_L)
        return dict(loss=loss, nll=base_nll, loss_D=loss_D, loss_L=loss_L, loss_R=torch.zeros_like(loss_L))

    def parameters_lr(self):
        return self.parameters()
    
    