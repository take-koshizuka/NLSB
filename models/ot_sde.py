from models.potential import Entropy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsde
import geomloss
import copy
from models.lagrangian import NullLagrangian
from mfg import opinion_lib

from tqdm import tqdm
import numpy as np
import math


def mahalanobis(u, v, cov):
    B, N, D = u.size()
    _, M, _ = v.size()
    u_ = u.repeat(M, 1, 1, 1).reshape(B, N, M, D)
    v_ = v.repeat(N, 1, 1, 1).reshape(B, N, M, D)
    delta = u_ - v_
    X = torch.matmul(delta, cov)
    m = torch.sum(delta * X, dim=-1)
    return m

def antiderivTanh(x): # activation function aka the antiderivative of tanh
    return torch.abs(x) + torch.log(1+torch.exp(-2.0*torch.abs(x)))

def derivTanh(x): # act'' aka the second derivative of the activation function antiderivTanh
    return 1 - torch.pow( torch.tanh(x) , 2 )

class ResNN(nn.Module):
    def __init__(self, d, m, nTh=2):
        """
            ResNet N portion of Phi
        :param d:   int, dimension of space input (expect inputs to be d+1 for space-time)
        :param m:   int, hidden dimension
        :param nTh: int, number of resNet layers , (number of theta layers)
        """
        super().__init__()

        if nTh < 2:
            print("nTh must be an integer >= 2")
            exit(1)

        self.input_dim = d
        self.m = m
        self.nTh = nTh
        self.layers = nn.ModuleList([])
        self.layers.append(nn.Linear(d, m, bias=True)) # opening layer
        self.layers.append(nn.Linear(m,m, bias=True)) # resnet layers
        for i in range(nTh-2):
            self.layers.append(copy.deepcopy(self.layers[1]))
        self.act = antiderivTanh
        self.h = 1.0 / (self.nTh-1) # step size for the ResNet

    def forward(self, x):
        """
            N(s;theta). the forward propogation of the ResNet
        :param x: tensor nex-by-d+1, inputs
        :return:  tensor nex-by-m,   outputs
        """

        x = self.act(self.layers[0].forward(x))

        for i in range(1,self.nTh):
            x = x + self.h * self.act(self.layers[i](x))

        return x

class Phi(nn.Module):
    def __init__(self, nTh, m, d, use_t=True, r=10):
        """
            neural network approximating Phi (see Eq. (9) in our paper)

            Phi( x,t ) = w'*ResNet( [x;t]) + 0.5*[x' t] * A'A * [x;t] + b'*[x;t] + c

        :param nTh:   int, number of resNet layers , (number of theta layers)
        :param m:     int, hidden dimension
        :param d:     int, dimension of space input (expect inputs to be d+1 for space-time)
        :param use_t: int, use time parameter or not
        :param r:    int, rank r for the A matrix
        """
        super().__init__()

        self.m    = m
        self.nTh  = nTh
        self.input_dim    = d
        self.use_t = use_t

        r = min(r,d+int(self.use_t)) # if number of dimensions is smaller than default r, use that

        self.A  = nn.Parameter(torch.zeros(r, d+int(self.use_t)) , requires_grad=True)
        self.A  = nn.init.xavier_uniform_(self.A)
        self.c  = nn.Linear( d+int(self.use_t), 1  , bias=True)  # b'*[x;t] + c
        self.w  = nn.Linear( m    , 1  , bias=False)

        self.N = ResNN(d+int(self.use_t), m, nTh=nTh)

        # set initial values
        self.w.weight.data = torch.ones(self.w.weight.data.shape)
        self.c.weight.data = torch.zeros(self.c.weight.data.shape)
        self.c.bias.data   = torch.zeros(self.c.bias.data.shape)

    def forward(self, t, x):
        """ calculating Phi(s, theta)...not used in OT-Flow """
        if self.use_t:
            x = F.pad(x, (0, 1, 0, 0), value=t)
        # force A to be symmetric
        symA = torch.matmul(torch.t(self.A), self.A) # A'A

        return self.w( self.N(x)) + 0.5 * torch.sum( torch.matmul(x , symA) * x , dim=1, keepdims=True) + self.c(x)

    def grad(self, t, x):
        """
        compute gradient of Phi wrt x and trace(Hessian of Phi); see Eq. (11) and Eq. (13), respectively
        recomputes the forward propogation portions of Phi
        :param x: input data, torch Tensor nex-by-d
        :param justGrad: boolean, if True only return gradient, if False return (grad, trHess)
        :return: gradient , trace(hessian)    OR    just gradient
        """

        # code in E = eye(d+1,d) as index slicing instead of matrix multiplication
        # assumes specific N.act as the antiderivative of tanh
        if self.use_t:
            x = F.pad(x, (0, 1, 0, 0), value=t)

        N    = self.N
        symA = torch.matmul(self.A.t(), self.A)

        u = [] # hold the u_0,u_1,...,u_M for the forward pass
        z = N.nTh*[None] # hold the z_0,z_1,...,z_M for the backward pass
        # preallocate z because we will store in the backward pass and we want the indices to match the paper

        # Forward of ResNet N and fill u
        opening     = N.layers[0].forward(x) # K_0 * S + b_0
        u.append(N.act(opening)) # u0
        feat = u[0]

        for i in range(1,N.nTh):
            feat = feat + N.h * N.act(N.layers[i](feat))
            u.append(feat)

        # going to be used more than once
        tanhopen = torch.tanh(opening) # act'( K_0 * S + b_0 )

        # compute gradient and fill z
        for i in range(N.nTh-1,0,-1): # work backwards, placing z_i in appropriate spot
            if i == N.nTh-1:
                term = self.w.weight.t()
            else:
                term = z[i+1]

            # z_i = z_{i+1} + h K_i' diag(...) z_{i+1}
            z[i] = term + N.h * torch.mm( N.layers[i].weight.t() , torch.tanh( N.layers[i].forward(u[i-1]) ).t() * term)

        # z_0 = K_0' diag(...) z_1
        z[0] = torch.mm( N.layers[0].weight.t() , tanhopen.t() * z[1] )
        grad = z[0] + torch.mm(symA, x.t() ) + self.c.weight.t()
        return grad.t()

    def diagHess(self, t, x):
        """
        compute gradient of Phi wrt x and trace(Hessian of Phi); see Eq. (11) and Eq. (13), respectively
        recomputes the forward propogation portions of Phi
        :param x: input data, torch Tensor nex-by-d
        :param justGrad: boolean, if True only return gradient, if False return (grad, trHess)
        :return: gradient , trace(hessian)    OR    just gradient
        """

        # code in E = eye(d+1,d) as index slicing instead of matrix multiplication
        # assumes specific N.act as the antiderivative of tanh
        N    = self.N
        m    = N.layers[0].weight.shape[0]
        nex  = x.shape[0] # number of examples in the batch
        d    = x.shape[1]
        symA = torch.matmul(self.A.t(), self.A)

        u = [] # hold the u_0,u_1,...,u_M for the forward pass
        z = N.nTh*[None] # hold the z_0,z_1,...,z_M for the backward pass
        # preallocate z because we will store in the backward pass and we want the indices to match the paper

        if self.use_t:
            x = F.pad(x, (0, 1, 0, 0), value=t)

        # Forward of ResNet N and fill u
        opening     = N.layers[0].forward(x) # K_0 * S + b_0
        u.append(N.act(opening)) # u0
        feat = u[0]

        for i in range(1,N.nTh):
            feat = feat + N.h * N.act(N.layers[i](feat))
            u.append(feat)

        # going to be used more than once
        tanhopen = torch.tanh(opening) # act'( K_0 * S + b_0 )

        # compute gradient and fill z
        for i in range(N.nTh-1,0,-1): # work backwards, placing z_i in appropriate spot
            if i == N.nTh-1:
                term = self.w.weight.t()
            else:
                term = z[i+1]

            # z_i = z_{i+1} + h K_i' diag(...) z_{i+1}
            z[i] = term + N.h * torch.mm( N.layers[i].weight.t() , torch.tanh( N.layers[i].forward(u[i-1]) ).t() * term)

        # z_0 = K_0' diag(...) z_1
        z[0] = torch.mm( N.layers[0].weight.t() , tanhopen.t() * z[1] )
        grad = z[0] + torch.mm(symA, x.t() ) + self.c.weight.t()

        # -----------------
        # trace of Hessian
        #-----------------

        # t_0, the trace of the opening layer
        Kopen = N.layers[0].weight[:,0:d]    # indexed version of Kopen = torch.mm( N.layers[0].weight, E  )
        temp  = derivTanh(opening.t()) * z[1]
        x = temp.reshape(m, -1, nex) * torch.pow(Kopen.unsqueeze(2), 2)
        trH  = torch.sum(temp.reshape(m, -1, nex) * torch.pow(Kopen.unsqueeze(2), 2), dim=0) # trH = t_0

        # grad_s u_0 ^ T
        temp = tanhopen.t()   # act'( K_0 * S + b_0 )
        Jac  = Kopen.unsqueeze(2) * temp.unsqueeze(1) # K_0' * act'( K_0 * S + b_0 )
        # Jac is shape m by d by nex

        # t_i, trace of the resNet layers
        # KJ is the K_i^T * grad_s u_{i-1}^T
        for i in range(1,N.nTh):
            KJ  = torch.mm(N.layers[i].weight , Jac.reshape(m,-1) )
            KJ  = KJ.reshape(m,-1,nex)
            if i == N.nTh-1:
                term = self.w.weight.t()
            else:
                term = z[i+1]

            temp = N.layers[i].forward(u[i-1]).t() # (K_i * u_{i-1} + b_i)
            t_i = torch.sum(  ( derivTanh(temp) * term ).reshape(m,-1,nex)  *  torch.pow(KJ,2) ,  dim=0)
            trH  = trH + N.h * t_i  # add t_i to the accumulate trace
            Jac = Jac + N.h * torch.tanh(temp).reshape(m, -1, nex) * KJ # update Jacobian

        diagHess = trH + torch.diag(symA[0:d,0:d], 0).unsqueeze(1)
        return grad.t(), diagHess.view(nex, d)
        # indexed version of: return grad.t() ,  trH + torch.trace( torch.mm( E.t() , torch.mm(  symA , E) ) )

class LipSwish(torch.nn.Module):
    def forward(self, x):
        return 0.909 * torch.nn.functional.silu(x)

class MLP(torch.nn.Module):
    def __init__(self, input_dim, out_dim, hidden_dim, num_layers, tanh):
        super().__init__()

        model = [torch.nn.Linear(input_dim, hidden_dim), LipSwish()]
        for _ in range(num_layers - 1):
            model.append(torch.nn.Linear(hidden_dim, hidden_dim))
            ###################
            # LipSwish activations are useful to constrain the Lipschitz constant of the discriminator.
            # (For simplicity we additionally use them in the generator, but that's less important.)
            ###################
            model.append(LipSwish())
        model.append(torch.nn.Linear(hidden_dim, out_dim))
        if tanh:
            model.append(torch.nn.Tanh())
        self._model = torch.nn.Sequential(*model)

    def forward(self, x):
        return self._model(x)
    
###################
# Now we define the SDEs.
###################

class ForwardSDE(torchsde.SDEIto):
    def __init__(self, noise_type, sigma_type, input_dim, brownian_size, drift_cfg, diffusion_cfg, criterion_cfg, solver_cfg, lagrangian=NullLagrangian()):
        super(ForwardSDE, self).__init__(noise_type=noise_type)
        
        self.noise_type = noise_type
        self.sigma_type = sigma_type
        self.input_dim = input_dim
        self.brownian_size = brownian_size
        self.drift_cfg = drift_cfg
        self.criterion_cfg = criterion_cfg
        self.solver_cfg = solver_cfg

        if noise_type == "scalar":
            assert self.brownian_size == 1
        elif noise_type == "diagonal":
            assert self.brownian_size == self.input_dim
        
        self.phi = Phi(**drift_cfg, d=input_dim)

        if 'use_t' in drift_cfg:
            self.use_t_f = drift_cfg['use_t']
        else:
            self.use_t_f = True

        if 'use_t' in diffusion_cfg:
            self.use_t_g = diffusion_cfg['use_t']
        else:
            self.use_t_g = True

        print(self.use_t_f, self.use_t_g)
        if sigma_type == "const":
            if noise_type == "scalar":
                self.register_buffer('sigma', torch.as_tensor(diffusion_cfg['sigma']))
            elif noise_type == "diagonal":
                self.register_buffer('sigma', torch.as_tensor(diffusion_cfg['sigma']))

        elif sigma_type == "param":
            if noise_type == "scalar":
                self.sigma = nn.Parameter(torch.randn(1, self.input_dim), requires_grad=True)
            elif noise_type == "diagonal":
                self.sigma = nn.Parameter(torch.randn(1, self.input_dim), requires_grad=True)
            else:
                raise NotImplementedError
        else:
            if sigma_type == "MLP-1":
                cfg = dict(
                    input_dim=input_dim+int(self.use_t_g),
                    out_dim=1,
                    hidden_dim=diffusion_cfg['hidden_dim'],
                    num_layers=diffusion_cfg['num_layers'],
                    tanh=diffusion_cfg['tanh']
                )
            
            elif sigma_type == "MLP":
                if noise_type == "diagonal":
                    cfg = dict(
                        input_dim=input_dim+int(self.use_t_g),
                        out_dim=input_dim,
                        hidden_dim=diffusion_cfg['hidden_dim'],
                        num_layers=diffusion_cfg['num_layers'],
                        tanh=diffusion_cfg['tanh']
                    )
                elif noise_type == "general":
                    cfg = dict(
                        input_dim=input_dim+int(self.use_t_g),
                        out_dim=input_dim*brownian_size,
                        hidden_dim=diffusion_cfg['hidden_dim'],
                        num_layers=diffusion_cfg['num_layers'],
                        tanh=diffusion_cfg['tanh']
                    )
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError
        
            self.sigma = MLP(**cfg)

        self.criterion_cfg = criterion_cfg
        self.solver_cfg = solver_cfg
        self.lagrangian = lagrangian

        self.sdeint_fn = torchsde.sdeint_adjoint if solver_cfg['adjoint'] else torchsde.sdeint
        self.loss_fn = geomloss.SamplesLoss(loss='sinkhorn', p=criterion_cfg['p'], blur=criterion_cfg['blur'])

    def f(self, t, x):  # Approximate posterior drift.
        # xt = F.pad(x, (0, 1, 0, 0), value=t)
        gradPhi = self.phi.grad(t, x)
        f = self.lagrangian.inv_L(t, x, -gradPhi[:, :self.input_dim])
        return f

    def g(self, t, x):  # Shared diffusion.
        if self.use_t_g:
            x = F.pad(x, (0, 1, 0, 0), value=t)
        else:
            x = x
        if self.sigma_type == "const" or self.sigma_type ==  "param":
            if self.noise_type == "scalar":
                g = self.sigma.repeat(x.size(0), 1, 1)
            elif self.noise_type == "diagonal":
                g = self.sigma.repeat(x.size(0), self.input_dim)
            
        elif self.sigma_type == "MLP-1":
            if self.noise_type == "diagonal":
                g = self.sigma(x).repeat(1, self.input_dim)
            else:
                g = self.sigma(x).repeat(1, self.input_dim, self.brownian_size)

        elif self.sigma_type == "MLP":
            if self.noise_type == "diagonal":
                g = self.sigma(x).view(-1, self.input_dim)
            else:
                g = self.sigma(x).view(-1, self.input_dim, self.brownian_size)
        return g

    def f_aug(self, t, x_aug):
        batch_size = len(x_aug)
        x = x_aug[:, :self.input_dim]
        # xt = F.pad(x, (0, 1, 0, 0), value=t)
        if self.noise_type == "scalar" or self.noise_type == "diagonal":
            gradPhi, diagHess = self.phi.diagHess(t, x)
        else:
            gradPhi = self.phi.grad(t, x)
        
        # drift
        f = self.lagrangian.inv_L(t, x, -gradPhi[:, :self.input_dim])
        # cal dv
        dv = self.lagrangian.L(t, x, f)
        # cal dr
        
        g = self.g(t, x)
        if self.noise_type == "scalar":
            D = 0.5 * torch.pow(g.squeeze(2), 2)
            prod_Hess_D = torch.sum(diagHess * D, axis=1)
        elif self.noise_type == "diagonal":
            D = 0.5 * torch.pow(g, 2)
            prod_Hess_D = torch.sum(diagHess * D, axis=1)
        else:
            batch_idx = torch.arange(len(x))
            hessian = torch.autograd.functional.jacobian(lambda x: self.phi.grad(t, x), x)[batch_idx, :self.input_dim, batch_idx, :self.input_dim]
            D = 0.5 * torch.bmm(g, g.transpose(1, 2))
            prod_Hess_D = torch.sum(hessian * D, dim=(1, 2))
        
        h = dv + torch.sum(gradPhi[:, :self.input_dim] * f, dim=1, keepdims=True)

        if self.use_t_f:
            # L1norm
            dr = torch.abs(gradPhi[:,self.input_dim].unsqueeze(1) + prod_Hess_D.unsqueeze(1) + h)
            #dr = torch.abs(gradPhi[:,self.input_dim].unsqueeze(1) + prod_Hess_D.unsqueeze(1) + h) ** 2
        else:
            dr = torch.abs(prod_Hess_D.unsqueeze(1) + h)
            # dr = torch.abs(prod_Hess_D.unsqueeze(1) + h) ** 2
        
        x = torch.cat([f, dv, dr], dim=1)
        return x
    
    def g_aug(self, t, x_aug):
        g = self.g(t, x_aug[:, :self.input_dim])
        if self.noise_type == "diagonal":
            x = F.pad(g, (0, 2, 0, 0), value=0)
        else:
            x = F.pad(g, (0, 0, 0, 2, 0, 0), value=0)
        return x

    def fv_aug(self, t, x_aug):
        batch_size = len(x_aug)
        x = x_aug[:, :self.input_dim]
        v = x_aug[:, -self.input_dim:]
        # xt = F.pad(x, (0, 1, 0, 0), value=t)
        if self.noise_type == "scalar" or self.noise_type == "diagonal":
            gradPhi, diagHess = self.phi.diagHess(t, x)
        else:
            gradPhi = self.phi.grad(t, x)
        # drift
        f = self.lagrangian.inv_L(t, x, -gradPhi[:, :self.input_dim], v)
        # cal dv
        dv = self.lagrangian.L(t, x, f, v)
        
        # cal dr
        g = self.g(t, x)
        if self.noise_type == "scalar":
            D = 0.5 * torch.pow(g.squeeze(2), 2)
            prod_Hess_D = torch.sum(diagHess * D, axis=1)
        elif self.noise_type == "diagonal":
            D = 0.5 * torch.pow(g, 2)
            prod_Hess_D = torch.sum(diagHess * D, axis=1)
        else:
            batch_idx = torch.arange(len(x))
            hessian = torch.autograd.functional.jacobian(lambda x: self.phi.grad(t, x), x)[batch_idx, :self.input_dim, batch_idx, :self.input_dim]
            D = 0.5 * torch.bmm(g, g.transpose(1, 2))
            prod_Hess_D = torch.sum(hessian * D, dim=(1, 2))
        
        h = dv + torch.sum(gradPhi[:, :self.input_dim] * f, dim=1, keepdims=True)

        if self.use_t_f:
            # L1norm
            dr = torch.abs(gradPhi[:,self.input_dim].unsqueeze(1) + prod_Hess_D.unsqueeze(1) + h)
            #dr = torch.abs(gradPhi[:,self.input_dim].unsqueeze(1) + prod_Hess_D.unsqueeze(1) + h) ** 2
        else:
            dr = torch.abs(prod_Hess_D.unsqueeze(1) + h)
            #dr = torch.abs(prod_Hess_D.unsqueeze(1) + h) ** 2
        
        x = torch.cat([f, dv, dr], dim=1)
        if self.noise_type == "diagonal":
            x = F.pad(x, (0, self.input_dim, 0, 0), value=0)
        else:
            x = F.pad(x, (0, 0, 0, self.input_dim, 0, 0), value=0)
        return x

    def gv_aug(self, t, x_aug):
        g = self.g(t, x_aug[:, :self.input_dim])
        if self.noise_type == "diagonal":
            x = F.pad(g, (0, 2+self.input_dim, 0, 0), value=0)
        else:
            x = F.pad(g, (0, 0, 0, 2+self.input_dim, 0, 0), value=0)
        return x

    def forward(self, ts, x0, v0=None, batch_idx=None):
        aug_x0 = F.pad(x0, (0, 2, 0, 0), value=0)
        if not v0 is None:
            aug_x0 = torch.cat((aug_x0, v0), dim=1)
            t1_ = min(ts[0] + self.solver_cfg['dt'], ts[1] - 1e-4 )
            ts_ = torch.tensor([ ts[0], t1_])
            xs = self.sdeint_fn(self, aug_x0, ts_,
                method=self.solver_cfg['method'],
                dt=self.solver_cfg['dt'],
                adaptive=self.solver_cfg['adaptive'],
                names={'drift': 'fv_aug', 'diffusion': 'gv_aug'})

            aug_x0 = xs[-1, :, :self.input_dim+2]
            ts[0] = t1_
        
        else:
            t1_ = min(ts[0] + self.solver_cfg['dt'], ts[1] - 1e-4 )
            ts_ = torch.tensor([ ts[0], t1_])
            xs = self.sdeint_fn(self, aug_x0, ts_,
                method=self.solver_cfg['method'],
                dt=self.solver_cfg['dt'],
                adaptive=self.solver_cfg['adaptive'],
                names={'drift': 'f_aug', 'diffusion': 'g_aug'})

            aug_x0 = xs[-1, :, :self.input_dim+2]
            ts[0] = t1_

        # xT: (len(ts), batch_size, d+2)
        if self.solver_cfg['adjoint']:
            xs = self.sdeint_fn(self, aug_x0, ts, 
                method=self.solver_cfg['method'], 
                dt=self.solver_cfg['dt'],
                adaptive=self.solver_cfg['adaptive'], 
                adjoint_adaptive=self.solver_cfg['adjoint_adaptive'],
                adjoint_method=self.solver_cfg['adjoint_method'], 
                names={'drift': 'f_aug', 'diffusion': 'g_aug'})
        else:
            xs = self.sdeint_fn(self, aug_x0, ts, 
                method=self.solver_cfg['method'], 
                dt=self.solver_cfg['dt'],
                adaptive=self.solver_cfg['adaptive'],
                names={'drift': 'f_aug', 'diffusion': 'g_aug'})

        #with open('rep.pickle', 'wb') as f:
        #    pickle.dump([ T, VALUE], f)
        loss_L = xs[-1][:, self.input_dim]
        loss_R = xs[-1][:, self.input_dim+1]

        xs[0] = F.pad(x0, (0, 2, 0, 0), value=0)
        
        assert xs.size(0) == len(ts)
        return dict(xs=xs[:, :, :self.input_dim], loss_L=loss_L, loss_R=loss_R)

    def criterion(self, x, x_hat, loss_L, loss_R, alpha_D, alpha_L, alpha_R):
        loss_D = self.loss_fn(x_hat, x)
        loss = alpha_D * loss_D + alpha_L * loss_L + alpha_R * loss_R
        return dict(loss=loss, loss_D=loss_D, loss_L=loss_L, loss_R=loss_R)

    def parameters_lr(self):
        return self.parameters()

    def clamp_parameters(self):
        if self.sigma_type == "param":
            self.sigma.data.clamp_(-5.0, 5.0)


class ReverseSDE(torchsde.SDEIto):
    def __init__(self, fsde):
        super(ReverseSDE, self).__init__(noise_type=fsde.noise_type)
        self.noise_type = fsde.noise_type
        self.sigma_type = fsde.sigma_type
        self.input_dim = fsde.input_dim
        self.brownian_size = fsde.brownian_size

        self.fsde = fsde
        self.df = Phi(**fsde.drift_cfg, d=self.input_dim)

        self.criterion_cfg = fsde.criterion_cfg
        self.solver_cfg = fsde.solver_cfg

        self.sdeint_fn = torchsde.sdeint
        self.loss_fn = geomloss.SamplesLoss(loss='sinkhorn', p=self.criterion_cfg['p'], blur=self.criterion_cfg['blur'])

    # --- sdeint ---
    def f(self, t, y):
        df = -self.df.grad(-t, y)[:, :self.input_dim]
        with torch.no_grad():
            f = self.fsde.f(-t, y)
        out = -(f - df)
        return out

    def g(self, t, y):
        with torch.no_grad():
            out = -self.fsde.g(-t, y)
        return out

    def forward(self, ts, xT):
        assert ts[0] >= ts[1]
        rev_ts = -torch.tensor(ts)
        xs = self.sdeint_fn(self, xT, rev_ts, 
            method=self.solver_cfg['method'], 
            dt=self.solver_cfg['dt'],
            adaptive=self.solver_cfg['adaptive'],
            names={'drift': 'f', 'diffusion': 'g'})
        
        assert xs.size(0) == len(ts)
        return dict(xs=xs[:, :, :self.input_dim])

    def criterion(self, x, x_hat):
        loss_D = self.loss_fn(x_hat, x)
        return dict(loss=loss_D, loss_D=loss_D)

    def parameters_lr(self):
        return self.df.parameters()


class OpinionSDE(torchsde.SDEIto):
    def __init__(self, noise_type, sigma_type, input_dim, brownian_size, drift_cfg, diffusion_cfg, criterion_cfg, solver_cfg, lagrangian=NullLagrangian()):
        super(OpinionSDE, self).__init__(noise_type=noise_type)
        
        self.noise_type = noise_type
        self.sigma_type = sigma_type
        self.input_dim = input_dim
        self.brownian_size = brownian_size
        self.drift_cfg = drift_cfg
        self.criterion_cfg = criterion_cfg
        self.solver_cfg = solver_cfg

        if noise_type == "scalar":
            assert self.brownian_size == 1
        elif noise_type == "diagonal":
            assert self.brownian_size == self.input_dim
        
        self.phi = Phi(**drift_cfg, d=input_dim)

        if 'use_t' in drift_cfg:
            self.use_t_f = drift_cfg['use_t']
        else:
            self.use_t_f = True

        if 'use_t' in diffusion_cfg:
            self.use_t_g = diffusion_cfg['use_t']
        else:
            self.use_t_g = True

        print(self.use_t_f, self.use_t_g)
        if sigma_type == "const":
            if noise_type == "scalar":
                self.register_buffer('sigma', torch.as_tensor(diffusion_cfg['sigma']))
            elif noise_type == "diagonal":
                self.register_buffer('sigma', torch.as_tensor(diffusion_cfg['sigma']))

        elif sigma_type == "param":
            if noise_type == "scalar":
                self.sigma = nn.Parameter(torch.randn(1, self.input_dim), requires_grad=True)
            elif noise_type == "diagonal":
                self.sigma = nn.Parameter(torch.randn(1, self.input_dim), requires_grad=True)
            else:
                raise NotImplementedError
        else:
            if sigma_type == "MLP-1":
                cfg = dict(
                    input_dim=input_dim+int(self.use_t_g),
                    out_dim=1,
                    hidden_dim=diffusion_cfg['hidden_dim'],
                    num_layers=diffusion_cfg['num_layers'],
                    tanh=diffusion_cfg['tanh']
                )
            
            elif sigma_type == "MLP":
                if noise_type == "diagonal":
                    cfg = dict(
                        input_dim=input_dim+int(self.use_t_g),
                        out_dim=input_dim,
                        hidden_dim=diffusion_cfg['hidden_dim'],
                        num_layers=diffusion_cfg['num_layers'],
                        tanh=diffusion_cfg['tanh']
                    )
                elif noise_type == "general":
                    cfg = dict(
                        input_dim=input_dim+int(self.use_t_g),
                        out_dim=input_dim*brownian_size,
                        hidden_dim=diffusion_cfg['hidden_dim'],
                        num_layers=diffusion_cfg['num_layers'],
                        tanh=diffusion_cfg['tanh']
                    )
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError
        
            self.sigma = MLP(**cfg)

        self.criterion_cfg = criterion_cfg
        self.solver_cfg = solver_cfg
        self.lagrangian = lagrangian

        self.sdeint_fn = torchsde.sdeint_adjoint if solver_cfg['adjoint'] else torchsde.sdeint
        #self.register_buffer('mu_true', torch.as_tensor(self.criterion_cfg['mu']))
        #self.register_buffer('sigma_true', torch.as_tensor(self.criterion_cfg['std']))
        #self.sigma_true = self.criterion_cfg['std'] * torch.eye(self.input_dim)
        self.loss_fn = geomloss.SamplesLoss(loss='sinkhorn', p=criterion_cfg['p'], blur=criterion_cfg['blur'])
        self.U = lagrangian.U
        self.ts = set()

    def f(self, t, x):  # Approximate posterior drift.
        # xt = F.pad(x, (0, 1, 0, 0), value=t)
        gradPhi = self.phi.grad(t, x)
        v = self.prior_f(t, x)
        f = self.lagrangian.inv_L(t, x, -gradPhi[:, :self.input_dim], v)
        return f

    def g(self, t, x):  # Shared diffusion.
        if self.use_t_g:
            x = F.pad(x, (0, 1, 0, 0), value=t)
        else:
            x = x
        if self.sigma_type == "const" or self.sigma_type ==  "param":
            if self.noise_type == "scalar":
                g = self.sigma.repeat(x.size(0), 1, 1)
            elif self.noise_type == "diagonal":
                g = self.sigma.repeat(x.size(0), self.input_dim)
            
        elif self.sigma_type == "MLP-1":
            if self.noise_type == "diagonal":
                g = self.sigma(x).repeat(1, self.input_dim)
            else:
                g = self.sigma(x).repeat(1, self.input_dim, self.brownian_size)

        elif self.sigma_type == "MLP":
            if self.noise_type == "diagonal":
                g = self.sigma(x).view(-1, self.input_dim)
            else:
                g = self.sigma(x).view(-1, self.input_dim, self.brownian_size)
        return g

    def null_f(self, t, x):
        self.ts.add(round(t.item(), 5))
        return x
    def null_g(self, t, x):
        return x

    def f_aug(self, t, x_aug):
        batch_size = len(x_aug)
        x = x_aug[:, :self.input_dim]
        v = self.prior_f(t, x)
        Ux = self.Uxt(t, x).to(x.device)
        # xt = F.pad(x, (0, 1, 0, 0), value=t)
        if self.noise_type == "scalar" or self.noise_type == "diagonal":
            gradPhi, diagHess = self.phi.diagHess(t, x)
        else:
            gradPhi = self.phi.grad(t, x)
        # drift
        f = self.lagrangian.inv_L(t, x, -gradPhi[:, :self.input_dim], v)
        # cal dv
        dv = self.lagrangian.L(t, x, f, v, Ux)
        
        # cal dr
        g = self.g(t, x)
        if self.noise_type == "scalar":
            D = 0.5 * torch.pow(g.squeeze(2), 2)
            prod_Hess_D = torch.sum(diagHess * D, axis=1)
        elif self.noise_type == "diagonal":
            D = 0.5 * torch.pow(g, 2)
            prod_Hess_D = torch.sum(diagHess * D, axis=1)
        else:
            batch_idx = torch.arange(len(x))
            hessian = torch.autograd.functional.jacobian(lambda x: self.phi.grad(t, x), x)[batch_idx, :self.input_dim, batch_idx, :self.input_dim]
            D = 0.5 * torch.bmm(g, g.transpose(1, 2))
            prod_Hess_D = torch.sum(hessian * D, dim=(1, 2))
        
        h = dv + torch.sum(gradPhi[:, :self.input_dim] * f, dim=1, keepdims=True)

        if self.use_t_f:
            # L1norm
            dr = torch.abs(gradPhi[:,self.input_dim].unsqueeze(1) + prod_Hess_D.unsqueeze(1) + h)
            #dr = torch.abs(gradPhi[:,self.input_dim].unsqueeze(1) + prod_Hess_D.unsqueeze(1) + h) ** 2
        else:
            dr = torch.abs(prod_Hess_D.unsqueeze(1) + h)
            #dr = torch.abs(prod_Hess_D.unsqueeze(1) + h) ** 2
        
        x = torch.cat([f, dv, dr], dim=1)
        return x

    def g_aug(self, t, x_aug):
        g = self.g(t, x_aug[:, :self.input_dim])
        if self.noise_type == "diagonal":
            x = F.pad(g, (0, 2, 0, 0), value=0)
        else:
            x = F.pad(g, (0, 0, 0, 2, 0, 0), value=0)
        return x

    def forward(self, ts, x0, v0=None, batch_idx=None):
        if len(self.ts) == 0:
            self.initialize_mf_drift(ts, x0)
        
        if batch_idx == 0 or len(self.Uxt_list) == 0:
            self.build_Uxt(x0, self.tseq)

        aug_x0 = F.pad(x0, (0, 2, 0, 0), value=0)
        t1_ = min(ts[0] + self.solver_cfg['dt'], ts[1] - 1e-4 )
        ts_ = torch.tensor([ ts[0], t1_])
        xs = self.sdeint_fn(self, aug_x0, ts_,
            method=self.solver_cfg['method'],
            dt=self.solver_cfg['dt'],
            adaptive=self.solver_cfg['adaptive'],
            names={'drift': 'f_aug', 'diffusion': 'g_aug'})

        aug_x0 = xs[-1, :, :self.input_dim+2]
        ts[0] = t1_
        # xT: (len(ts), batch_size, d+2)
        if self.solver_cfg['adjoint']:
            xs = self.sdeint_fn(self, aug_x0, ts, 
                method=self.solver_cfg['method'], 
                dt=self.solver_cfg['dt'],
                adaptive=self.solver_cfg['adaptive'], 
                adjoint_adaptive=self.solver_cfg['adjoint_adaptive'],
                adjoint_method=self.solver_cfg['adjoint_method'], 
                names={'drift': 'f_aug', 'diffusion': 'g_aug'})
        else:
            xs = self.sdeint_fn(self, aug_x0, ts, 
                method=self.solver_cfg['method'], 
                dt=self.solver_cfg['dt'],
                adaptive=self.solver_cfg['adaptive'],
                names={'drift': 'f_aug', 'diffusion': 'g_aug'})

        #with open('rep.pickle', 'wb') as f:
        #    pickle.dump([ T, VALUE], f)
        loss_L = xs[-1][:, self.input_dim]
        loss_R = xs[-1][:, self.input_dim+1]

        xs[0] = F.pad(x0, (0, 2, 0, 0), value=0)
        
        assert xs.size(0) == len(ts)
        return dict(xs=xs[:, :, :self.input_dim], loss_L=loss_L, loss_R=loss_R)

    def criterion(self, x, x_hat, loss_L, loss_R, alpha_D, alpha_L, alpha_R):
        loss_D = self.loss_fn(x_hat, x)
        loss = alpha_D * loss_D + alpha_L * loss_L + alpha_R * loss_R
        return dict(loss=loss, loss_D=loss_D, loss_L=loss_L, loss_R=loss_R)

    def parameters_lr(self):
        return self.parameters()

    def clamp_parameters(self):
        if self.sigma_type == "param":
            self.sigma.data.clamp_(-5.0, 5.0)

    @torch.no_grad()
    def initialize_mf_drift(self, ts, x0) -> None:
        ts_c = copy.deepcopy(ts)
        t1_ = min(ts_c[0] + self.solver_cfg['dt'], ts_c[1] - 1e-4 )
        Ts_c = torch.tensor([ ts_c[0], t1_])
        _ = self.sdeint_fn(self, x0[:2], Ts_c,
            method=self.solver_cfg['method'],
            dt=self.solver_cfg['dt'],
            adaptive=self.solver_cfg['adaptive'],
            names={'drift': 'null_f', 'diffusion': 'null_g'})
        ts_c[0] = t1_
        _ = self.sdeint_fn(self, x0[:2], ts_c, 
            method=self.solver_cfg['method'], 
            dt=self.solver_cfg['dt'],
            adaptive=self.solver_cfg['adaptive'],
            names={'drift': 'null_f', 'diffusion': 'null_g'})
        
        self.tseq = torch.tensor(sorted(list(self.ts)))
        self.f_mul = self.build_f_mul(self.tseq)
        self.xis = self.build_xis(len(self.tseq), self.input_dim)
        self.polarize_strength = 1.0 if self.input_dim == 2 else 6.0
        self.tseq = self.tseq.cpu().numpy()
        self.Uxt_list = []

    @torch.no_grad()
    def prior_f(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        b, nx = x.shape
        assert nx == self.input_dim
        t_idx = self.t_to_idx(t)
        xi = self.xis[t_idx].to(x.device)

        mf_drift = opinion_lib.compute_mean_drift_term(x, xi).to(x.device)
        
        fmul = self.f_mul[t_idx].to(x.device).unsqueeze(-1)
        assert xi.shape == mf_drift.shape == (nx, )

        f = self.polarize_strength * opinion_lib.opinion_f(x, mf_drift, xi)
        assert f.shape == x.shape

        f = fmul * f
        assert f.shape == x.shape

        return f

    def build_f_mul(self, ts) -> torch.Tensor:
        # set f_mul with some heuristic so that it doesn't diverge exponentially fast
        # and yield bad normalization, since the more polarized the opinion is the faster it will grow
        coeff = 8.0
        f_mul = torch.clip(1.0 - torch.exp(coeff * (ts - ts[-1])) + 1e-5, min=1e-4, max=1.0)
        f_mul = f_mul ** 5.0
        return f_mul


    def build_xis(self, interval, x_dim) -> torch.Tensor:
        # Generate random unit vectors.
        rng = np.random.default_rng(seed=4078213)
        xis = rng.standard_normal([interval, x_dim])

        # Construct a xis that has some degree of "continuous" over time, as a brownian motion.
        xi = xis[0]
        bm_xis = [xi]
        std = 0.4
        for t in range(1, interval):
            xi = xi - (2.0 * xi) * 0.01 + std * math.sqrt(0.01) * xis[t]
            bm_xis.append(xi)
        assert len(bm_xis) == xis.shape[0]

        xis = torch.Tensor(np.stack(bm_xis))
        xis /= torch.linalg.norm(xis, dim=-1, keepdim=True)
        return xis

    def t_to_idx(self, t):
        return int(((t - self.tseq[0]) + 1e-6) / self.solver_cfg['dt'])
    
    @torch.no_grad()
    def build_Uxt(self, x0, ts=None):
        if len(self.ts) == 0:
            self.initialize_mf_drift(ts, x0)

        del self.Uxt_list
        xs = self.sdeint_fn(self, x0, torch.from_numpy(self.tseq).float(), 
                method=self.solver_cfg['method'], 
                dt=self.solver_cfg['dt'],
                adaptive=self.solver_cfg['adaptive'],
                names={'drift': 'f', 'diffusion': 'g'})

        Uxt = []
        t_size = len(xs)
        mu_init = None
        for ti in tqdm(range(t_size), leave=False):
            Ux = self.U(self.lagrangian.pca_proj, xs[ti, :], num_components=5, mu_init=mu_init)
            Uxt.append(Ux)
            params = Ux.params()
            mu_init = params['mu']
        self.Uxt_list = Uxt
        return Uxt
    
    def Uxt(self, t, x):
        t_idx = self.t_to_idx(t)
        Uxt = torch.nan_to_num(self.Uxt_list[t_idx](x))
        return Uxt

    """
    def loss_fn(self, x, x_hat):
        mu_hat = torch.mean(x_hat, dim=0)
        sigma_hat = torch.cov(x_hat.T)
        return self._calculate_frechet_distance(mu_hat, sigma_hat)

        # Modified from: https://github.com/bioinf-jku/TTUR/blob/master/fid.py
    def _calculate_frechet_distance(self, mu, sigma):
        diff = mu - self.mu_true
        # product might be almost singular
        M = self.sigma_true * sigma
        S = torch.linalg.eigvals(M.float()) + 1e-15
        tr_covmean = S.sqrt().abs().sum()
        return torch.sum(torch.pow(diff, 2)) + torch.trace(sigma) + (self.input_dim * self.sigma_true) - 2 * tr_covmean.to(mu)
    """

    