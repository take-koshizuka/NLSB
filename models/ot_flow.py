import numpy as np
from numpy.core.multiarray import dot
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
from torchdiffeq import odeint_adjoint as odeint

def antiderivTanh(x): # activation function aka the antiderivative of tanh
    return torch.abs(x) + torch.log(1+torch.exp(-2.0*torch.abs(x)))

def derivTanh(x): # act'' aka the second derivative of the activation function antiderivTanh
    return 1 - torch.pow( torch.tanh(x) , 2 )

def linear_interp(t0, z0, t1, z1, t):
    if t0 <= t <= t1:
        y = (t1 - t) / (t1 - t0) * z0 + (t - t0) / (t1 - t0) * z1
    elif t1 <= t <= t0:
        y = (t0 - t) / (t0 - t1) * z0 + (t - t1) / (t0 - t1) * z1
    else:
        raise ValueError(f"Incorrect time order for linear interpolation: t0={t0}, t={t}, t1={t1}.")
    return y

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

        self.d = d
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
        :param x: tensor nex-by-d, inputs
        :return:  tensor nex-by-m,   outputs
        """

        x = self.act(self.layers[0].forward(x))

        for i in range(1,self.nTh):
            x = x + self.h * self.act(self.layers[i](x))

        return x

class Phi(nn.Module):
    def __init__(self, nTh, m, d, use_t=True, r=10, alph=[1.0] * 5):
        """
            neural network approximating Phi (see Eq. (9) in our paper)
            Phi( x,t ) = w'*ResNet( [x;t]) + 0.5*[x' t] * A'A * [x;t] + b'*[x;t] + c
        :param nTh:  int, number of resNet layers , (number of theta layers)
        :param m:    int, hidden dimension
        :param d:    int, dimension of space input (expect inputs to be d+1 for space-time)
        :param r:    int, rank r for the A matrix
        :param alph: list, alpha values / weighted multipliers for the optimization problem
        """
        super().__init__()

        self.m    = m
        self.nTh  = nTh
        self.d    = d
        self.alph = alph
        self.use_t = use_t

        r = min(r,d+int(self.use_t)) # if number of dimensions is smaller than default r, use that

        self.A  = nn.Parameter(torch.zeros(r, d+int(self.use_t)) , requires_grad=True)
        self.A  = nn.init.xavier_uniform_(self.A)
        self.c  = nn.Linear( d+int(self.use_t)  , 1  , bias=True)  # b'*[x;t] + c
        self.w  = nn.Linear( m    , 1  , bias=False)

        self.N = ResNN(d+int(self.use_t), m, nTh=nTh)

        # set initial values
        self.w.weight.data = torch.ones(self.w.weight.data.shape)
        self.c.weight.data = torch.zeros(self.c.weight.data.shape)
        self.c.bias.data   = torch.zeros(self.c.bias.data.shape)

    def forward(self, t, x):
        """ calculating Phi(s, theta)...not used in OT-Flow """
        
        if self.use_t:
            x = torch.cat([x, t.repeat(x.size(0), 1)], dim=1)

        # force A to be symmetric
        symA = torch.matmul(torch.t(self.A), self.A) # A'A
        return self.w( self.N(x)) + 0.5 * torch.sum( torch.matmul(x , symA) * x , dim=1, keepdims=True) + self.c(x)


    def trHess(self, t, x, justGrad=False):
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
            x = torch.cat([x, t.repeat(x.size(0), 1)], dim=1)
            # x = F.pad(x, (0, 1, 0, 0), value=t)
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

        if justGrad:
            return grad.t()

        # -----------------
        # trace of Hessian
        #-----------------

        # t_0, the trace of the opening layer
        Kopen = N.layers[0].weight[:,0:d]    # indexed version of Kopen = torch.mm( N.layers[0].weight, E  )
        temp  = derivTanh(opening.t()) * z[1]
        trH  = torch.sum(temp.reshape(m, -1, nex) * torch.pow(Kopen.unsqueeze(2), 2), dim=(0, 1)) # trH = t_0

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
            t_i = torch.sum(  ( derivTanh(temp) * term ).reshape(m,-1,nex)  *  torch.pow(KJ,2) ,  dim=(0, 1) )
            trH  = trH + N.h * t_i  # add t_i to the accumulate trace
            Jac = Jac + N.h * torch.tanh(temp).reshape(m, -1, nex) * KJ # update Jacobian

        return grad.t(), trH + torch.trace(symA[0:d,0:d])
        # indexed version of: return grad.t() ,  trH + torch.trace( torch.mm( E.t() , torch.mm(  symA , E) ) )


def stepRK4(odefun, z, Phi, alph, t0, t1):
    """
        Runge-Kutta 4 integration scheme
    :param odefun: function to apply at every time step
    :param z:      tensor nex-by-d+4, inputs
    :param Phi:    Module, the Phi potential function
    :param alph:   list, the 3 alpha values for the OT-Flow Problem
    :param t0:     float, starting time
    :param t1:     float, end time
    :return: tensor nex-by-d+4, features at time t1
    """

    h = t1 - t0 # step size
    z0 = z

    K = h * odefun(z0, t0, Phi, alph=alph)
    z = z0 + (1.0/6.0) * K

    K = h * odefun( z0 + 0.5*K , t0+(h/2) , Phi, alph=alph)
    z += (2.0/6.0) * K

    K = h * odefun( z0 + 0.5*K , t0+(h/2) , Phi, alph=alph)
    z += (2.0/6.0) * K

    K = h * odefun( z0 + K , t0+h , Phi, alph=alph)
    z += (1.0/6.0) * K

    return z

def stepRK1(odefun, z, Phi, alph, t0, t1):
    """
        Runge-Kutta 1 / Forward Euler integration scheme.  Added for comparison, but we recommend stepRK4.
    :param odefun: function to apply at every time step
    :param z:      tensor nex-by-d+4, inputs
    :param Phi:    Module, the Phi potential function
    :param alph:   list, the 3 alpha values for the mean field game problem
    :param t0:     float, starting time
    :param t1:     float, end time
    :return: tensor nex-by-d+4, features at time t1
    """
    z += (t1 - t0) * odefun(z, t0, Phi, alph=alph)
    return z

def C(z):
    """Expected negative log-likelihood; see Eq.(3) in the paper"""
    d = z.shape[1]-3
    l = z[:,d] # log-det
    return -( torch.sum(  -0.5 * math.log(2*math.pi) - torch.pow(z[:,0:d],2) / 2  , 1 , keepdims=True ) + l.unsqueeze(1) )

class ODEfunc(nn.Module):
    def __init__(self, input_dim, alph, potential_cfg, use_t=True):
        super(ODEfunc, self).__init__()
        self.input_dim = input_dim
        self.phi = Phi(**potential_cfg, d=input_dim)
        self.alph = alph
        self.use_t = use_t

    def forward(self, t, x_aug):
        x = x_aug[:, :self.input_dim]
        # x = F.pad(x, (0, 1, 0, 0), value=float(t))
        gradPhi, trH = self.phi.trHess(t, x)
        dx = -(1.0/self.alph) * gradPhi[:,0:self.input_dim]
        dl = -(1.0/self.alph) * trH.unsqueeze(1)
        dv = 0.5 * torch.sum(torch.pow(dx, 2) , 1 ,keepdims=True)
        if self.use_t:
            dr = torch.abs(-gradPhi[:,-1].unsqueeze(1) + self.alph * dv  )
        else:
            dr = torch.abs(self.alph * dv)
        return torch.cat((dx, dl, dv, dr) , 1)

class OT_flow(nn.Module):
    def __init__(self, input_dim, int_t_to_noise, time_scale, potential_cfg, criterion_cfg, solver_cfg) -> None:
        super(OT_flow, self).__init__()
        self.odefunc = ODEfunc(input_dim, solver_cfg['alph'], potential_cfg)
        self.int_t_to_noise = int_t_to_noise
        self.time_scale = time_scale
        self.criterion_cfg = criterion_cfg
        self.solver_cfg = solver_cfg
        
    def forward(self, ts, x0):
        # xT -> x0 -> z(t=0)
        z = F.pad(x0, (0, 3, 0, 0), value=0)

        zs = odeint(
            self.odefunc,
            z,
            torch.tensor(ts, device=z.device),
            atol=self.solver_cfg['atol'],
            rtol=self.solver_cfg['rtol'],
            method=self.solver_cfg['method'],
            options=dict(step_size=self.solver_cfg['dt'])
        )

        # ASSUME all examples are equally weighted
        xs = zs[:, :, :-3]
        log_det = zs[-1, :, -3]
        loss_L  = torch.abs(zs[-1, :, -2])
        loss_R  = torch.abs(zs[-1, :, -1])
        # shape(xs) = (t_size, batch_size, dim)
        return dict(xs=xs, log_det=log_det, loss_L=loss_L, loss_R=loss_R)

    def criterion(self,  batch_size, base_nll, log_det, losses_L, losses_R, alpha_L, alpha_R):
        nll = base_nll - log_det
        base_nll = nll[:-batch_size]
        loss_D = torch.mean(nll[-batch_size:])
        loss_L = torch.mean(losses_L)
        loss_R = torch.mean(losses_R)
        loss = (loss_D + alpha_L * loss_L + alpha_R * loss_R) 
        return dict(loss=loss, nll=base_nll, loss_D=loss_D, loss_L=loss_L, loss_R=loss_R)

    def parameters_lr(self):
        return self.parameters()