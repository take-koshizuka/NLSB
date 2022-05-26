import torch
from models.gmm import TimeDependentGMM
from models.potential_2d import Box, Slit, HarmonicOscillator, Hill

POTENTIAL_NAME = {
    "box" : Box,
    "slit": Slit,
    "harmonic_oscillator": HarmonicOscillator,
    "hill": Hill
}

def lambda_t(t, lms, intervals):
    for i, (t0, t1) in enumerate(intervals):
        if t0 <= t < t1:
            return lms[i]
    if t == t1:
        return lms[-1]
    
    raise ValueError
    
class PotentialFreeLagrangian:
    def L(self, t, x, u):
        return 0.5 * torch.sum(torch.pow(u, 2), 1, keepdims=True)

    def inv_L(self, t, x, f):
        return f

class CellularLagrangian:
    def __init__(self, Xs, ts, n_components_list, lm_u2=1.0, lm_U=1.0, lm_v=1.0, intervals=None, device='cpu'):
        if intervals is None:
            t_set = list(sorted(list(set(ts.numpy()))))
            intervals = []
            for i in range(len(t_set) - 1):
                intervals.append((t_set[i], t_set[i+1]))
        self.U = TimeDependentGMM(Xs, ts, n_components_list=n_components_list, intervals=intervals).to(device)
        self.intervals = intervals
        self.lm_u2_value = lm_u2
        self.lm_U_value = lm_U
        self.lm_v_value = lm_v
    
    def lm_u2(self, t):
        if not type(self.lm_u2_value) is list:
            return self.lm_u2_value
        else:
            return lambda_t(t, self.lm_u2_value, self.intervals)
    
    def lm_U(self, t):
        if not type(self.lm_U_value) is list:
            return self.lm_U_value
        else:
            return lambda_t(t, self.lm_U_value, self.intervals)
    
    def lm_v(self, t):
        if not type(self.lm_v_value) is list:
            return self.lm_v_value
        else:
            return lambda_t(t, self.lm_v_value, self.intervals)

    def L(self, t, x, u, v=None):
        Uxt = self.U(x, t).unsqueeze(1).float()
        if v is None:
            return self.lm_u2(t) * 0.5 * torch.sum(torch.pow(u, 2), 1, keepdims=True) - self.lm_U(t) * Uxt
        else:
            return self.lm_u2(t) * 0.5 * torch.sum(torch.pow(u, 2), 1, keepdims=True) - self.lm_U(t) * Uxt \
                                + self.lm_v(t) * 0.5 * torch.sum(torch.pow(u - v, 2), 1, keepdims=True)

    def inv_L(self, t, x, f, v=None):
        if v is None:
            return f
        else:
            return (self.lm_v(t) * v + f) / (1 + self.lm_v(t))

class NewtonianLagrangian:
    def __init__(self, M, U_cfg, lm_u2=1.0, lm_U=1.0):
        self.M = M
        self.U = POTENTIAL_NAME[U_cfg["name"]]()
        self.lm_u2 = lm_u2
        self.lm_U = lm_U
    
    def L(self, t, x, u):
        Ux = self.U(x, t).float()
        return self.lm_u2 * 0.5 * self.M * torch.sum(torch.pow(u, 2), dim=1, keepdims=True) - self.lm_U * Ux

    def inv_L(self, t, x, f):
        return (1 / self.M) * f
    
class NullLagrangian:
    def L(self, t, x, u):
        return torch.zeros((len(u), 1))

    def inv_L(self, t, x, f):
        return f
    