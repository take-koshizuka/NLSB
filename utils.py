import ot
import torch
import numpy as np
import math

class EarlyStopping(object):
    def __init__(self, monitor='loss', direction='min'):
        self.monitor = monitor
        self.direction = direction
        self.best_state = None
        if direction == 'min':
            self.monitor_values = { self.monitor : float('inf') }
        elif direction == 'max':
            self.monitor_values = { self.monitor : -float('inf') }
        else:
            raise ValueError("args: [direction] must be min or max")

    def judge(self, values):
        return (self.direction == 'min' and self.monitor_values[self.monitor] > values[self.monitor]) \
                    or (self.direction == 'max' and self.monitor_values[self.monitor] < values[self.monitor])

    def update(self, values):
        self.monitor_values[self.monitor] = values[self.monitor]

    def get_value(self):
        return self.monitor_values[self.monitor]

class Accumulator:
    def __init__(self, score_init=0, data_size_init=0):
        self.score = score_init
        self.data_size = data_size_init

    def update(self, score, data_size):
        self.score += score
        self.data_size += data_size

    def compute(self):
        return float(self.score) / float(self.data_size)


def conditional_distribution_discrepancy(ref_traj, pred_traj, int_time, eval_idx=None, p=2):
    if not eval_idx is None:
        ref_traj = ref_traj[:, eval_idx, :, :]
        pred_traj = pred_traj[:, eval_idx, :, :]
        int_time = int_time[eval_idx]

    data_size, t_size, num_repeat, dim = ref_traj.size()
    res = {}
    for j in range(1, t_size):
        losses = []
        for i in range(data_size):
            ref_dist = ref_traj[i, j]
            pred_dist = pred_traj[i, j]
            M = torch.cdist(ref_dist, pred_dist, p=p)
            a, b = ot.unif(ref_dist.size()[0]), ot.unif(pred_dist.size()[0])
            loss = ot.emd2(a, b, M.cpu().detach().numpy())
            losses.append(loss)
        res[f't={int_time[j].item()}'] = sum(losses) / data_size
    return res

def marginal_distribution_discrepancy(ref_traj, pred_traj, int_time, eval_idx=None, p=2):
    if not eval_idx is None:
        ref_traj = ref_traj[:, eval_idx, :]
        pred_traj = pred_traj[:, eval_idx, :]
        int_time = int_time[eval_idx]
    
    if pred_traj.ndim == 3:
        data_size, t_size, dim = pred_traj.size()
        res = {}
        for j in range(1, t_size):
            ref_dist = ref_traj[:, j]
            pred_dist = pred_traj[:, j]
            M = torch.cdist(ref_dist, pred_dist, p=p)
            a, b = ot.unif(ref_dist.size()[0]), ot.unif(pred_dist.size()[0])
            loss = ot.emd2(a, b, M.cpu().detach().numpy())
            res[f't={int_time[j].item()}'] = { 'mean' : loss }
        
        return res

    elif pred_traj.ndim == 4:
        data_size, t_size, num_repeat, dim = pred_traj.size()
        res = {}
        for j in range(1, t_size):
            losses = []
            for i in range(num_repeat):
                ref_dist = ref_traj[:, j]
                pred_dist = pred_traj[:, j, i]
                M = torch.cdist(ref_dist, pred_dist, p=p)
                a, b = ot.unif(ref_dist.size()[0]), ot.unif(pred_dist.size()[0])
                loss = ot.emd2(a, b, M.cpu().detach().numpy())
                losses.append(loss)
            
            res[f't={int_time[j].item()}'] = { 'mean' : np.mean(losses), 'std' : np.std(losses) }
        return res

def decode(data, param=None):
    if param is None:
        return data
    
    if data.ndim == 2:
        scale = param['scale'].repeat(data.size(0),  1)
        mean = param['mean'].repeat(data.size(0),  1)
    elif data.ndim == 3:
        scale = param['scale'].repeat(data.size(0), data.size(1), 1)
        mean = param['mean'].repeat(data.size(0), data.size(1), 1)
    elif data.ndim == 4:
        scale = param['scale'].repeat(data.size(0), data.size(1), data.size(2), 1)
        mean = param['mean'].repeat(data.size(0), data.size(1), data.size(2), 1)
    return (data * scale) + mean


def compute_emd2(ref_data, pred_data, p=2):
    M = torch.cdist(pred_data, ref_data, p=p)
    a, b = ot.unif(pred_data.size()[0]), ot.unif(ref_data.size()[0])
    loss = ot.emd2(a, b, M.cpu().detach().numpy(),  numItermax=1000000)
    return loss

def gaussian_nll(z):
    return -(torch.sum(-0.5 * math.log(2*math.pi) - torch.pow(z, 2) / 2, axis=1))