import ot
import torch
import numpy as np
import math
from sklearn import metrics
import geomloss

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
            M = torch.pow(torch.cdist(ref_dist, pred_dist, p=p),p)
            a, b = ot.unif(ref_dist.size()[0]), ot.unif(pred_dist.size()[0])
            loss = ot.emd2(a, b, M.cpu().detach().numpy())
            if p == 2:
                loss = np.sqrt(loss)
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
        for j in range(t_size):
            ref_dist = ref_traj[:, j]
            pred_dist = pred_traj[:, j]
            M = torch.pow(torch.cdist(ref_dist, pred_dist, p=p),p)
            a, b = ot.unif(ref_dist.size()[0]), ot.unif(pred_dist.size()[0])
            loss = ot.emd2(a, b, M.cpu().detach().numpy())
            if p == 2:
                loss = np.sqrt(loss)
            res[f't={int_time[j].item()}'] = { 'mean' : loss }  
        return res

    elif pred_traj.ndim == 4:
        data_size, t_size, num_repeat, dim = pred_traj.size()
        res = {}
        for j in range(t_size):
            losses = []
            for i in range(num_repeat):
                ref_dist = ref_traj[:, j]
                pred_dist = pred_traj[:, j, i]
                M = torch.pow(torch.cdist(ref_dist, pred_dist, p=p),p)
                a, b = ot.unif(ref_dist.size()[0]), ot.unif(pred_dist.size()[0])
                loss = ot.emd2(a, b, M.cpu().detach().numpy())
                if p == 2:
                    loss = np.sqrt(loss)
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
    M = torch.pow(torch.cdist(ref_data, pred_data, p=p),p)
    a, b = ot.unif(pred_data.size()[0]), ot.unif(ref_data.size()[0])
    loss = ot.emd2(a, b, M.cpu().detach().numpy(),  numItermax=1000000)
    if p == 2:
        loss = np.sqrt(loss)
    return loss

def compute_MMD(ref_data, pred_data, gamma=None):
    # loss_fn = geomloss.SamplesLoss(loss='gaussian')
    # MMD = loss_fn(ref_data, pred_data)
    if gamma is None:
        gamma = 1.0 / ref_data.shape[1]
    mmd = mmd_rbf(ref_data, pred_data, gamma)
    return mmd

def mmd_rbf(X, Y, gamma):
    """MMD using rbf (gaussian) kernel (i.e., k(x,y) = exp(-gamma * ||x-y||^2 / 2))
    Arguments:
        X {[n_sample1, dim]} -- [X matrix]
        Y {[n_sample2, dim]} -- [Y matrix]
    Keyword Arguments:
        gamma {float} -- [kernel parameter] (default: {1.0})
    Returns:
        [scalar] -- [MMD value]
    """
    XX = metrics.pairwise.rbf_kernel(X, X, gamma)
    YY = metrics.pairwise.rbf_kernel(Y, Y, gamma)
    XY = metrics.pairwise.rbf_kernel(X, Y, gamma)
    return XX.mean() + YY.mean() - 2 * XY.mean()

def gaussian_nll(z):
    return -(torch.sum(-0.5 * math.log(2*math.pi) - torch.pow(z, 2) / 2, axis=1))

def pca(x: torch.Tensor, low_dim=100):
    B, nx = x.shape

    mean_pca_x = torch.mean(x, dim=0, keepdim=True)
    y = x - mean_pca_x

    if B > 200:
        rand_idxs = torch.randperm(B)[:200]
        y = y[rand_idxs]
    # U: (batch, k)
    # S: (k, k)
    # VT: (k, nx)
    U, S, VT = torch.linalg.svd(y)
    D = min(low_dim, nx)
    # log.info("Singular values of xs_f at final timestep:")
    # log.info(S)
    # Keep the first and last directions.
    VT = VT[:D, :]
    #VT = VT[[0, -1], :]
    # assert VT.shape == (D, nx)
    V = VT.T

    proj_x = y @ V

    return proj_x, V, mean_pca_x

def loss_fn(x, x_1):
    mu = torch.mean(x, dim=0)
    sigma = torch.cov(x.T)
    W2 = calculate_frechet_distance(mu, sigma)
    nll = calculate_kl(x)
    z, V, mean = pca(x, low_dim=5)
    z_1 = (x_1 - mean) @ V
    L = geomloss.SamplesLoss(loss='sinkhorn', p=2)
    W2_ = L(x, x_1)
    W2_z = L(z, z_1)
    return W2, torch.mean(nll), W2_, W2_z

# Modified from: https://github.com/bioinf-jku/TTUR/blob/master/fid.py
def calculate_frechet_distance(mu, sigma):
    mu_true = 0
    sigma_true = 1.0
    input_dim = mu.size()[0]
    diff = mu - mu_true
    # product might be almost singular
    M = sigma_true * sigma
    S = torch.linalg.eigvals(M.float()) + 1e-15
    tr_covmean = S.sqrt().abs().sum()
    return torch.sum(torch.pow(diff, 2)) + torch.trace(sigma) + (input_dim * sigma_true) - 2 * tr_covmean.to(mu)

def calculate_kl(x):
    mu_true = 0
    sigma_true = 1.0
    z = x - mu_true
    nlogp = -(torch.sum(-0.5 * np.log(2*np.pi*sigma_true) - torch.pow(z, 2) / (2*sigma_true), axis=1))
    return nlogp
