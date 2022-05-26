# ======================
# Import Libs
# ======================
import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
from utils import gaussian_nll, compute_emd2

# from models.resnet import ResNet50
from models.trajectorynet import TrajectoryNet
from models.ot_sde import OT_SDEIto
from models.ot_flow import OT_flow
from models.lagrangian import NullLagrangian, PotentialFreeLagrangian, NewtonianLagrangian, CellularLagrangian

ODE_MODEL_NAME = {
    "trajectorynet": TrajectoryNet,
    "ot-flow": OT_flow
}
SDE_MODEL_NAME = {
    # "model_name" : model class
    # "resnet50" : ResNet50
    "ito": OT_SDEIto
}

LAGRANGIAN_NAME = {
    "null": NullLagrangian,
    "potential-free": PotentialFreeLagrangian,
    "cellular": CellularLagrangian,
    "newtonian": NewtonianLagrangian
}

class SDENet(nn.Module):
    def __init__(self, net, device):
        super(SDENet, self).__init__()
        self.net = net
        self.device = device
        self.criterion = self.net.criterion
        self.criterion_cfg = self.net.criterion_cfg

    def forward(self, ts, x0, v0=None):
        return self.net(ts, x0, v0)

    def step(self, batch, batch_idx, t_set, T0):
        xs = batch['x'].float().to(self.device)
        ts = batch['t'].float().to(self.device)
        use_v = ('v' in batch)
        vs = batch['v'].float().to(self.device) if use_v else None
        score = {}
        for i, t in enumerate(t_set):
            score[i] = dict()
            xt = xs[ts == t]
            vt = vs[ts == t] if use_v else None
            
            if i == 0:
                x0 = batch['base']['X'].to(self.device)
                v0 = batch['base']['V'].to(self.device) if use_v else None
                # batch_size = len(x0)
                int_time = [float(T0), float(t)]
            else:
                x0, v0 = prev_x, prev_v
                int_time = [float(t_set[i-1]), float(t)]

            res = self.forward(int_time, x0, v0)
            xt_hat = res['xs'][-1]

            alpha_D = self.criterion_cfg['alpha_D'][i] if type(self.criterion_cfg['alpha_D']) is list else self.criterion_cfg['alpha_D']
            alpha_L = self.criterion_cfg['alpha_L'][i] if type(self.criterion_cfg['alpha_L']) is list else self.criterion_cfg['alpha_L']
            alpha_R = self.criterion_cfg['alpha_R'][i] if type(self.criterion_cfg['alpha_R']) is list else self.criterion_cfg['alpha_R']
            
            loss_L = torch.mean(res['loss_L'])
            loss_R = torch.mean(res['loss_R'])
            losses = self.criterion(xt, xt_hat, loss_L, loss_R, alpha_D, alpha_L, alpha_R)

            score[i][f'loss'] = losses['loss']
            score[i][f'loss_D'] = losses['loss_D']
            score[i][f'loss_L'] = losses['loss_L']
            score[i][f'loss_R'] = losses['loss_R']

            prev_x, prev_v = xt, vt

            """
            score[i][f'loss'] = score[i][f'loss_D'] = score[i][f'loss_L'] = score[i][f'loss_R'] = 0
            
            for k in range(i+1):
                idx = torch.arange(k*batch_size, (k+1)*batch_size)
                loss_L, loss_R = torch.mean(res['loss_L'][idx]), torch.mean(res['loss_R'][idx])
                losses = self.criterion(xt, xt_hat[idx], loss_L, loss_R, alpha_D, alpha_L, alpha_R)
                score[i][f'loss'] += losses['loss']
                score[i][f'loss_D'] += losses['loss_D']
                score[i][f'loss_L'] += losses['loss_L']
                score[i][f'loss_R'] += losses['loss_R']
            
            prev_x = torch.cat((xt_hat, xt), dim=0)
            """

        score['loss'] = sum([ score[j]['loss'] for j in score.keys() ])
        return score

    def training_step(self, batch, batch_idx, t_set, T0=0.0):
        self.train()
        score = self.step(batch, batch_idx, t_set, T0)
        return score
    
    @torch.no_grad()    
    def validation_step(self, batch, batch_idx, t_set, T0=0.0):
        self.eval()
        score = self.step(batch, batch_idx, t_set, T0)
        return score

    @torch.no_grad()
    def validation_step_full(self, batch, batch_idx, t_set, T0=0.0):
        self.eval()
        t_set = [int(T0)] + list(t_set)
        int_time = torch.from_numpy(np.array(t_set))
        x0 = batch['base']['X'].to(self.device)

        res = self.forward(int_time, x0)['xs']
        res = res[1:, :, :]
        
        xs = batch['x'].float().to(self.device)
        ts = batch['t'].float().to(self.device)

        losses = {}
        for i, t in enumerate(t_set[1:]):
            xt = xs[ts == t]
            loss = compute_emd2(xt.cpu().detach(), res[i].cpu().detach())
            losses[i] = { 'emd' : loss }
        return losses
    
    def training_epoch_end(self, outputs):
        score = {}
        ti_set = [ key for key in outputs[0].keys() if not key in ['loss'] ]
        sum_avg_loss = 0.0
        for i in ti_set:
            score[f'k={i}'] = dict(
                avg_loss=torch.mean(torch.tensor([ out[i]['loss'].item() for out in outputs ]).flatten()).item(),
                avg_loss_D=torch.mean(torch.tensor([ out[i]['loss_D'].item() for out in outputs ]).flatten()).item(),
                avg_loss_L=torch.mean(torch.tensor([ out[i]['loss_L'].item() for out in outputs ]).flatten()).item(),
                avg_loss_R=torch.mean(torch.tensor([ out[i]['loss_R'].item() for out in outputs ]).flatten()).item(),
            )
            sum_avg_loss +=  score[f'k={i}'][f"avg_loss"]
        
        avg_loss =  sum_avg_loss / len(ti_set)
        logs = { 'avg_loss' : avg_loss  }
        logs.update(score)
        return { 'avg_loss' : avg_loss, 'log' : logs  }

    def validation_epoch_end(self, outputs):
        score = {}
        ti_set = [ key for key in outputs[0].keys() if not key in ['loss'] ]
        sum_avg_loss = sum_avg_loss_D = 0.0
        for i in ti_set:
            score[f'k={i}'] = dict(
                avg_loss=torch.mean(torch.tensor([ out[i]['loss'].item() for out in outputs ]).flatten()).item(),
                avg_loss_D=torch.mean(torch.tensor([ out[i]['loss_D'].item() for out in outputs ]).flatten()).item(),
                avg_loss_L=torch.mean(torch.tensor([ out[i]['loss_L'].item() for out in outputs ]).flatten()).item(),
                avg_loss_R=torch.mean(torch.tensor([ out[i]['loss_R'].item() for out in outputs ]).flatten()).item()
            )
            
            sum_avg_loss +=  score[f'k={i}'][f"avg_loss"]
            sum_avg_loss_D += score[f'k={i}'][f"avg_loss_D"]
        
        avg_loss =  sum_avg_loss / len(ti_set) 
        avg_loss_D = sum_avg_loss_D / len(ti_set)
        
        logs = { 'avg_loss' : avg_loss, 'avg_loss_D' : avg_loss_D }
        logs.update(score)
        return { 'avg_loss' : avg_loss, 'avg_loss_D' : avg_loss_D , 'log' : logs  }

    def validation_epoch_end_full(self, outputs):
        score = {}
        ti_set = outputs[0].keys()
        sum_avg_emd = 0.0
        for i in ti_set:
            score[f'k={i}'][f"avg_emd"] = torch.mean(torch.tensor([ out[i]['emd'] for out in outputs ]).flatten()).item()
            sum_avg_emd += score[f'k={i}'][f"avg_emd"]
        
        avg_emd = sum_avg_emd / len(ti_set)
        logs = { 'avg_emd' : avg_emd  }
        logs.update(score)
        return { 'avg_emd' : avg_emd, 'log' : logs  }

    @torch.no_grad()
    def validation(self, ds, t_set):
        emds = []
        for i in range(len(t_set)):
            target_idx = ds.get_subset_index(t_set[i])
            target_X = ds.get_data(target_idx)["X"].float()
            if i == 0:
                source = ds.base_sample(len(target_X))
                int_time = [ float(ds.T0), float(t_set[i])]
            else:
                source_idx = ds.get_subset_index(t_set[i - 1])
                source = ds.get_data(source_idx)
                int_time = [float(t_set[i-1]), float(t_set[i])]

            source_X = source["X"].float()
            source_V = source["V"].float() if "V" in source else None

            pred_sample = self.sample(source_X, int_time, source_V)
            emd = compute_emd2(target_X.cpu(), pred_sample[:, -1].cpu())
            emds.append(emd)
        return { 'avg_emd' : np.mean(emds), 'emds': emds }

    @torch.no_grad()
    def sample(self, x0, int_time, v0=None):
        self.eval()
        if not v0 is None:
            v0 = v0.float().to(self.device)
        if int_time[0] <= int_time[1]:
            res = self.forward(int_time, x0.float().to(self.device), v0)
            ys = res["xs"].transpose(0, 1)
        else:
            raise NotImplementedError
        return ys
    
    @torch.no_grad()
    def sample_with_uncertainty(self, x0, int_time, num_repeat, v0=None):
        batch_size, data_dim = x0.size()
        x0_ex = x0.repeat(1, num_repeat+1)
        x0_ex = x0_ex.view(-1, data_dim)
        if not v0 is None:
            v0_ex = v0.repeat(1, num_repeat+1).view(-1, data_dim)
            v0_ex = v0_ex.float().to(self.device)
        else:
            v0_ex = None
        
        self.eval()
        if int_time[0] <= int_time[1]:
            res = self.forward(int_time, x0_ex.float().to(self.device), v0_ex)
            ys = res["xs"].view(len(int_time), batch_size, num_repeat+1, data_dim).transpose(0, 1)
        else:
            raise NotImplementedError
        # (batch_size, int_time, num_repeat, data_dim)
        return ys

    def state_dict(self, optimizer, scheduler=None):
        dic =  {
            "net": deepcopy(self.net.state_dict()),
            "optimizer": deepcopy(optimizer.state_dict())
        }
        if not scheduler is None:
            dic["scheduler"] = deepcopy(scheduler.state_dict())
        
        if AMP:
            dic['amp'] = deepcopy(amp.state_dict())
        return dic

    def load_model(self, checkpoint, amp=False):
        print(f"load model of epoch={checkpoint['epochs']}")
        self.net.load_state_dict(checkpoint["net"])
        if amp:
            amp.load_state_dict(checkpoint["amp"])

    def parameters_lr(self):
        return self.net.parameters_lr()

    def clamp_parameters(self):
        self.net.clamp_parameters()

class ODENet(nn.Module):
    def __init__(self, net, device):
        super(ODENet, self).__init__()
        self.net = net
        self.device = device
        self.criterion = self.net.criterion
        self.criterion_cfg = self.net.criterion_cfg
        self.int_t_to_noise = self.net.int_t_to_noise
        self.time_scale = self.net.time_scale

    def forward(self, ts, x0):
        return self.net(ts, x0)

    def step(self, batch, batch_idx, t_set, T0):
        t_set = [int(T0)] + list(t_set)
        t_set_rev = list(reversed(t_set))
        ts_aug = np.array(t_set) * self.time_scale + self.int_t_to_noise
        ts_aug_rev = list(reversed(ts_aug))

        xs = batch['x'].float().to(self.device)
        ts = batch['t'].float().to(self.device)
        scores = {}
        xt_hat = None
        for i, t in enumerate(t_set_rev):
            idx = t_set.index(t)
            scores[idx] = dict()
            if t == int(T0):
                # T0 -> gaussian (t=0)
                xt = batch['base']['X'].to(self.device)
                batch_size = xt.size(0)
                int_time = [ float(ts_aug_rev[i]) , 0.0]
            else:
                xt = xs[ts == t]
                int_time = [ float(ts_aug_rev[i]), float(ts_aug_rev[i+1])]

            if not xt_hat is None:
                xt = torch.cat((xt_hat, xt))
            res = self.forward(int_time, xt)
            xt_hat = res['xs'][-1, :, :]
            scores[idx] = { k: res[k] for k in ['log_det', 'loss_L', 'loss_R'] }

        z = res['xs'][-1, :, :] # z ~ N(0, I), shape(z) = (batch_size*t_size, dim)
        losses = {}
        nll_T = gaussian_nll(z) - scores[t_set_rev[-1]]['log_det']
        base_nll = nll_T[:-batch_size]
        loss_D = torch.mean(nll_T[-batch_size:])
        loss_L = loss_R = torch.tensor([0.0])
        losses[0] = { 'loss' : loss_D, 'loss_D': loss_D, 'nll': base_nll, 'loss_L': loss_L, 'loss_R': loss_R }
        
        for i, t in enumerate(t_set[1:], start=1):
            loss_t = scores[i]
            alpha_L = self.criterion_cfg['alpha_L'][i - 1] if type(self.criterion_cfg['alpha_L']) is list else self.criterion_cfg['alpha_L']
            if 'alpha_R' in self.criterion_cfg:
                alpha_R = self.criterion_cfg['alpha_R'][i - 1] if type(self.criterion_cfg['alpha_R']) is list else self.criterion_cfg['alpha_R']
            else:
                alpha_R = 0.0
            res = self.criterion(batch_size, base_nll, loss_t['log_det'], loss_t['loss_L'], loss_t['loss_R'], alpha_L, alpha_R)
            losses[i] = res
            base_nll = res['nll']
        return losses

    def training_step(self, batch, batch_idx, t_set, T0=0.0):
        self.train()
        return self.step(batch, batch_idx, t_set, T0)

    @torch.no_grad()
    def validation_step(self, batch, batch_idx, t_set, T0=0.0):
        self.eval()
        return self.step(batch, batch_idx, t_set, T0)

    @torch.no_grad()
    def validation_step_full(self, batch, batch_idx, t_set, T0=0.0):
        self.eval()
        t_set = [int(T0)] + list(t_set)
        int_time = torch.from_numpy(np.array(t_set) * self.time_scale + self.int_t_to_noise)

        x0 = batch['base']['X'].to(self.device)

        res = self.forward(int_time, x0)['xs']
        
        xs = batch['x'].float().to(self.device)
        ts = batch['t'].float().to(self.device)

        losses = {}
        for i, t in enumerate(t_set[1:], start=1):
            xt = xs[ts == t]
            emd = compute_emd2(xt.cpu().detach(), res[i].cpu().detach())
            losses[i] = { 'emd' : emd }

        return losses

    @torch.no_grad()
    def validation(self, ds, t_set):
        emds = []
        for i in range(len(t_set)):
            target_idx = ds.get_subset_index(t_set[i])
            target_X = ds.get_data(target_idx)["X"].float()
            if i == 0:
                source_X = ds.base_sample(len(target_X))["X"].float()
                int_time = [ float(ds.T0), float(t_set[i])]
            else:
                source_idx = ds.get_subset_index(t_set[i - 1])
                source_X = ds.get_data(source_idx)["X"].float()
                int_time = [float(t_set[i-1]), float(t_set[i])]
            
            pred_sample = self.sample(source_X, int_time)
            emd = compute_emd2(target_X.cpu(), pred_sample[:, -1].cpu())
            emds.append(emd)
        return { 'avg_emd' : np.mean(emds), 'emds': emds }

    def training_epoch_end(self, outputs):
        score = {}
        ti_set = outputs[0].keys()
        sum_avg_loss = 0.0

        for i in ti_set:
            score[f'k={i}'] = dict(
                avg_loss=torch.mean(torch.tensor([ out[i]['loss'].item() for out in outputs ]).flatten()).item(),
                avg_loss_D=torch.mean(torch.tensor([ out[i]['loss_D'].item() for out in outputs ]).flatten()).item(),
                avg_loss_L=torch.mean(torch.tensor([ out[i]['loss_L'].item() for out in outputs ]).flatten()).item(),
                avg_loss_R=torch.mean(torch.tensor([ out[i]['loss_R'].item() for out in outputs ]).flatten()).item(),
            )
            sum_avg_loss +=  score[f'k={i}'][f"avg_loss"]

        avg_loss =  sum_avg_loss / len(ti_set)
        logs = { 'avg_loss' : avg_loss  }
        logs.update(score)
        return { 'avg_loss' : avg_loss, 'log' : logs  }

    def validation_epoch_end(self, outputs):
        score = {}
        ti_set = outputs[0].keys()
        sum_avg_loss = sum_avg_loss_D = 0.0
        for i in ti_set:
            score[f'k={i}'] = dict(
                avg_loss=torch.mean(torch.tensor([ out[i]['loss'].item() for out in outputs ]).flatten()).item(),
                avg_loss_D=torch.mean(torch.tensor([ out[i]['loss_D'].item() for out in outputs ]).flatten()).item(),
                avg_loss_L=torch.mean(torch.tensor([ out[i]['loss_L'].item() for out in outputs ]).flatten()).item(),
                avg_loss_R=torch.mean(torch.tensor([ out[i]['loss_R'].item() for out in outputs ]).flatten()).item()
            )
            sum_avg_loss +=  score[f'k={i}'][f"avg_loss"]
            sum_avg_loss_D += score[f'k={i}'][f"avg_loss_D"]
        
        avg_loss =  sum_avg_loss / len(ti_set) 
        avg_loss_D = sum_avg_loss_D / len(ti_set)
        logs = { 'avg_loss' : avg_loss, 'avg_loss_D' : avg_loss_D,  }
        logs.update(score)
        return { 'avg_loss' : avg_loss, 'avg_loss_D' : avg_loss_D , 'log' : logs  }

    @torch.no_grad()
    def validation_epoch_end_full(self, outputs):
        score = {}
        ti_set = outputs[0].keys()
        sum_avg_emd = 0.0
        for i in ti_set:
            score[f'k={i}'][f"avg_emd"] = torch.mean(torch.tensor([ out[t]['emd'] for out in outputs ]).flatten()).item()
            sum_avg_emd += score[f'k={i}'][f"avg_emd"]
        
        avg_emd = sum_avg_emd / len(ti_set)
        logs = { 'avg_emd' : avg_emd }
        logs.update(score)
        return { 'avg_emd' : avg_emd , 'log' : logs  }

    @torch.no_grad()
    def sample(self, x0, int_time):
        self.eval()
        int_time = np.array(int_time) * self.time_scale + self.int_t_to_noise
        res = self.net(int_time, x0.float().to(self.device))
        return res["xs"].transpose(0, 1)

    def state_dict(self, optimizer, scheduler=None):
        dic =  {
            "net": deepcopy(self.net.state_dict()),
            "optimizer": deepcopy(optimizer.state_dict())
        }
        if not scheduler is None:
            dic["scheduler"] = deepcopy(scheduler.state_dict())
        
        if AMP:
            dic['amp'] = deepcopy(amp.state_dict())
        return dic

    def load_model(self, checkpoint, amp=False):
        print(f"load model of epoch={checkpoint['epochs']}")
        self.net.load_state_dict(checkpoint["net"])
        if amp:
            amp.load_state_dict(checkpoint["amp"])

    def parameters_lr(self):
        return self.net.parameters_lr()