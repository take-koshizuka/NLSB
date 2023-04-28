import random
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import time
from utils import EarlyStopping
from tqdm import tqdm
from pathlib import Path
import argparse

from model import ODENet, SDENet, SDE_MODEL_NAME, ODE_MODEL_NAME, LAGRANGIAN_NAME
from dataset import OrnsteinUhlenbeckSDE_Dataset, BalancedBatchSampler, scRNASeq, UniformDataset, HighDimOrnsteinUhlenbeckSDE_Dataset

try:
    import apex.amp as amp
    AMP = True
except ImportError:
    AMP = False

PAT = 200

def fix_seed(seed):
    # random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    
def main(cfg, checkpoint_dir="checkpoints/tmp", resume_path="", gpu=0, me=5000):
    checkpoint_dir = Path(checkpoint_dir)
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    fix_seed(cfg['seed'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    times = []
    emd_history = []
    if cfg['dataset']['name'] == "ornstein-uhlenbeck-sde":
        tr_ds = OrnsteinUhlenbeckSDE_Dataset(device=device, t_size=cfg['dataset']['t_size'], data_size=cfg['train_size'], 
                                        t_0=cfg['dataset']['t_0'], t_T = cfg['dataset']['t_T'], sigma=cfg['dataset']['sigma'])
        va_ds = OrnsteinUhlenbeckSDE_Dataset(device=device, t_size=cfg['dataset']['t_size'], data_size=cfg['val_size'], 
                                        t_0=cfg['dataset']['t_0'], t_T = cfg['dataset']['t_T'], sigma=cfg['dataset']['sigma'])
    
    elif cfg['dataset']['name'] == "highdim-ornstein-uhlenbeck-sde":
        tr_ds = HighDimOrnsteinUhlenbeckSDE_Dataset(device=device, t_size=cfg['dataset']['t_size'], z_dim=cfg['dataset']['z_dim'], data_size=cfg['train_size'], 
                                            t_0=cfg['dataset']['t_0'], t_T = cfg['dataset']['t_T'], sigma=cfg['dataset']['sigma'])
        va_ds = HighDimOrnsteinUhlenbeckSDE_Dataset(device=device, t_size=cfg['dataset']['t_size'], z_dim=cfg['dataset']['z_dim'], data_size=cfg['val_size'], 
                                            t_0=cfg['dataset']['t_0'], t_T = cfg['dataset']['t_T'], sigma=cfg['dataset']['sigma'])

    elif cfg['dataset']['name'] == "scRNA":
        tr_ds = scRNASeq([cfg['dataset']['train_data_path']], cfg['dataset']['dim'], use_v=cfg['dataset']['use_v'], LMT=cfg['LMT'])
        va_ds = scRNASeq([cfg['dataset']['val_data_path']], cfg['dataset']['dim'], use_v=cfg['dataset']['use_v'], LMT=cfg['LMT'], scaler=tr_ds.get_scaler())

    elif cfg['dataset']['name'] == "uniform":
        tr_ds = UniformDataset(device=device, t_0=cfg['dataset']['t_0'], t_T = cfg['dataset']['t_T'], data_size=cfg['train_size'])
        va_ds = UniformDataset(device=device, t_0=cfg['dataset']['t_0'], t_T = cfg['dataset']['t_T'], data_size=cfg['val_size'])

    else:
        raise ValueError("The dataset name does not exist.")
    
    t_set = tr_ds.get_label_set()
    # idx = np.arange(len(ds))
    LMT = cfg['LMT'] in t_set[:-1]
    train_t_set = t_set[:]
    if LMT:
        print("LMT")
        # leave middle points at t=cfg['LMT']
        train_t_set = set(train_t_set)
        train_t_set.discard(cfg['LMT'])
        train_t_set = list(train_t_set)
    
    ## define dataloaders
    batch_sampler_tr = BalancedBatchSampler(tr_ds, cfg['dataset']['batch_size'])
    batch_sampler_va = BalancedBatchSampler(va_ds, cfg['dataset']['val_batch_size'])
    tr_dl = DataLoader(tr_ds, batch_sampler=batch_sampler_tr)
    va_dl = DataLoader(va_ds, batch_sampler=batch_sampler_va)

    # Define model
    model_name = cfg['model_name'].lower()
    if model_name in SDE_MODEL_NAME:
        if cfg['lagrangian_name'] == "null" or cfg['lagrangian_name'] == "potential-free":
            L = LAGRANGIAN_NAME[cfg['lagrangian_name']]()
        elif cfg['lagrangian_name'] == "latent-potential-free":
            L = LAGRANGIAN_NAME[cfg['lagrangian_name']](P=tr_ds.P)
        elif cfg['lagrangian_name'] == "cellular":
            L = LAGRANGIAN_NAME["cellular"](tr_ds.full_data['X'], tr_ds.full_data['t'], **cfg['lagrangian'], device=device)
        else:
            raise NotImplementedError
        net = SDE_MODEL_NAME[model_name](**cfg['model'], lagrangian=L)
        model = SDENet(net, device)
        MODEL = 'sde'
    elif model_name in ODE_MODEL_NAME:
        net = ODE_MODEL_NAME[model_name](**cfg['model'])
        model = ODENet(net, device)
        MODEL = 'ode'
    else:
        raise ValueError("The model name does not exist.")
    model.to(device)

    # Define optimizer and scheduler (optional)
    optimizer = optim.Adam(model.parameters_lr(), lr=cfg['optim']['lr'])
    # scheduler = 

    early_stopping = EarlyStopping('avg_emd', 'min')
    init_epochs = 1
    max_epochs = cfg['epochs']

    if AMP:
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

    if not resume_path == "":
        checkpoint = torch.load(resume_path, map_location=lambda storage, loc: storage)
        model.load_model(checkpoint)
        optimizer.load_state_dict(checkpoint["optimizer"])
        # scheduler.load_state_dict(checkpoint["scheduler"])
        init_epochs = checkpoint['epochs'] + 1
        model.to(device)

    writer = SummaryWriter(log_dir=f"./runs/{str(Path(*checkpoint_dir.parts[1:]))}")

    if MODEL == "sde":
        ts = train_t_set
    elif MODEL == "ode":
        ts = [int(tr_ds.T0)] + train_t_set
    

    cnt = 0
    v = float('inf')
    for i in tqdm(range(init_epochs, max_epochs + 1)):
        # training phase
        outputs = []
        torch.cuda.synchronize()
        start = time.time()
        for batch_idx, train_batch in enumerate(tqdm(tr_dl, leave=False)):
            train_batch['base'] = tr_ds.base_sample(cfg['dataset']['batch_size'])
            optimizer.zero_grad()
            out = model.training_step(train_batch, batch_idx, train_t_set, tr_ds.T0)
            outputs.append(out)
            loss = out['loss']
            if AMP:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            del train_batch
        
            optimizer.step()

            if hasattr(model, 'clamp_parameters'):
                model.clamp_parameters()
            # scheduler.step()
        
        torch.cuda.synchronize()
        elapsed_time = time.time() - start
        times.append(elapsed_time)
        train_result = model.training_epoch_end(outputs)

        writer.add_scalar(f'loss/train', train_result['log']['avg_loss'], i)
        for j, t in enumerate(ts):
            writer.add_scalars(f'loss_{t}/train', train_result['log'][f'k={j}'], i)

        # validation phase
        outputs = []    
        for batch_idx, val_batch in enumerate(tqdm(va_dl, leave=False)):
            val_batch['base'] = va_ds.base_sample(cfg['dataset']['val_batch_size'])
            out = model.validation_step(val_batch, batch_idx, train_t_set, va_ds.T0)
            outputs.append(out)
            del val_batch

        val_result = model.validation_epoch_end(outputs)
        emd_result = model.validation(va_ds, train_t_set)
        val_result.update(emd_result)
        emd_history.append(emd_result['avg_emd'])

        writer.add_scalar(f'loss/val', val_result['log']['avg_loss'], i)
        for j, t in enumerate(ts):
            writer.add_scalars(f'loss_{t}/val', val_result['log'][f'k={j}'], i)
        for j, t in enumerate(train_t_set):
            writer.add_scalar(f'emd_{t}', emd_result['emds'][j], i)
        
        # early_stopping
        if early_stopping.judge(val_result):
            early_stopping.update(val_result)
            state_dict = model.state_dict(optimizer)
            state_dict['epochs'] = i
            early_stopping.best_state = state_dict

        if i % cfg['checkpoint_period'] == 0:
            state_dict = model.state_dict(optimizer)
            state_dict['epochs'] = i
            torch.save(state_dict, str(checkpoint_dir / f"model-{i}.pt"))
        
        if i >= me and len(emd_history) >= 10:
            mavg_emd = np.mean(emd_history[-10:])
            if mavg_emd < v:
                v = mavg_emd
                cnt = 0
                state_dict_i = model.state_dict(optimizer)
                state_dict_i['epochs'] = i
                state_dict_i['times'] = np.sum(times)
            else:
                cnt += 1

            if cnt > PAT:
                torch.save(state_dict_i, str(checkpoint_dir / f"model-t-{state_dict_i['epochs']}.pt"))
                v = -float('inf')
                break

    best_state = early_stopping.best_state
    model.load_model(best_state)
    torch.save(best_state, str(checkpoint_dir / "best-model.pt"))
    with open(str(checkpoint_dir / "train_config.json"), "w") as f:
        json.dump(cfg, f, indent=4)
    
    np.savez(str(checkpoint_dir / "time-emd"), t=np.array(times), emd=np.array(emd_history))
    writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', '-c', help="Path to the configuration file for training.", type=str, required=True)
    parser.add_argument('-dir', '-d', help="Path to the directory where the checkpoint of the model is stored.", type=str, required=True)
    parser.add_argument('-resume', '-r', help="Path to the checkpoint of the model you want to resume training.", type=str, default="")
    parser.add_argument('-gpu', type=int, default=0)
    parser.add_argument('-seed', '-s', type=int, default=57)
    parser.add_argument('-me', type=int, default=1000)
    args = parser.parse_args()

    ## example
    # args.config = "config/train.json"
    # args.dir = "checkpoints/tmp"
    # args.resume = "checkpoints/tmp/best-model.pt"
    ##

    with open(args.config, 'r') as f:
        cfg = json.load(f)
    cfg['seed'] = args.seed
    main(cfg, Path(args.dir), args.resume, args.gpu, args.me)
