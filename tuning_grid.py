import json
import optuna
import argparse
from pathlib import Path
import random
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.optim as optim

from utils import EarlyStopping
from tqdm import tqdm
from pathlib import Path
import argparse

from model import ODENet, SDENet, SDE_MODEL_NAME, ODE_MODEL_NAME, LAGRANGIAN_NAME
from dataset import OrnsteinUhlenbeckSDE_Dataset, BalancedBatchSampler, scRNASeq, PotentialSDE_Dataset

try:
    import apex.amp as amp
    AMP = True
except ImportError:
    AMP = False

def fix_seed(seed):
    # random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    

def objective(trial, cfg, name, gpu, leave):
    alpha_L = trial.suggest_float('alpha_L', 0.0, 0.5)
    alpha_R = trial.suggest_float('alpha_R', 0.0, 0.5)
    #alpha_3 = trial.suggest_float('alpha_3', 0.0, 0.5)
    #alpha_4 = trial.suggest_float('alpha_4', 0.0, 0.5)
    

    cfg['model']['criterion_cfg']['alpha_L'] = alpha_L
    cfg['model']['criterion_cfg']['alpha_R'] = alpha_R

    checkpoint_dir=f"checkpoints/{name}/exp"
    resume_path=""

    checkpoint_dir = Path(checkpoint_dir)
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    fix_seed(cfg['seed'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    ## define dataset
    if cfg['dataset']['name'] == "ornstein-uhlenbeck-sde":
        tr_ds = OrnsteinUhlenbeckSDE_Dataset(device=device, t_size=cfg['dataset']['t_size'], data_size=cfg['train_size'], 
                                        mu=cfg['dataset']['mu'], theta=cfg['dataset']['theta'], sigma=cfg['dataset']['sigma'])
        va_ds = OrnsteinUhlenbeckSDE_Dataset(device=device, t_size=cfg['dataset']['t_size'], data_size=cfg['val_size'], 
                                        mu=cfg['dataset']['mu'], theta=cfg['dataset']['theta'], sigma=cfg['dataset']['sigma'])
    
    elif cfg['dataset']['name'] == "scRNA":
        tr_ds = scRNASeq([cfg['dataset']['train_data_path']], cfg['dataset']['dim'], use_v=cfg['dataset']['use_v'], LMT=leave)
        va_ds = scRNASeq([cfg['dataset']['val_data_path']], cfg['dataset']['dim'], use_v=cfg['dataset']['use_v'], LMT=leave, scaler=tr_ds.get_scaler())
    
    elif cfg['dataset']['name'] == "potential-sde":
        tr_ds = PotentialSDE_Dataset(device=device, t_size=cfg['dataset']['t_size'], data_size=cfg['train_size'], t_0=cfg['dataset']['t_0'], t_T = cfg['dataset']['t_T'],
                                        a=cfg['dataset']['a'], sigma=cfg['dataset']['sigma'])
        va_ds = PotentialSDE_Dataset(device=device, t_size=cfg['dataset']['t_size'], data_size=cfg['val_size'], t_0=cfg['dataset']['t_0'], t_T = cfg['dataset']['t_T'], 
                                        a=cfg['dataset']['a'], sigma=cfg['dataset']['sigma'])
    else:
        raise ValueError("The dataset name does not exist.")
    
    t_set = tr_ds.get_label_set()
    # idx = np.arange(len(ds))
    LMT = leave in t_set[:-1]
    train_t_set = t_set[:]
    if LMT:
        print("LMT")
        # leave middle points at t=cfg['LMT']
        train_t_set = set(train_t_set)
        train_t_set.discard(leave)
        train_t_set = list(train_t_set)
    
    ## define dataloaders
    batch_sampler_tr = BalancedBatchSampler(tr_ds, cfg['dataset']['batch_size'])
    batch_sampler_va = BalancedBatchSampler(va_ds, cfg['dataset']['val_batch_size'])
    tr_dl = DataLoader(tr_ds, batch_sampler=batch_sampler_tr)
    va_dl = DataLoader(va_ds, batch_sampler=batch_sampler_va)

    early_stopping = EarlyStopping('avg_emd', 'min')
    init_epochs = 1
    max_epochs = cfg['epochs']

    model_name = cfg['model_name'].lower()
    if model_name in SDE_MODEL_NAME:
        if cfg['lagrangian_name'] == "null" or cfg['lagrangian_name'] == "potential-free":
            L = LAGRANGIAN_NAME[cfg['lagrangian_name']]()
        elif cfg['lagrangian_name'] == "cellular":
            L = LAGRANGIAN_NAME["cellular"](tr_ds.full_data['X'], tr_ds.full_data['t'], **cfg['lagrangian'], device=device)
        elif cfg['lagrangian_name'] == "newtonian":
            if cfg['dataset']['name'] == "potential-sde":
                L = LAGRANGIAN_NAME["newtonian"](**cfg['lagrangian'], U=tr_ds.potential)
            else:
                L = LAGRANGIAN_NAME["newtonian"](**cfg['lagrangian'])
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
    if AMP:
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

    if not resume_path == "":
        checkpoint = torch.load(resume_path, map_location=lambda storage, loc: storage)
        model.load_model(checkpoint)
        optimizer.load_state_dict(checkpoint["optimizer"])
        # scheduler.load_state_dict(checkpoint["scheduler"])
        init_epochs = checkpoint['epochs'] + 1
        model.to(device)

    with open(str(checkpoint_dir / "train_config.json"), "w") as f:
        json.dump(cfg, f, indent=4)

    if MODEL == "sde":
        ts = train_t_set
    elif MODEL == "ode":
        ts = [int(tr_ds.T0)] + train_t_set

    for i in tqdm(range(init_epochs, max_epochs + 1)):
        # training phase
        outputs = []
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
        val_result = model.validation(va_ds, train_t_set)

        if early_stopping.judge(val_result):
            early_stopping.update(val_result)
            state_dict = model.state_dict(optimizer)
            state_dict['epochs'] = i
            early_stopping.best_state = state_dict

        if i % cfg['checkpoint_period'] == 0:
            state_dict = model.state_dict(optimizer)
            state_dict['epochs'] = i
            torch.save(state_dict, str(checkpoint_dir / f"model-{i}.pt"))

        # report
        intermediate_value = early_stopping.get_value()
        trial.report(intermediate_value, i)

    best_state = early_stopping.best_state
    model.load_model(best_state)
    torch.save(best_state, str(checkpoint_dir / "best-model.pt"))
    
    return early_stopping.get_value()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-name', '-n', type=str, required=True)
    parser.add_argument('-config', '-c', help="Path to the configuration file for training.", type=str, required=True)
    parser.add_argument('-trial', '-t', type=int, default=20)
    parser.add_argument('-gpu', type=int, default=0)
    parser.add_argument('-leave', type=int, default=(-1))
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = json.load(f)

    search_space = {
        "alpha_L": [0.1, 0.01, 0.001],
        "alpha_R": [0.1, 0.01, 0.001]
    }

    study = optuna.create_study(study_name=args.name, direction='minimize', storage=f'sqlite:///{args.name}.db', load_if_exists=True, 
                                    sampler=optuna.samplers.GridSampler(search_space))
    study.optimize(lambda trial: objective(trial, cfg, args.name, args.gpu, args.leave), n_trials=args.trial)

    df = study.trials_dataframe(attrs=('number', 'value', 'params', 'state'))

    Path("tuning").mkdir(parents=True, exist_ok=True)
    df.to_csv(f'tuning/{args.name}.csv')

