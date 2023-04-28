import torch
import torch.nn.functional as F
import numpy as np
import json
import random
import argparse
from tqdm import tqdm
from pathlib import Path

from utils import compute_MMD, compute_emd2, marginal_distribution_discrepancy
from dataset import OrnsteinUhlenbeckSDE_Dataset, scRNASeq
from model import ODENet, SDENet, SDE_MODEL_NAME, ODE_MODEL_NAME, LAGRANGIAN_NAME

def fix_seed(seed):
    # random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def main(eval_cfg, checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_dir = Path(checkpoint_path).parent
    fix_seed(eval_cfg['seed'])
    train_config_path = Path(checkpoint_dir) / "train_config.json"
    with open(train_config_path, 'r') as f:
        train_cfg = json.load(f)

    if train_cfg['dataset']['name'] == "ornstein-uhlenbeck-sde":
        tr_ds = OrnsteinUhlenbeckSDE_Dataset(device=device, t_size=train_cfg['dataset']['t_size'], data_size=train_cfg['train_size'], mu=train_cfg['dataset']['mu'], 
                                        theta=train_cfg['dataset']['theta'], sigma=train_cfg['dataset']['sigma'])
        ds = OrnsteinUhlenbeckSDE_Dataset(device=device, t_size=train_cfg['dataset']['t_size'], data_size=eval_cfg['num_points'], mu=train_cfg['dataset']['mu'], 
                                        theta=train_cfg['dataset']['theta'], sigma=train_cfg['dataset']['sigma'])
    elif train_cfg['dataset']['name'] == "scRNA":
        # LMTを今だけ
        use_v = train_cfg['dataset']['use_v'] if 'use_v' in train_cfg['dataset'] else False
        tr_ds = scRNASeq([train_cfg['dataset']['train_data_path']], train_cfg['dataset']['dim'], use_v=use_v, LMT=train_cfg['LMT'])
        # va_ds = scRNASeq([train_cfg['dataset']['val_data_path']], train_cfg['dataset']['dim'], use_v=use_v, scaler=tr_ds.get_scaler())
        ds = scRNASeq([train_cfg['dataset']['test_data_path']], train_cfg['dataset']['dim'], use_v=use_v, scaler=tr_ds.get_scaler())
    else:
        raise ValueError("The dataset name does not exist.")

    # Define model
    model_name = train_cfg['model_name'].lower()
    if model_name in SDE_MODEL_NAME:
        if train_cfg['lagrangian_name'] == "null" or train_cfg['lagrangian_name'] == "potential-free":
            L = LAGRANGIAN_NAME[train_cfg['lagrangian_name']]()
        elif train_cfg['lagrangian_name'] == "cellular":
            L = LAGRANGIAN_NAME["cellular"](tr_ds.full_data['X'], tr_ds.full_data['t'], **train_cfg['lagrangian'], device=device)
        else:
            raise NotImplementedError
        net = SDE_MODEL_NAME[model_name](**train_cfg['model'], lagrangian=L)
        model = SDENet(net, device)
        MODEL = 'sde'
    elif model_name in ODE_MODEL_NAME:
        net = ODE_MODEL_NAME[model_name](**train_cfg['model'])
        model = ODENet(net, device)
        MODEL = 'ode'
    else:
        raise ValueError("The model name does not exist.")
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    model.load_model(checkpoint)
    model.to(device)
    
    
    if train_cfg['dataset']['name'] == "scRNA":
        if MODEL == 'sde':
            one_step_L1, all_step_L1, one_step_L2, all_step_L2 = eval_rna_sde(ds, model, eval_cfg)
        else:
            one_step_L1, all_step_L1, one_step_L2, all_step_L2 = eval_rna_ode(ds, model, eval_cfg)
        
        with open(str(checkpoint_dir / f"mdd.json"), "w") as f:
            json.dump({
                'seed' : eval_cfg['seed'],
                'L1': { 'one-step' : one_step_L1, 'all-step' : all_step_L1 },
                'L2': { 'one-step' : one_step_L2, 'all-step' : all_step_L2 },
            }, f, indent=4)
    
    else:

        one_step, all_step = eval_ou_sde(ds, model, eval_cfg, MODEL)
        with open(str(checkpoint_dir / f"mdd.json"), "w") as f:
            json.dump({
                'seed' : eval_cfg['seed'],
                'L2': { 'one-step' : one_step, 'all-step' : all_step },
            }, f, indent=4)

def eval_rna_sde(ds, model, eval_cfg):
    t_set = ds.get_label_set()
    int_times = [float(ds.T0)] + [ float(t) for t in t_set ]
    int_times = torch.tensor(int_times)
    ### forward
    ## one-step
    forward_one_step_L1 = {}
    forward_one_step_L2 = {}

    sum_emds_L1 = []
    sum_emds_L2 = []
    for i in tqdm(range(len(t_set))):
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
        t1 = t_set[i]
        pred_traj = model.sample_with_uncertainty(source_X, int_time, eval_cfg['num_repeat'], source_V)
        # pred_sample = torch.mean(pred_sample, axis=2)
        emds_L1 = []
        emds_L2 = []
        for j in range(eval_cfg['num_repeat']):
            emd_L1 = compute_emd2(target_X.cpu(), pred_traj[:, -1, j].cpu(), p=1)
            emd_L2 = compute_emd2(target_X.cpu(), pred_traj[:, -1, j].cpu(), p=2)
            emds_L1.append(emd_L1)
            emds_L2.append(emd_L2)
        
        sum_emds_L1.append(np.mean(emds_L1))
        sum_emds_L2.append(np.mean(emds_L2))
        forward_one_step_L1[f't={t1}'] = f'{np.mean(emds_L1)} ± {np.std(emds_L1)}'
        forward_one_step_L2[f't={t1}'] = f'{np.mean(emds_L2)} ± {np.std(emds_L2)}'

    print("[forward] one-step L1:", forward_one_step_L1, 'average: ', np.mean(sum_emds_L1) )
    print("[forward] one-step L2:", forward_one_step_L2, 'average: ', np.mean(sum_emds_L2))

    ## all-step
    source = ds.base_sample()
    source_X = source["X"].float()
    source_V = source["V"].float() if "V" in source else None
    pred_traj = model.sample_with_uncertainty(source_X, int_times, eval_cfg['num_repeat'], source_V)
    forward_all_step_L1 = {}
    forward_all_step_L2 = {}
    for i in tqdm(range(len(t_set))):
        target_idx = ds.get_subset_index(t_set[i])
        target_X = ds.get_data(target_idx)["X"].float()
        emds_L1 = []
        emds_L2 = []
        for j in range(eval_cfg['num_repeat']):
            emd_L1 = compute_emd2(target_X.cpu(), pred_traj[:, i+1, j].cpu(), p=1)
            emd_L2 = compute_emd2(target_X.cpu(), pred_traj[:, i+1, j].cpu(), p=2)   
            emds_L1.append(emd_L1)
            emds_L2.append(emd_L2)

        t1 = t_set[i]
        forward_all_step_L1[f't={t1}'] = f'{np.mean(emds_L1)} ± {np.std(emds_L1)}'
        forward_all_step_L2[f't={t1}'] = f'{np.mean(emds_L2)} ± {np.std(emds_L2)}'
    print("[forward] all-step: L1", forward_all_step_L1)
    print("[forward] all-step: L2", forward_all_step_L2)
    return  forward_one_step_L1, forward_all_step_L1,\
            forward_one_step_L2, forward_all_step_L2
    
def eval_rna_ode(ds, model, eval_cfg):
    t_set = ds.get_label_set()
    int_times = [float(ds.T0)] + [ float(t) for t in t_set ]
    int_times = torch.tensor(int_times)
    ### forward
    ## one-step
    forward_one_step_L1 = {}
    forward_one_step_L2 = {}

    sum_emds_L1 = []
    sum_emds_L2 = []
    for i in tqdm(range(len(t_set))):
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
        t1 = t_set[i]
        pred_traj = model.sample(source_X, int_time)
        emd_L1 = compute_emd2(target_X.cpu(), pred_traj[:, -1].cpu(), p=1)
        emd_L2 = compute_emd2(target_X.cpu(), pred_traj[:, -1].cpu(), p=2)
        
        sum_emds_L1.append(emd_L1)
        sum_emds_L2.append(emd_L2)
        
        forward_one_step_L1[f't={t1}'] = emd_L1
        forward_one_step_L2[f't={t1}'] = emd_L2

    print("[forward] one-step L1:", forward_one_step_L1, np.mean(sum_emds_L1))
    print("[forward] one-step L2:", forward_one_step_L2, np.mean(sum_emds_L2))
    ## all-step
    source_X = ds.base_sample()["X"].float()
    pred_traj = model.sample(source_X, int_times)
    forward_all_step_L1 = {}
    forward_all_step_L2 = {}
    for i in tqdm(range(len(t_set))):
        target_idx = ds.get_subset_index(t_set[i])
        target_X = ds.get_data(target_idx)["X"].float()
        t1 = t_set[i]
        emd_L1 = compute_emd2(target_X.cpu(), pred_traj[:, i+1].cpu(), p=1)
        emd_L2 = compute_emd2(target_X.cpu(), pred_traj[:, i+1].cpu(), p=2)
        forward_all_step_L1[f't={t1}'] = emd_L1
        forward_all_step_L2[f't={t1}'] = emd_L2

    print("[forward] all-step: L1", forward_all_step_L1)
    print("[forward] all-step: L2", forward_all_step_L2)

    return  forward_one_step_L1, forward_all_step_L1, forward_one_step_L2, forward_all_step_L2


def eval_ou_sde(ds, model, eval_cfg, MODEL):
    t_set = ds.get_label_set()
    int_times = [ds.T0]
    for i in range(len(t_set)):
        if i == 0:
            int_time = torch.linspace(ds.T0, t_set[i], eval_cfg["num_timepoints"])
        else:
            int_time = torch.linspace(t_set[i-1], t_set[i], eval_cfg["num_timepoints"])
        int_times.extend(list(int_time.cpu().numpy())[1:])
    int_times = torch.tensor(int_times)

    
    #### forward
    # one-step
    forward_one_step = {}
    """
    for i in tqdm(range(len(t_set))):
        if i == 0:
            source_X = ds.base_sample()["X"].float()
            int_time = torch.linspace(ds.T0, t_set[i], eval_cfg["num_timepoints"])
            t0 = ds.T0
        else:
            source_idx = ds.get_subset_index(t_set[i - 1])
            source_X = ds.get_data(source_idx)["X"].float()
            int_time = torch.linspace(t_set[i-1], t_set[i], eval_cfg["num_timepoints"])
            t0 = t_set[i-1]
        
        ref_traj = ds.sample(source_X, int_time)
        if MODEL == 'sde':
            pred_traj = model.sample_with_uncertainty(source_X, int_time, eval_cfg['num_repeat'])
        else:
            pred_traj = model.sample(source_X, int_time)
        
        mdd_dict = marginal_distribution_discrepancy(ref_traj, pred_traj, int_time)
        forward_one_step[f't0={t0}'] = mdd_dict
    """
    ## all-step
    forward_all_step = {}
    source_X = ds.base_sample()["X"].float()
    ref_traj = ds.sample(source_X, int_times)
    if MODEL == 'sde':
        pred_traj = model.sample_with_uncertainty(source_X, int_times, eval_cfg['num_repeat'])
    else:
        pred_traj = model.sample(source_X, int_times)
    mdd_dict = marginal_distribution_discrepancy(ref_traj, pred_traj, int_times)
    forward_all_step[f't0={ds.T0}'] = mdd_dict
    print('mean', [ x['mean'] for x in mdd_dict.values() ])
    print('std', [ x['std'] for x in mdd_dict.values() ])
    
    return forward_one_step, forward_all_step


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', '-c', help="Path to the configuration file for conversion.", type=str, required=True)
    parser.add_argument('-path', '-p', help="Path to the checkpoint of the model", type=str, required=True)
    parser.add_argument('-seed', '-s', type=int, default=57)
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        cfg = json.load(f)
    cfg['seed'] = args.seed
    main(cfg, Path(args.path))