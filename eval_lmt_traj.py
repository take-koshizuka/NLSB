
import torch
import numpy as np
import json
import random
import argparse
from pathlib import Path

from dataset import scRNASeq
from model import SDENet, SDE_MODEL_NAME, LAGRANGIAN_NAME
from utils import conditional_distribution_discrepancy

def fix_seed(seed):
    # random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def main(eval_cfg, checkpoint_path, checkpoint_path_lmt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fix_seed(eval_cfg['seed'])

    checkpoint_dir = Path(checkpoint_path).parent
    train_config_path = Path(checkpoint_dir) / "train_config.json"
    with open(train_config_path, 'r') as f:
        train_cfg = json.load(f)

    checkpoint_dir_lmt = Path(checkpoint_path_lmt).parent
    train_config_path_lmt = Path(checkpoint_dir_lmt) / "train_config.json"
    with open(train_config_path_lmt, 'r') as f:
        train_cfg_lmt = json.load(f)

    if not 'use_v' in train_cfg['dataset']:
        train_cfg['dataset']['use_v'] = False
    if not 'use_v' in train_cfg_lmt['dataset']:
        train_cfg_lmt['dataset']['use_v'] = False
    
    assert train_cfg['dataset']['name'] == "scRNA"
    tr_ds = scRNASeq([train_cfg['dataset']['train_data_path']], train_cfg['dataset']['dim'], use_v=train_cfg['dataset']['use_v'])
    scaler = tr_ds.get_scaler()
    ds = scRNASeq([train_cfg['dataset']['test_data_path']], train_cfg['dataset']['dim'], use_v=train_cfg['dataset']['use_v'], scaler=scaler)

    tr_ds_lmt = scRNASeq([train_cfg['dataset']['train_data_path']], train_cfg['dataset']['dim'], use_v=train_cfg_lmt['dataset']['use_v'], LMT=train_cfg_lmt['LMT'])
    scaler_lmt = tr_ds_lmt.get_scaler()
    ds_lmt = scRNASeq([train_cfg['dataset']['test_data_path']], train_cfg['dataset']['dim'], use_v=train_cfg_lmt['dataset']['use_v'], scaler=scaler_lmt)
    
    model_name = train_cfg['model_name'].lower()
    if train_cfg['lagrangian_name'] == "null" or train_cfg['lagrangian_name'] == "potential-free":
        L = LAGRANGIAN_NAME[train_cfg['lagrangian_name']]()
    elif train_cfg['lagrangian_name'] == "cellular":
        L = LAGRANGIAN_NAME["cellular"](tr_ds.full_data['X'], tr_ds.full_data['t'], **train_cfg['lagrangian'], device=device)
    else:
        raise NotImplementedError
    net = SDE_MODEL_NAME[model_name](**train_cfg['model'], lagrangian=L)
    model = SDENet(net, device)
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    model.load_model(checkpoint)
    model.to(device)
    #############

    # Define model
    model_name = train_cfg_lmt['model_name'].lower()
    if not "lagrangian_name" in train_cfg_lmt:
        train_cfg_lmt["lagrangian_name"] = "null"
    if "lagrangian_cfg" in train_cfg_lmt['model']:
        train_cfg_lmt['model'].pop("lagrangian_cfg")
    
    if train_cfg_lmt['lagrangian_name'] == "null" or train_cfg_lmt['lagrangian_name'] == "potential-free":
        L_lmt = LAGRANGIAN_NAME[train_cfg_lmt['lagrangian_name']]()
    elif train_cfg['lagrangian_name'] == "cellular":
        L_lmt = LAGRANGIAN_NAME["cellular"](tr_ds_lmt.full_data['X'], tr_ds_lmt.full_data['t'], **train_cfg_lmt['lagrangian'], device=device)
    else:
        raise NotImplementedError
    net_lmt = SDE_MODEL_NAME[model_name](**train_cfg_lmt['model'], lagrangian=L_lmt)
    model_lmt = SDENet(net_lmt, device)
    checkpoint_lmt = torch.load(checkpoint_path_lmt, map_location=lambda storage, loc: storage)
    model_lmt.load_model(checkpoint_lmt)
    model_lmt.to(device)

    ## define data
    t_set = ds.get_label_set()
    t_set = [int(ds.T0)] + list(t_set)
    assert train_cfg_lmt['LMT'] in t_set[:-1]

    LMT_idx = t_set.index(train_cfg_lmt['LMT'])
    te_idx = ds.get_subset_index(train_cfg_lmt['LMT'])
    ref_n = len(te_idx)

    t0 = t_set[LMT_idx - 1]
    t1 = t_set[LMT_idx + 1]

    param = ds.scaler_params()
    param_lmt = ds_lmt.scaler_params()

    if LMT_idx == 1:
        test_source_data_ = ds.base_sample()['X'].float()
        n = min(ref_n, len(test_source_data_))
        source = ds.base_sample(n)
    else:
        test_source_idx = ds.get_subset_index(t_set[LMT_idx - 1])
        n = min(ref_n, len(test_source_idx))
        test_source_idx = ds.get_subset_index(t_set[LMT_idx - 1], n)
        source = ds.get_data(test_source_idx)

    test_source_X = source["X"].float()
    data_size = test_source_X.size(0)
    test_source_X_lmt = ((test_source_X * param['scale']) + param['mean'] - param_lmt['mean']) / param_lmt['scale']

    int_time = torch.linspace(float(t0), float(t1), eval_cfg["num_timepoints"])
    pred_traj = model.sample_with_uncertainty(test_source_X, int_time, eval_cfg["num_repeat"])
    pred_traj_lmt = model_lmt.sample_with_uncertainty(test_source_X_lmt, int_time, eval_cfg["num_repeat"])
    
    data_size, t_size,  num_repeat, dim = pred_traj_lmt.size()
    param_lmt = ds_lmt.scaler_params()
    param = ds.scaler_params()

    scale_lmt = param_lmt['scale'].repeat(data_size, t_size, num_repeat, 1).to(device)
    mean_lmt = param_lmt['mean'].repeat(data_size, t_size, num_repeat, 1).to(device)
    scale = param['scale'].repeat(data_size, t_size, num_repeat, 1).to(device)
    mean = param['mean'].repeat(data_size, t_size, num_repeat, 1).to(device)

    pred_traj_lmt =  (((pred_traj_lmt * scale_lmt) + mean_lmt - mean) / scale)
    cdd_dict_L2 = conditional_distribution_discrepancy(pred_traj.float(), pred_traj_lmt.float(), int_time, p=2)

    print("L2-distance")
    print(cdd_dict_L2)
    cdd_dict_L2['mean'] = np.mean(list(cdd_dict_L2.values()))
    print(f"average: {cdd_dict_L2['mean']}")

    cdd_dict_L1 = conditional_distribution_discrepancy(pred_traj.float(), pred_traj_lmt.float(), int_time, p=1)

    print("L1-distance")
    print(cdd_dict_L1)
    cdd_dict_L1['mean'] = np.mean(list(cdd_dict_L1.values()))
    print(f"average: {cdd_dict_L1['mean']}")

    with open(str(checkpoint_dir / f"lmt_traj.json"), "w") as f:
        json.dump({ 
            'seed': eval_cfg['seed'],
            'forward' : { 'L2' : cdd_dict_L2, 'L1' : cdd_dict_L1 },
            }, f, indent=4)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', '-c', help="Path to the configuration file for conversion.", type=str, required=True)
    parser.add_argument('-path', '-p', help="Path to the checkpoint of the model", type=str, required=True)
    parser.add_argument('-path_lmt', '-pl', help="Path to the checkpoint of the model with LMT", type=str, required=True)
    parser.add_argument('-seed', '-s', type=int, default=57)
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        cfg = json.load(f)
    cfg['seed'] = args.seed
    main(cfg, Path(args.path), Path(args.path_lmt))
