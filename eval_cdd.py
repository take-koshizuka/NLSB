
import torch
import torch.nn.functional as F
import numpy as np
import json
import random
import argparse
from tqdm import tqdm
from pathlib import Path

from utils import conditional_distribution_discrepancy, Accumulator
from dataset import OrnsteinUhlenbeckSDE_Dataset
from model import SDENet, SDE_MODEL_NAME, LAGRANGIAN_NAME


def fix_seed(seed):
    # random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def main(eval_config_path, checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_dir = Path(checkpoint_path).parent
    with open(eval_config_path, 'r') as f:
        eval_cfg = json.load(f)
    fix_seed(eval_cfg['seed'])

    train_config_path = Path(checkpoint_dir) / "train_config.json"
    with open(train_config_path, 'r') as f:
        train_cfg = json.load(f)

    data_size = eval_cfg['num_points']
    assert train_cfg['dataset']['name'] == "ornstein-uhlenbeck-sde"
    ds = OrnsteinUhlenbeckSDE_Dataset(device=device, t_size=train_cfg['dataset']['t_size'], data_size=data_size, mu=train_cfg['dataset']['mu'], 
                                            theta=train_cfg['dataset']['theta'], sigma=train_cfg['dataset']['sigma'])

    # Define model
    model_name = train_cfg['model_name'].lower()
    if train_cfg['lagrangian_name'] == "null" or train_cfg['lagrangian_name'] == "potential-free":
        L = LAGRANGIAN_NAME[train_cfg['lagrangian_name']]()
    else:
        raise NotImplementedError
    
    net = SDE_MODEL_NAME[model_name](**train_cfg['model'], lagrangian=L)
    model = SDENet(net, device)
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    model.load_model(checkpoint)
    model.to(device)

    t_set = ds.get_label_set()
    cdd_res = {}
    cdd_acc = Accumulator()
    int_times = [ds.T0]
    for i in tqdm(range(len(t_set))):
        target_idx = ds.get_subset_index(t_set[i])
        target_X = ds.get_data(target_idx)["X"].float()
        if i == 0:
            source_X = ds.base_sample(len(target_X))["X"].float()
            int_time = torch.linspace(ds.T0, t_set[i], eval_cfg["num_timepoints"])
            t0 = ds.T0
        else:
            source_idx = ds.get_subset_index(t_set[i - 1])
            source_X = ds.get_data(source_idx)["X"].float()
            int_time = torch.linspace(t_set[i-1], t_set[i], eval_cfg["num_timepoints"])
            t0 = t_set[i-1]
        
        pred_traj = model.sample_with_uncertainty(source_X, int_time, eval_cfg["num_repeat"])
        ref_traj = ds.sample_with_uncertainty(source_X, int_time, eval_cfg["num_repeat"])

        cdd_dict = conditional_distribution_discrepancy(ref_traj, pred_traj, int_time)
        sum_score = sum(list(cdd_dict.values()))
        cdd_acc.update(sum_score, len(cdd_dict))
        cdd_res[f't0={t0}'] = cdd_dict

        int_times.extend(list(int_time.cpu().numpy())[1:])

    cdd_res['all'] = cdd_acc.compute()

    cdd_res = {}
    cdd_acc = Accumulator()
    source_X = ds.base_sample(len(target_X))["X"].float()
    int_time = torch.tensor(int_times)
    ref_traj = ds.sample_with_uncertainty(source_X, int_time, eval_cfg['num_repeat'])
    pred_traj = model.sample_with_uncertainty(source_X, int_time, eval_cfg['num_repeat'])
    
    cdd_dict = conditional_distribution_discrepancy(ref_traj, pred_traj, int_time)

    with open(str(checkpoint_dir / f"cdd.json"), "w") as f:
        json.dump({ 'one-step' : cdd_res, 'all-step' : cdd_dict }, f, indent=4)
    
    print({ 'one-step' : cdd_res, 'all-step' : cdd_dict })

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', '-c', help="Path to the configuration file for conversion.", type=str, required=True)
    parser.add_argument('-path', '-p', help="Path to the checkpoint of the model", type=str, required=True)

    args = parser.parse_args()
    main(args.config, Path(args.path))
