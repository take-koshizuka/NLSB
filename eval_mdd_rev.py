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
from model import ODENet, SDENet, SDE_MODEL_NAME, ODE_MODEL_NAME, LAGRANGIAN_NAME, ReverseSDENet

def fix_seed(seed):
    # random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def main(eval_cfg, checkpoint_path, fsde_checkpoint_dir="checkpoints/ou-sde/neuralSDE/NLSB/expM"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_dir = Path(checkpoint_path).parent
    fix_seed(eval_cfg['seed'])
    train_config_path = Path(checkpoint_dir) / "train_config.json"
    with open(train_config_path, 'r') as f:
        train_cfg = json.load(f)

    fsde_train_config_path = Path(fsde_checkpoint_dir) / "train_config.json"
    with open(fsde_train_config_path, 'r') as f:
        fsde_train_cfg = json.load(f)

    
    tr_ds = OrnsteinUhlenbeckSDE_Dataset(device=device, t_size=train_cfg['dataset']['t_size'], data_size=train_cfg['train_size'], mu=train_cfg['dataset']['mu'], 
                                        theta=train_cfg['dataset']['theta'], sigma=train_cfg['dataset']['sigma'])
    ds = OrnsteinUhlenbeckSDE_Dataset(device=device, t_size=train_cfg['dataset']['t_size'], data_size=eval_cfg['num_points'], mu=train_cfg['dataset']['mu'], 
                                        theta=train_cfg['dataset']['theta'], sigma=train_cfg['dataset']['sigma'])

    # Define model
    if 'model_name' in train_cfg:
        model_name = train_cfg['model_name'].lower()
    else:
        model_name = "ito"

    # Define model
    # model_name = train_cfg['model_name'].lower()
    if model_name in SDE_MODEL_NAME:
        fnet = SDE_MODEL_NAME["ito"](**fsde_train_cfg['model'])
        fsde = SDENet(fnet, device)
        bnet = SDE_MODEL_NAME["rev-sde"](fsde.net)
        model = ReverseSDENet(bnet, device)
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

    eval_ou_sde(ds, model, eval_cfg, MODEL)

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
    
    ## all-step
    source_X = ds.base_sample()["X"].float()
    ref_traj = ds.sample(source_X, int_times)
    source = ref_traj[:, -1, :].float()
    rev_int_times = torch.flip(int_times, dims=(0, ))
    
    if MODEL == 'sde':
        pred_traj = model.sample_with_uncertainty(source, rev_int_times, eval_cfg['num_repeat'])
    else:
        pred_traj = model.sample(source, rev_int_times)

    mdd_dict = marginal_distribution_discrepancy(torch.flip(ref_traj, dims=[1]), pred_traj, rev_int_times)
    print(mdd_dict)


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