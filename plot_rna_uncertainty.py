
import torch
import torch.nn.functional as F
import numpy as np
import json
import random
import argparse

from pathlib import Path

from utils import decode
from dataset import scRNASeq
from model import SDENet, ODENet, SDE_MODEL_NAME, ODE_MODEL_NAME
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

plt.style.use(['science', 'notebook'])

def fix_seed(seed):
    # random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


sample_colors = plt.get_cmap("tab10")
cmap_colors = [ '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd' ]
cmap_colors_fill = [plt.get_cmap("Blues"), plt.get_cmap("Oranges"), plt.get_cmap("Greens"), plt.get_cmap('Reds'), plt.get_cmap('Purples') ]

fill_color = '#9ebcda'
mean_color = '#4d004b'

def main(eval_cfg, checkpoint_path_sde, checkpoint_path_ode, out_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_dir = str(Path(checkpoint_path_sde).parent)
    train_config_path = Path(checkpoint_dir) / "train_config.json"
    with open(train_config_path, 'r') as f:
        train_cfg_sde = json.load(f)

    checkpoint_dir = str(Path(checkpoint_path_ode).parent)
    train_config_path = Path(checkpoint_dir) / "train_config.json"
    with open(train_config_path, 'r') as f:
        train_cfg_ode = json.load(f)
    
    fix_seed(eval_cfg['seed'])
    Path(out_dir).mkdir(exist_ok=True, parents=True)

    
    assert train_cfg_sde['dataset']['name'] == "scRNA"
    use_v = train_cfg_sde['dataset']['use_v'] if 'use_v' in train_cfg_sde['dataset'] else False
    tr_ds = scRNASeq([train_cfg_sde['dataset']['train_data_path']], train_cfg_sde['dataset']['dim'], use_v=use_v)
    ds = scRNASeq([train_cfg_sde['dataset']['val_data_path'], train_cfg_sde['dataset']['test_data_path']], train_cfg_sde['dataset']['dim'], scaler=tr_ds.get_scaler())
    param = tr_ds.scaler_params()

    # Define model
    sde_name = train_cfg_sde['model_name'].lower()
    net = SDE_MODEL_NAME[sde_name](**train_cfg_sde['model'])
    sde = SDENet(net, device)
    
    checkpoint = torch.load(checkpoint_path_sde, map_location=lambda storage, loc: storage)
    sde.load_model(checkpoint)
    sde.to(device)
    sde.eval()

    ode_name = train_cfg_ode['model_name'].lower()
    net = ODE_MODEL_NAME[ode_name](**train_cfg_ode['model'])
    ode = ODENet(net, device)

    checkpoint = torch.load(checkpoint_path_ode, map_location=lambda storage, loc: storage)
    ode.load_model(checkpoint)
    ode.to(device)
    ode.eval()

    # evaluation on test data
    t_set = ds.get_label_set()
    source_X = ds.base_sample(eval_cfg["num_trajectory"])["X"].float()
    
    sde_idx = [0, 150, 300, 450, 600]
    int_time = torch.linspace(ds.T0, t_set[-1], 150*len(t_set)+1) 

    data = {}
    data[int(ds.T0)] = decode(ds.base_sample(eval_cfg["num_points"])["X"].float(), param)
    for i in range(len(t_set)):
        data[int(t_set[i])] = decode(ds.get_data(ds.get_subset_index(t_set[i], eval_cfg["num_points"]))["X"].float(), param)
    
    pred_traj_sde = sde.sample_with_uncertainty(source_X, int_time, eval_cfg["num_repeat"])
    pred_traj_ode = ode.sample(source_X, int_time)

    pred_traj_sde = decode(pred_traj_sde, param)
    pred_traj_ode = decode(pred_traj_ode, param)

    fig, axes = plt.subplots(nrows=2, ncols=eval_cfg["num_trajectory"], figsize=(5*eval_cfg["num_trajectory"], 10))
    heatmap(axes, data, pred_traj_sde, pred_traj_sde[:, sde_idx, :, :], pred_traj_ode, pred_traj_ode[:, sde_idx, :])
    save_path = Path(out_dir) / f'heatmap_{eval_cfg["num_timepoints"]}.png'
    plt.savefig(save_path, dpi=300)


def heatmap(axes, data, sde_trajs, sde_samples, ode_trajs, ode_samples):
    data_size, t_size, n_traj, dim = sde_samples.size()
    mean_sde_trajs = torch.mean(sde_trajs, axis=2)
    axes[0, 0].set_ylabel('PC2')
    axes[1, 0].set_ylabel('PC2')

    for k in range(data_size):
        sde_sample = sde_samples[k, :, :, :].cpu().numpy()
        mean_sde_traj = mean_sde_trajs[k, :, :]
        ode_traj = ode_trajs[k, :, :]
        ode_sample = ode_samples[k, :, :].cpu().numpy()

        idx = t_size - 1
        for m in range(2):
            for i, key in enumerate(data.keys()):
                t = key
                axes[m, k].scatter(data[key][:, 0], data[key][:, 1], color=sample_colors((idx + i) % t_size), alpha=0.2, s=2.0, label=f"$t_{int(t)}$: Day {6*int(t)} to {6*int(t)+3}")
            axes[m, k].set_xlabel('PC1')

        axes[0, k].plot(mean_sde_traj[:, 0], mean_sde_traj[:, 1], color=mean_color, label='NLSB (E+D+V)')
        axes[1, k].plot(ode_traj[:, 0], ode_traj[:, 1], color=mean_color, label='TrajectoryNet + OT')

        xmin, xmax = axes[0, k].get_xlim()
        ymin, ymax = axes[0, k].get_ylim()
        xx,yy = np.meshgrid(np.linspace(xmin, xmax, 50), np.linspace(ymin, ymax, 50))
        
        idx = t_size - 1
        axes[0, k].scatter([sde_sample[0, 0, 0]], [sde_sample[0, 0, 1]], s=50.0, alpha=1.0, color=sample_colors(t_size - 1), marker="s")
        axes[1, k].scatter([ode_traj[0, 0]], [ode_traj[0, 1]], s=50.0, alpha=1.0, color=sample_colors(t_size - 1), marker="s")
        for i, ti in enumerate(range(1, t_size)):
            axes[0, k].scatter([np.mean(sde_sample[ti, :, 0])], [np.mean(sde_sample[ti, :, 1])], s=50.0, alpha=1.0, color=sample_colors(i), marker="s")
            axes[1, k].scatter([ode_sample[ti, 0]], [ode_sample[ti, 1]], s=50.0, alpha=1.0, color=sample_colors(i), marker="s")
            # kernel = gaussian_kde(value, bw_method=0.1)
            gmm = GaussianMixture(n_components=5, covariance_type="full")
            gmm.fit(sde_sample[ti, :, :2])
            F = np.vectorize(lambda x, y: gmm.score_samples(np.array([[x, y]])))
            f = F(xx, yy)
            levels = np.linspace(np.min(f), np.max(f), 40)
            axes[0, k].contourf(xx, yy, f, cmap=cmap_colors_fill[i], levels=levels[-3:], alpha=0.3, zorder=0)
            axes[0, k].contour(xx, yy, f, colors=cmap_colors[i], linewidths=1.0, levels=levels[-3:], zorder=1,)
            axes[1, k].axes.xaxis.set_ticklabels([])
            axes[1, k].axes.yaxis.set_ticklabels([])
            
    axes[0, 0].legend(markerscale=5.0, fontsize="large", loc='lower left')
    axes[1, 0].legend(markerscale=5.0, fontsize="large", loc='lower left')
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', '-c', help="Path to the configuration file for conversion.", type=str, required=True)
    parser.add_argument('-path_sde', '-ps', help="Path to the checkpoint of the sde model", type=str, required=True)
    parser.add_argument('-path_ode', '-po', help="Path to the checkpoint of the ode model", type=str, required=True)
    parser.add_argument('-outdir', '-o', help="Path to the output directory", type=str, required=True)
    parser.add_argument('-seed', '-s', type=int, default=1)
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        cfg = json.load(f)
    cfg['seed'] = args.seed
    main(cfg, Path(args.path_sde), Path(args.path_ode), args.outdir)
