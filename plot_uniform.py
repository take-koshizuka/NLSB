

import torch
import numpy as np
import json
import random
import argparse

from pathlib import Path
from dataset import UniformDataset, UniformDataset2
from model import SDENet, SDE_MODEL_NAME, LAGRANGIAN_NAME
import matplotlib.pyplot as plt
import scienceplots
import seaborn as sns

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

sample_colors = plt.get_cmap("Spectral")
fill_color = '#9ebcda'
mean_color = '#4d004b'

def main(eval_cfg, checkpoint_path, out_dir, checkpoint_path_L=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_dir = str(Path(checkpoint_path).parent)
    fix_seed(eval_cfg['seed'])

    train_config_path = Path(checkpoint_dir) / "train_config.json"
    with open(train_config_path, 'r') as f:
        train_cfg = json.load(f)

    checkpoint_dir_L = str(Path(checkpoint_path_L).parent)
    if not checkpoint_path_L is None:
        checkpoint_dir_L = str(Path(checkpoint_path_L).parent)
        train_config_path_L = Path(checkpoint_dir_L) / "train_config.json"
        with open(train_config_path_L, 'r') as f:
            train_cfg_L = json.load(f)
    
    Path(out_dir).mkdir(exist_ok=True, parents=True)
    
    assert train_cfg['dataset']['name'] == "uniform" or train_cfg['dataset']['name'] == "uniform2"
    if train_cfg['dataset']['name'] == "uniform":
        ds = UniformDataset(device=device, t_0=train_cfg['dataset']['t_0'], t_T=train_cfg['dataset']['t_T'], data_size=eval_cfg['num_points'])
    elif train_cfg['dataset']['name'] == "uniform2":
        ds = UniformDataset2(device=device, t_0=train_cfg['dataset']['t_0'], t_T=train_cfg['dataset']['t_T'], data_size=eval_cfg['num_points'])
    else:
        raise NotImplementedError
    # Define model
    model_name = train_cfg['model_name'].lower()
    if train_cfg['lagrangian_name'] == "newtonian":
        if not checkpoint_path_L is None:
            L = LAGRANGIAN_NAME["newtonian"](**train_cfg_L['lagrangian'])
        else:
            L = LAGRANGIAN_NAME["newtonian"](**train_cfg['lagrangian'])
    elif train_cfg['lagrangian_name'] == "LQ":
        L = LAGRANGIAN_NAME["LQ"](**train_cfg['lagrangian'], device=device)
    
    net = SDE_MODEL_NAME[model_name](**train_cfg['model'], lagrangian=L)
    model = SDENet(net, device)
    
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    model.load_model(checkpoint)
    model.to(device)
    model.eval()

    # evaluation on test data
    t_set = ds.get_label_set()
    
    samples = {}
    colors = {}
    samples[int(ds.T0)] = ds.base_sample(eval_cfg["num_points"])["X"].float().cpu().numpy()
    colors[int(ds.T0)] = sample_colors(len(t_set))
    for i, t in enumerate(t_set):
        samples[int(t)] = ds.get_data(ds.get_subset_index(t, eval_cfg["num_points"]))["X"].float().cpu().numpy()
        colors[int(t)] = sample_colors(i)

    pred_samples, pred_traj = one_step_prediction(model, ds, eval_cfg)
    fig, axes = plt.subplots(nrows=1, ncols=6, sharex=True, sharey=True, figsize=(30, 5))
    feature_x = np.arange(-2.0, 2.0, 0.1)
    feature_y = np.arange(-2.0, 2.0, 0.1)
    X, Y = np.meshgrid(feature_x, feature_y)
        # CS = axes[i].contourf(feature_x, feature_y, Z, cmap='Blues_r')
        # fig.colorbar(CS, ax=axes[i], shrink=0.9)
    uf = np.vectorize(lambda x, y: L.U(torch.tensor([[x, y]]), 0))
    #uf = np.vectorize(lambda x,y : 0)
    Z = uf(X, Y)
    for i in range(len(axes)):
        axes[i].contourf(feature_x, feature_y, Z, cmap='Greys_r')
        axes[i].set_xlim([-1.5, 1.5])
        axes[i].set_ylim([-1.5, 1.4])
    
    #plot_samples_and_trajectory(axes[0], samples, pred_traj, colors=colors, reduce_type=None)
    #plot_samples_and_trajectory(axes[1], samples, pred_traj, colors=colors, reduce_type='mean')
    #legend_setting(axes[0])
    #legend_setting(axes[1])
    colors = [ sample_colors(i) for i in range(sample_colors.N) ]
    make_snapshot_animation(axes, pred_traj, colors)

    save_path = Path(out_dir) / f'anime.png'
    plt.savefig(save_path, dpi=300)


def legend_setting(ax, xlim=[-1.5, 1.5], ylim=[-1.5, 1.4], legend=False):
    #ax.axes.xaxis.set_ticklabels([])
    #ax.axes.yaxis.set_ticklabels([])
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    if legend:
        ax.legend(markerscale=5.0, fontsize="large", loc='lower left')

def one_step_prediction(model, ds, eval_cfg):
    t_set = ds.get_label_set()

    t0, t1 = ds.T0, t_set[0]
    source = ds.base_sample(eval_cfg["num_points"])["X"].float()
    
    int_time = torch.linspace(t0, t1, 151)

    if hasattr(model, 'sample_with_uncertainty'):
        traj = model.sample_with_uncertainty(source, int_time, eval_cfg["num_repeat"])
    else:
        traj = model.sample(source, int_time)
        traj = traj.unsqueeze(2)
    
    r = torch.randperm(len(traj))
    traj = traj[r]
    traj = traj.cpu()
    pred_traj = traj.numpy()
    pred_samples = traj[:, -1, 0].numpy()
    
    return pred_samples, pred_traj

def plot_samples_and_trajectory(ax, samples, pred_traj, colors, reduce_type='mean'):
    ts = samples.keys()
    for i, t in enumerate(ts):
        ax.scatter(samples[t][:, 0], samples[t][:, 1], color=colors[t], alpha=0.8, s=4.0, label=f"$t$={t}", zorder=1)
    
    intervals = pred_traj.keys()
    for i, T in enumerate(intervals):
        trajectory = pred_traj[T]
        if trajectory.ndim == 3:
            reduced_trajectory = trajectory
        elif (trajectory.ndim == 4) and (reduce_type == 'mean'):
            reduced_trajectory = trajectory.mean(axis=2)
        elif (trajectory.ndim == 4) and (reduce_type is None):
            reduced_trajectory = trajectory[:, :, 0, :]
        else:
            raise ValueError("Invalid reduced_type")
        
        n_trajectory = len(reduced_trajectory)
        for j in range(n_trajectory):
            ax.plot(reduced_trajectory[j, :, 0], reduced_trajectory[j, :, 1], color=mean_color, zorder=2)

    if reduce_type is None:
        ax.set_title(r"$\hat{X}(k)$")
        
    elif reduce_type == 'mean':
        ax.set_title(r"$E[\hat{X}(k)]$")

def make_snapshot_animation(axes, trajectory, colors):
    reduced_trajectory = trajectory[:, :, 0, :]
    L = len(axes)
    idx = torch.linspace(0, reduced_trajectory.shape[1]-1, L).long()
    cidx = torch.linspace(0, len(colors)-1, L).long()
    for i, t in enumerate(idx):
        p = i/(len(idx)-1)
        ci = cidx[i]
        axes[i].scatter(reduced_trajectory[:, t, 0], reduced_trajectory[:, t, 1], s=4.0, alpha=0.8, color=colors[ci])
        if i == 0:
            axes[i].set_title(r"$t=0$")
        elif i == len(idx) - 1:
            axes[i].set_title(r"$t=T$")
        else:
            axes[i].set_title(r"$t=$"+f"${p}$"+r"$T$")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', '-c', help="Path to the configuration file for conversion.", type=str, required=True)
    parser.add_argument('-path', '-p', help="Path to the checkpoint of the model", type=str, required=True)
    parser.add_argument('-outdir', '-o', help="Path to the output directory", type=str, required=True)
    parser.add_argument('-seed', '-s', type=int, default=10)
    parser.add_argument('-path_L', '-pL', help="Path to the checkpoint of the Lagrangian model", type=str)
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        cfg = json.load(f)
    cfg['seed'] = args.seed
    if args.path_L is None:
        args.path_L = args.path
    main(cfg, Path(args.path), args.outdir, Path(args.path_L))
