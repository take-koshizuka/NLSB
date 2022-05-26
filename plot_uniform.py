

import torch
import numpy as np
import json
import random
import argparse

from pathlib import Path
from dataset import UniformDataset
from model import SDENet, SDE_MODEL_NAME, LAGRANGIAN_NAME
import matplotlib.pyplot as plt

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

sample_colors = plt.get_cmap("Set1")
fill_color = '#9ebcda'
mean_color = '#4d004b'

def main(eval_config_path, checkpoint_path, out_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_dir = str(Path(checkpoint_path).parent)
    with open(eval_config_path, 'r') as f:
        eval_cfg = json.load(f)
    fix_seed(eval_cfg['seed'])

    train_config_path = Path(checkpoint_dir) / "train_config.json"
    with open(train_config_path, 'r') as f:
        train_cfg = json.load(f)
    Path(out_dir).mkdir(exist_ok=True, parents=True)
    
    assert train_cfg['dataset']['name'] == "uniform"
    ds = UniformDataset(device=device, t_0=train_cfg['dataset']['t_0'], t_T=train_cfg['dataset']['t_T'], data_size=eval_cfg['num_points'])

    # Define model
    model_name = train_cfg['model_name'].lower()
    
    assert train_cfg['lagrangian_name'] == "newtonian"
    L = LAGRANGIAN_NAME["newtonian"](**train_cfg['lagrangian'])
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
    fig, axes = plt.subplots(nrows=1, ncols=2,sharex=True, sharey=True, figsize=(10, 5))
    feature_x = np.arange(-1.5, 1.5, 0.1)
    feature_y = np.arange(-1.5, 1.5, 0.1)
    X, Y = np.meshgrid(feature_x, feature_y)
    uf = np.vectorize(lambda x, y: L.U(torch.tensor([[x, y]]), 0))
    # uf = np.vectorize(lambda x,y : 0)
    Z = uf(X, Y)
    for i in range(2):
        axes[i].contourf(feature_x, feature_y, Z, cmap='Blues_r')
    
    plot_samples_and_trajectory(axes[0], samples, pred_traj, colors=colors, reduce_type=None)
    plot_samples_and_trajectory(axes[1], samples, pred_traj, colors=colors, reduce_type='mean')
    legend_setting(axes[0])
    legend_setting(axes[1])

    save_path = Path(out_dir) / f'prediction.png'
    plt.savefig(save_path, dpi=300)


def legend_setting(ax, legend=False):
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    if legend:
        ax.legend(markerscale=5.0, fontsize="large", loc='lower left')

def one_step_prediction(model, ds, eval_cfg):
    pred_samples = {}
    pred_traj = {}
    t_set = ds.get_label_set()
    
    for i in range(len(t_set)):
        if i == 0:
            t0, t1 = ds.T0, t_set[i]
            source = ds.base_sample(eval_cfg["num_points"])["X"].float()
        else:
            t0, t1 = t_set[i-1], t_set[i]
            source = ds.get_data(ds.get_subset_index(t0, eval_cfg["num_points"]))["X"].float()
        
        int_time = torch.linspace(t0, t1, 151)

        if hasattr(model, 'sample_with_uncertainty'):
            traj = model.sample_with_uncertainty(source, int_time, eval_cfg["num_repeat"])
        else:
            traj = model.sample(source, int_time)
            traj = traj.unsqueeze(2)
            
        traj = traj.cpu()
        pred_traj[f'{int(t0)}-{int(t1)}'] = traj[:eval_cfg["num_trajectory"]].numpy()
        pred_samples[int(t1)] = traj[:, -1, 0].numpy()
    
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', '-c', help="Path to the configuration file for conversion.", type=str, required=True)
    parser.add_argument('-path', '-p', help="Path to the checkpoint of the model", type=str, required=True)
    parser.add_argument('-outdir', '-o', help="Path to the output directory", type=str, required=True)

    args = parser.parse_args()
    main(args.config, Path(args.path), args.outdir)
