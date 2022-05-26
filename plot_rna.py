
import torch
import numpy as np
import json
import random
import argparse

from pathlib import Path

from utils import decode
from dataset import scRNASeq
from model import ODENet, SDENet, SDE_MODEL_NAME, ODE_MODEL_NAME, LAGRANGIAN_NAME
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

sample_colors = plt.get_cmap("tab10")
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

    assert train_cfg['dataset']['name'] == "scRNA"
    use_v = train_cfg['dataset']['use_v'] if 'use_v' in train_cfg['dataset'] else False
    tr_ds = scRNASeq([train_cfg['dataset']['train_data_path']], train_cfg['dataset']['dim'], use_v=use_v, LMT=train_cfg['LMT'])
    ds = scRNASeq([train_cfg['dataset']['test_data_path']], train_cfg['dataset']['dim'], use_v=use_v, scaler=tr_ds.get_scaler())
    param = tr_ds.scaler_params()

    # Define model
    model_name = train_cfg['model_name'].lower()
    if model_name in SDE_MODEL_NAME:
        if not 'lagrangian_name' in train_cfg:
            L = LAGRANGIAN_NAME['potential-free']()
        elif train_cfg['lagrangian_name'] == "null" or train_cfg['lagrangian_name'] == "potential-free":
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
    model.eval()

    # evaluation on test data
    t_set = ds.get_label_set()
    
    samples = {}
    colors = {}
    samples[int(ds.T0)] = decode(ds.base_sample(eval_cfg["num_points"])["X"].float(), param).cpu().numpy()
    colors[int(ds.T0)] = sample_colors(len(t_set))
    for i, t in enumerate(t_set):
        samples[int(t)] = decode(ds.get_data(ds.get_subset_index(t, eval_cfg["num_points"]))["X"].float(), param).cpu().numpy()
        colors[int(t)] = sample_colors(i)
    pred_samples, pred_traj = one_step_prediction(model, ds, eval_cfg)
    
    if MODEL == 'sde':
        fig, axes = plt.subplots(nrows=1, ncols=2,sharex=True, sharey=True, figsize=(10, 5))
        plot_samples_and_trajectory(axes[0], samples, pred_traj, colors=colors, reduce_type=None)
        plot_samples_and_trajectory(axes[1], samples, pred_traj, colors=colors, reduce_type='mean')
        legend_setting(axes[0])
        legend_setting(axes[1])
    else:
        fig, ax = plt.subplots(nrows=1, ncols=1,sharex=True, sharey=True, figsize=(5, 5))
        plot_samples_and_trajectory(ax, samples, pred_traj, colors=colors, reduce_type=None)
        legend_setting(ax)

    save_path = Path(out_dir) / f'one_step_prediction.png'
    plt.savefig(save_path, dpi=300)

    if len(t_set) >= 2:
        # split figure
        fig, axes = plt.subplots(nrows=1, ncols=len(t_set), sharex=True, sharey=True, figsize=(5*len(t_set), 5))
        plot_samples_and_trajectory_split(axes, samples, pred_traj, colors, reduce_type='mean')
        save_path = Path(out_dir) / f'one_step_prediction_split.png'
        plt.savefig(save_path, dpi=300)

        # split figure
        fig, axes = plt.subplots(nrows=1, ncols=len(t_set), sharex=True, sharey=True, figsize=(5*len(t_set), 5))
        plot_samples_split(axes, samples, pred_samples, colors, ds.T0)
        for i in range(len(t_set)):
            axes[i].set_xlim([40, 58])
            axes[i].set_ylim([-15, 20])
            legend_setting(axes[i], legend=True)
        save_path = Path(out_dir) / f'samples.png'
        plt.savefig(save_path, dpi=300)
    
    else:
        # split figure
        fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True, figsize=(5*len(t_set), 5))
        plot_samples_split([ax], samples, pred_samples, colors, ds.T0)
        for i in range(len(t_set)):
            axes[i].set_xlim([40, 58])
            axes[i].set_ylim([-15, 20])
            legend_setting(axes[i], legend=True)
        save_path = Path(out_dir) / f'samples.png'
        plt.savefig(save_path, dpi=300)


    pred_samples, pred_traj = all_step_prediction(model, ds, eval_cfg)
    if MODEL == 'sde':
        fig, axes = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=(10, 5))
        plot_samples_and_trajectory_full(axes[0], samples, pred_traj, pred_samples, colors=colors, reduce_type=None)
        plot_samples_and_trajectory_full(axes[1], samples, pred_traj, pred_samples, colors=colors, reduce_type='mean')
        legend_setting(axes[0], legend=True)
        legend_setting(axes[1])
    else:
        fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True, figsize=(5, 5))
        plot_samples_and_trajectory_full(ax, samples, pred_traj, pred_samples, colors=colors, reduce_type=None)
        legend_setting(ax)
    save_path = Path(out_dir) / f'all_step_prediction.png'
    plt.savefig(save_path, dpi=300)


def legend_setting(ax, legend=False, dimX=0, dimY=1, xlim=[48, 58], ylim=[-15, 20]):
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    ax.set_xlabel(f'PC{dimX+1}')
    ax.set_ylabel(f'PC{dimY+1}')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    if legend:
        ax.legend(markerscale=5.0, fontsize="large", loc='lower left')

def one_step_prediction(model, ds, eval_cfg):
    pred_samples = {}
    pred_traj = {}

    param = ds.scaler_params() if hasattr(ds, "scaler_params") else None
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
            
        traj = decode(traj.cpu(), param)
        pred_traj[f'{int(t0)}-{int(t1)}'] = traj[:eval_cfg["num_trajectory"]].numpy()
        pred_samples[int(t1)] = traj[:, -1, 0].numpy()
    
    return pred_samples, pred_traj


def one_step_prediction_backward(model, ds, eval_cfg):
    pred_samples = {}
    pred_traj = {}

    param = ds.scaler_params() if hasattr(ds, "scaler_params") else None
    t_set = ds.get_label_set()

    for i in range(len(t_set)):
        t0 = t_set[i]
        t1 = t_set[i-1] if i != 0 else ds.T0

        source = ds.get_data(ds.get_subset_index(t0))["X"].float()
        rev_int_time = torch.linspace(t0, t1, 151)
        
        if hasattr(model, 'sample_with_uncertainty'):
            traj = model.sample_with_uncertainty(source, rev_int_time, eval_cfg["num_repeat"])
        else:
            traj = model.sample(source, rev_int_time)
            traj = traj.unsqueeze(2)
        
        traj = decode(traj.cpu(), param)
        pred_traj[f'{int(t0)}-{int(t1)}'] = traj[:eval_cfg["num_trajectory"]].numpy()
        pred_samples[int(t1)] = traj[:, -1, 0, :].numpy()
    
    return pred_samples, pred_traj

def all_step_prediction(model, ds, eval_cfg):
    pred_samples = {}

    param = ds.scaler_params() if hasattr(ds, "scaler_params") else None
    t_set = ds.get_label_set()

    source = ds.base_sample(eval_cfg["num_points"])["X"].float()
    int_time = torch.linspace(ds.T0, t_set[-1], 160*len(t_set)+1)

    if hasattr(model, 'sample_with_uncertainty'):
        traj = model.sample_with_uncertainty(source[:eval_cfg["num_trajectory"]], int_time, eval_cfg["num_repeat"])
    else:
        traj = model.sample(source[:eval_cfg["num_trajectory"]], int_time)
        traj = traj.unsqueeze(2)

    traj = decode(traj.cpu(), param)
    pred_traj = traj[:eval_cfg["num_trajectory"]]

    for i, t1 in enumerate(t_set):
        ti = 160*(i+1)
        pred_samples[int(t1)] = pred_traj[:, ti, :, :].numpy()
    
    return pred_samples, pred_traj.numpy()

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
        
        [t0, t1] = T.split('-')
        n_trajectory = len(reduced_trajectory)
        for j in range(n_trajectory):
            ax.plot(reduced_trajectory[j, :, 0], reduced_trajectory[j, :, 1], color=mean_color, zorder=2)
        
        ax.scatter(reduced_trajectory[:, 0, 0], reduced_trajectory[:, 0, 1], color=colors[int(t0)], marker='*', zorder=3)
        ax.scatter(reduced_trajectory[:, -1, 0], reduced_trajectory[:, -1, 1], color=colors[int(t1)], marker='*', zorder=3)

    if reduce_type is None:
        ax.set_title(r"$\hat{X}(k), k=0,1, \dots$")
    elif reduce_type == 'mean':
        ax.set_title(r"$E[\hat{X}(k)], k=0,1, \dots$")


def plot_samples_and_trajectory_split(axes, samples, pred_traj, colors, reduce_type='mean'):
    # plot samples
    ts = list(samples.keys())
    for i in range(len(ts) - 1):
        t0, t1 = ts[i], ts[i+1]
        axes[i].scatter(samples[t0][:, 0], samples[t0][:, 1], color=colors[t0], alpha=0.8, s=4.0, label=f"$t$={t0}", zorder=1)
        axes[i].scatter(samples[t1][:, 0], samples[t1][:, 1], color=colors[t1], alpha=0.8, s=4.0, label=f"$t$={t1}", zorder=1)
        if f'{int(t0)}-{int(t1)}' in pred_traj:
            trajectory = pred_traj[f'{int(t0)}-{int(t1)}']
        
        elif f'{int(t1)}-{int(t0)}' in pred_traj:
            trajectory = pred_traj[f'{int(t1)}-{int(t0)}']
        
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
            axes[i].plot(reduced_trajectory[j, :, 0], reduced_trajectory[j, :, 1], color=mean_color, zorder=2)

def plot_samples_split(axes, ref_samples, samples, colors, T0):
    # plot samples
    ts = [T0] + list(samples.keys())
    for i in range(len(ts) - 1):
        t0, t1 = ts[i], ts[i+1]
        axes[i].scatter(ref_samples[t0][:, 0], ref_samples[t0][:, 1], color=colors[t0], alpha=0.8, s=4.0, label=f"$t_{i}$: Day {6*i} to {6*(i)+3} (data)", zorder=1)
        axes[i].scatter(samples[t1][:, 0], samples[t1][:, 1], color=colors[t1], alpha=0.8, s=4.0, label=f"$t_{i+1}$: Day {6*(i+1)} to {6*(i+1)+3} (pred)", zorder=1)


def plot_samples_and_trajectory_full(ax, samples, pred_traj, pred_samples, colors, reduce_type='mean'):
    ts = samples.keys()
    for i, t in enumerate(ts):
        ax.scatter(samples[t][:, 0], samples[t][:, 1], color=colors[t], alpha=0.8, s=4.0, label=f"$t_{i}$: Day {6*i} to {6*i+3}", zorder=1)

    trajectory = pred_traj
    if trajectory.ndim == 3:
        reduced_trajectory = trajectory
    elif (trajectory.ndim == 4) and (reduce_type == 'mean'):
        reduced_trajectory = trajectory.mean(axis=2)
    elif (trajectory.ndim == 4) and (reduce_type is None):
        reduced_trajectory = trajectory[:, :, 0, :]
    else:
        raise ValueError("Invalid reduced_type")
    
    n_trajectory = len(reduced_trajectory)
    for i in range(n_trajectory):
        ax.plot(reduced_trajectory[i, :, 0], reduced_trajectory[i, :, 1], color=mean_color, zorder=2)
    
    ts = pred_samples.keys()
    for i, t in enumerate(ts): 
        if reduce_type is None:
            ax.scatter(pred_samples[t][:, 0, 0], pred_samples[t][:, 0, 1], color=colors[t], marker='*', zorder=3)
        elif reduce_type == 'mean':
            XY = np.mean(pred_samples[t], axis=1)
            ax.scatter(XY[:, 0], XY[:, 1], color=colors[t], marker='*', zorder=3)
        else:
            raise ValueError("Invalid reduced_type")

    if reduce_type is None:
        ax.set_title(r"$\hat{X}(t) \mid X(0)$")
    elif reduce_type == 'mean':
        ax.set_title(r"$E[\hat{X}(t) \mid X(0)]$")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', '-c', help="Path to the configuration file for conversion.", type=str, required=True)
    parser.add_argument('-path', '-p', help="Path to the checkpoint of the model", type=str, required=True)
    parser.add_argument('-outdir', '-o', help="Path to the output directory", type=str, required=True)

    args = parser.parse_args()
    main(args.config, Path(args.path), args.outdir)
