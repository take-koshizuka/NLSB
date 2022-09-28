
import torch
import numpy as np
import json
import random
import argparse
from pathlib import Path

from dataset import OrnsteinUhlenbeckSDE_Dataset
from model import ODENet, SDENet, ReverseSDENet, SDE_MODEL_NAME, ODE_MODEL_NAME
import matplotlib.pyplot as plt


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

def main(eval_cfg, checkpoint_path, out_dir, fsde_checkpoint_dir="checkpoints/ou-sde/neuralSDE/NLSB/expM"):
    plt.style.use(['science', 'notebook'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_dir = str(Path(checkpoint_path).parent)
    fix_seed(eval_cfg['seed'])

    train_config_path = Path(checkpoint_dir) / "train_config.json"
    with open(train_config_path, 'r') as f:
        train_cfg = json.load(f)
    Path(out_dir).mkdir(exist_ok=True, parents=True)

    fsde_train_config_path = Path(fsde_checkpoint_dir) / "train_config.json"
    with open(fsde_train_config_path, 'r') as f:
        fsde_train_cfg = json.load(f)

    data_size = eval_cfg['num_points']
    assert train_cfg['dataset']['name'] == "ornstein-uhlenbeck-sde"
    tr_ds = OrnsteinUhlenbeckSDE_Dataset(device=device, t_size=train_cfg['dataset']['t_size'], data_size=train_cfg['train_size'], mu=train_cfg['dataset']['mu'], 
                                        theta=train_cfg['dataset']['theta'], sigma=train_cfg['dataset']['sigma'])
    ds = OrnsteinUhlenbeckSDE_Dataset(device=device, t_size=train_cfg['dataset']['t_size'], data_size=data_size, mu=train_cfg['dataset']['mu'], 
                                        theta=train_cfg['dataset']['theta'], sigma=train_cfg['dataset']['sigma'])
    
    t_set = ds.get_label_set()
    samples = {}
    colors = {}
    samples[round(ds.T0, 2)] = ds.base_sample(eval_cfg["num_points"])["X"].float()
    colors[round(ds.T0, 2)] = sample_colors(len(t_set))
    for i, t in enumerate(t_set):
        samples[round(t, 2)] = ds.get_data(ds.get_subset_index(t, eval_cfg["num_points"]))["X"].float()
        colors[round(t, 2)] = sample_colors(i)

    _, ref_traj = one_step_prediction(ds, ds, eval_cfg)
    fig, axes = plt.subplots(nrows=1, ncols=2,sharex=True, sharey=True, figsize=(10, 5))
    plot_samples_and_trajectory(axes[0], samples, ref_traj, colors=colors, reduce_type=None)
    plot_samples_and_trajectory(axes[1], samples, ref_traj, colors=colors, reduce_type='mean')
    save_path = Path(out_dir) / f'ground-truth sde(one_step).png'
    plt.savefig(save_path, dpi=300)

    ref_samples, ref_traj = all_step_prediction(ds, ds, eval_cfg)
    int_time = torch.linspace(ds.T0, t_set[-1], 150*len(t_set)+1)
    fig, axes = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=(10, 5))
    plot_samples_and_trajectory_full(axes[0], int_time, samples, ref_traj, ref_samples, colors=colors, reduce_type=None)
    plot_samples_and_trajectory_full(axes[1], int_time, samples, ref_traj, ref_samples, colors=colors, reduce_type='mean')
    
    save_path = Path(out_dir) / f'ground-truth sde(all_step).png'
    plt.savefig(save_path, dpi=300)

    # Define model
    if 'model_name' in train_cfg:
        model_name = train_cfg['model_name'].lower()
    else:
        model_name = "ito"
    
    if model_name in SDE_MODEL_NAME:
        fnet = SDE_MODEL_NAME["ito"](**fsde_train_cfg['model'])
        fsde = SDENet(fnet, device)
        bnet = SDE_MODEL_NAME["rev-sde"](fsde.net)
        model = ReverseSDENet(bnet, device)
    elif model_name in ODE_MODEL_NAME:
        net = ODE_MODEL_NAME[model_name](**train_cfg['model'])
        model = ODENet(net, device)
    else:
        raise ValueError("The model name does not exist.")

    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    model.load_model(checkpoint)
    model.to(device)

    model.eval()
    # evaluation on test data
    pred_samples, pred_traj = one_step_prediction_backward(model, ds, eval_cfg)
    fig, axes = plt.subplots(nrows=1, ncols=2,sharex=True, sharey=True, figsize=(10, 5))
    plot_samples_and_trajectory(axes[0], samples, pred_traj, colors=colors, reduce_type=None)
    plot_samples_and_trajectory(axes[1], samples, pred_traj, colors=colors, reduce_type='mean')
    save_path = Path(out_dir) / f'one_step_prediction.png'
    plt.savefig(save_path, dpi=300)

    pred_samples, pred_traj = all_step_prediction_backward(model, ds, eval_cfg)
    int_time = torch.linspace(t_set[-1], ds.T0, 150*len(t_set)+1)
    # fig, axes = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=(10, 5))
    fig, axes = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True, figsize=(5, 5))
    plot_samples_and_trajectory_full(axes, int_time, samples, pred_traj, pred_samples, colors=colors, reduce_type=None)
    #plot_samples_and_trajectory_full(axes[1], int_time, samples, pred_traj, pred_samples, colors=colors, reduce_type='mean')
    # axes.set_ylim([-3.5, 8.5])
    save_path = Path(out_dir) / f'all_step_prediction.png'
    plt.savefig(save_path, dpi=300)


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
            pred_traj[f'{round(t0, 2)}-{round(t1, 2)}'] = traj[:eval_cfg["num_trajectory"]].cpu().numpy()
            pred_samples[round(t1, 2)] = traj[:, -1, 0, :].cpu().numpy()
        else:
            traj = model.sample(source, int_time)
            pred_traj[f'{round(t0, 2)}-{round(t1, 2)}'] = traj[:eval_cfg["num_trajectory"]].cpu().numpy()
            pred_samples[round(t1, 2)] = traj[:, -1, :].cpu().numpy()
    
    return pred_samples, pred_traj


def one_step_prediction_backward(model, ds, eval_cfg):
    pred_samples = {}
    pred_traj = {}

    t_set = ds.get_label_set()

    for i in range(len(t_set)):
        t0 = t_set[i]
        t1 = t_set[i-1] if i != 0 else ds.T0

        source = ds.get_data(ds.get_subset_index(t0))["X"].float()
        rev_int_time = torch.linspace(t0, t1, 151)
        
        if hasattr(model, 'sample_with_uncertainty'):
            traj = model.sample_with_uncertainty(source, rev_int_time, eval_cfg["num_repeat"])
            pred_traj[f'{round(t0, 2)}-{round(t1, 2)}'] = traj[:eval_cfg["num_trajectory"]].cpu().numpy()
            pred_samples[round(t1, 2)] = traj[:, -1, 0, :].cpu().numpy()
        else:
            traj = model.sample(source, rev_int_time)
            pred_traj[f'{round(t0, 2)}-{round(t1, 2)}'] = traj[:eval_cfg["num_trajectory"]].cpu().numpy()
            pred_samples[round(t1, 2)] = traj[:, -1, :].cpu().numpy()
    
    return pred_samples, pred_traj

def all_step_prediction(model, ds, eval_cfg):
    pred_samples = {}

    t_set = ds.get_label_set()

    source = ds.base_sample(eval_cfg["num_points"])["X"].float()
    int_time = torch.linspace(ds.T0, t_set[-1], 150*len(t_set)+1)

    if hasattr(model, 'sample_with_uncertainty'):
        traj = model.sample_with_uncertainty(source, int_time, eval_cfg["num_repeat"])
    else:
        traj = model.sample(source, int_time)

    traj = traj.cpu()
    pred_traj = traj[:eval_cfg["num_trajectory"]].numpy()

    for i, t1 in enumerate(t_set):
        ti = 150*(i+1)
        pred_samples[round(t1, 2)] = traj[:, ti].numpy()
    
    return pred_samples, pred_traj


def all_step_prediction_backward(model, ds, eval_cfg):
    pred_samples = {}

    t_set = ds.get_label_set()

    source = ds.get_data(ds.get_subset_index(t_set[-1]))["X"].float()
    int_time = torch.linspace(ds.T0, t_set[-1], 150*len(t_set)+1)
    rev_int_time = torch.flip(int_time, dims=(0, ))

    if hasattr(model, 'sample_with_uncertainty'):
        traj = model.sample_with_uncertainty(source, rev_int_time, eval_cfg['num_repeat'])
    else:
        traj = model.sample(source, rev_int_time)

    traj = traj.cpu()
    pred_traj = traj[:eval_cfg["num_trajectory"]].numpy()

    ts = list(reversed(t_set[:-1])) + [ds.T0]
    for i, t1 in enumerate(ts):
        ti = 150*(i+1)
        pred_samples[round(t1, 2)] = traj[:, ti, :].numpy()
    
    return pred_samples, pred_traj


def plot_samples_and_trajectory(ax, samples, pred_traj, colors, reduce_type='mean'):
    ts = samples.keys()
    for i, t in enumerate(ts):
        ax.scatter([t]*len(samples[t]), samples[t][:, 0], color=colors[t], alpha=0.8, s=4.0, label=f"$t$={t}", zorder=1)
    
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
        t0, t1 = float(t0), float(t1)
        n_trajectory = len(reduced_trajectory)

        int_time = torch.linspace(t0, t1, 151)
        for j in range(n_trajectory):
            ax.plot(int_time, reduced_trajectory[j, :], color=mean_color, zorder=2)
            if reduce_type =='mean':
                confidence_interval(ax, int_time, trajectory[j], alpha=0.2, facecolor='pink', edge_color='deepskyblue')
        
        ax.scatter([t0]*len(reduced_trajectory), reduced_trajectory[:, 0], color=colors[round(t0, 2)], marker='*')
        ax.scatter([t1]*len(reduced_trajectory), reduced_trajectory[:, -1], color=colors[round(t1, 2)], marker='*')

    if reduce_type is None:
        ax.set_title(r"$\hat{X}(k), k=0,1, \dots$")
    elif reduce_type == 'mean':
        ax.set_title(r"$E[\hat{X}(k)], k=0,1, \dots$")

def plot_samples_and_trajectory_full(ax, int_time, samples, pred_traj, pred_samples, colors, reduce_type='mean'):
    ts = samples.keys()
    for i, t in enumerate(ts):
        ax.scatter([t]*len(samples[t]), samples[t][:, 0], color=colors[t], alpha=0.8, s=4.0, label=f"$t$={t}", zorder=1)

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
        ax.plot(int_time, reduced_trajectory[i, :, 0], color=mean_color, zorder=2)
        if reduce_type =='mean':
            confidence_interval(ax, int_time, trajectory[i], alpha=0.2, facecolor='pink', edge_color='deepskyblue')
    """
    ts = pred_samples.keys()
    for i, t in enumerate(ts):
        if pred_samples[t].ndim == 2:
            ax.scatter([t]*len(pred_samples[t]), pred_samples[t][:, 0], color=colors[t], marker='*')
        elif reduce_type is None:
            ax.scatter([t]*len(pred_samples[t]), pred_samples[t][:, 0], color=colors[t], marker='*')
        elif reduce_type == 'mean':
            ax.scatter([t]*len(pred_samples[t]), np.mean(pred_samples[t], axis=1), color=colors[t], marker='*')
        else:
            raise ValueError("Invalid reduced_type")
    """
    if reduce_type is None:
        ax.set_title(r"$\hat{X}(t) \mid X(0)$")
    elif reduce_type == 'mean':
        ax.set_title(r"$E[\hat{X}(t) \mid X(0)]$")
    
def plot_samples(ax, pred_samples, colors, reduce_type):
    ts = pred_samples.keys()
    for i, t in enumerate(ts):
        if pred_samples[t].ndim == 2:
            ax.scatter([t]*len(pred_samples[t]), pred_samples[t][:, 0], color=colors[t], alpha=0.8, s=4.0, label=f"$t$={t}")
        elif reduce_type is None:
            ax.scatter([t]*len(pred_samples[t]), pred_samples[t][:, 0, 0], color=colors[t], alpha=0.8, s=4.0, label=f"$t$={t}")
        else:
            raise ValueError("Invalid reduced_type")

def plot_trajectory(ax, data, property, show_cov=False, reduce_type=None):
    # show_samples
    t_set = sorted(list(data.keys()))
    T0 = min(t_set) - 1
    # show mean
    for i, t in enumerate(t_set):
        if property == 'pred':
            trajectory = data[t]['pred_traj']
        elif property == 'ref':
            trajectory = data[t]['ref_traj']

        int_time = data[t]['int_time']
        t1 = int_time[-1].cpu().numpy()
        if i == 0:
            ax.scatter([T0]*len(data[T0+1]['source']), data[T0+1]['source'], color=sample_colors(len(t_set)), s=4.0, alpha=0.8, label=f"$t$={T0}")
        ax.scatter([t1]*len(data[t]['target']), data[t]['target'], color=sample_colors(i), s=4.0, alpha=0.8, label=f"$t$={t}")

        if trajectory.ndim == 3:
            reduced_trajectory = trajectory
        elif (trajectory.ndim == 4) and (reduce_type == 'mean'):
            reduced_trajectory = trajectory.mean(axis=2)
        elif (trajectory.ndim == 4) and (reduce_type is None):
            reduced_trajectory = trajectory[:, :, 0, :]
        else:
            raise ValueError("Invalid reduced_type")
        
        n_trajectory = len(reduced_trajectory)
        # 2: trajectory_num, 150: t_size
        c = sample_colors(len(t_set)) if i == 0 else sample_colors(i-1)
        
        if i == 0:
            ax.scatter([T0]*n_trajectory, reduced_trajectory[: ,0], color=c, marker='*')
        else:
            ax.scatter([t_set[i-1]]*n_trajectory, reduced_trajectory[: ,0], color=c, marker='*')
        ax.scatter([t_set[i]]*n_trajectory, reduced_trajectory[:, -1], color=sample_colors(i), marker='*')
        
        for j in range(n_trajectory):
            ax.plot(int_time, reduced_trajectory[j, :], color=mean_color)
            if reduce_type =='mean' and show_cov:
                confidence_interval(ax, int_time, trajectory[j], alpha=0.2, facecolor='pink', edge_color='deepskyblue')

def confidence_interval(ax, ts, trajectory, alpha, facecolor, edge_color):
    mu = trajectory.mean(axis=1).squeeze()
    sigma = trajectory.std(axis=1).squeeze()
    ax.fill_between(ts, mu+sigma, mu-sigma, facecolor=facecolor, edgecolor=edge_color,alpha=alpha)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', '-c', help="Path to the configuration file for conversion.", type=str, required=True)
    parser.add_argument('-path', '-p', help="Path to the checkpoint of the model", type=str, required=True)
    parser.add_argument('-outdir', '-o', help="Path to the output directory", type=str, required=True)
    parser.add_argument('-seed', '-s', type=int, default=57)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = json.load(f)
    cfg['seed'] = args.seed
    main(cfg, Path(args.path), args.outdir)
