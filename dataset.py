from collections import namedtuple
import torch
import torchsde
from mfg.mfg import MFG
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, BatchSampler
import numpy as np

class TrajectoryInferenceDataset(Dataset):
    def __init__(self):
        self.has_velocity = False
        self.resample = False
        pass

    def get_subset_index(self, t, n=None):
        idxs = np.arange(self.ncells)[self.labels == t]
        if not n is None:
            idxs = np.random.choice(idxs, size=n)
        return  idxs

    def get_data(self, index):
        data = dict(X=self.X[index], t=self.labels[index])
        if self.has_velocity:
            data['V'] = self.V[index]
        return data

    def get_label_set(self):
        return self.t_set

    @property
    def T0(self):
        return self.t_0

    def __len__(self):
        return self.ncells

    def __getitem__(self, index):
        x, t = self.X[index], self.labels[index]
        data = dict(x=x, t=t)
        if self.has_velocity:
            data['v'] = self.V[index]
        return data


class OrnsteinUhlenbeckSDE_Dataset(TrajectoryInferenceDataset):
    def __init__(self, device, t_size, t_0=0.0, t_T=4.0, data_size=5000, mu=None, theta=None, sigma=0.4):
        super().__init__()
        class OrnsteinUhlenbeckSDE(torch.nn.Module):
            sde_type = 'ito'
            noise_type = 'scalar'

            def __init__(self, mu, theta, sigma, t_size):
                super().__init__()
                self.register_buffer('mu', torch.as_tensor(mu))
                self.register_buffer('theta', torch.as_tensor(theta))
                self.register_buffer('sigma', torch.as_tensor(sigma))
                self.t_size = t_size

            def f(self, t, y):
                return self.mu * t - self.theta * y

            def g(self, t, y):
                return self.sigma.expand(y.size(0), len(mu), 1) * (2 * t / self.t_size)
        self.data_size = data_size
        self.dim = len(mu)
        self.device = device
        self.ou_sde = OrnsteinUhlenbeckSDE(mu=mu, theta=theta, sigma=sigma, t_size=t_size).to(device)
        # 0, ... 1, ... , t_size
        # self.ts = torch.linspace(0, t_size, t_size*150+1, device=device)
        self.t_0, self.t_T = t_0, t_T
        ts = torch.linspace(self.t_0, self.t_T, t_size+1, device=device)
        y0 = self.base_sample(data_size)["X"]
        #bm = torchsde.BrownianInterval(t0=ts[0], t1=ts[-1], size=(data_size, brownian_size), device=device)
        ys = torchsde.sdeint(self.ou_sde, y0, ts).view(len(ts), data_size, self.dim)
        self.X = ys[1:].view(-1, self.dim)
        self.labels = np.repeat(ts[1:].cpu(), data_size)
        self.ncells = self.X.shape[0]
        self.t_set = sorted(list(set(self.labels.cpu().numpy())))

    # t = 0
    def base_sample(self, batch_size=None):
        if batch_size is None:
            batch_size = self.data_size
        return dict(X=torch.rand(batch_size, self.dim).to(self.device) * 2 - 1)

    def sample(self, y0, int_time):
        # shape(y0) = (batch_size, dim)
        data_size, dim = y0.size()
        y0, int_time = y0.to(self.device), int_time.to(self.device)
        ys = torchsde.sdeint(self.ou_sde, y0, int_time).view(len(int_time), data_size, dim)
        traj = ys.transpose(0, 1)
        return traj

    def sample_with_uncertainty(self, y0, int_time, num_repeat):
        # shape(y0) = (batch_size, dim)
        data_size, dim = y0.size()
        y0, int_time = y0.to(self.device), int_time.to(self.device)
        y0 = y0.repeat(1, num_repeat+1).view(-1, dim)
        ys = torchsde.sdeint(self.ou_sde, y0, int_time).view(len(int_time), data_size, num_repeat+1, dim)
        traj = ys.transpose(0, 1)
        return traj

class HighDimOrnsteinUhlenbeckSDE_Dataset(TrajectoryInferenceDataset):
    def __init__(self, device, t_size, z_dim=5, t_0=0.0, t_T=4.0, data_size=5000, 
                        mu=None, theta=None, sigma=0.4):
        super().__init__()
        if mu is None:
            mu = torch.linspace(0.5, 1.0, 50)
            theta = torch.linspace(-1.0, 0.5, 50)
        
        class OrnsteinUhlenbeckSDE(torch.nn.Module):
            sde_type = 'ito'
            noise_type = 'scalar'

            def __init__(self, mu, theta, sigma, t_size):
                super().__init__()
                self.register_buffer('mu', torch.as_tensor(mu))
                self.register_buffer('theta', torch.as_tensor(theta))
                self.register_buffer('sigma', torch.as_tensor(sigma))
                self.t_size = t_size

            def f(self, t, y):
                return self.mu * t - self.theta * y

            def g(self, t, y):
                return self.sigma.expand(y.size(0), len(mu), 1) * (2 * t / self.t_size)

        self.data_size = data_size
        self.dim = len(mu)
        self.z_dim = z_dim
        self.device = device
        self.ou_sde = OrnsteinUhlenbeckSDE(mu=mu, theta=theta, sigma=sigma, t_size=t_size).to(device)
        # 0, ... 1, ... , t_size
        # self.ts = torch.linspace(0, t_size, t_size*150+1, device=device)
        self.t_0, self.t_T = t_0, t_T
        ts = torch.linspace(self.t_0, self.t_T, t_size+1, device=device)
        y0 = self.base_sample_y(data_size)["X"]
        #bm = torchsde.BrownianInterval(t0=ts[0], t1=ts[-1], size=(data_size, brownian_size), device=device)
        ys = torchsde.sdeint(self.ou_sde, y0, ts).view(-1, self.dim)

        self.pca = PCA(n_components=z_dim)
        zs = torch.from_numpy(self.pca.fit_transform(ys.cpu()))
        zs = zs.view(len(ts), data_size, self.z_dim)

        self.X = zs[1:].view(-1, self.z_dim).float()
        self.labels = np.repeat(ts[1:].cpu(), data_size)
        self.ncells = self.X.shape[0]
        self.t_set = sorted(list(set(self.labels.cpu().numpy())))
        self.P = torch.from_numpy(self.pca.components_).float().to(self.device)

    def base_sample_y(self, batch_size=None):
        if batch_size is None:
            batch_size = self.data_size
        return dict(X=torch.rand(batch_size, self.dim).to(self.device) * 2 - 1)

    def base_sample(self, batch_size=None):
        X = self.base_sample_y(batch_size)["X"]
        return dict(X=torch.from_numpy(self.pca.transform(X.cpu())).float().to(self.device))

    def sample(self, z0, int_time, y_space=False):
        # shape(y0) = (batch_size, dim)
        data_size, dim = z0.size()
        assert dim == self.z_dim
        z0, int_time = z0.to(self.device), int_time.to(self.device)
        zs = torchsde.sdeint(self.ou_sde, z0, int_time).view(len(int_time), data_size, dim)
        traj = zs.transpose(0, 1)
        if y_space:
            ys = self.pca.inverse_transform(zs.view(-1, dim)).view(len(int_time), data_size, self.dim)
            traj = torch.from_numpy(ys.transpose(0, 1)).float()
        return traj

    def sample_with_uncertainty(self, z0, int_time, num_repeat, y_space=False):
        # shape(y0) = (batch_size, dim)
        data_size, dim = z0.size()
        assert dim == self.z_dim
        z0, int_time = z0.to(self.device), int_time.to(self.device)
        z0 = z0.repeat(1, num_repeat+1).view(-1, dim)
        zs = torchsde.sdeint(self.ou_sde, z0, int_time).view(len(int_time), data_size, num_repeat+1, dim)
        traj = zs.transpose(0, 1)

        if y_space:
            ys = self.pca.inverse_transform(zs.view(-1, dim)).view(len(int_time), data_size, self.dim)
            traj = torch.from_numpy(ys.transpose(0, 1)).float()
        
        return traj


class PotentialSDE_Dataset(TrajectoryInferenceDataset):
    def __init__(self, device, t_size, dim=3, t_0=0.0, t_T=4.0, data_size=5000, a=[1.0], sigma=0.1):
        super().__init__()
        class PotentialSDE(torch.nn.Module):
            sde_type = 'ito'
            noise_type = 'scalar'

            def __init__(self, sigma, t_size):
                super().__init__()
                self.register_buffer('a', torch.as_tensor(a))
                self.register_buffer('sigma', torch.as_tensor(sigma))
                self.t_size = t_size

            def f(self, t, y):
                return -self.a * (y**2)

            def g(self, t, y):
                return self.sigma.expand(y.size(0), dim, 1)
        
        class Cubic:
            def __init__(self, a):
                self.a = torch.as_tensor(a).to(device)

            def __call__(self, x, t):
                U = torch.sum(self.a * ((x ** 3) / 3), axis=1)
                return U.unsqueeze(1)
    
        self.data_size = data_size
        self.potential = Cubic(a=a)
        self.dim = dim
        self.device = device
        self.potential_sde = PotentialSDE(sigma=sigma, t_size=t_size).to(device)
        # 0, ... 1, ... , t_size
        # self.ts = torch.linspace(0, t_size, t_size*150+1, device=device)
        self.t_0, self.t_T = t_0, t_T
        ts = torch.linspace(self.t_0, self.t_T, t_size+1, device=device)
        y0 = self.base_sample(data_size)["X"]
        #bm = torchsde.BrownianInterval(t0=ts[0], t1=ts[-1], size=(data_size, brownian_size), device=device)
        ys = torchsde.sdeint(self.potential_sde, y0, ts).view(len(ts), data_size, self.dim)
        self.X = ys[1:].view(-1, self.dim)
        self.labels = np.repeat(ts[1:].cpu(), data_size)
        self.ncells = self.X.shape[0]
        self.t_set = sorted(list(set(self.labels.cpu().numpy())))

    # t = 0
    def base_sample(self, batch_size=None):
        if batch_size is None:
            batch_size = self.data_size
        return dict(X=torch.rand(batch_size, self.dim).to(self.device) + 5.0)

    def sample(self, y0, int_time):
        # shape(y0) = (batch_size, dim)
        data_size, dim = y0.size()
        y0, int_time = y0.to(self.device), int_time.to(self.device)
        ys = torchsde.sdeint(self.potential_sde, y0, int_time).view(len(int_time), data_size, dim)
        traj = ys.transpose(0, 1)
        return traj

    def sample_with_uncertainty(self, y0, int_time, num_repeat):
        # shape(y0) = (batch_size, dim)
        data_size, dim = y0.size()
        y0, int_time = y0.to(self.device), int_time.to(self.device)
        y0 = y0.repeat(1, num_repeat+1).view(-1, dim)
        ys = torchsde.sdeint(self.potential_sde, y0, int_time).view(len(int_time), data_size, num_repeat+1, dim)
        traj = ys.transpose(0, 1)
        return traj

class scRNASeq(TrajectoryInferenceDataset):
    def __init__(self, data_path_list, dim,  use_v=True, LMT=-1, scaler=None):
        super().__init__()
        self.dim = dim
        self.data_path_list = data_path_list
        self.t_0, self.t_T = 0.0, 4.0
        self.has_velocity = use_v

        X, ts, V = [], [], []
        for data_path in data_path_list:
            npzfile = np.load(data_path)
            X.append(npzfile['X'])
            ts.append(npzfile['ts'])
            if use_v:
                V.append(npzfile['v'])

        X = np.concatenate(X, axis=0)
        ts = np.concatenate(ts, axis=0)
        if self.has_velocity:
            V = np.concatenate(V, axis=0)
        t_set = sorted(list(set(ts[ts > 0])))

        if LMT in t_set[:-1]:
            X = X[ts != LMT]
            if self.has_velocity:
                V = V[ts != LMT]
            ts = ts[ts != LMT]

        if scaler is None:
            self.scaler = StandardScaler()
            self.scaler.fit(X)
        else:
            self.scaler = scaler
        X = self.scaler.transform(X)
        # t_set = sorted(list(set(ts)))
        # X, ts = self.reduce_subset(X, ts, t_set)

        self._full_data = dict(X=torch.from_numpy(X[:, :dim]), t=torch.from_numpy(ts))

        self.y0 = X[ts == 0, :dim]
        self.labels = torch.from_numpy(ts[ts > 0])
        self.X = torch.from_numpy(X[ts > 0, :dim])

        self.ncells = self.X.shape[0]
        self.t_set = sorted(list(set(self.labels.numpy())))

        if self.has_velocity:
            V /= self.scaler.scale_
            self._full_data['V'] = V
            self.v0 = V[ts == 0, :dim]
            self.V = torch.from_numpy(V[ts > 0, :dim])

    @property
    def full_data(self):
        return self._full_data
    
    def get_scaler(self):
        return self.scaler
    
    def scaler_params(self):
        return { 'mean' : torch.from_numpy(self.scaler.mean_[:self.dim]).float(), 'scale' : torch.from_numpy(self.scaler.scale_[:self.dim]).float() }

    def base_sample(self, batch_size=None):
        if batch_size is None:
            x = torch.from_numpy(self.y0).float()
            if self.has_velocity:
                v = torch.from_numpy(self.v0).float()
                return dict(X=x, V=v)
        else:
            idx = np.random.choice(np.arange(len(self.y0)), size=batch_size, replace=False)
            x = torch.from_numpy(self.y0[idx]).float()

            if self.has_velocity:
                v = torch.from_numpy(self.v0[idx]).float()
                return dict(X=x, V=v)

        return dict(X=x)

class UniformDataset(TrajectoryInferenceDataset):
    dim = 2
    def __init__(self, device, t_0=0.0, t_T=1.0, data_size=5000):
        super().__init__()
        t_size = 1
        self.data_size = data_size
        self.device = device
        # 0, ... 1, ... , t_size
        self.t_0, self.t_T = t_0, t_T
        x = (-0.25) * torch.rand(data_size, 1) + 1.25
        y = (-2) * torch.rand(data_size, 1) + 1
        self.X = torch.cat([x, y], dim=1).float()
        self.labels = torch.ones(data_size) * self.t_T
        self.ncells = self.X.shape[0]
        self.t_set = sorted(list(set(self.labels.cpu().numpy())))

    # t = 0
    def base_sample(self, batch_size=None):
        if batch_size is None:
            batch_size = self.data_size

        x = (-0.25) * torch.rand(batch_size, 1) - 1
        y = (-2) * torch.rand(batch_size, 1) + 1
        init = torch.cat([x, y], dim=1).float()
        return dict(X=init.to(self.device))

class UniformDataset2(TrajectoryInferenceDataset):
    dim = 2
    def __init__(self, device, t_0=0.0, t_T=1.0, data_size=5000):
        super().__init__()
        t_size = 1
        self.data_size = data_size
        self.device = device
        # 0, ... 1, ... , t_size
        self.t_0, self.t_T = t_0, t_T
        x1 = 0.5 * torch.rand(data_size // 2, 1) + 0.75
        y1 = 0.5 * torch.rand(data_size // 2, 1) + 0.75
        X1 = torch.cat([x1, y1], dim=1).float()
        x2 = 0.5 * torch.rand(data_size - data_size // 2, 1) - 1.25
        y2 = 0.5 * torch.rand(data_size - data_size // 2, 1) - 1.25
        X2 = torch.cat([x2, y2], dim=1).float()
        self.X = torch.cat([X1, X2], dim=0).float()
        self.labels = torch.ones(data_size) * self.t_T
        self.ncells = self.X.shape[0]
        self.t_set = sorted(list(set(self.labels.cpu().numpy())))

    # t = 0
    def base_sample(self, batch_size=None):
        if batch_size is None:
            batch_size = self.data_size
        x1 = 0.5 * torch.rand(batch_size // 2, 1) - 1.25
        y1 = 0.5 * torch.rand(batch_size // 2, 1) + 0.75
        X1 = torch.cat([x1, y1], dim=1).float()

        x2 = 0.5 * torch.rand(batch_size - batch_size // 2, 1) + 0.75
        y2 = 0.5 * torch.rand(batch_size - batch_size // 2, 1) - 1.25
        X2 = torch.cat([x2, y2], dim=1).float()
        init = torch.cat([X1, X2], dim=0).float()
        return dict(X=init.to(self.device))


class MFGDataset(TrajectoryInferenceDataset):
    def __init__(self, problem_name, dim, t_0=0.0, t_T=1.0, data_size=5000, batch_size=512,interval=100, diffusion_std=1.0, MF_cost=0.0, device='cpu'):
        super().__init__()
        self.resample =True
        self.dim = dim
        self.data_size = data_size
        self.batch_size = batch_size
        self.device = device
        self.t_0, self.t_T = t_0, t_T
        
        Option = namedtuple('Option', ['problem_name', 't0', 'T', 'interval', 'diffusion_std', 'samp_bs', 'MF_cost', 'x_dim', 'device'])
        self.opt = Option(
            problem_name=problem_name,
            t0=self.t_0,
            T=self.t_T,
            interval=interval,
            diffusion_std=diffusion_std,
            samp_bs=data_size,
            MF_cost=MF_cost,
            x_dim=self.dim,
            device=device,
        )

        self.mfg = MFG(self.opt)
        self.X = self.mfg.sde.pT.sample(batch=self.data_size)
        self.labels = torch.ones(self.data_size) * self.t_T
        self.ncells = self.data_size
        self.t_set = [self.t_T]
        if self.dim > 100:
            self.pca_X, self.V, self.mean = self.pca(self.X, 100)

    def resampling(self):
        del self.X
        self.X = self.mfg.sde.pT.sample(batch=self.data_size)
        
    def base_sample(self, batch_size=None):
        if batch_size is None:
            batch_size = self.data_size
        return dict(X=self.mfg.sde.p0.sample(batch=batch_size).to(self.device))

    def get_subset_index(self, t, n=None):
        idxs = np.arange(self.data_size)[self.labels == t]
        if not n is None:
            idxs = np.random.choice(idxs, size=n)
        return  idxs
    
    def pca(self, x: torch.Tensor, low_dim=100):
        B, nx = x.shape

        mean_pca_x = torch.mean(x, dim=0, keepdim=True)
        y = x - mean_pca_x

        if B > 200:
            rand_idxs = torch.randperm(B)[:200]
            y = y[rand_idxs]
        # U: (batch, k)
        # S: (k, k)
        # VT: (k, nx)
        U, S, VT = torch.linalg.svd(y)
        D = min(low_dim, nx)
        # log.info("Singular values of xs_f at final timestep:")
        # log.info(S)
        # Keep the first and last directions.
        VT = VT[:D, :]
        #VT = VT[[0, -1], :]
        # assert VT.shape == (D, nx)
        V = VT.T

        proj_x = y @ V

        self.V = V
        self.mean = mean_pca_x
        return proj_x, V, mean_pca_x

    def pca_proj(self, x):
        if self.dim > 100:
            return (x - self.mean) @ self.V
        else:
            return x

class BalancedBatchSampler(BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, dataset, n_samples):
        L = len(dataset)
        self.labels = np.array([ dataset[i]['t'] for i in range(L) ])
        self.labels_set = list(set(self.labels))
        self.label_to_indices = {label: np.where(self.labels == label)[0] for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = len(self.labels_set)
        self.n_samples = n_samples
        self.dataset = dataset
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size <= len(self.dataset):
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return len(self.dataset) // self.batch_size