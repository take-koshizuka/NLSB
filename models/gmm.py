import abc
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch import distributions
from sklearn.mixture import GaussianMixture

def compute_log_det_cholesky(matrix_chol, n_features):
    """Compute the log-det of the cholesky decomposition of matrices.
    Parameters
    ----------
    matrix_chol : array-like
        Cholesky decompositions of the matrices.
        'full' : shape of (n_components, n_features, n_features)
        'tied' : shape of (n_features, n_features)
        'diag' : shape of (n_components, n_features)
        'spherical' : shape of (n_components,)
    n_features : int
        Number of features.
    Returns
    -------
    log_det_precision_chol : array-like of shape (n_components,)
        The determinant of the precision matrix for each component.
    """
    
    n_components, _, _ = matrix_chol.shape
    log_det_chol = torch.sum(
        torch.log(matrix_chol.view(n_components, -1)[:, :: n_features + 1]), 1
    )
    return log_det_chol

class GenerativeModel(abc.ABC, nn.Module):
    """Base class inherited by all generative models in pytorch-generative.
    Provides:
        * An abstract `sample()` method which is implemented by subclasses that support
          generating samples.
        * Variables `self._c, self._h, self._w` which store the shape of the (first)
          image Tensor the model was trained with. Note that `forward()` must have been
          called at least once and the input must be an image for these variables to be
          available.
        * A `device` property which returns the device of the model's parameters.
    """

    @property
    def device(self):
        return next(self.parameters()).device

    @abc.abstractmethod
    def sample(self, n_samples):
        ...


class MixtureModel(GenerativeModel):
    """Base class inherited by all mixture models in pytorch-generative.
    Provides:
        * A generic `forward()` method which returns the log likelihood of the input
          under the distribution.` The log likelihood of the component distributions
          must be defined by the subclasses via `_component_log_prob()`.
        * A generic `sample()` method which returns samples from the distribution.
          Samples from the component distribution must be defined by the subclasses via
          `_component_sample()`.
    """

    def __init__(self, n_components, n_features):
        """Initializes a new MixtureModel instance.
        Args:
            n_components: The number of component distributions.
            n_features: The number of features (i.e. dimensions) in each component.
        """
        super().__init__()
        self.n_components = n_components
        self.n_features = n_features

    @abc.abstractmethod
    def _component_log_prob(self):
        """Returns the log likelihood of the component distributions."""
        
    def forward(self, x):
        mixture_log_prob = torch.log(self.mixture_logits)
        log_prob = mixture_log_prob + self._component_log_prob(x)
        return torch.logsumexp(log_prob, dim=-1)

    @abc.abstractmethod
    def _component_sample(self, idxs):
        """Returns samples from the component distributions conditioned on idxs."""

    def sample(self, n_samples):
        with torch.no_grad():
            shape = (n_samples,)
            idxs = distributions.Categorical(logits=self.mixture_logits).sample(shape)
            sample = self._component_sample(idxs)
            return sample.view(n_samples, *self._original_shape[1:])


class GaussianMixtureModel(MixtureModel):
    """A categorical mixture of Gaussian distributions with diagonal covariance."""

    def __init__(self, X, n_components):
        super().__init__(n_components, X.shape[1])
        gm = GaussianMixture(n_components=n_components, covariance_type="full").fit(X)
        self.gm = gm
        self.register_buffer('means', torch.from_numpy(gm.means_).float())
        self.register_buffer('precisions_cholesky', torch.from_numpy(gm.precisions_cholesky_).float())
        self.register_buffer('mixture_logits', torch.from_numpy(gm.weights_).float())

    def _component_log_prob(self, x):
        n_samples, n_features = x.shape
        n_components, _ = self.means.shape
        log_det = compute_log_det_cholesky(self.precisions_cholesky, n_features)
        log_prob = torch.empty((n_samples, n_components)).to(log_det)
        for k, (mu, prec_chol) in enumerate(zip(self.means, self.precisions_cholesky)):
            y = torch.matmul(x, prec_chol) - torch.matmul(mu, prec_chol)
            log_prob[:, k] = torch.sum(torch.square(y), axis=1)
        return -0.5 * (n_features * np.log(2 * np.pi) + log_prob) + log_det

    def _component_sample(self, idxs):
        mean, std = self.mean[idxs], self.std[idxs]
        return distributions.Normal(mean, std).sample()


class TimeDependentGMM(nn.Module):
    def __init__(self, Xs, ts, n_components_list, intervals=None):
        super(TimeDependentGMM, self).__init__()
        if intervals is None:
            t_set = list(sorted(list(set(ts.numpy()))))
            assert len(n_components_list) == (len(t_set) - 1)
            intervals = []
            for i in range(len(t_set) - 1):
                intervals.append((t_set[i], t_set[i+1]))

        GMMs = []
        for i, (t0, t1) in enumerate(intervals):
            Xt = Xs[torch.logical_or(ts == t0, ts == t1)]
            GMMs.append(GaussianMixtureModel(Xt, n_components_list[i]))
            
        self.GMMs = nn.ModuleList(GMMs)
        self.intervals = intervals
    
    def forward(self, x, t):
        for i, (t0, t1) in enumerate(self.intervals):
            if t0 <= t < t1:
                return self.GMMs[i](x)
        if t == t1:
            return self.GMMs[-1](x)
        raise ValueError