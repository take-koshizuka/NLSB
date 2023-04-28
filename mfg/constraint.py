import logging
from math import dist
from typing import NamedTuple, Optional

import numpy as np
import torch
import torch.distributions as td

log = logging.getLogger(__file__)


"""
class Sampler:
    def __init__(self, distribution: td.Distribution, batch_size: int, device: str):
        self.distribution = distribution
        self.batch_size = batch_size
        self.device = device

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(x)

    def sample(self, batch: Optional[int] = None) -> torch.Tensor:
        if batch is None:
            batch = self.batch_size
        return self.distribution.sample([batch]).to(self.device)
"""
class Sampler:
    def __init__(self, distributions, batch_size: int, device: str):
        self.distributions = distributions
        self.n_dist = len(self.distributions)
        self.batch_size = batch_size
        self.device = device

    def log_prob(self, x: torch.Tensor) -> torch.Tensor: 
        log_prob = 0.0
        for dist in self.distributions:
            log_prob += 1/self.n_dist * dist.log_prob(x)
        return log_prob

    def sample(self, batch: Optional[int] = None) -> torch.Tensor:
        if batch is None:
            batch = self.batch_size
        N = [batch//self.n_dist] * self.n_dist
        for i in range(batch % self.n_dist): 
            N[i] += 1
        samples = []
        for i in range(self.n_dist):
            samples.append(self.distributions[i].sample([N[i]]))
        return torch.cat(samples, dim=0).to(self.device)

class ProblemDists(NamedTuple):
    p0: Sampler
    pT: Sampler


def build_constraint(problem_name: str, batch_size: int, device: str) -> ProblemDists:
    log.info("build distributional constraints ...")

    distribution_builder = {
        "GMM": gmm_builder,
        "Stunnel": stunnel_builder,
        "Vneck": vneck_builder,
        "opinion": opinion_builder,
        "opinion_1k": opinion_1k_builder,
    }.get(problem_name)

    return distribution_builder(batch_size, device)


def gmm_builder(batch_size: int, device: str) -> ProblemDists:

    # ----- pT -----
    radius, num = 16, 8
    arc = 2 * np.pi / num
    xs = [np.cos(arc * idx) * radius for idx in range(num)]
    ys = [np.sin(arc * idx) * radius for idx in range(num)]

    mix = td.Categorical(
        torch.ones(
            num,
        )
    )
    comp = td.Independent(td.Normal(torch.Tensor([[x, y] for x, y in zip(xs, ys)]), torch.ones(num, 2)), 1)
    dist = td.MixtureSameFamily(mix, comp)
    pT = Sampler(dist, batch_size, device)

    # ----- p0 -----
    dist = td.MultivariateNormal(torch.zeros(2), torch.eye(2))
    p0 = Sampler(dist, batch_size, device)

    return ProblemDists(p0, pT)


def vneck_builder(batch_size: int, device: str) -> ProblemDists:

    # ----- pT -----
    dist = td.MultivariateNormal(torch.Tensor([7, 0]), 0.2 * torch.eye(2))
    pT = Sampler(dist, batch_size, device)

    # ----- p0 -----
    dist = td.MultivariateNormal(-torch.Tensor([7, 0]), 0.2 * torch.eye(2))
    p0 = Sampler(dist, batch_size, device)

    return ProblemDists(p0, pT)


def stunnel_builder(batch_size: int, device: str) -> ProblemDists:

    # ----- pT -----
    dist = td.MultivariateNormal(torch.Tensor([11, 1]), 0.5 * torch.eye(2))
    pT = Sampler(dist, batch_size, device)

    # ----- p0 -----
    dist = td.MultivariateNormal(-torch.Tensor([11, 1]), 0.5 * torch.eye(2))
    p0 = Sampler(dist, batch_size, device)

    return ProblemDists(p0, pT)


def opinion_builder(batch_size: int, device: str) -> ProblemDists:

    p0_std = 0.25
    pT_std = 3.0

    # ----- p0 -----
    mu0 = torch.zeros(2)
    covar0 = p0_std * torch.eye(2)

    # Start with kind-of polarized opinions.
    covar0[0, 0] = 0.5

    # ----- pT -----
    muT = torch.zeros(2)
    # Want to finish with more homogenous opinions.
    covarT = pT_std * torch.eye(2)

    dist = td.MultivariateNormal(muT, covarT)
    pT = Sampler([dist], batch_size, device)

    dist = td.MultivariateNormal(mu0, covar0)
    p0 = Sampler([dist], batch_size, device)

    return ProblemDists(p0, pT)
"""
def opinion_builder(batch_size: int, device: str) -> ProblemDists:

    p0_std = 1.0
    pT_std = 3.0

    # ----- p0 -----
    mu0_1 = torch.zeros(2) + 1.
    covar0_1 = p0_std * torch.eye(2)

    mu0_2 = torch.zeros(2) - 1.
    covar0_2 = p0_std * torch.eye(2)

    # Start with kind-of polarized opinions.
    # covar0[0, 0] = 0.5

    # ----- pT -----
    muT = torch.zeros(2)
    # Want to finish with more homogenous opinions.
    covarT = pT_std * torch.eye(2)

    dist = td.MultivariateNormal(muT, covarT)
    pT = Sampler([dist], batch_size, device)

    dist_1 = td.MultivariateNormal(mu0_1, covar0_1)
    dist_2 = td.MultivariateNormal(mu0_2, covar0_2)
    p0 = Sampler([dist_1, dist_2], batch_size, device)

    return ProblemDists(p0, pT)
"""

def opinion_1k_builder(batch_size: int, device: str) -> ProblemDists:
    p0_std = 2.0
    pT_std = 2.0

    # ----- p0 -----
    mu0 = torch.zeros(1000)
    covar0 = p0_std * torch.eye(1000)

    # Start with kind-of polarized opinions.
    #rand_idxs = torch.randperm(1000)[:300]
    #covar0[rand_idxs, rand_idxs] = 10.0
    covar0[0, 0] = 10.0
    #covar0[-1, -1] = 20.0

    # ----- pT -----
    muT = torch.zeros(1000)
    # Want to finish with more homogenous opinions.
    covarT = pT_std * torch.eye(1000)

    dist = td.MultivariateNormal(muT, covarT)
    pT = Sampler([dist], batch_size, device)

    dist = td.MultivariateNormal(mu0, covar0)
    p0 = Sampler([dist], batch_size, device)
    
    return ProblemDists(p0, pT)

"""
def opinion_1k_builder(batch_size: int, device: str) -> ProblemDists:

    p0_std = 1.0
    pT_std = 1.0

    # ----- p0 -----
    mu0_1 = torch.zeros(1000)
    mu0_1[0] = 4.
    covar0_1 = p0_std * torch.eye(1000)
    covar0_1[0, 0] = 5.0

    mu0_2 = torch.zeros(1000)
    mu0_2[0] = -4.
    covar0_2 = p0_std * torch.eye(1000)
    covar0_2[0, 0] = 5.0

    # ----- pT -----
    muT = torch.zeros(1000)
    # Want to finish with more homogenous opinions.
    covarT = pT_std * torch.eye(1000)

    dist = td.MultivariateNormal(muT, covarT)
    pT = Sampler([dist], batch_size, device)

    dist_1 = td.MultivariateNormal(mu0_1, covar0_1)
    dist_2 = td.MultivariateNormal(mu0_2, covar0_2)
    p0 = Sampler([dist_1, dist_2], batch_size, device)
    
    return ProblemDists(p0, pT)

    p0_std = 0.5
    pT_std = 3.0

    # ----- p0 -----
    mu0_1 = torch.zeros(2) + 1.
    covar0_1 = p0_std * torch.eye(2)

    mu0_2 = torch.zeros(2) - 1.
    covar0_2 = p0_std * torch.eye(2)

    # Start with kind-of polarized opinions.
    # covar0[0, 0] = 0.5

    # ----- pT -----
    muT = torch.zeros(2)
    # Want to finish with more homogenous opinions.
    covarT = pT_std * torch.eye(2)

    dist = td.MultivariateNormal(muT, covarT)
    pT = Sampler([dist], batch_size, device)

    dist_1 = td.MultivariateNormal(mu0_1, covar0_1)
    dist_2 = td.MultivariateNormal(mu0_2, covar0_2)
    p0 = Sampler([dist_1, dist_2], batch_size, device)

    return ProblemDists(p0, pT)
"""