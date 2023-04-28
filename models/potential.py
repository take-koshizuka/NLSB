import torch
import logging
try:
    from pycave.bayes import GaussianMixture
    logging.getLogger("pycave").setLevel(logging.WARNING)
except ImportError:
    pass

class Cubic:
    def __init__(self, a):
        self.a = a

    def __call__(self, x, t):
        U = torch.sum(x ** 3 / 3, axis=1)
        return -self.a * U
    
    def grad(self, x, t):
        return self.a * (x**2)

class Entropy:
    def __init__(self, proj, x, num_components=5, mu_init=None):
        self.device  = "gpu" if torch.cuda.is_available() else "cpu"
        self.proj = proj
        
    #def fit(self, x, num_components=5, mu_init=None):
        z = self.proj(x)
        if mu_init is None:
            self.score = GaussianMixture(num_components=num_components, covariance_type='spherical', trainer_params=dict(accelerator=self.device, devices=1, progress_bar_refresh_rate=0, logger=False))
        else:
            self.score = GaussianMixture(num_components=num_components, init_means=mu_init, covariance_type='spherical', trainer_params=dict(max_epochs=50, accelerator=self.device, progress_bar_refresh_rate=0, devices=1, logger=False))
        
        with torch.no_grad():
            self.score.fit(z)

        self.score.model_.to('cpu')

    def params(self):
        return dict(mu=self.score.model_.means)

    def __call__(self, x):
        z = self.proj(x)
        return self.score.score_samples(z.float()).transpose(0, 1).to(x.device) + 1.0
