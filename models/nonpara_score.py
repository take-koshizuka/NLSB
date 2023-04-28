import torch
# import kscore
"""
def get_estimator(kernel, est, lam, subsample_rate):
    kernel_dicts = {
        "diagonal_adaptive_gaussian": kscore.kernels.DiagonalAdaptiveGaussian,
        "curlfree_adaptive_gaussian": kscore.kernels.CurlFreeAdaptiveGaussian,
        "curlfree_imq": kscore.kernels.CurlFreeIMQ,
        "curlfree_rbf": kscore.kernels.CurlFreeGaussian,
        "diagonal_imq": kscore.kernels.DiagonalIMQ,
        "diagonal_rbf": kscore.kernels.DiagonalGaussian,
    }

    estimator_dicts = {
        "tikhonov": kscore.estimators.Tikhonov,
        "nu": kscore.estimators.NuMethod,
        "landweber": kscore.estimators.Landweber,
        "spectral_cutoff": kscore.estimators.SpectralCutoff,
        "stein": kscore.estimators.Stein,
    }

    if est == "tikhonov_nystrom":
        estimator = kscore.estimators.Tikhonov(
            lam=lam, subsample_rate=subsample_rate, kernel=kernel_dicts[kernel]()
        )
    else:
        estimator = estimator_dicts[est](lam=lam, kernel=kernel_dicts[kernel]())
    return estimator

class ScoreEstimator:
    def __init__(self, kernel, estimator, lam, subsample_rate):
        self.estimator = get_estimator(kernel, estimator, lam, subsample_rate)

    def fit(self, X):
        with torch.no_grad():
            self.estimator.fit(X)

    def logp(self, X):
        logp = self.estimator.compute_energy(X)
        return logp
"""