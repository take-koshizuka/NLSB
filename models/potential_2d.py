import torch

class Box:
    def __init__(self, M=1e1, temp=1e-2, ymin=-0.5, ymax=0.5, xmin=-0.5, xmax=0.5):
        self.M = M
        self.xmin, self.xmax = xmin, xmax
        self.ymin, self.ymax = ymin, ymax
        self.temp = temp

    def __call__(self, x, t):
        Ux = (torch.sigmoid((x[:, 0] - self.xmin) / self.temp) - torch.sigmoid((x[:, 0] - self.xmax) / self.temp)) * self.M
        Uy = (torch.sigmoid((x[:, 1] - self.ymin) / self.temp) - torch.sigmoid((x[:, 1] - self.ymax) / self.temp)) * self.M
        U = -Ux * Uy
        return U.unsqueeze(1)
    
class Slit:
    def __init__(self, M=1e1, temp=1e-2, xmin=-0.1, xmax=0.1, ymin=-0.25, ymax=0.25):
        self.M = M
        self.ymin, self.ymax = ymin, ymax
        self.xmin, self.xmax = xmin, xmax
        self.temp = temp

    def __call__(self, x, t):
        Ux = (torch.sigmoid((x[:, 0] - self.xmin) / self.temp) - torch.sigmoid((x[:, 0] - self.xmax) / self.temp)) * self.M
        Uy = (torch.sigmoid((x[:, 1] - self.ymin) / self.temp) - torch.sigmoid((x[:, 1] - self.ymax) / self.temp)) * self.M - self.M
        U = Ux * Uy
        return U.unsqueeze(1)

class HarmonicOscillator:
    def __init__(self, k=5.0):
        self.k = k

    def __call__(self, x, t):
        return -0.5 * self.k * torch.sum(torch.pow(x, 2), dim=1, keepdims=True)

class Hill:
    def __init__(self, mu=0, sigma=1.0, h=10.0):
        self.mu = mu
        self.sigma = sigma
        self.h = h
    def __call__(self, x, t):
        return -self.h * torch.exp(-torch.sum(torch.pow(x - self.mu, 2), dim=1, keepdims=True) / self.sigma)

