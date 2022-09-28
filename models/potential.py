import torch

class Cubic:
    def __init__(self, a):
        self.a = a

    def __call__(self, x, t):
        U = torch.sum(x ** 3 / 3, axis=1)
        return -self.a * U
    
    def grad(self, x, t):
        return self.a * (x**2)
    