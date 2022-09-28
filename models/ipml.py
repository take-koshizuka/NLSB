from models.gp_sinkhorn.SDE_solver import solve_sde_RK

class SBSDE:
    def __init__(self, sigma, kernel, noise, gp_mean_function):
        self.noise_type = fsde.noise_type
        self.sigma_type = fsde.sigma_type
        self.input_dim = fsde.input_dim
        self.brownian_size = fsde.brownian_size

        self.fsde = fsde
        self.df = Phi(**fsde.drift_cfg, d=self.input_dim)

        self.criterion_cfg = fsde.criterion_cfg
        self.solver_cfg = fsde.solver_cfg

        self.sigma = sigma
        self.gp_drift_model = None
        self.gp_b_drift_model = None
        self.kernel = kernel
        self.noise = noise
        self.gp_mean_function = gp_mean_function
        
    def f(self, t, y):
        return self.gp_drift_model.predict(y, debug=False)
    
    def fb(self, t, y):
        return self.gp_b_drift_model.predict(y, debug=False)

    def g(self, t, y):
        return self.sigma

    def forward(self, ts, x0):
        assert ts[0] <= ts[1]
        N = int((ts[-1] - ts[0]) / self.solver_cfg['dt'])
        t, xs = solve_sde_RK(b_drift=self.f, sigma=self.sigma, X0=x0, dt=self.solver_cfg['dt'], N=N, t0=ts[0])
        return dict(xs=xs[:, :, :self.input_dim])

    def backward(self, ts, xT):
        assert ts[0] >= ts[1]
        N = int((ts[0] - ts[-1]) / self.solver_cfg['dt'])
        t, xs = solve_sde_RK(b_drift=self.fb, sigma=self.sigma, X0=xT, dt=self.solver_cfg['dt'], N=N, t0=ts[0])

    def criterion(self, x, x_hat):
        loss_D = self.loss_fn(x_hat, x)
        return dict(loss=loss_D, loss_D=loss_D)

    def parameters_lr(self):
        return self.df.parameters()