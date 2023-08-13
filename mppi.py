import numpy as np

# TODO: use in-place operation
class MPPI:
    def __init__(self, u_dim, x_dim, N_smp, T_pred, pred_dT, variance, lmd = 100, alpha=0.8):
        self.u_dim = u_dim
        self.x_dim = x_dim
        self.N_smp = N_smp
        self.T_pred = T_pred
        self.pred_dT = pred_dT
        self.variance = variance
        self.lmd = lmd
        self.alpha = alpha
        self.reset()

    def reset(self):
        self.u_pred = np.zeros((self.T_pred, self.u_dim))
        self.debug_trace = np.zeros((self.T_pred, self.N_smp, self.x_dim))
        self.w = None

    def gen_delta_u(self):
        delta_u = np.zeros((self.T_pred, self.N_smp, self.u_dim))
        a = int(self.N_smp * self.alpha)
        du = np.random.normal(size=self.T_pred * (a) * self.u_dim).reshape((self.T_pred, -1, self.u_dim))
        delta_u[:, :a] = self.variance * du
        return delta_u

    # x_init shape :(1, state_dim)
    # max_u  shape : (u_dim)
    # input shape of update_fun/cost_fun : (N_smp, state_dim) or (1, state_dim)
    # return shape: (T_pred, u_dim)
    def step(self, x_init, max_u, update_fun, cost_fun):
        # reuse previous prediction
        for i in range(self.T_pred-1):
            self.u_pred[i] = self.u_pred[i+1]

        delta_u = self.gen_delta_u()
        Stk = np.zeros(self.N_smp)
        x = np.tile(x_init, (self.N_smp,1))
        self.debug_trace[0] = x

        for i in range(self.T_pred-1):
            u_cur = np.clip(self.u_pred[i] + delta_u[i], -max_u, max_u)
            x = update_fun(x, u_cur, self.pred_dT)
            self.debug_trace[i+1] = x
            c = cost_fun(x, u_cur)
            Stk += c

        beta = np.min(Stk)
        w = np.exp(-(1 / self.lmd) * (Stk-beta))
        sw = np.sum(w)
        w = w / sw
        #du = delta_u @ w
        du = np.tensordot(delta_u, w, axes=(1,0))
        self.u_pred = np.clip(self.u_pred + du, -max_u, max_u)
        sorted_idx = sorted(range(self.N_smp), key=lambda i: -w[i])
        self.w = w[sorted_idx]
        self.debug_trace = self.debug_trace[:, sorted_idx]

        return self.u_pred[:1]

