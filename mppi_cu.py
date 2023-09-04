import numpy as np
import cupy as cp
import sys
import time

# TODO: use in-place operation
class MPPI:
    def __init__(self, u_dim, x_dim, N_smp, T_pred, pred_dT, variance, lmd = 100, alpha=0.8):
        self.u_dim = u_dim
        self.x_dim = x_dim
        self.N_smp = N_smp
        self.block_size = 128
        assert self.N_smp % self.block_size == 0
        self.grid_size = self.N_smp // self.block_size
        self.T_pred = T_pred
        self.pred_dT = pred_dT
        self.variance = variance
        self.lmd = lmd
        self.alpha = alpha
        self.u_pred = np.zeros((self.T_pred, self.u_dim))
        self.cu_u_cur = cp.zeros((self.T_pred, self.u_dim, self.N_smp))
        self.cu_c = cp.zeros(self.N_smp)
        self.cu_x = cp.zeros((self.x_dim, self.N_smp))

        self.reset()

    def reset(self):
        self.u_pred.fill(0)
        self.x_tmp = cp.zeros((self.T_pred, self.x_dim, self.N_smp))
        self.debug_trace = np.zeros((self.T_pred, self.x_dim, self.N_smp))
        self.w = None

    def gen_delta_u(self):
        delta_u = np.zeros((self.T_pred, self.u_dim, self.N_smp))
        a = int(self.N_smp * self.alpha)
        du = np.random.normal(size=self.T_pred * (a) * self.u_dim).reshape((self.T_pred, self.u_dim, -1))
        delta_u[:, :, :a] = self.variance * du
        return delta_u

    def step(self, x_init, max_u, update_kernel, cost_kernel):
        # reuse previous prediction
        for i in range(self.T_pred-1):
            self.u_pred[i] = self.u_pred[i+1]

        delta_u = self.gen_delta_u()
        self.cu_u_cur.set(self.u_pred.reshape((self.T_pred, self.u_dim, 1)) + delta_u)
        self.cu_c.fill(0)
        self.cu_x.set(np.tile(x_init, (self.N_smp,1)).T)
        self.x_tmp[0] = self.cu_x

        start_time = time.perf_counter()

        for i in range(self.T_pred-1):
            update_kernel((self.grid_size,), (self.block_size,), (*self.x_tmp[i], *self.x_tmp[i+1], *self.cu_u_cur[i], self.pred_dT))
            cost_kernel((self.grid_size,), (self.block_size,), (*self.x_tmp[i+1], *self.cu_u_cur[i], self.cu_c))

        end_time = time.perf_counter()
        print('time1 = {} Seconds'.format(end_time - start_time))


        start_time = time.perf_counter()
        stk = self.cu_c.get()
        beta = np.min(stk)
        w = np.exp(-(1 / self.lmd) * (stk-beta))
        sw = np.sum(w)
        w = w / sw

        du = np.tensordot(delta_u, w, axes=(2,0))
        self.u_pred = np.clip(self.u_pred + du, -max_u, max_u)

        sorted_idx = sorted(range(self.N_smp), key=lambda i: -w[i])
        self.w = w[sorted_idx]
        self.debug_trace = self.x_tmp.get()
        self.debug_trace = self.debug_trace[:, :, sorted_idx]
        end_time = time.perf_counter()
        print('time2 = {} Seconds'.format(end_time - start_time))

        return self.u_pred[0]

