from linktree import *
from jointlink import *
from simulation import *
from draw_cmd import *
from numpy import sin, cos
from mppi import MPPI
from jet_color import jet

np.random.seed(seed=10)
r = 0.05
l = 0.25
g = 9.81
mc = 1
mp = 0.01

def update_carpole_scalar(x, dx, th, dth, u, dt):
    ddx   = (u+(g*mp*cos(th)+dth**2*l*mp)*sin(th))/(mp*sin(th)**2+mc)
    ddth  = -(cos(th)*u+(dth**2*l*mp*cos(th)+g*mp+g*mc)*sin(th))/(l*mp*sin(th)**2+l*mc)
    new_dx  = dx  + ddx * dt
    new_dth = dth + ddth * dt
    new_x   = x   + new_dx * dt
    new_th  = th  + new_dth * dt
    return new_x, new_dx, new_th, new_dth

def update_carpole(s, us, dt):
    x, dx, th, dth = s[:,0], s[:,1], s[:,2], s[:,3]
    u = us[:,0]
    new_x, new_dx, new_th, new_dth = update_carpole_scalar(x, dx, th, dth, u, dt)
    return np.array([new_x, new_dx, new_th, new_dth]).T

def cost_function_info(x, dx, th, dth, u):
    R = 0.001;
    cx = 60*x**2
    cdx = 0.1 * dx ** 2
    cth = 12 * (1 + np.cos(th)) ** 2
    cdth = 0.1 * dth ** 2
    cu = 0.5*R*u**2
    return cx, cdx, cth, cdth, cu

def cost_function(s, u):
    x, dx, th, dth = s[:,0], s[:,1], s[:,2], s[:,3]
    cx, cdx, cth, cdth, cu = cost_function_info(x, dx, th, dth, u[:,0])
    return cx + cdx + cth + cdth + cu

N_smp = 1000
T_pred = 50
pred_dT = 0.02

DEFAULT_TRACE_NUM = 10

class CarPole(Simulator):
    def __init__(self):
        self.reset_state()
        self.pause()

    def reset_state(self):
        self.x = 0
        self.dx = 0
        u_dim = 1
        state_dim = 4
        #self.th = np.pi
        self.th = 0
        self.dth = 0
        self.u = 0
        self.t = 0
        self.itr = 0
        self.u_pred = np.zeros((T_pred,1))
        #self.max_u = np.inf
        self.max_u = 10
        self.TRACE_NUM = DEFAULT_TRACE_NUM # debug sample
        self.trace_offset = 0
        variance = 5
        self.mppi = MPPI(u_dim, state_dim, N_smp, T_pred, pred_dT, variance)

    def update(self, dt):
        self.itr = 0
        if self.itr % int(pred_dT / dt) == 0:
            s_init = np.array([self.x, self.dx, self.th, self.dth])
            self.u_pred = self.mppi.step(s_init, self.max_u, update_carpole, cost_function)
            self.u = self.u_pred[0,0]

        self.x, self.dx, self.th, self.dth = update_carpole_scalar(self.x, self.dx, self.th, self.dth, self.u, dt)
        self.t = self.t + dt
        self.itr = self.itr + 1

    def draw(self):
        ret = []
        ret = ret + draw_circle_cmd((self.x, 0), r)
        ret = ret + draw_lineseg_cmd((self.x, 0), (self.x + l * np.cos(self.th + np.pi /2 * 3), l * np.sin(self.th+np.pi/2*3)))
        if self.mppi.w is not None:
            for i in range(self.trace_offset, self.trace_offset + self.TRACE_NUM):
                if i >= N_smp:
                    break
                color = jet(5*self.mppi.w[i])
                x_th = self.mppi.debug_trace[:,i,[0,2]]
                x = x_th[:,0]
                th = x_th[:,1]
                px = x + l * np.cos(th + np.pi /2 * 3)
                py = l * np.sin(th+np.pi/2*3)
                ps = [(px[i],py[i]) for i in range(T_pred)]
                ret = ret + plot_points_with_line_cmd(ps, color)

        return ret

    def draw_text(self):
        cx, cdx, cth, cdth, cu = cost_function_info(self.x, self.dx, self.th, self.dth, self.u)
        return [f"t = {self.t:.03f} u = {self.u:.01f}", f"cx = {cx:.01f}", f"cdx = {cdx:.01f}", f"cth = {cth:.01f}", f"cdth = {cdth:.01f}"]

def test():
    model = CarPole()

    dt = 0.01
    def event_handler(key, type, shifted):
        if key == 'l' and type == "DOWN":
            model.u = 2
        elif key == 'h' and type == "DOWN":
            model.u = -2
        elif key == 'd' and type == "DOWN":
            model.TRACE_NUM = DEFAULT_TRACE_NUM
        elif key == 'u' and type == "DOWN":
            model.TRACE_NUM = N_smp
        elif key == 'i' and type == "DOWN":
            model.TRACE_NUM = 10
            model.trace_offset = (model.trace_offset + model.TRACE_NUM) % N_smp
        elif type == "UP":
            model.u = 0

    run(model, event_handler, dt)

if __name__ == '__main__':
    test()
