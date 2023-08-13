from sympy import *
from sympy.parsing.sympy_parser import parse_expr
from draw_cmd import *
from pprint import pprint
import numpy as np
import os


#----------------
# simulation
#----------------
STATE_PAUSE = "pause"
STATE_ONE_STEP = "onestep"
STATE_TERMINATED = "teminated"
STATE_RUN = "run"

class Simulator:
    def __init__(self):
        self.state = STATE_PAUSE
        self.t = 0

    def reset_state(self):
        self.t = 0

    def step(self, dt):
        if self.state in [STATE_PAUSE, STATE_TERMINATED]:
            return

        self.update(dt)

        if self.state == STATE_ONE_STEP:
            self.state = STATE_PAUSE

    def terminate(self):
        self.state = STATE_TERMINATED

    def pause(self):
        self.state = STATE_PAUSE

    def onestep(self):
        self.state = STATE_ONE_STEP

    def toggle_pause(self):
        if self.state == STATE_PAUSE:
            self.state = STATE_RUN
        elif self.state == STATE_RUN:
            self.state = STATE_PAUSE

    def update(self, dt):
        self.t = self.t + dt

    def draw(self):
        return []

    def draw_text(self):
        return [f"t = {self.t:.03f} {self.state}"]

    def is_terminate(self):
        return False

class LinkTreeSimulator(Simulator):
    def __init__(self, linktree, context={}, name=None):
        self.name = name
        super().__init__()
        self.NB = linktree.NB
        self.reset_state()
        self.calc_ddq = self._gen_ddqf(linktree, context)
        self.draw_model_cmds = self._gen_draw_model_cmds(linktree, context)

    def _allsyms(self, linktree):
        return linktree.q() +  linktree.dq() + self.sim_input_syms()

    def _H_rhs(self, linktree, simp=False):
        H = linktree.H
        C = linktree.calc_counter_joint_force()
        tau = Matrix([jl.active_joint_force() for jl in linktree.jointlinks])
        rhs = tau - C
        for i in range(self.NB):
            if linktree.jointlinks[i].locked:
                rhs[i] = 0
                for j in range(self.NB):
                    H[i,j] = 0
                H[i,i] = 1
        if simp:
            H = simplify(H)
            rhs = simplify(rhs)
        return H, rhs

    def _gen_ddqf(self, linktree, context):
        allsyms = self._allsyms(linktree)
        H, rhs = self._H_rhs(linktree)
        H_f = lambdify(allsyms, H.subs(context))
        rhs_f = lambdify(allsyms, rhs.subs(context))
        def ddq_f(qv, dqv, uv):
            b = rhs_f(*qv, *dqv, *uv).reshape(-1).astype(np.float64)
            A = H_f(*qv, *dqv, *uv)
            return np.linalg.solve(A, b)
        return ddq_f

    def _gen_draw_model_cmds(self, linktree, context):
        allsyms = self._allsyms(linktree)
        sym_cmds = sum([jl.draw_cmds() for jl in linktree.jointlinks], [])
        sym_cmds = sym_cmds + plot_points_cmd([linktree.cog], 0.01, color="blue", name="cog")
        return [gen_eval_draw_cmd_f(cmd, allsyms, context) for cmd in sym_cmds]

    def sim_input_syms(self):
        return []

    def sim_input_vals(self):
        return []

    def reset_state(self):
        super().reset_state()
        self.q_v = np.zeros(self.NB)
        self.dq_v = np.zeros(self.NB)

    def update_env(self):
        pass

    def update(self, dt):
        self.update_env()
        ddq_v = self.calc_ddq(self.q_v, self.dq_v, self.sim_input_vals())
        self.dq_v = self.dq_v + ddq_v * dt
        self.q_v = self.q_v + self.dq_v * dt
        super().update(dt)

    def draw(self):
        ret = []
        for dc in self.draw_model_cmds:
            ret.append(dc(self.q_v, self.dq_v, self.sim_input_vals()))
        return ret


def run(simulater, eh=None, dt=0.001, Hz=None, scale=200):
    import graphic
    viewer = graphic.Viewer(scale, offset=[0, 0.2])

    if Hz is None:
        Hz = 1./dt

    def event_handler(key, type, shifted):
        if eh is not None:
            eh(key, type, shifted)
        if key == 'q':
            sys.exit()
        elif key == 's' and type == "DOWN":
            simulater.toggle_pause()
        elif key == 'a' and type == "DOWN":
            simulater.onestep()
        elif key == 'r' and type == "DOWN":
            simulater.pause()
            simulater.reset_state()

    while True:
        cmds = simulater.draw()
        text = simulater.draw_text()
        viewer.handle_event(event_handler)
        viewer.clear()
        viewer.text(text)
        viewer.draw(cmds)
        viewer.draw_horizon(0)
        viewer.flush(Hz)

        simulater.step(dt)


