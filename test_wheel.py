from linktree import *
from jointlink import *
from simulation import *
from draw_cmd import *
from sympy import pi, symbols

r,Iw,mw = symbols('r Iw mw')
uw = symbols('uw') # motor torq
g = symbols('g')
fx, fy = symbols('fx fy') # friction force and normal force from ground
context = { r: 0.1, mw: 1, Iw: 0, g: 9.81 }

class Wheel(LinkTreeSimulator):
    def __init__(self):
        # virtual joint with length == 0
        vjl_x    = StickJointLink("x", 0, 0, PrismaticJoint(), XT=Xpln(pi/2, 0, 0))
        vjl_y    = StickJointLink("y", 0, 0, PrismaticJoint(), XT=Xpln(-pi/2, 0, 0))
        jl_wheel = WheelJointLink("q", mw, r, RevoluteJoint(), XT=Xpln(pi/2, 0, 0))
        linktree = LinkTreeModel([vjl_x, vjl_y, jl_wheel], g)

        xy = jl_wheel.joint_position()
        jl_wheel.apply_force(fx, fy, xy[0], xy[1] - r)

        super().__init__(linktree, context)

    def reset_state(self):
        super().reset_state()
        self.q_v = np.array([0, 1, 0])
        self.dq_v = np.array([0.5, 0, 0])
        self.v_fx = 0
        self.v_fy = 0

    def sim_input_syms(self):
        return [fx, fy]

    def sim_input_vals(self):
        return np.array([self.v_fx, self.v_fy])

    def update_env(self):
        Kn = 10000
        Dn = 100
        Dt = 100
        mu = 1
        if self.q_v[1] <= context[r]:
            self.v_fy = max(0, -Kn * (self.q_v[1] - context[r]) - Dn * self.dq_v[1])
            self.v_fx = np.clip(-Dt * (self.dq_v[0] + context[r] * self.dq_v[2]), -mu * self.v_fy, mu * self.v_fy)
        else:
            self.v_fx = 0
            self.v_fy = 0


def test():
    model = Wheel()

    dt = 0.001
    def event_handler(key, type, shifted):
        if key == 'l' and type == "DOWN":
            model.u = 0.8
        elif key == 'h' and type == "DOWN":
            model.u = -0.8
        elif type == "UP":
            model.u = 0

    run(model, event_handler)

if __name__ == '__main__':
    test()
