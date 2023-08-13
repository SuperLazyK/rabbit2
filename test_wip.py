from linktree import *
from jointlink import *
from simulation import *
from draw_cmd import *
from mppi_control import pendulum_dynamics

r,l,Iw,Ib,mw,mb = symbols('r l Iw Ib mw mb')
uw = symbols('uw') # motor torq
g = symbols('g')

context = { l: 0.25, r: 0.05,
        mw: 0.01, mb: 1,
        Iw: 0, Ib: 0,
        g: 9.81
        }

class WIPG(LinkTreeSimulator):
    def __init__(self):
        jl1 = WheelJointLink("qw", mw, r, RackPinionJoint(r, 0), XT=Xpln(pi/2, 0, 0), Icog=Iw)
        jl2 = StickJointLink("qs", mb, l, RevoluteJoint(), I=transInertia(mcI(mb, [l, 0], Ib), Xpln(0, 0, 0)), tau=uw)
        linktree = LinkTreeModel([jl1, jl2], g, X0=Xpln(0, 0, 0))
        super().__init__(linktree, context)

    def reset_state(self):
        super().reset_state()
        self.u = 0

    def sim_input_syms(self):
        return [uw]

    def sim_input_vals(self):
        return np.array([self.u])

    def draw(self):
        return self.draw_model()


def test():
    #model = WIPG()
    model = WIP()

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
