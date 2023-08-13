from linktree import *
from jointlink import *
from simulation import *
from draw_cmd import *
from sympy import pi, symbols

u0, u1, u2 = symbols('u0 u1 u2') # motor torq
fx, fy = symbols('fx fy') # friction force and normal force from ground
cf0, cf1, cf2 = symbols('cf0 cf1 cf2') # constraint force aligned axis
cpx, cpy = symbols('cpx cpy') # contact point
g = symbols('g')
l_stick, l_shin, l_thigh, l_ubody = symbols('lstick lshin lthigh lubody')
m_foot, m_stick, m_shin, m_thigh, m_ubody = symbols('mfoot mstick mshin mthigh mubody')

lstick= 0.83
lshin= 0.4
lthigh= 0.5
lubody= 0.5
mfoot= 5
mstick= 7
mshin= 10
mthigh= 30
mubody= 40
context = { 
        l_stick: lstick, l_shin: lshin, l_thigh: lthigh, l_ubody: lubody,
        m_foot: mfoot, m_stick: mstick, m_shin: mshin, m_thigh: mthigh,  m_ubody: mubody,
        g: 9.81
        }
z0 = 0.55


def contact_distance(x,y,qa):
    if y <= 0:
        return 0
    ca = np.cos(qa)
    if abs(ca) < 0.05:
        return np.inf if ca > 0 else -np.inf
    return y / ca


def contact_force(x,y,qa,q0,q1,q2,dx,dy,dqa,dq0,dq1,dq2,dt):
    K = 10000
    Dt = 1./dt
    mu = 1
    #mu = np.inf

    d = contact_distance(x,y,qa)
    if d <= z0 and d >= 0:
        ca = np.cos(qa)
        sa = np.sin(qa)
        pc = (x + d * sa, 0)
        vx = dx + y/(ca**2)*dqa + dy * sa/ca
        f_spring_x = K * (z0 - d) * sa
        f_spring_y = max(0, K * (z0 - d) * ca)
        f_fric_x   = np.clip(-Dt * vx, -mu * f_spring_y, mu * f_spring_y)
        fx = f_spring_x + f_fric_x
        fy = f_spring_y
        cpx = pc[0]
        cpy = pc[1]
    else:
        fx = 0
        fy = 0
        cpx = 0
        cpy = 0

    return fx,fy,cpx,cpy


def constraint_force(q0,q1,q2):
    cf0 = 0
    cf1 = 0
    cf2 = 0
    return cf0, cf1, cf2


def solve33(H00,H01,H02,H10,H11,H12,H20,H21,H22,rhs0,rhs1,rhs2):
    adjH00 = H11*H22-H12*H21
    adjH01 = H02*H21-H01*H22
    adjH02 = H01*H12-H02*H11
    adjH10 = H12*H20-H10*H22
    adjH11 = H00*H22-H02*H20
    adjH12 = H02*H10-H00*H12
    adjH20 = H10*H21-H11*H20
    adjH21 = H01*H20-H00*H21
    adjH22 = H00*H11-H01*H10
    detH   = H00*adjH00+H01*adjH10+H02*adjH20
    sol0 = (adjH00 * rhs0 + adjH01 * rhs1 + adjH02 * rhs2)/detH
    sol1 = (adjH10 * rhs0 + adjH11 * rhs1 + adjH12 * rhs2)/detH
    sol2 = (adjH20 * rhs0 + adjH21 * rhs1 + adjH22 * rhs2)/detH
    return sol0, sol1, sol2

class Rabbit(LinkTreeSimulator):
    def __init__(self):
        vjl_x       = StickJointLink("x", 0, 0, PrismaticJoint(), XT=Xpln(pi/2, 0, 0))
        vjl_y       = StickJointLink("y", 0, 0, PrismaticJoint(), XT=Xpln(0, 0, 0))
        jl_stick    = StickJointLink("qa", m_stick, l_stick, RevoluteJoint(), XT=Xpln(0, 0, 0), Icog=0) # jl_shin is connected to joint of jl_stick
        jl_shin     = StickJointLink("q0", m_shin, l_shin, RevoluteJoint(), Icog=0, tau=u0+cf0)
        jl_thigh    = StickJointLink("q1", m_thigh, l_thigh, RevoluteJoint(), Icog=0, tau=u1+cf1)
        jl_ubody    = StickJointLink("q2", m_ubody, l_ubody, RevoluteJoint(), Icog=0, tau=u2+cf2)
        linktree = LinkTreeModel([vjl_x, vjl_y, jl_stick, jl_shin, jl_thigh, jl_ubody], g)
        jl_shin.locked     = True
        jl_thigh.locked    = True
        jl_ubody.locked    = True

        jl_stick.apply_force(fx, fy, cpx, cpy)

        self.ref_q0 = np.pi/6
        self.ref_q1 = -np.pi/3
        self.ref_q2 = np.pi/6

        super().__init__(linktree, context, 'rabbit')

    def reset_state(self):
        super().reset_state()
        self.u0 = 0
        self.u1 = 0
        self.u2 = 0
        self.cf0 = 0
        self.cf1 = 0
        self.cf2 = 0
        self.cpx = 0
        self.cpy = 0
        self.fx = 0
        self.fy = 0

        self.q_v[1] = 1
        self.q_v[3] = np.pi/6
        self.q_v[4] = -np.pi/3
        self.q_v[5] = np.pi/6
        self.message = [""]

    def sim_input_syms(self):
        return [u0, u1, u2, cf0, cf1, cf2, fx, fy, cpx, cpy]

    def sim_input_vals(self):
        return np.array([self.u0, self.u1, self.u2, self.cf0, self.cf1, self.cf2, self.fx, self.fy, self.cpx, self.cpy])

    def update(self, dt):
        x,y,qa,q0,q1,q2 = self.q_v[0],self.q_v[1],self.q_v[2],self.q_v[3],self.q_v[4],self.q_v[5]
        dx,dy,dqa,dq0,dq1,dq2 = self.dq_v[0],self.dq_v[1],self.dq_v[2],self.dq_v[3],self.dq_v[4],self.dq_v[5]
        self.fx,self.fy,self.cpx,self.cpy = contact_force(x,y,qa,q0,q1,q2,dx,dy,dqa,dq0,dq1,dq2,dt)
        #self.u0 = 10000 * (self.ref_q0 - q0) - 100 * dq0
        #self.u1 = 10000 * (self.ref_q1 - q1) - 100 * dq1
        #self.u2 = 10000 * (self.ref_q2 - q2) - 100 * dq2

        self.cf0, self.cf1, self.cf2 = constraint_force(q0,q1,q2)
        super().update(dt)

    def draw_text(self):
        return super().draw_text() + self.message

    def draw(self):
        x,y,qa,q0,q1,q2 = self.q_v[0],self.q_v[1],self.q_v[2],self.q_v[3],self.q_v[4],self.q_v[5]

        pbase = np.array([x, y])
        pknee = pbase + lshin * np.array([-sin(q0 + qa), cos(q0 + qa)])
        pwest = pknee + lthigh* np.array([-sin(q0 + q1 + qa), cos(q0 + q1 + qa)])
        psholder = pwest + lubody * np.array([-sin(q0 + q1 + q2 + qa), cos(q0 + q1 + q2 + qa)])
        phandle = pbase + lstick * np.array([-sin(qa), cos(qa)])
        if pbase[1] < 0 or psholder[1] < 0 or phandle[1] < 0:
            self.terminate()

        ret = plot_points_cmd([phandle, pbase, pknee, pwest, psholder], 0.05)
        ret = ret + plot_points_with_line_cmd([phandle, pbase, pknee, pwest, psholder])

        # sprint is shrinked
        if y <= 0:
            self.message = ["contact"]
            return ret

        d = contact_distance(x,y,qa)
        if d < 0:
            d = z0
            self.message = ["air"]
        elif d > z0:
            d = z0
            self.message = ["air"]
        else:
            self.message = ["contact"]
        pbottom = (x + d * np.sin(qa), y - d * np.cos(qa))

        ret = ret + draw_circle_cmd(pbottom, 0.05)
        ret = ret + draw_lineseg_cmd(pbottom, pbase)
        return ret

    def _gen_ddqf(self, linktree, context):
        if self.name is not None and os.path.isfile(f"{self.name}_H.txt") and  os.path.isfile(f"{self.name}_rhs.txt"):
            with open(f"{self.name}_H.txt") as f:
                H = parse_expr(f.read())
            with open(f"{self.name}_rhs.txt") as f:
                rhs = parse_expr(f.read())
        else:
            H, rhs = self._H_rhs(linktree, simp=True)
            Hrhs = zeros(self.NB, self.NB+1)
            Hrhs[:,:self.NB] = H
            Hrhs[:,self.NB] = rhs
            Hrhs[0,:] = Hrhs[0,:]
            Hrhs[1,:] = Hrhs[1,:]
            for i in range(2,self.NB):
                Hrhs[i,:] = Hrhs[0,0] * Hrhs[i,:] - Hrhs[i,0]*Hrhs[0,:] - Hrhs[i,1]*Hrhs[1,:]
            Hrhs[self.NB-1,self.NB-1] = simplify(Hrhs[self.NB-1,self.NB-1])
            for i in range(self.NB-1):
                Hrhs[i,:] = Hrhs[self.NB-1,self.NB-1] * Hrhs[i,:] - Hrhs[i,self.NB-1] * Hrhs[self.NB-1,:]

            Hrhs[0,:] = Hrhs[0,:]/Hrhs[0,0]
            Hrhs[1,:] = Hrhs[1,:]/Hrhs[1,1]
            Hrhs[self.NB-1,:] = Hrhs[self.NB-1,:]/Hrhs[self.NB-1,self.NB-1]

            H = Hrhs[:,:self.NB]
            rhs = Hrhs[:,self.NB]

            with open(f"{self.name}_H.txt", "w") as f:
                f.write(str(H))
            with open(f"{self.name}_rhs.txt", "w") as f:
                f.write(str(rhs))

        allsyms = self._allsyms(linktree)
        H_f = lambdify(allsyms, H.subs(context))
        rhs_f = lambdify(allsyms, rhs.subs(context))

        def ddq_f(qv, dqv, uv):
            rhsv = rhs_f(*qv, *dqv, *uv).reshape(-1).astype(np.float64)
            Hv = H_f(*qv, *dqv, *uv)
            subHv=Hv[2:-1,2:-1]
            subrhsv=rhsv[2:-1]
            vqa, vq0, vq1 = solve33(subHv[0,0],subHv[0,1],subHv[0,2],subHv[1,0],subHv[1,1],subHv[1,2],subHv[2,0],subHv[2,1],subHv[2,2],subrhsv[0],subrhsv[1],subrhsv[2])
            sol = np.array([vqa, vq0, vq1])
            vq2 = rhsv [-1] - Hv[-1,2:-1] @ sol
            vx  = rhsv [0] - Hv[0,2:-1] @ sol
            vy  = rhsv [1] - Hv[1,2:-1] @ sol
            return np.array([vx, vy, vqa, vq0, vq1, vq2])

        return ddq_f

def test():
    model = Rabbit()
    dt = 0.001
    def event_handler(key, type, shifted):
        if key == 'l' and type == "DOWN":
            model.u = 0.8
        elif key == 'h' and type == "DOWN":
            model.u = -0.8
        elif type == "UP":
            model.u = 0

    run(model, event_handler, dt=dt, scale=50)

if __name__ == '__main__':
    test()
