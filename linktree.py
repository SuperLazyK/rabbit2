from sympy import *
from jointlink import *
import scipy
import numpy as np
#from sympy import init_printing
#init_printing() 


# NOTE: the original index 1,2,... and 0 means base body(basically not moved)
# NOTE: body velocity in the body coordinate does not mean 0 (5.14)
# symbolic calculation should be executed only once
# NOTE
#  Ic[i] is i-th body corrdinate inertia matrix of accumlated bodies after i
#  but it does NOT depend on its joint angle.
# X_r_to[i] transfer matrix to i-th body local coordinate from global(root) coordinate
# NOTE:
# i-th body local coordinate is transfered with its joint to the parent!!
# => X_r_to[0] may not be same as X0
class LinkTreeModel:

    def __init__(self, jointlinks, g, X0=eye(3), parent_idx=None):
        self.jointlinks = jointlinks
        self.NB = len(jointlinks)
        self.dim = 3 # num of dimension of spatial/planar vector
        self._X_parent_to = [zeros(self.dim, self.dim) for i in range(self.NB)]
        self.g = g
        self.accb = zeros(self.dim, 1)
        self.velb = zeros(self.dim, 1)
        self.X0 = X0

        if parent_idx is None:
            # list
            self.parent_idx = list(range(-1, self.NB-1))
        else:
            for i in parent_idx:
                assert -1 <= i and i < self.NB
            self.parent_idx = parent_idx

        self.Ic = [zeros(self.dim, self.dim) for i in range(self.NB)]

        # symbolic calculation should be executed only once
        self._update_vel_X()
        self._composite_inertia()

        self.cog = self._calc_cog()
        self.joint_pos = Matrix([self._pos(i).T for i in range(self.NB)])
        self.joint_vel = Matrix([self._vel(i).T for i in range(self.NB)])


    def _update_vel_X(self):
        for i in range(self.NB):
            I  = self.jointlinks[i].I
            XJ = self.jointlinks[i].XJ()
            j = self._parent(i)
            if j != -1: # parent is not root
                XT = self.jointlinks[j].XT
            else:
                XT = self.X0
            X_j_to_i = XJ * XT
            self._X_parent_to[i] = X_j_to_i

            vJ = self.jointlinks[i].vJ()
            if j != -1: # parent is not root
                self.jointlinks[i].X_r_to = X_j_to_i * self.jointlinks[j].X_r_to
                self.jointlinks[i].vel = X_j_to_i * self.jointlinks[j].vel + vJ
            else:
                self.jointlinks[i].X_r_to = X_j_to_i
                self.jointlinks[i].vel = X_j_to_i * self.velb + vJ


    def _parent(self, i):
        return self.parent_idx[i]

    # recursive newton-euler on each body coordinate
    # NOTE: fext : ext-force to i-th body in root coordinate
    def _inverse_dynamics(self, ddq, fext, impulse=False):
        q = self.q()
        dq = self.dq()
        NB = self.NB
        dim = self.dim

        assert NB == len(ddq)
        assert NB == len(fext)
        assert (dim, 1) == fext[0].shape

        acc = [zeros(dim, 1) for i in range(NB)]
        tau = [0 for i in range(NB)]
        f = [zeros(dim, 1) for i in range(NB)]

        S = [self.jointlinks[i].S() for i in range(NB)]

        for i in range(NB):
            I  = self.jointlinks[i].I
            vJ = self.jointlinks[i].vJ()
            cJ = self.jointlinks[i].cJ()
            j = self._parent(i)
            X_j_to_i = self._X_parent_to[i]

            if j != -1: # parent is not root
                accj = acc[j]
            else:
                accj = self.accb

            vel = self.jointlinks[i].vel
            acc[i] = X_j_to_i * accj  + S[i] * ddq[i] + cJ + crm(vel) * vJ

            f_g = self.jointlinks[i].gravity_force(self.g)
            # Euler's body coordinate dynamics equation
            # Inertia and angular velocity always changes in reference cordinate!!
            # NOTE w, I, M(tora) are descirbed in body coordinate
            # M = I dw/dt + w x (Iw)
            # f[i] is interaction force to i "from its parent j"
            # passive joint force is eliminated
            if impulse:
                f[i] = - dualm(self.jointlinks[i].X_r_to) * fext[i]
            else:
                f[i] = I * acc[i] + crf(vel) * I * vel - dualm(self.jointlinks[i].X_r_to) * fext[i] - f_g

        for i in range(NB-1, -1, -1):
            # parent force projeced to S(non-constraint-dimension)
            if impulse:
                tau[i] = S[i].dot(f[i])
            else:
                tau[i] = S[i].dot(f[i]) - self.jointlinks[i].passive_joint_force()
            j = self._parent(i)
            if j != -1: # parent is root
                X_j_to_i = self._X_parent_to[i]
                f[j] = f[j] + X_j_to_i.T * f[i]

        return Matrix(tau)


    def _composite_inertia(self):
        NB = self.NB
        dim = self.dim
        H = zeros(NB, NB)
        S = [self.jointlinks[i].S() for i in range(NB)]

        Ic = [zeros(dim, dim) for i in range(NB)]
        for i in range(NB):
            Ic[i] = self.jointlinks[i].I

        for i in range(NB-1, -1, -1):
            j = self._parent(i)
            if j != -1: # parent is root
                X_j_to_i = self._X_parent_to[i]
                Ic[j] = Ic[j] + X_j_to_i.T * Ic[i] * X_j_to_i
            F = Ic[i] * S[i]
            H[i,i] = S[i].T * F

            j = i
            while self._parent(j) != -1:
                F = self._X_parent_to[j].T * F
                j = self._parent(j)
                H[i,j] = F.T * S[j]
                H[j,i] = H[i,j]

        self.Ic = Ic
        # local to root
        #for i in range(NB):
        #    self.Ic[i] = self.jointlinks[i].X_r_to.T * Ic[i] * self.jointlinks[i].X_r_to

        self.H = H

    def _calc_cog(self, ith=0): # global coordinate
        _, cx, cy, _ = I2mc(transInertia(self.Ic[ith], self.jointlinks[ith].X_r_to))
        return Matrix([cx, cy])

    def _pos(self, ith=0):
        _, _, x, y = Xtoscxy(self.jointlinks[ith].X_r_to)
        return Matrix([x, y])

    def _vel(self, ith=0):
        p = self._pos(ith)
        return p.diff(symbols("t")).subs({q.diff():dq for q, dq in zip(self.q(), self.dq())})

    def q(self):
        return [jl.q for jl in self.jointlinks]

    def dq(self):
        return [jl.dq for jl in self.jointlinks]

    def ddq(self):
        return [jl.ddq for jl in self.jointlinks]

    def joints_coordinates(self, simp=True):
        for jl in self.jointlinks:
            print(jl.name, fromX(jl.X_r_to, simp))

    def joints_position(self, simp=True):
        for jl in self.jointlinks:
            print(jl.name, simplify(jl.joint_position()))

    # NOTE: fext is updated after tree construction
    # joint force to keep the attitude
    def calc_counter_joint_force(self):
        fext = [Matrix([jl.fa, jl.fx, jl.fy]) for jl in self.jointlinks]
        return self._inverse_dynamics([0 for i in range(self.NB)], fext)

    #def equation(self):
    #    tau = Matrix([jl.active_joint_force() for jl in self.jointlinks])
    #    return self.H * Matrix(self.ddq()) + self.C - tau

    #def kinetic_energy(self):
    #    # other approach: Sum of dq H dq/2
    #    # NOTE: vJ is relative. So Sum of vJ I vJ/2 is NOT kinetic energy
    #    return sum([jl.kinetic_energy() for jl in self.jointlinks])

    #def potential_energy(self):
    #    return sum([jl.potential_energy(self.g) for jl in self.jointlinks])

