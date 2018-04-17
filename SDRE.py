import numpy as np
import control as ctrl
import math

def LQR_wrap(A,B,Q,R):


    #LQR Test
    # J = 0.0475
    # m = 1.5
    # r = 0.25
    # g = 10
    # gamma = 0.51
    # d = 0.2
    # l = 0.05
    #
    # A = np.array([[0, 0, 0, 1, 0, 0],
    # [0,    0,    0,    0,    1,    0],
    # [0,    0,    0,    0,    0,    1],
    # [0,    0,     - gamma,     - d / m,    0,    0],
    # [0,    0,    0,    0,     - d / m,    0],
    # [0,    0,     - m * g * l / J,    0,    0,    0]])
    # B = np.array([[0., 0.], [0., 0.], [0., 0.], [1/m, 0.], [0., 1/m], [r/J, 0.]])
    # Q = np.eye(6)
    # R = 0.1 * np.eye(2)

    # print(A, B, Q, R)
    k, s, e = ctrl.lqr(A, B, Q, R)
    # print('k:', k)
    return k

class NonlinearController(object):
    def __init__(self, dim, cdim):      #dim: # of generalized coordinates
        self.D = np.zeros((dim, dim))
        self.C = np.zeros((dim, dim))
        self.G = np.zeros((dim,1))
        self.H = np.zeros((dim, cdim))
        self.dim = dim
        self.cdim = cdim
        self.dGdTheta_0 = np.zeros(dim)
        self.Gsd = np.zeros(dim)
        self.ctrlForceScale = 0
        #outputs of update functions
        #: dot(x) = Ax + Bu

        self.invD = np.zeros((dim, dim))
        self.A = np.zeros((dim * 2, dim * 2))
        self.A[0, 2] = 1
        self.A[1, 3] = 1
        self.B = np.zeros((dim * 2, cdim))


    def calcControlForce(self, x, U, Q, R, K, xd):
        dim = self.myCon.dim
        self.update_sdre(x[0:dim, 0], x[dim:dim*2, 0])
        useSDRE = False
        # useSDRE = True

        if useSDRE:
            K = LQR_wrap(self.myCon.A, self.myCon.B, Q, R)
        U = np.dot(K, xd)

        return U - np.dot(K, x)

    def oneStep(self, x, U, dt, maxForce, Q, R, K, xd):
        dim = self.myCon.dim
        self.cf = self.calcControlForce(x, U, Q, R, K, xd)
        cf = self.cf
        return self.oneStepRaw(x, cf, dt, maxForce, Q, R, K, xd)

    def oneStepRaw(self, x, cf, dt, maxForce, Q, R, K, xd):
        dim = self.myCon.dim
        ctrlForceScale= self.myCon.ctrlForceScale

        def clamp(mat, val):
            if val:
                print(" MaxForce Exist -> inside!!!!!")
                for i in range(0, mat.shape[0]):
                    for j in range (0, mat.shape[1]):
                        v = mat[i,j]
                        if v > val:
                            mat[i,j] = val
                        elif v < -2 * val:
                            mat[i,j] = -1 * val


            return mat

        useLSim = False

        if useLSim:
            Ad = M.fromMatrixn(self.A)
            Bd = M.fromMatrixn(self.B)
            M.c2d(Ad, Bd, dt)
            Ad = M.toMatrixn(Ad)
            Bd = M.toMatrixn(Bd)

            x = Ad * x + Bd * clamp(cf, maxForce)
        elif False:
            xdot = self.A * x
            if ctrlForceScale:
                xdot = np.add(xdot, self.B * clamp(cf, maxForce) * ctrlForceScale)
            else:
               xdot = np.add(xdot, self.B * clamp(cf, maxForce))

            x = np.add(x, xdot * dt)
        else:

            theta = x[0:dim, 0:1]
            dtheta = x[dim:dim*2, 0:1]

            # Dddtheta
            if ctrlForceScale:
                Dddtheta = self.myCon.H * clamp(cf, maxForce) * ctrlForceScale - self.myCon.G - np.dot(self.myCon.C, dtheta)
            else:
                Dddtheta = self.myCon.H * clamp(cf, maxForce) - self.myCon.G - np.dot(self.myCon.C, dtheta)

            ddtheta = np.dot(self.myCon.invD, Dddtheta)

            dtheta = dtheta + ddtheta*dt
            theta = theta + dtheta*dt

            x[0:dim, 0:1] = theta
            x[dim: dim * 2, 0: 1] = dtheta
            return ddtheta

    def _update_sdre(self, theta):
        #assumes that D, C, G is already calculated.

        #self.invD: pseudoInverse(self.D)
        self.invD = np.linalg.pinv(self.D)

        dim = self.dim
        cdim = self.cdim

        Gsd = self.Gsd
        G = self.G
        dGdTheta_0 = self.dGdTheta_0

        thr = 0.0001

        for i in range(0, Gsd.size):
            if theta[i] < thr:
                Gsd[i] = dGdTheta_0[i]
            else:
                Gsd[i] =G[i] / theta[i]

        self.Gsd = Gsd

        a3 = self.A[dim:dim * 2, 0:dim]

        #multAdiagB --> Samples/scripts/module.

        #a3: multAdiagB(self.invD, Gsd)
        for i in range(0, self.invD.shape[0]):
            a3[i] = self.invD[i,:]*Gsd[i]

        #a3:rmult(-1)
        a3 = -1 * a3

        self.A[dim:dim * 2, 0:dim] = a3

        # print("after a3: ", self.A)

        #a4 = self.A:range(dim, dim * 2, dim, dim * 2)
        #a4:mult(self.invD, self.C)
        # a4: rmult(-1)

        a4 = self.A[dim:dim * 2, dim:dim * 2]
        a4 = np.matmul(self.invD, self.C)
        a4 = -1* a4

        self.A[dim:dim * 2, dim:dim * 2] = a4

        # print("after a4: ", self.A)

        # b2 = self.B:range(dim, dim * 2, 0, cdim)
        b2 = self.B[dim : dim * 2, 0 : cdim]
        #b2: mult(self.invD, self.H)
        b2 = np.matmul(self.invD, self.H)
        self.B[dim: dim * 2, 0: cdim] = b2

        # print("after b2: ", self.B)


class IPC(NonlinearController):
    def __init__(self, M, m, b, I, g,l):
        self.M = M
        self.m = m
        self.b = b
        self.I = I
        self.g = g
        self.l = l

        myCon = NonlinearController(2, 1)
        self.myCon = myCon
        # self.dGdTheta_0:setValues(0, -m * g * l)

        self.myCon.dGdTheta_0[0] = 0
        self.myCon.dGdTheta_0[1] = -m * g * l
        #self.H:setValues(1, 0)
        self.myCon.H[0] = 1
        self.myCon.H[1] = 0

    def update_sdre(self, x, dx):
        M = self.M
        m = self.m
        b = self.b
        I = self.I
        g = self.g
        l = self.l

        myCon = self.myCon

        theta = x[1]
        dtheta = dx[1]
        # print("theta", theta, dtheta)

        # print("dd", M + m, -m * l * math.cos(theta), I + m * l * l)
        # self.D:setSymmetric(M + m, -m * l * math.cos(theta), I + m * l * l)

        list = [M + m, -m * l * math.cos(theta), I + m * l * l]

        # print(myCon.D.shape[0],  myCon.D.shape[1])
        c = 0
        for i in range(0, myCon.D.shape[0]):
            for j in range(i, myCon.D.shape[1]):
                myCon.D[i, j] = list[c]
                myCon.D[j, i] = list[c]
                c=c+1
        # self.C:setValues(b, m * l * dtheta * math.sin(theta), 0, 0)
        myCon.C[0, 0] = b
        myCon.C[0, 1] = m * l * dtheta * math.sin(theta)
        myCon.C[1, 0] = 0
        myCon.C[1, 1] = 0

        # print("myCon.C", myCon.C)
        # self.G:setValues(0, -m * g * l * math.sin(theta))
        myCon.G[0] = 0
        myCon.G[0] = -m * g * l * math.sin(theta)
        # print("myCon.G", myCon.G)
        myCon._update_sdre(x)

        self.myCon = myCon


class IPCview(object):
    def setQ(self, mode, q, qd):
        if self.mode != mode:
            self.Q.fill(0)
            if mode == 0:
                self.Q[0, 0] = q
                self.Q[2, 2] = qd
            else:           #velocity control

                self.Q[1, 1] = q
                self.Q[2, 2] = qd


            self.mode = mode
            self.K = LQR_wrap(self.IPC.myCon.A, self.IPC.myCon.B, self.Q, self.R)

    def __init__(self, M,m,b,i,g,l,dt,q, qd=None):

        self.maxForce = None
        self.IPC = IPC(M, m, b, i, g, l)
        self.IPC.update_sdre(np.zeros((2, 1)), np.zeros((2, 1)))

        self.mode = 0       # i made

        self.l = l
        self.Q = np.zeros((4, 4))
        self.R = np.eye(1)
        if qd is None:
            qd = q
        self.setQ(1, q, qd)     #velocity control

        # print(np.zeros((4,4)))

        # print("Info:", self.IPC.myCon.A, self.IPC.myCon.B, self.Q, self.R, self.K)

        #prepare euler integration

        self.t = 0
        self.dt = dt
        self.x = np.zeros((4, 1))

        #self.U = CT.mat(1, 1, 0)
        self.U = np.zeros(1)

        self.xd = np.zeros((4, 1))

    def setMaximumForce(self, force):
        self.maxForce = force

    def oneStep(self):
        # print("IPC ONE STEP")
        #self.setMaximumForce(np.array([0,0,0]))
        # self.x[2] = 3
        self.ddtheta = self.IPC.oneStep(self.x, self.U, self.dt, self.maxForce, self.Q, self.R, self.K, self.xd)
        # print("return: ", self.ddtheta)
        self.t = self.t + self.dt

    def setParam(self, speed):
        xd = self.xd

        if self.mode == 0:
            xd[0, 0] = speed
        else:
            xd[2, 0] = speed

        self.xd = xd
    #def draw(self):

