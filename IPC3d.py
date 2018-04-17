import numpy as np
import pyquaternion as quater
from SDRE import IPCview
import math

class IPC3d():
    def __init__(self, skel, b, g, dt, q, qd=None):
        self.dt = dt
        self.skel = skel

        # self.theta = vectorn()
        self.theta = np.zeros(6)
        self.theta[0] = self.skel.q['j_cart_x']
        self.theta[1] = self.skel.q['j_cart_z']

        # self.loader:getPoseDOF(self.theta)
        #convert euler angle to quaternion
        rot_v = np.array([self.skel.q['j_pole_x'], self.skel.q['j_pole_y'], self.skel.q['j_pole_z']])
        rot_v_norm = np.linalg.norm(rot_v)

        if rot_v_norm == 0:
            rot_q = quater.Quaternion([0., 0., 0., 0.])
        else:
            rot_v_normalized = rot_v / rot_v_norm
            # print("rot_v", rot_v)
            # print("norm", rot_v_norm)
            # print("rot_v_normalized: ", rot_v_normalized)
            rot_q = quater.Quaternion(axis = rot_v_normalized, angle = rot_v_norm)

        # rot_q = quater.Quaternion([1.0, 0.1, 0.1, 0.1])
        self.theta[2] = rot_q[0]
        self.theta[3] = rot_q[1]
        self.theta[4] = rot_q[2]
        self.theta[5] = rot_q[3]

        # print("rot_q", rot_q)
        # print("theta: ", self.theta)

        self.dtheta = np.zeros(6)
        # print(self.dtheta)

        cart = self.skel.body('h_cart')
        pole = self.skel.body('h_pole')
        self.cart = cart
        self.pole = pole
        M = cart.m
        m = pole.m
        l = pole.local_com()[1]     #localCOM.y
        i = pole.inertia()[0, 0]       #inertia x

        # print("pole.inertia()", pole.inertia())
        self.friction = b

        print(M, m, l, i)

        self.pendX = IPCview(M, m, b, i, g, l, dt, q, qd)
        self.pendZ = IPCview(M, m, b, i, g, l, dt, q, qd)
        self.q = q
        self.dt = dt
        self.initState()

    def initState(self):
        self.mode = 1                                      #velocity control
        self.ori = quater.Quaternion(1, 0, 0, 0)         #cart orientation
        self.desiredVel = np.array([0, 0, 0])
        self.desiredVelGlobal = np.array([0, 0, 0])
        self.desiredPos = np.array([0, 0, 0])
        self.useDesiredPos = False
        self.desiredHip = np.array([0, 0, 0])
        self.desiredHipGlobal = np.array([0, 0, 0])
        self.controlForce = np.array([0, 0, 0])
        self.dv = np.array([0, 0, 0])
        self.dw = np.array([0, 0, 0])
        self.Y = np.array([0])
        # self.Y = np.array([[0,1],[2,3]])
        self.numFrame = 0

    def setDesiredParam(self, velOrPos):
        self.desiredVelGlobal = velOrPos
        self.desiredVel = velOrPos
        # self.desiredVel:rotate(self.ori.inverse())
        self.desiredVel = self.ori.inverse.rotate(self.desiredVel)

    def setDesiredVelocity(self, vel):
        if self.mode == 0:
            self.mode = 1
            q = self.q
            pxq = 1
            self.pendX.setQ(1,pxq*q,(1.1-pxq)*q)
            self.pendZ.setQ(1,q,0.1*q)

        self.setDesiredParam(vel)

    def calcCOMpos(self):
        i =2
        q = quater.Quaternion([self.theta[i+0], self.theta[i+1], self.theta[i+2], self.theta[i+3]])
        v0 = np.array([self.theta[0], 0, self.theta[1]])
        # v0 = np.array([self.theta[1], 0, self.theta[0]])

        # v1 = q.rotate(self.pole.local_com())
        if q.__nonzero__():
            inv_q = q.inverse
        else:
            inv_q = q

        v_temp = q * quater.Quaternion([0, self.pole.local_com()[0], self.pole.local_com()[1], self.pole.local_com()[2]]) * inv_q

        v1 = np.zeros(3)
        v1[0] = v_temp[1]
        v1[1] = v_temp[2]
        v1[2] = v_temp[3]
        # print("pole local com", self.pole.local_com())
        return v0+v1

    def oneStep(self):
        self.__step(1)
        self.Y.resize((max(self.Y.shape[0], self.numFrame + 1), 6))
        self.numFrame = self.numFrame + 1
        lastRow = self.numFrame - 1
        self.Y[lastRow, :] = self.theta

    def calcCartPos(self):
        pos = np.zeros(3)
        pos[0] = self.theta[0]
        pos[1] = 0
        pos[2] = self.theta[1]

        return pos

    def __step(self, ntimes):
        xx = self.pendX.x
        zx = self.pendZ.x

        # self.pendX: setParam(self.desiredVel.x)
        # self.pendZ: setParam(self.desiredVel.z)
        self.pendX.setParam(self.desiredVel[0])
        self.pendZ.setParam(self.desiredVel[2])

        pos = self.calcCartPos()
        # print("pos: ",pos)
        d = np.array([self.dtheta[0], self.dtheta[1], self.dtheta[2]])

        #q = self.theta:toQuater(2)
        i = 2
        q_temp = quater.Quaternion(self.theta[i + 0], self.theta[i + 1], self.theta[i + 2], self.theta[i + 3])
        # print("__step() theta?", q)
        #w = self.dtheta:toVector3(3)
        w = np.array([self.dtheta[3], self.dtheta[4], self.dtheta[5]])

        q = projectQuaternion(q_temp)

        # theta = q.rotationVector()
        # out.log( in);
        theta = q.axis * q.angle
        # print("q axis", q.axis, type(q.axis))
        # print("q yaw pitch roll", q.yaw_pitch_roll, type(q.yaw_pitch_roll))

        # theta[0] = q.yaw_pitch_roll[0]
        # theta[1] = q.yaw_pitch_roll[1]
        # theta[2] = q.yaw_pitch_roll[2]

        # print("theta", theta)
        toLocal = self.ori.inverse
        # print("toLocal: ",  toLocal)

        pos = toLocal.rotate(pos)
        ldesiredPos = self.desiredPos
        # print("ldesiredPos: ", ldesiredPos)
        ldesiredPos = toLocal.rotate(ldesiredPos)
        w = toLocal.rotate(w)
        theta = toLocal.rotate(theta)
        # print("after rotate wrt toLocal", theta)
        d = toLocal.rotate(d)

        if True :
            xx[0, 0] = pos[0]
            xx[1, 0] = theta[2]
            xx[2, 0] = d[0]
            xx[3, 0] = w[0]

            zx[0, 0] = pos[2]
            zx[1, 0] = -theta[0]
            zx[2, 0] = d[2]
            zx[3, 0] = -w[2]

        prevDt = self.dt
        self.pendX.dt = prevDt * ntimes
        self.pendZ.dt = prevDt * ntimes

        self.pendX.x = xx
        self.pendZ.x = zx

        self.pendX.oneStep()
        self.pendZ.oneStep()

        self.pendX.dt = prevDt
        self.pendZ.dt = prevDt

        xx = self.pendX.x
        zx = self.pendZ.x

        if True :
            #convert back to 3d states
            pos[0] = xx[0, 0]
            theta[2] = xx[1, 0]
            d[0] = xx[2, 0]
            w[0] = xx[3, 0]

            pos[2] = zx[0, 0]
            theta[0] = -zx[1, 0]
            d[2] = zx[2, 0]
            w[2] = -zx[3, 0]

        self.controlForce[0] = self.pendX.IPC.cf[0, 0]
        self.controlForce[2] = self.pendZ.IPC.cf[0, 0]

        self.dv[0] = self.pendX.ddtheta[0, 0]
        self.dv[2] = self.pendZ.ddtheta[0, 0]

        self.dw[2] = self.pendX.ddtheta[1, 0]
        self.dw[0] = -self.pendZ.ddtheta[1, 0]

        toGlobal = self.ori

        # print("333theta: ", theta)
        pos = toGlobal.rotate(pos)
        w = toGlobal.rotate(w)
        theta = toGlobal.rotate(theta)
        # print("after toGlobal theta: ", theta)
        d = toGlobal.rotate(d)
        self.controlForce = toGlobal.rotate(self.controlForce)

        # if false then -- debug draw
        # self.pendX.y: assign(xx:column(0): range(0, 3))
        # self.pendX: draw()
        # -- self.pendZ.y: assign(zx:column(0): range(0, 3))
        # -- self.pendZ: draw()


        #q: setRotation(theta)
        theta_norm = np.linalg.norm(theta)

        if theta_norm == 0:
            print("zero")
            q = quater.Quaternion([1., 0., 0., 0.])
        else:
            theta_normalized = theta / theta_norm
            # print("rot_v", rot_v)
            # print("norm", rot_v_norm)
            # print("rot_v_normalized: ", rot_v_normalized)
            q = quater.Quaternion(axis=theta_normalized, angle=theta_norm)

        self.theta[0] = pos[2]
        self.theta[1] = pos[0]

        #self.theta: setQuater(2, q)
        self.theta[2] = q[0]
        self.theta[3] = q[1]
        self.theta[4] = q[2]
        self.theta[5] = q[3]

        #self.dtheta: setVec3(0, d)
        self.dtheta[0] = d[0]
        self.dtheta[1] = d[1]
        self.dtheta[2] = d[2]
        #self.dtheta: setVec3(3, w)
        self.dtheta[3] = w[0]
        self.dtheta[4] = w[1]
        self.dtheta[5] = w[2]

        # print("xx", self.pendX.x)
        # print("theta", self.theta)
        # print("dtheta", self.dtheta)

    def setOrientation(self, a):
        self.ori = quater.Quaternion(axis=[0, 1, 0], angle = a)
        self.setDesiredVelocity(self.desiredVelGlobal)

def projectQuaternion(q):
    rkAxis = np.array([0, 1, 0])

    #kRotatedAxis.rotate((*this),rkAxis);

    kRotatedAxis = np.zeros(3)

    if q.__nonzero__() == True:
        inv_a = q.inverse
    else:
        inv_a = q

    c = q * quater.Quaternion([0, rkAxis[0], rkAxis[1], rkAxis[2]]) * inv_a

    kRotatedAxis[0] = c[1]
    kRotatedAxis[1] = c[2]
    kRotatedAxis[2] = c[3]

    #rkNoTwist.axisToAxis(rkAxis, kRotatedAxis);

    if np.linalg.norm(rkAxis) == 0:
        vA = rkAxis
    else:
        vA = rkAxis / np.linalg.norm(rkAxis)

    if np.linalg.norm(kRotatedAxis) == 0:
        vB = kRotatedAxis
    else:
        vB = kRotatedAxis / np.linalg.norm(kRotatedAxis)

    # print("q", q)
    # print("c", c)
    # print("vA", vA)
    # print("vB", vB)

    if vA[0]*vB[0] + vA[1]*vB[1]+ vA[2]*vB[2] < -0.999:
        print("Singular!!", vA[0]*vB[0] + vA[1]*vB[1]+ vA[2]*vB[2])
    # else:
        # print("nononononoonooonononno!!!", vA[0]*vB[0] + vA[1]*vB[1]+ vA[2]*vB[2])
    vHalf = vA + vB
    vHalf = vHalf / np.linalg.norm(vHalf)
    # print("vHalf", vHalf, type(vHalf))
    #unitAxisToUnitAxis2(vA, vHalf);
    vAxis = np.zeros(3)
    vAxis[0] = vA[1] * vHalf[2] - vA[2] * vHalf[1]
    vAxis[1] = vA[2] * vHalf[0] - vA[0] * vHalf[2]
    vAxis[2] = vA[0] * vHalf[1] - vA[1] * vHalf[0]

    # print("vAxis", vAxis, type(vAxis))
    # print("angle", vA[0] * vHalf[0] + vA[1] * vHalf[1] + vA[2] * vHalf[2])
    rkNoTwist = quater.Quaternion(vA[0] * vHalf[0] + vA[1] * vHalf[1] + vA[2] * vHalf[2],vAxis[0], vAxis[1],vAxis[2] )

    # print("rkNoTwist", rkNoTwist)

    #rkTwist.mult(rkNoTwist.inverse(),*this);
    q_a = rkNoTwist.inverse
    q_b = q

    rkTwist = quater.Quaternion([q_a[0]*q_b[0] - q_a[1]*q_b[1] - q_a[2]*q_b[2] - q_a[3]*q_b[3],q_a[0]*q_b[1] + q_a[1]*q_b[3] + q_a[2]*q_b[3] - q_a[3]*q_b[2],q_a[0]*q_b[2] + q_a[2]*q_b[3] + q_a[3]*q_b[1] - q_a[1]*q_b[3],q_a[0]*q_b[3] + q_a[3]*q_b[3] + q_a[1]*q_b[2] - q_a[2]*q_b[1]])
    # rkTwist[0] = q_a[0]*q_b[0] - q_a[1]*q_b[1] - q_a[2]*q_b[2] - q_a[3]*q_b[3]
    # rkTwist[1] = q_a[0]*q_b[1] + q_a[1]*q_b[3] + q_a[2]*q_b[3] - q_a[3]*q_b[2]
    # rkTwist[2] = q_a[0]*q_b[2] + q_a[2]*q_b[3] + q_a[3]*q_b[1] - q_a[1]*q_b[3]
    # rkTwist[3] = q_a[0]*q_b[3] + q_a[3]*q_b[3] + q_a[1]*q_b[2] - q_a[2]*q_b[1]

    return rkNoTwist

