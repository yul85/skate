import pydart2 as pydart
import numpy as np
from IPC3d import IPC3d
import math
from scipy.spatial import distance
import sys
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import *
import pyquaternion as quater
import BinaryFIle
import struct


mTime = 0


class Window(QWidget):
    def __init__(self, parent=None):
        super(Window, self).__init__(parent)

        grid = QGridLayout()
        grid.addWidget(self.createExampleGroup(), 0, 0)
        # grid.addWidget(self.createExampleGroup(), 1, 0)
        # grid.addWidget(self.createExampleGroup(), 0, 1)
        # grid.addWidget(self.createExampleGroup(), 1, 1)
        self.setLayout(grid)
        self.desired_x_val = 3
        self.desired_z_val = 3
        self.desired_ori = 0
        self.lqr_strength = 0
        self.usePosCon = 0

        self.setWindowTitle("Control Window")
        self.resize(400,300)

    def createExampleGroup(self):
        groupBox = QGroupBox("Control param")

        label1 = QLabel("set desired speed X")
        slider1 = QSlider(Qt.Horizontal)
        # slider1.setFocusPolicy(Qt.StrongFocus)
        slider1.setTickPosition(QSlider.TicksBothSides)
        slider1.setMaximum(5)
        slider1.setMinimum(-5)
        slider1.setTickInterval(1)
        slider1.setSingleStep(1)
        label2 = QLabel("set desired speed Z")
        slider2 = QSlider(Qt.Horizontal)
        slider2.setFocusPolicy(Qt.StrongFocus)
        slider2.setTickPosition(QSlider.TicksBothSides)
        slider2.setMaximum(5)
        slider2.setMinimum(-5)
        slider2.setTickInterval(1)
        slider2.setSingleStep(1)
        label3 = QLabel("set orientation")
        slider3 = QSlider(Qt.Horizontal)
        slider3.setFocusPolicy(Qt.StrongFocus)
        slider3.setTickPosition(QSlider.TicksBothSides)
        slider3.setMaximum(180)
        slider3.setMinimum(0)
        slider3.setTickInterval(10)
        slider3.setSingleStep(1)
        label4 = QLabel("lqr strength")
        slider4 = QSlider(Qt.Horizontal)
        slider4.setFocusPolicy(Qt.StrongFocus)
        slider4.setTickPosition(QSlider.TicksBothSides)
        slider4.setMaximum(10)
        slider4.setMinimum(0)
        slider4.setTickInterval(1)
        slider4.setSingleStep(1)

        self.slider1 = slider1
        self.slider2 = slider2
        self.slider3 = slider3
        self.slider4 = slider4


        radio1 = QRadioButton("&Use position control")
        radio1.setChecked(False)
        self.radio1 = radio1
        radio1.toggled.connect(lambda:self.btnstate(self.radio1))

        vbox = QVBoxLayout()
        vbox.addWidget(label1)
        vbox.addWidget(slider1)
        self.slider1.valueChanged.connect(self.valueChanged)
        vbox.addWidget(label2)
        vbox.addWidget(slider2)
        self.slider2.valueChanged.connect(self.valueChanged)
        vbox.addWidget(label3)
        vbox.addWidget(slider3)
        self.slider3.valueChanged.connect(self.valueChanged)
        vbox.addWidget(label4)
        vbox.addWidget(slider4)
        self.slider4.valueChanged.connect(self.valueChanged)
        vbox.addWidget(radio1)
        vbox.addStretch(1)
        groupBox.setLayout(vbox)

        return groupBox

    def valueChanged(self):
        self.desired_x_val = self.slider1.value()
        self.desired_z_val = self.slider2.value()
        self.desired_ori = self.slider3.value()
        self.lqr_strength = self.slider4.value()/10.

        print("desired x", self.desired_x_val)
        print("desired z", self.desired_z_val)
        print("desired ori", self.desired_ori)
        print("desired lqr strength", self.lqr_strength)

    def btnstate(self, b):
        if b.isChecked() == True:
            self.usePosCon = 1
        else:
            self.usePosCon = 0

        print("usePosCon", self.usePosCon)


class MyWorld(pydart.World):
    def __init__(self, ):
        """
        """
        pydart.World.__init__(self, 1.0 / 2000.0, './data/skel/cart_pole.skel')
        self.force = None
        self.duration = 0
        # self.skeletons[0].body('ground').set_friction_coeff(0.02)

        myWin = Window()
        self.myWin = myWin
        myWin.show()
        print('control window OK')

        mPendulum1 = IPC3d(self.skeletons[1], 0, 9.8, 1.0 / 60, 500000)
        # mPendulum2 = IPC3d(self.skeletons[1], 0, 9.8, 1.0 / 60, 500000)

        # read binary file
        # s = BinaryFIle.BinaryFile('./data/work/optimizedTraj1_muaythai8.tbl', "rb")

        # s = open('./data/work/optimizedTraj1_muaythai8.tbl', "rb")
        # print("s", s)
        #
        # tid, n = struct.unpack('ii', s.read(8))
        # # print( "out", tid, n)
        # numRef = n
        # print("numRef: ", numRef)
        #
        # # _refToTable = []
        # _refToTable = []
        # # _refToTable = [[] for i in range(numRef)]
        # print("_refToTable 1: ", _refToTable)
        # for i in range(numRef-1, -1, -1):
        #     print("i", i)
        #     _refToTable.insert(i, {})
        #     print("_refToTable 2: ", _refToTable)
        #     tbl = _refToTable[i]
        #
        #     print("table!!!!", tbl)
        #     tid, n = struct.unpack('ii', s.read(8))
        #     print("middle: ", tid, n)
        #     while n == 1:
        #         ref = n
        #         if ref == 1:
        #             # print("check", n)
        #             k = ''.join(map(bytes.decode, struct.unpack("<%dc" % n, s.read(n))))
        #
        #             # print("this: ", "<%dc" % n, type("<%dc" % n))
        #             # print("s.read(n)", s.read(n), type(s.read(n)))
        #             # k = ''.join(struct.unpack("<%dc" % n, s.read(n)))
        #             # k = n
        #             iii = n
        #
        #             # print ("iii", iii)
        #             v = _refToTable[iii-1]
        #             # print("hh",struct.unpack("<%dc" % n, s.read(n)))
        #             print("if k: ", k, type(k))
        #             print("if v: ", v)
        #         else:
        #             tid, n = struct.unpack('ii', s.read(8))
        #
        #             # k = ''.join(struct.unpack("<%dc" % n, s.read(n)))
        #             # v = ''.join(struct.unpack("<%dc" % n, s.read(n)))
        #             k = n
        #             v = n
        #             # print("else k: ", k, type(k))
        #             # print("else v: ", v)
        #         # tbl.append(v)
        #         tbl[k] = v
        #     _refToTable[i] = tbl
        #     # print("=======================")
        #
        # s.close()
        #
        # optimizedTraj1 = _refToTable[1]
        #
        # print("optimizedTraj1: ", optimizedTraj1)

        motion_dof_row = 4000

        desiredTraj1 = np.zeros((motion_dof_row, 6))
        # desiredTraj2 = np.zeros((motion_dof_row, 6))

        # for i in range(0, motion_dof_row):
        for i in range(0, 2000):
            # print("i", i)
            # if i > 2000:
            #     x_step = i * 0.1 + 0.05
            # else:
            #     x_step = 2001*0.1 + 0.05
            x_step = i * 0.01 + 0.01
            desiredTraj1[i] = np.array([x_step, 0, 0, 0, 0, 0])
            # y_step = i * 0.01 + 3
            # desiredTraj2[i] = np.array([0, y_step, 0, 0, 0, 0])

        for i in range(2000, motion_dof_row):
            x_step = 2001*0.01 + 0.01
            desiredTraj1[i] = np.array([x_step, 0, 0, 0, 0, 0])

        # print(desiredTraj1[999])

        self.desiredTraj1 = desiredTraj1
        # self.desiredTraj2 = desiredTraj2

        # optimizedTraj1 = util.loadTable("optimizedTraj1_muaythai8.tbl")
        # optimizedTraj2 = util.loadTable("optimizedTraj2_muaythai8.tbl")
        # desiredTraj1 = optimizedTraj1.pendState
        # desiredTraj2 = optimizedTraj2.pendState
        # desiredVel1 = optimizedTraj1.pendDState
        # desiredVel2 = optimizedTraj2.pendDState
        # param1 = optimizedTraj1.controlParam
        # param2 = optimizedTraj2.controlParam

        mVel = np.array([0, 0, 0])
        self.mVel = mVel

        self.mPendulum1 = mPendulum1
        # self.mPendulum2 = mPendulum2

        self.motion_dof_row = motion_dof_row
        pendulumPos1 = np.zeros((motion_dof_row, 6))
        # pendulumPos2 = np.zeros((motion_dof_row, 6))
        pendulumVel1 = np.zeros((motion_dof_row, 6))
        # pendulumVel2 = np.zeros((motion_dof_row, 6))

        segNum = 0
        for fr in range(0, motion_dof_row):
            print("fr : " , fr)
            # pendState1, pendState2 = run(fr, segFinder:startFrame(segNum))
            # pendState1, pendState2 = self.run(fr, 0)
            pendState1 = self.run(fr, 0)

            pendulumPos1[fr, :] = pendState1[0, :]
            # pendulumPos2[fr, :] = pendState2[0, :]
            pendulumVel1[fr, :] = pendState1[1, :]
            # pendulumVel2[fr, :] = pendState2[1, :]


        self.pendulumPos1 = pendulumPos1
        # self.pendulumPos2 = pendulumPos2
        self.pendulumVel1 = pendulumVel1
        # self.pendulumVel2 = pendulumVel2

    def run(self, cFrame, startFrame):

        mPendulum1 = self.mPendulum1
        # mPendulum2 = self.mPendulum2

        desiredTraj1 = self.desiredTraj1
        # desiredTraj2 = self.desiredTraj2

        if cFrame == startFrame:
            initVel = np.zeros(6)

            mPendulum1.theta = desiredTraj1[cFrame, :]
            # mPendulum2.theta = desiredTraj2[cFrame, :]
            # mPendulum1.theta = initVel
            # mPendulum2.theta = initVel
            mPendulum1.dtheta = initVel
            # mPendulum2.dtheta = initVel

        param = np.zeros(4)
        param[0] = desiredTraj1[cFrame, 0]
        # print("param[0]", param[0])
        # param[0] = 0.5
        # param[0] = 50.0
        param[1] = 0
        param[2] = desiredTraj1[cFrame+1, 0] if cFrame < 3999 else desiredTraj1[cFrame, 0]
        param[3] = 10

        # param: set(0, param1:row(cFrame)(0))
        # param: set(1, param1:row(cFrame)(1))
        # param: set(2, param2:row(cFrame)(0))
        # param: set(3, param2:row(cFrame)(1))

        # dVel1, dVel2 = self.calcDesiredVel(param, cFrame)
        dVel1 = self.calcDesiredVel(param, cFrame)
        mPendulum1.setDesiredVelocity(dVel1)
        # # print("dVel1", dVel1)
        # print("dVel2", dVel2)

        # mPendulum2.setDesiredVelocity(dVel2);

        # self.mVel[0] = self.myWin.desired_x_val
        # self.mVel[2] = self.myWin.desired_z_val

        # print("desired velocity", self.mVel)
        # mPendulum1.ctrlForceScale = self.myWin.lqr_strength

        # mPendulum1.setDesiredVelocity(self.mVel)
        # mPendulum2.setDesiredVelocity(self.mVel)

        # mPendulum1.setOrientation(self.myWin.desired_ori);
        # mPendulum2.setOrientation(self.myWin.desired_ori);

        mPendulum1.oneStep()
        # mPendulum2.oneStep()

        pend1State = np.zeros((2, 6))
        # pend2State = np.zeros((2, 6))

        pend1State[0, :] = mPendulum1.theta
        # pend2State[0, :] = mPendulum2.theta

        pend1State[1, :] = mPendulum1.dtheta
        # pend2State[1, :] = mPendulum2.dtheta

        # mPendulum1: draw()
        # mPendulum2: draw()

        # skel = self.skeletons[1]
        # q = skel.q
        # q["j_cart_x"] = mPendulum1.theta[0]
        # q["j_cart_z"] = mPendulum1.theta[1]
        # #
        # # #convert quaternion to axis angle
        # pole_quaternion = quater.Quaternion([mPendulum1.theta[2], mPendulum1.theta[3], mPendulum1.theta[4], mPendulum1.theta[5]])
        # axis_angle = pole_quaternion.axis * pole_quaternion.angle
        # # euler = pole_quaternion.yaw_pitch_roll
        # # print("pole_quaternion: ", pole_quaternion)
        # # print("euler: ", euler)
        # # # test_quaternion = quater.Quaternion([0.7071, 0.7071, 0, 0])
        # # # ea = test_quaternion.yaw_pitch_roll
        # # # print("test_quaternion: ", test_quaternion)
        # # # print("angle: ", ea)
        # #
        # q["j_pole_x"] = axis_angle[0]
        # q["j_pole_y"] = axis_angle[1]
        # q["j_pole_z"] = axis_angle[2]
        #
        # skel.set_positions(q)

        # print("pend1State: ", pend1State)
        # return pend1State, pend2State
        return pend1State

    def calcDesiredVel(self, pos, iFrame):
        mPendulum1 = self.mPendulum1
        # mPendulum2 = self.mPendulum2

        # print("mPendulum1.theta", mPendulum1.theta)
        # print("mPendulum2.theta", mPendulum2.theta)

        yVec = np.array([0, 1, 0])

        distance1 = pos[0]
        avoidance1 = pos[1]
        distance2 = pos[2]
        avoidance2 = pos[3]

        # print(mPendulum1.calcCOMpos())
        # print(mPendulum2.calcCOMpos())

        # facingVec1 = mPendulum2.calcCOMpos() - mPendulum1.calcCOMpos()
        facingVec1 = np.array([0., 0., 0.]) - mPendulum1.calcCOMpos()
        # print("facingVec1 1:", facingVec1)
        print("pendulum COM Pos: ", mPendulum1.calcCOMpos())

        facingVec1[1] = 0
        facingVec1 = facingVec1/math.sqrt(facingVec1[0]**2+facingVec1[1]**2+facingVec1[2]**2)
        avoidingVec1 = np.cross(facingVec1, yVec)

        # print("facingVec1 : ", facingVec1)
        # print("avoidingVec1 : ", avoidingVec1)
        #
        # facingVec2 = mPendulum1.calcCOMpos() - mPendulum2.calcCOMpos()
        # facingVec2[1] = 0
        # facingVec2=facingVec2/math.sqrt(facingVec2[0]**2+facingVec2[1]**2+facingVec2[2]**2)
        # avoidingVec2 = np.cross(facingVec2, yVec)
        #
        # print("distance : ", distance.euclidean(mPendulum1.calcCOMpos(), mPendulum2.calcCOMpos()), distance1)
        # desiredVel1 = (distance.euclidean(mPendulum1.calcCOMpos(), mPendulum2.calcCOMpos()) - distance1) * facingVec1 + avoidance1 * avoidingVec1
        # desiredVel2 = (distance.euclidean(mPendulum2.calcCOMpos(), mPendulum1.calcCOMpos()) - distance2) * facingVec2 + avoidance2 * avoidingVec2

        # desiredVel1 = (distance.euclidean(mPendulum1.calcCOMpos(), np.array([0, 0, 0])) - distance1) * facingVec1 + avoidance1 * avoidingVec1

        #desiredVel1 = 10.*(np.array([pos[0], 0., 0.]) - mPendulum1.calcCOMpos())
        desiredVel1 = np.array([10., 0., 0.])
        # desiredVel1 = (distance.euclidean(np.array([self.skeletons[1].q["j_cart_x"], -0.92, self.skeletons[1].q["j_cart_z"]]), [0, -0.92, 0]) - distance1) * facingVec1
        # print("desiredVel1: ", desiredVel1)
        # return desiredVel1, desiredVel2
        return desiredVel1

    def step(self):
        global mTime
        mPendulum1 = self.mPendulum1
        # mPendulum2 = self.mPendulum2

        pendulumPos1 = self.pendulumPos1
        # pendulumPos2 = self.pendulumPos2
        # pendulumVel1 = self.pendulumVel1
        # pendulumVel2 = self.pendulumVel2

        mTime = mTime + 1

        # self.mVel[0] = self.myWin.desired_x_val
        # self.mVel[2] = self.myWin.desired_z_val
        #
        # mPendulum1.setDesiredVelocity(self.mVel)
        # mPendulum2.setDesiredVelocity(self.mVel)
        # print(self.mVel)

        if mTime > self.motion_dof_row-1:
            mPendulum1.theta = pendulumPos1[self.motion_dof_row-1, :]
            # mPendulum2.theta = pendulumPos2[self.motion_dof_row-1, :]
            # mPendulum1.theta = np.zeros(6)
            # mPendulum1.theta[0] = 1.
            # mPendulum1.theta[1] = 0.
            # mPendulum1.theta[2] = 0.
            # mPendulum1.theta[3] = 0.
            # mPendulum1.theta[4] = 0.
            # mPendulum1.theta[5] = 0.
        else:
            print("mTime", mTime)
            mPendulum1.theta = pendulumPos1[mTime, :]
            # mPendulum2.theta = pendulumPos2[mTime, :]
            # print(mPendulum1.theta)

        # print("1 theta: ", mPendulum1.theta)
        # print(mPendulum2.theta)
        mPendulum1.oneStep()
        # mPendulum2.oneStep()
        # mPendulum1.draw()
        # mPendulum2.draw()

        skel = world.skeletons[1]
        q = skel.q
        q["j_cart_x"] = mPendulum1.theta[0]
        q["j_cart_z"] = mPendulum1.theta[1]

        #convert quaternion to axis angle
        pole_quaternion = quater.Quaternion([mPendulum1.theta[2], mPendulum1.theta[3], mPendulum1.theta[4], mPendulum1.theta[5]])
        axis_angle = pole_quaternion.axis * pole_quaternion.angle
        # euler = pole_quaternion.yaw_pitch_roll
        q["j_pole_x"] = axis_angle[0]
        q["j_pole_y"] = axis_angle[1]
        q["j_pole_z"] = axis_angle[2]

        # q["j_pole_x"] = mPendulum1.theta[3]
        # q["j_pole_y"] = mPendulum1.theta[4]
        # q["j_pole_z"] = mPendulum1.theta[5]

        print("q", q)

        skel.set_positions(q)

        super(MyWorld, self).step()


    def on_key_press(self, key):
        if key == '1':
            self.force = np.array([50.0, 0.0, 0.0])
            self.duration = 100
            print('push backward: f = %s' % self.force)
        elif key == '2':
            self.force = np.array([-50.0, 0.0, 0.0])
            self.duration = 100
            print('push backward: f = %s' % self.force)

    def render_with_ri(self, ri):
        # if self.force is not None and self.duration >= 0:
        #     p0 = self.skeletons[1].body('h_spine').C
        #     p1 = p0 + 0.01 * self.force
        #     ri.set_color(1.0, 0.0, 0.0)
        #     ri.render_arrow(p0, p1, r_base=0.05, head_width=0.1, head_len=0.1)
        ri.render_sphere(self.mPendulum1.calcCOMpos(), 0.05)        #mPendulum1 COM position

        # ri.render_sphere(np.array([2.0, 0.0, 0.0]), 0.05)
        # ri.render_sphere(np.array([-2.0, 0.0, 0.0]), 0.05)

        # for ii in range(len(self.desiredTraj1)):
        #     ri.render_sphere([self.desiredTraj1[ii, 0], 0.2, 0], 0.01)
        # ri.render_sphere([self.desiredTraj1[2000, 0], 0.2, 0], 0.02)
        ri.render_sphere([0.5, 0.2, 0], 0.02)
        # render axes
        ri.render_axes(np.array([0, 0, 0]), 0.5)
        # ri.set_color(0., 0., 0.)
        # ri.render_sphere(np.array([0, 0, 0]), 0.01)
        # ri.set_color(1.0, 0., 0.)
        # ri.render_line(np.array([0, 0, 0]), np.array([0.3, 0, 0]))
        # ri.set_color(0.0, 1., 0.)
        # ri.render_line(np.array([0, 0, 0]), np.array([0.0, 0.3, 0]))
        # ri.set_color(0.0, 0., 1.)
        # ri.render_line(np.array([0, 0, 0]), np.array([0.0, 0, 0.3]))

class Controller(object):
    def __init__(self, skel, h):
        self.h = h
        self.skel = skel
        ndofs = self.skel.ndofs
        self.qhat = self.skel.q
        self.Kp = np.diagflat([400.0] * ndofs)
        self.Kd = np.diagflat([40.0] * ndofs)
        self.preoffset = 0.0

    def compute(self):
        skel = self.skel

        # Algorithm:
        # Stable Proportional-Derivative Controllers.
        # Jie Tan, Karen Liu, Greg Turk
        # IEEE Computer Graphics and Applications, 31(4), 2011.

        invM = np.linalg.inv(skel.M + self.Kd * self.h)
        p = -self.Kp.dot(skel.q + skel.dq * self.h - self.qhat)
        d = -self.Kd.dot(skel.dq)
        qddot = invM.dot(-skel.c + p + d + skel.constraint_forces())
        return p + d - self.Kd.dot(qddot) * self.h

if __name__ == '__main__':
    print('Example: test IPC in 3D')

    app = QApplication(sys.argv)

    pydart.init()
    print('pydart initialization OK')

    world = MyWorld()
    print('MyWorld  OK')

    # skel = world.skeletons[1]
    # myq = skel.q
    # myq["j_cart_x"] = 11.0
    #
    # skel.set_positions(myq)
    #
    # # print("dt", world.dt)
    # # skel.set_controller(Controller(skel, world.dt))
    # # print('create controller OK')

    # print("zz: ", myq)
    #
    # print("cart_x: ",  myq["j_cart_x"])
    # print("cart_z: ", myq["j_cart_z"])

    # print("pole_x", myq["j_pole_x"])
    # print("pole_y", myq["j_pole_y"])
    # print("pole_z", myq["j_pole_z"])

    pydart.gui.viewer.launch_pyqt5(world)

    sys.exit(app.exec_())