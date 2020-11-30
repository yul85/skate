import numpy as np
import pydart2 as pydart
import sys
from PyQt5.QtWidgets import (QWidget, QFrame, QApplication)
from PyQt5.QtGui import QColor

class StateWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):

        self.col = QColor(0,0,0)

        self.square1 = QFrame(self)
        self.square1.setGeometry(50, 50, 50, 50)
        self.square1.setStyleSheet("QWidget { background-color: %s }" %
                                  self.col.name())

        self.col = QColor(1, 0, 0)
        self.square2 = QFrame(self)
        self.square2.setGeometry(150, 50, 50, 50)
        self.square2.setStyleSheet("QWidget { background-color: %s }" %
                                  self.col.name())
        self.col = QColor(0, 1, 0)
        self.square3 = QFrame(self)
        self.square3.setGeometry(150, 150, 50, 50)
        self.square3.setStyleSheet("QWidget { background-color: %s }" %
                                  self.col.name())
        self.col = QColor(0, 0, 1)
        self.square4 = QFrame(self)
        self.square4.setGeometry(50, 150, 50, 50)
        self.square4.setStyleSheet("QWidget { background-color: %s }" %
                                  self.col.name())

        self.setGeometry(300, 300, 300, 300)
        self.setWindowTitle('Finite State Machine')
        self.show()


class State(object):
    def __init__(self, dt, c_d, c_v, angles):
        self.dt = dt
        self.c_d = c_d
        self.c_v = c_v
        self.angles = angles

class Controller:
    def __init__(self, skel, h):
        self.h = h
        self.skel = skel
        ndofs = self.skel.ndofs
        self.qhat = self.skel.q
        # self.q_desired = self.skel.q
        self.Kp = np.diagflat([0.0] * 3 + [300.0] * (ndofs - 3))
        self.Kd = np.diagflat([0.0] * 3 + [30.0] * (ndofs - 3))
        self.preoffset = 0.0

        self.swing = "left"

    def balance_feedback(self, state, swing_foot):
        skel = self.skel
        c_d = state.c_d
        c_v = state.c_v
        v = skel.com_velocity()

        # print("skel.com(): ", skel.com())
        # print("skel.body: ", skel.body("h_heel_left").to_world([0., 0., 0.]))

        if swing_foot == "left":
            self.swing = "left"
            self.qhat[0] = state.angles[0]
            self.qhat[3] = state.angles[1]
            self.qhat[4] = state.angles[2]
            self.qhat[5] = state.angles[3]
            self.qhat[7] = state.angles[4]
            self.qhat[8] = state.angles[5]
            d = skel.com()[0] - skel.body("h_heel_left").to_world([0., 0., 0.])[0]
            # print("1:", type(self.qhat[3]), self.qhat[3])
            # print("2:", type(c_d * d + c_v * v), c_d * d + c_v * v[0])
            self.qhat[3] = self.qhat[3] + c_d * d + c_v * v[0]
        else:
            self.swing = "right"
            self.qhat[0] = state.angles[0]
            self.qhat[4] = state.angles[4]
            self.qhat[5] = state.angles[5]
            self.qhat[6] = state.angles[1]
            self.qhat[7] = state.angles[2]
            self.qhat[8] = state.angles[3]
            d = skel.com()[0] - skel.body("h_heel_right").to_world([0., 0., 0.])[0]
            self.qhat[6] = self.qhat[6] + c_d * d + c_v * v[0]

    def compute(self):
        skel = self.skel

        # Algorithm:
        # Stable Proportional-Derivative Controllers.
        # Jie Tan, Karen Liu, Greg Turk
        # IEEE Computer Graphics and Applications, 31(4), 2011.

        # print("q", skel.q)

        invM = np.linalg.inv(skel.M + self.Kd * self.h)
        p = -self.Kp.dot(skel.q + skel.dq * self.h - self.qhat)
        d = -self.Kd.dot(skel.dq)
        qddot = invM.dot(-skel.c + p + d + skel.constraint_forces())
        tau = p + d - self.Kd.dot(qddot) * self.h

        if self.swing == "left":
            tau[6] = -tau[0] - tau[3]
        else:
            tau[3] = -tau[0] - tau[6]

        # # Check the balance
        # COP = skel.body('h_heel_left').to_world([0.05, 0, 0])
        # offset = skel.C[0] - COP[0]
        # preoffset = self.preoffset
        #
        # # Adjust the target pose -- translated from bipedStand app of DART
        # foot = skel.dof_indices(["j_heel_left_1", "j_toe_left",
        #                          "j_heel_right_1", "j_toe_right"])
        # if 0.0 < offset < 0.1:
        #     k1, k2, kd = 200.0, 100.0, 10.0
        #     k = np.array([-k1, -k2, -k1, -k2])
        #     tau[foot] += k * offset + kd * (preoffset - offset) * np.ones(4)
        #     self.preoffset = offset
        # elif -0.2 < offset < -0.05:
        #     k1, k2, kd = 2000.0, 100.0, 100.0
        #     k = np.array([-k1, -k2, -k1, -k2])
        #     tau[foot] += k * offset + kd * (preoffset - offset) * np.ones(4)
        #     self.preoffset = offset

        # Make sure the first six are zero
        tau[:6] = 0
        return tau


class MyWorld(pydart.World):
    def __init__(self, ):
        pydart.World.__init__(self, 1.0 / 1000.0, './data/skel/2D_body.skel')

        self.force = None
        self.duration = 0

        # if dt == -1, fc(foot contact)
        state0 = State(0.3, 0.0, 0.2, np.array([0.0, 0.4, -1.1, 0.2, -0.05, 0.2]))
        state1 = State(-1, 2.2, 0.0, np.array([0.0, -0.7, -0.05, 0.2, -0.1, 0.2]))
        state2 = State(0.3, 0.0, 0.2, np.array([0.0, 0.4, -1.1, 0.2, -0.05, 0.2]))
        state3 = State(-1, 2.2, 0.0, np.array([0.0, -0.7, -0.05, 0.2, -0.1, 0.2]))

        self.state_list = [state0, state1, state2, state3]

        self.curr_state = self.state_list[0]
        self.elapsedTime = 0.0
        self.curr_state_index = 0
        self.swing_leg = "left"

    def step(self):
        if self.force is not None and self.duration >= 0:
            self.duration -= 1
            self.skeletons[1].body('h_spine').add_ext_force(self.force)

        if skel.body("h_heel_right") in self.collision_result.contacted_bodies:
            self.skeletons[1].controller.balance_feedback(self.curr_state, "left")
        if skel.body("h_heel_left") in self.collision_result.contacted_bodies:
            self.skeletons[1].controller.balance_feedback(self.curr_state, "right")

        # print("t: ", self.time())
        # todo : state transition
        print("currunt_state dt: ", self.curr_state.dt)
        # if self.curr_state_index % 2 == 1:
        # if self.curr_state.dt < 0.:
        #     print("fcfcfcfcfcf")
            # if self.swing_leg == "left":
            #     if skel.body("h_heel_left") in self.collision_result.contacted_bodies:
            #         print("LEFT foot contact")
            #         self.swing_leg = "right"
            #         self.curr_state_index = self.curr_state_index + 1
            #         self.curr_state_index = self.curr_state_index % 4
            #         self.elapsedTime = self.time()
            #         self.curr_state = self.state_list[self.curr_state_index]
            #         print("state_", self.curr_state_index)
            # if self.swing_leg == "right":
            #     if skel.body("h_heel_right") in self.collision_result.contacted_bodies:
            #         print("RIGHT foot contact")
            #         self.swing_leg = "left"
            #         self.curr_state_index = self.curr_state_index + 1
            #         self.curr_state_index = self.curr_state_index % 4
            #         self.elapsedTime = self.time()
            #         self.curr_state = self.state_list[self.curr_state_index]
            #         print("state_", self.curr_state_index)
        if self.curr_state_index == 1:
            if skel.body("h_heel_left") in self.collision_result.contacted_bodies:
                print("LEFT foot contact")
                self.skeletons[1].controller.balance_feedback(self.curr_state, self.swing_leg)
                self.swing_leg = "right"
                self.curr_state_index = self.curr_state_index + 1
                self.curr_state_index = self.curr_state_index % 4
                self.elapsedTime = self.time()
                self.curr_state = self.state_list[self.curr_state_index]
                print("state_", self.curr_state_index)
        elif self.curr_state_index == 3:
            if skel.body("h_heel_right") in self.collision_result.contacted_bodies:
                print("RIGHT foot contact")
                self.skeletons[1].controller.balance_feedback(self.curr_state, self.swing_leg)
                self.swing_leg = "left"
                self.curr_state_index = self.curr_state_index + 1
                self.curr_state_index = self.curr_state_index % 4
                self.elapsedTime = self.time()
                self.curr_state = self.state_list[self.curr_state_index]
                print("state_", self.curr_state_index)
        else:
            # print("deltattttttttt")
            # print("time:", self.time())
            # print("elapsed:",self.elapsedTime)
            # print("--:", self.time() - self.elapsedTime)
            # self.skeletons[1].controller.balance_feedback(self.curr_state, self.swing_leg)
            if self.curr_state.dt < self.time() - self.elapsedTime:
                print("here!!!", self.curr_state_index)
                self.curr_state_index = self.curr_state_index + 1
                self.curr_state_index = self.curr_state_index % 4
                self.elapsedTime = self.time()
                self.curr_state = self.state_list[self.curr_state_index]
                print("state_", self.curr_state_index)

        super(MyWorld, self).step()

    def on_key_press(self, key):
        if key == '1':
            self.force = np.array([50.0, 0.0, 0.0])
            self.duration = 1000
            print('push backward: f = %s' % self.force)
        elif key == '2':
            self.force = np.array([-50.0, 0.0, 0.0])
            self.duration = 100
            print('push backward: f = %s' % self.force)

    def render_with_ri(self, ri):
        if self.force is not None and self.duration >= 0:
            p0 = self.skeletons[1].body('h_spine').C
            p1 = p0 + 0.01 * self.force
            ri.set_color(1.0, 0.0, 0.0)
            ri.render_arrow(p0, p1, r_base=0.05, head_width=0.1, head_len=0.1)
        # render axes
        ri.render_axes(np.array([0, 0, 0]), 0.5)

if __name__ == '__main__':
    print('Example: FSM test ')

    app = QApplication(sys.argv)
    ex = StateWindow()

    pydart.init()
    print('pydart initialization OK')

    world = MyWorld()
    print('MyWorld  OK')

    # Use SkelVector to configure the initial pose
    skel = world.skeletons[1]
    q = skel.q
    # q["j_pelvis_y"] = 0
    # q["j_thigh_left"] = 0.4
    # q["j_shin_left"] = -1.1
    # q["j_heel_left"] = 0.2
    # # q["j_thigh_left"] = 0
    # q["j_shin_left"] = -0.05
    # q["j_heel_left"] = 0.2
    # # # q["default"] = -0.2
    # # q["j_thigh_left", "j_shin_left", "j_heel_left"] = 0.15, -0.4, 0.25
    # # q["j_thigh_right", "j_shin_right", "j_heel_right"] = 0.15, -0.4, 0.25
    # skel.set_positions(q)
    print('skeleton position OK')

    # Initialize the controller
    skel.set_controller(Controller(skel, world.dt))
    print('create controller OK')

    # print('[Joint]')
    # for joint in skel.joints:
    #     print("\t" + str(joint))
    #     print("\t\tparent = " + str(joint.parent_bodynode))
    #     print("\t\tchild = " + str(joint.child_bodynode))
    #     print("\t\tdofs = " + str(joint.dofs))

    pydart.gui.viewer.launch_pyqt5(world)

    sys.exit(app.exec_())