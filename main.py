# Copyright (c) 2015, Disney Research
# All rights reserved.
#
# Author(s): Sehoon Ha <sehoon.ha@disneyresearch.com>
# Disney Research Robotics Group
import pydart2 as pydart
import numpy as np
import cma
from scipy.optimize import minimize
from pydart2.gui.trackball import Trackball
from pydart2.gui.opengl.scene import OpenGLScene

# class MyCamera(pydart.OpenGLScene):
#     def __init__camera(self, ):
#
#         self.add_camera(
#             Trackball(
#                 rot=[-0.152, 0.045, -0.002, 0.987],
#                 trans=[0.150, 0.310, -2.600]),
#             "YUL Camera")


# class MyDartWindow(Pydartwindow):
#     def __init__(self):
#         self.__init__()

class Controller(object):
    def __init__(self, skel, h):
        self.h = h
        self.skel = skel
        ndofs = self.skel.ndofs
        print("ndofs",ndofs )
        self.qhat = self.skel.q
        self.Kp = np.diagflat([0.0] * 6 + [400.0] * (ndofs - 6))
        self.Kd = np.diagflat([0.0] * 6 + [40.0] * (ndofs - 6))
        self.preoffset = 0.0

    def compute(self):
        skel = self.skel

        # # Algorithm:
        # # Stable Proportional-Derivative Controllers.
        # # Jie Tan, Karen Liu, Greg Turk
        # # IEEE Computer Graphics and Applications, 31(4), 2011.
        #
        # invM = np.linalg.inv(skel.M + self.Kd * self.h)
        # p = -self.Kp.dot(skel.q + skel.dq * self.h - self.qhat)
        # d = -self.Kd.dot(skel.dq)
        # qddot = invM.dot(-skel.c + p + d + skel.constraint_forces())
        # tau = p + d - self.Kd.dot(qddot) * self.h
        #
        # # Check the balance
        # COP = skel.body('h_heel_left').to_world([0.05, 0, 0])
        # offset = skel.C[0] - COP[0] + 0.03
        # preoffset = self.preoffset
        #
        # # Adjust the target pose -- translated from bipedStand app of DART
        # foot = skel.dof_indices(["j_heel_left_1", "j_toe_left",
        #                          "j_heel_right_1", "j_toe_right"])
        # # if 0.0 < offset < 0.1:
        # #     k1, k2, kd = 200.0, 100.0, 10.0
        # #     k = np.array([-k1, -k2, -k1, -k2])
        # #     tau[foot] += k * offset + kd * (preoffset - offset) * np.ones(4)
        # #     self.preoffset = offset
        # # elif -0.2 < offset < -0.05:
        # #     k1, k2, kd = 2000.0, 100.0, 100.0
        # #     k = np.array([-k1, -k2, -k1, -k2])
        # #     tau[foot] += k * offset + kd * (preoffset - offset) * np.ones(4)
        # #     self.preoffset = offset
        #
        # # # High-stiffness
        # # k1, k2, kd = 200.0, 10.0, 10.0
        # # k = np.array([-k1, -k2, -k1, -k2])
        # # tau[foot] += k * offset + kd * (preoffset - offset) * np.ones(4)
        # # print("offset = %s" % offset)
        # # self.preoffset = offset
        #
        # # Low-stiffness
        # k1, k2, kd = 20.0, 1.0, 10.0
        # k = np.array([-k1, -k2, -k1, -k2])
        # tau[foot] += k * offset + kd * (preoffset - offset) * np.ones(4)
        # if offset > 0.03:
        #     tau[foot] += 0.3 * np.array([-100.0, -50.0, -100.0, -50.0])
        #     print("Discrete A")
        # if offset < -0.02:
        #     tau[foot] += -1.0 * np.array([-100.0, 0.0, -100.0, 0.0])
        #     print("Discrete B")
        # print("offset = %s" % offset)
        # self.preoffset = offset

        # normal force
        g = 9.81
        f_s = 1 / 4 * skel.m * g

        cp1 = [0.17, 0.95, -0.20]  # contact point
        cp2 = [0.17, 0.95, -0.22]  # contact point
        cp3 = [0.17, 0.95, -0.24]  # contact point
        cp4 = [0.17, 0.95, -0.26]  # contact point
        body1 = skel.body('h_index_finger_distal_left')
        body2 = skel.body('h_middle_finger_distal_left')
        body3 = skel.body('h_ring_finger_distal_left')
        body4 = skel.body('h_little_finger_distal_left')
        #print(body1.world_jacobian(body1.to_local(cp1)))

        f1 = np.zeros(3)
        f1[1] = f_s
        f2 = np.zeros(3)
        f2[1] = f_s
        f3 = np.zeros(3)
        f3[1] = f_s
        f4 = np.zeros(3)
        f4[1] = f_s

        tau1 = np.dot(body1.world_jacobian(body1.to_local(cp1))[3:, :].transpose(), f1)
        tau2 = np.dot(body2.world_jacobian(body2.to_local(cp2))[3:, :].transpose(), f2)
        tau3 = np.dot(body3.world_jacobian(body3.to_local(cp3))[3:, :].transpose(), f3)
        tau4 = np.dot(body4.world_jacobian(body4.to_local(cp4))[3:, :].transpose(), f4)

        tau = tau1+tau2+tau3+tau4
        #tau = np.zeros(61)
        print(tau.shape, tau)

        # Make sure the first six are zero
        tau[:6] = 0
        #print(tau.shape, tau)
        return tau


class MyWorld(pydart.World):
    def __init__(self, ):

        self.is_control = 0
        """
        """
        #pydart.World.__init__(self, 1.0 / 2000.0, './data/skel/fullbody1.skel')
        #pydart.World.__init__(self, 1.0 / 2000.0, './data/skel/hand1.skel')
        pydart.World.__init__(self, 1.0 / 2000.0, './data/skel/hand1_fullbody.skel')
        self.ik_skel = self.skeletons[0]
        print("dododoff", self.ik_skel.ndofs)
        #self.theta = 0.0 * np.pi
        self.update_target()
        self.solve()


        self.count = 0
        #self.force = None
        #self.duration = 0

    def update_target(self, ):
        # th, r = self.theta - 0.5 * np.pi, 0.6
        # x, y = r * np.cos(th) + 0.4, r* np.sin(th)
        # self.target = np.array([x, y, 0.3])


        self.target = np.array([0.145, 0.9, -0.1509])

        x = 0.17
        y = 0.9
        self.t1 = np.array([x, y, -0.20])
        self.t2 = np.array([x, y, -0.22])
        self.t3 = np.array([x, y, -0.24])
        self.t4 = np.array([x, y, -0.26])
        #self.target = np.array([x, y, -0.1259])

    def set_params(self, x):
        q = self.skeletons[0].positions()
        q[6:] = x
        self.skeletons[0].set_positions(q)

    def get_params(self):
        return self.skeletons[0].positions()

    def f(self, x):
        q_temp = self.get_params()
        self.set_params(x)

        for i in range(len(x)):
            if i < 20-6-1:              #legs
                x[i] = q_temp[i+6]
            if i == 22-6-1:             #head
                x[i] = q_temp[i+6]
            if i > 31-6-1:              #right arm
                x[i] = q_temp[i+6]

        palm = self.ik_skel.body("h_hand_left").to_world([0.0, 0.0, 0.0])
        # f1 = self.ik_skel.body("h_index_finger_distal_left").to_world([0.0, 0.0, 0.0])
        # f2 = self.ik_skel.body("h_middle_finger_distal_left").to_world([0.0, 0.0, 0.0])
        # f3 = self.ik_skel.body("h_ring_finger_distal_left").to_world([0.0, 0.0, 0.0])
        # f4 = self.ik_skel.body("h_little_finger_distal_left").to_world([0.0, 0.0, 0.0])

        # return 0.5 * np.linalg.norm(f1-self.t1) ** 2 + 0.5 * np.linalg.norm(f2-self.t2) ** 2 + 0.5 * np.linalg.norm(f3-self.t3) ** 2 + 0.5 * np.linalg.norm(f4-self.t4) ** 2
        return 0.5 * np.linalg.norm(palm-self.target) ** 2 # + 0.5 * np.linalg.norm(f2-self.t2) ** 2 + 0.5 * np.linalg.norm(f3-self.t3) ** 2 + 0.5 * np.linalg.norm(f4-self.t4) ** 2

    def g(self, x):
        self.set_params(x)

        lhs = self.ik_skel.body("h_hand_left").to_world([0.0, 0.0, 0.0])
        rhs = self.target
        J = self.ik_skel.body("h_hand_left").world_jacobian()[3:]
        g = (lhs - rhs).dot(J)[6:]

        return g

    def solve(self, ):

        #print("initial x", self.ik_skel.positions()[6:])
        res = minimize(self.f,
                       x0=self.ik_skel.positions()[6:],
                       jac=self.g,
                       method="SLSQP",
                       # tol=0.00001
                       )
        #print(res)

    def step(self, ):
        # if self.force is not None and self.duration >= 0:
        #     self.duration -= 1
        #     self.skeletons[1].body('h_spine').add_ext_force(self.force)
        #print("cnt", self.count)
        if self.count < 1000:
            self.update_target()
            self.solve()
            self.count = self.count+1
        else:
            self.skeletons[1].body('ground').set_collidable(False)
            self.is_control = 1

        cfn = pydart.papi.collisionresult__getCollidingBodyNodes(0)
        # cf = pydart.papi.collisionresult__getContacts(2, )
        print(cfn)

        super(MyWorld, self).step()

    def on_key_press(self, key):
        print(key)
        if key == '1':
            self.force = np.array([50.0, 0.0, 0.0])
            self.duration = 100
            print('push backward: f = %s' % self.force)
        elif key == '2':
            self.force = np.array([-50.0, 0.0, 0.0])
            self.duration = 100
            print('push backward: f = %s' % self.force)
        elif key == 'S':
            super(MyWorld, self).step()
        elif key == 'C':
            self.is_control = 1


    # def render_with_ri(self, ri):
    #     if self.force is not None and self.duration >= 0:
    #         p0 = self.skeletons[1].body('h_spine').C
    #         p1 = p0 + 0.01 * self.force
    #         ri.set_color(1.0, 0.0, 0.0)
    #         ri.render_arrow(p0, p1, r_base=0.05, head_width=0.1, head_len=0.1)


if __name__ == '__main__':
    print('Example: bipedStand')

    pydart.init()
    print('pydart initialization OK')

    world = MyWorld()
    print('MyWorld  OK')


    # Use SkelVector to configure the initial pose
    skel = world.skeletons[0]
    q = skel.q
    q["j_pelvis_pos_y"] = -0.05
    q["j_pelvis_rot_y"] = -0.2
    q["j_thigh_left_z", "j_shin_left", "j_heel_left_1"] = 0.15, -0.4, 0.25
    q["j_thigh_right_z", "j_shin_right", "j_heel_right_1"] = 0.15, -0.4, 0.25
    q["j_abdomen_2"] = 0.0

    # q["j_bicep_left_x", "j_bicep_left_y", "j_bicep_left_z"] =0.0, 0.397146, -0.169809
    # q['j_bicep_left_x'] = -0.5


    #Q : How to set the joint limit?
    q["j_bicep_left_y"] = 0.0

    # skel.joint(15)
    # skel.getdof(15).set_position_lower_limit(0.0)
    # q["j_bicep_left_y"].set_position_lower_limit(0.0)
    # q["j_bicep_left_y"].set_position_lower_limit(0.0)
    # skel->getDof(15).setPositionLimits(0.0, 0.0)
    # skel->getDof(dt + 0)->setPositionLimits(-0.0, 0.0);
    print('skeleton position OK')

    # Initialize the controller
    # skel.set_controller(Controller(skel, world.dt))
    # world.skeletons[1].body('ground').set_collidable(False)
    # if world.is_control == 1:
    #     skel.set_controller(Controller(skel, world.dt))

    # print('create controller OK')

    # print("'1'--'2': programmed interaction")
    # print("    '1': push forward")
    # print("    '2': push backward")

    print("Camera:")
    print("    drag: rotate camera")
    print("    shift-drag: zoom camera")
    print("    control-drag: translate camera")

    # OpenGLScene.add_camera(OpenGLScene,
    #     Trackball(
    #             rot=[-0.152, 0.045, -0.002, 0.987],
    #             trans=[-0.050, 0.210, -3.500]),
    #     "YUL camera")

    pydart.gui.viewer.launch_pyqt5(world)


    #Contact force modeling
    # g = 9.8
    # f_s = skel.M * -9.81
    # v1 = [1,2,3]
    # #skel.body('h_index_finger_distal_left').position()
    # v2 = [4,5,6]
    # n = np.cross(v1, v2)
    # alpha = 0.1
    # f = alpha*n*np.transpose(n)*f_s

    cf = pydart.world.CollisionResult
    print(cf)




    #CMA-ES
    # objective_func = cf
    # res = cma.fmin(-objective_func)
    # print(res)