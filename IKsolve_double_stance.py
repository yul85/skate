import numpy as np
import math
from scipy.optimize import minimize

class IKsolver(object):
    def __init__(self, skel, ref_skel, h):
        self.skel = skel
        self.ref_skel = skel
        # self.margin = np.array([0, 0, 0.05])
        self.h = h
        self.target_foot = self.skel.C + self.skel.dC * self.h
        self.prev_state = -1

        self.offset_list = [[-0.1040 + 0.0216, +0.80354016 - 0.85354016, 0.0], [0.1040+0.0216, +0.80354016-0.85354016, 0.0]]

    def update_target(self, ground_height):
        # self.state = state
        self.target_foot_left1 = self.skel.body("h_blade_left").to_world(self.offset_list[0]) + np.array([self.skel.body("h_blade_left").dC[0], 0, 0]) * self.h
        self.target_foot_left2 = self.skel.body("h_blade_left").to_world(self.offset_list[1]) + np.array([self.skel.body("h_blade_left").dC[0], 0, 0]) * self.h
        self.target_foot_right1 = self.skel.body("h_blade_right").to_world(self.offset_list[0]) + np.array([self.skel.body("h_blade_right").dC[0], 0, 0]) * self.h
        self.target_foot_right2 = self.skel.body("h_blade_right").to_world(self.offset_list[1]) + np.array([self.skel.body("h_blade_right").dC[0], 0, 0]) * self.h
        # self.target_foot_left1 = self.ref_skel.body("h_blade_left").to_world(self.offset_list[0]) + np.array([self.ref_skel.body("h_blade_left").dC[0], 0, 0]) * self.h
        # self.target_foot_left2 = self.ref_skel.body("h_blade_left").to_world(self.offset_list[1]) + np.array([self.ref_skel.body("h_blade_left").dC[0], 0, 0]) * self.h
        # self.target_foot_right1 = self.ref_skel.body("h_blade_right").to_world(self.offset_list[0]) + np.array([self.ref_skel.body("h_blade_right").dC[0], 0, 0]) * self.h
        # self.target_foot_right2 = self.ref_skel.body("h_blade_right").to_world(self.offset_list[1]) + np.array([self.ref_skel.body("h_blade_right").dC[0], 0, 0]) * self.h

        self.target_foot_left1[1] = ground_height
        self.target_foot_left2[1] = ground_height
        self.target_foot_right1[1] = ground_height
        self.target_foot_right2[1] = ground_height

        self.target_foot_vel = self.skel.dC
        self.target_foot_vel[1] = 0
        # self.target_foot[1] = self.skel.body("h_heel_right").to_world([0.0, 0.0, 0.0])[1]
        # print("target: ", self.target_foot)

    def set_params(self, x):
        q = self.skel.q
        q[6:] = x
        self.skel.set_positions(q)

    def f(self, x):
        self.set_params(x)

        foot_state_left1 = self.skel.body("h_blade_left").to_world(self.offset_list[0])
        foot_state_left2 = self.skel.body("h_blade_left").to_world(self.offset_list[1])
        foot_state_right1 = self.skel.body("h_blade_right").to_world(self.offset_list[0])
        foot_state_right2 = self.skel.body("h_blade_right").to_world(self.offset_list[1])

        return 0.5 * (np.linalg.norm(foot_state_left1 - self.target_foot_left1) ** 2 + np.linalg.norm(foot_state_left2 - self.target_foot_left2) ** 2 +
                      np.linalg.norm(foot_state_right1 - self.target_foot_right1) ** 2 + np.linalg.norm(foot_state_right2 - self.target_foot_right2) ** 2)

        # return 0.5 * (np.linalg.norm(foot_state_right1 - self.target_foot_right1) ** 2 + np.linalg.norm(foot_state_right2 - self.target_foot_right2) ** 2)
        # return 0.5 * (np.linalg.norm(foot_state_left1 - self.target_foot_left1) ** 2 + np.linalg.norm(foot_state_left2 - self.target_foot_left2) ** 2)

    def g(self, x):
        self.set_params(x)

        foot_state_left1 = self.skel.body("h_blade_left").to_world(self.offset_list[0])
        foot_state_left2 = self.skel.body("h_blade_left").to_world(self.offset_list[1])
        foot_state_right1 = self.skel.body("h_blade_right").to_world(self.offset_list[0])
        foot_state_right2 = self.skel.body("h_blade_right").to_world(self.offset_list[1])

        J_foot_left = self.skel.body("h_blade_left").linear_jacobian()
        J_foot_right = self.skel.body("h_blade_right").linear_jacobian()
        AA1 = np.append(foot_state_left1 - self.target_foot_left1, foot_state_left2 - self.target_foot_left2)
        AA2 = np.append(foot_state_right1 - self.target_foot_right1, foot_state_right2 - self.target_foot_right2)
        AA = np.append(AA1, AA2)

        J1 = np.vstack((J_foot_left, J_foot_left))
        J2 = np.vstack((J_foot_right, J_foot_right))
        J = np.vstack((J1, J2))

        g = AA.dot(J)

        # g = AA2.dot(J2)     # only right foot
        # g = AA1.dot(J1)  # only right foot

        return g[6:]

    def solve(self, ):
        q_backup = self.skel.q
        res = minimize(self.f, x0=self.skel.q[6:], jac=self.g, method="SLSQP")
        # res = minimize(self.f, x0=self.skel.q, jac=self.g, method="SLSQP")
        # print("res: ", res)

        self.skel.set_positions(q_backup)

        return res['x']