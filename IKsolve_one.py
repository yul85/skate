import numpy as np
import math
from scipy.optimize import minimize

class IKsolver(object):
    def __init__(self, skel, h):
        self.skel = skel
        self.margin = np.array([0, 0, 0.05])
        self.h = h
        self.target_foot = self.skel.C + self.skel.dC * self.h
        self.prev_state = -1

    def update_target(self, state):
        self.state = state
        ground_height = -0.85

        if state == "state2":
            self.target_foot = self.skel.C + self.skel.dC * self.h
            self.prev_state = "state2"
        elif state == "state3":
            self.target_foot = self.skel.C + self.skel.dC * self.h
            self.prev_state = "state3"
        else:
            # print("foot velocity: ", self.skel.body("h_heel_right").dC[0])
            if self.prev_state == "state2":
                self.target_foot = self.skel.body("h_heel_right").to_world([0.0, 0.0, 0.0]) + np.array([self.skel.body("h_heel_right").dC[0], 0, 0]) * self.h
            elif self.prev_state == "state3":
                self.target_foot = self.skel.body("h_heel_left").to_world([0.0, 0.0, 0.0]) + np.array([self.skel.body("h_heel_left").dC[0], 0, 0]) * self.h
            else:
                self.prev_state = "state0"
                self.target_foot = self.skel.C + self.skel.dC * self.h


        self.target_foot[1] = ground_height

        self.target_foot_vel = self.skel.dC
        self.target_foot_vel[1] = 0
        # self.target_foot[1] = self.skel.body("h_heel_right").to_world([0.0, 0.0, 0.0])[1]

        # print("target: ", self.target_foot)

    def set_params(self, x):
        q = self.skel.q
        if self.state == "state2":
            q[12:18] = x
        elif self.state == "state3":
            q[6:12] = x
        else:
            q[6:18] = x
        # q = x
        self.skel.set_positions(q)

    def f(self, x):

        def rotationMatrixToAxisAngles(R):
            temp_r = np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]])
            angle = math.acos((R[0, 0]+R[1, 1]+R[2, 2] - 1)/2.)
            temp_r_norm = np.linalg.norm(temp_r)
            if temp_r_norm < 0.000001:
                return np.zeros(3)

            return temp_r/temp_r_norm * angle

        self.set_params(x)
        if self.state == "state2":
            body_name = ["h_blade_right"]
        elif self.state == "state3":
            body_name = ["h_blade_left"]
        else:
            body_name = ["h_blade_left", "h_blade_right"]

        if len(body_name) == 1:
            foot_state = self.skel.body(body_name[0]).to_world([0.0, 0.0, 0.0])
            foot_vel_state = self.skel.body(body_name[0]).dC
            foot_vel_state[1] = 0
            # print("foot pos : ", foot_state)
            # foot_ori = self.skel.body(body_name[0]).world_transform()
        else:
            # print("name?: ", body_name[0], body_name[1])
            foot_state1 = self.skel.body(body_name[0]).to_world([0.0, 0.0, 0.0])
            foot_vel_state1 = self.skel.body(body_name[0]).dC
            foot_vel_state1[1] = 0
            # foot_ori1 = self.skel.body(body_name[0]).world_transform()
            foot_state2 = self.skel.body(body_name[1]).to_world([0.0, 0.0, 0.0])
            foot_vel_state2 = self.skel.body(body_name[1]).dC
            foot_vel_state2[1] = 0
            # foot_ori2 = self.skel.body(body_name[1]).world_transform()
            # print("target vel: ", self.target_foot_vel)
            # print("foot_vel_state1: ", foot_vel_state1)


        # print("foot_ori: ", left_foot_ori[:3, :3], type(left_foot_ori[:3, :3]))


        # x_axis = np.array([0., 0., self.left_der[0][self.gtime]])
        # x_axis = np.array([1.0, 0., 1.0])
        # x_axis = x_axis / np.linalg.norm(x_axis)  # normalize
        # y_axis = np.array([0., 1., 0.])
        # z_axis = np.cross(x_axis, y_axis)
        # z_axis = z_axis / np.linalg.norm(z_axis)  # normalize
        # R_des = np.stack((x_axis, y_axis, z_axis), axis=-1)
        # # print("x_axis: \n")
        # # print(x_axis)
        # # print("y_axis: \n")
        # # print(y_axis)
        # # print("z_axis: \n")
        # # print(z_axis)
        # # print("R: \n")
        # # print(R_des)
        #
        # # left_axis_angle =rotationMatrixToAxisAngles(left_foot_ori[:3, :3])
        # # right_axis_angle = rotationMatrixToAxisAngles(right_foot_ori[:3, :3])
        #
        # res_rot = np.transpose(R_des).dot(foot_ori[:3, :3])
        # axis_angle = rotationMatrixToAxisAngles(res_rot)

        # print("axis_angle", axis_angle)

        # return 0.5 *  np.linalg.norm(left_foot_state - self.target_foot) ** 2
        # return 0.5 * (np.linalg.norm(foot_state - self.target_foot) ** 2)
                      # + np.linalg.norm(axis_angle - np.array([0., 0., 0.])) ** 2)

        if len(body_name) == 1:
            return 0.5 * (np.linalg.norm(foot_state - self.target_foot) ** 2)
        else:
            if self.prev_state == "state2":
                # print("22222222222")
                return 0.5 * (np.linalg.norm(foot_state1 - self.target_foot + np.array([0.0, 0.0, -0.1])) ** 2
                              + np.linalg.norm(foot_state2 - self.target_foot) ** 2)
            elif self.prev_state == "state3":
                # print("333333333")
                return 0.5 * (np.linalg.norm(foot_state1 - self.target_foot) **2
                              + np.linalg.norm(foot_state2 - self.target_foot + self.margin) ** 2)
            else:
                # print("11111111111")
                return 0.5 * (np.linalg.norm(foot_state1 - self.target_foot - self.margin) ** 2
                              + np.linalg.norm(foot_state2 - self.target_foot + self.margin) ** 2)

    def g(self, x):
        self.set_params(x)
        # print("x", x)

        if self.state == "state2":
            body_name = ["h_blade_right"]
        elif self.state == "state3":
            body_name = ["h_blade_left"]
        else:
            body_name = ["h_blade_left", "h_blade_right"]

        if len(body_name) == 1:
            foot_state = self.skel.body(body_name[0]).to_world([0.0, 0.0, 0.0])
            J_foot = self.skel.body(body_name[0]).linear_jacobian()
            A = foot_state - self.target_foot
            g = A.dot(J_foot)
        else:
            foot_state1 = self.skel.body(body_name[0]).to_world([0.0, 0.0, 0.0])
            J_foot1 = self.skel.body(body_name[0]).linear_jacobian()
            foot_state2 = self.skel.body(body_name[1]).to_world([0.0, 0.0, 0.0])
            J_foot2 = self.skel.body(body_name[1]).linear_jacobian()
            AA = np.append(foot_state1 - self.target_foot + self.margin, foot_state2 - self.target_foot - self.margin)
            J = np.vstack((J_foot1, J_foot2))
            g = AA.dot(J)

        # print("g:", g)
        if self.state == "state2":
            return g[12:18]
        elif self.state == "state3":
            return g[6:12]
        else:
            return g[6:18]


    def solve(self, ):
        q_backup = self.skel.q
        if self.state == "state2":
            res = minimize(self.f, x0= self.skel.q[12:18], jac=self.g, method="SLSQP")
        elif self.state == "state3":
            res = minimize(self.f, x0=self.skel.q[6:12], jac=self.g, method="SLSQP")
        else:
            res = minimize(self.f, x0=self.skel.q[6:18], jac=self.g, method="SLSQP")
        # res = minimize(self.f, x0=self.skel.q, jac=self.g, method="SLSQP")
        # print(res)
        self.skel.set_positions(q_backup)

        return res['x']