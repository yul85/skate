import numpy as np
import math
from scipy.optimize import minimize

class IKsolver(object):
    def __init__(self, skel1, skel2, left_foot_traj, right_foot_traj, t, pend):
        self.left_foot_traj = left_foot_traj
        self.right_foot_traj = right_foot_traj
        self.skel1 = skel1
        self.skel2 = skel2
        self.gtime = t
        self.pendulum = pend

    def update_target(self, ):
        # print("self.gtime:", self.gtime)
        # self.target_foot = np.array([skel1.q["j_cart_x"], -0.92, skel1.q["j_cart_z"]])

        # ground_height = -1.1
        ground_height = -0.92
        self.target_left_foot = np.array(
            [self.skel1.q["j_cart_x"], ground_height, self.left_foot_traj[1][self.gtime] - 0.05 - 0.3])
        self.target_right_foot = np.array(
            [self.skel1.q["j_cart_x"], ground_height, self.right_foot_traj[1][self.gtime] + 0.05 + 0.3])
        # l = self.skeletons[1].body('h_pole').local_com()[1]
        l = self.pendulum.length
        # self.target_pelvis = np.array([0, 0, 0])
        self.target_pelvis = np.array(
            [self.skel1.q["j_cart_x"] - l * math.sin(self.pendulum.theta), -0.92 + l * math.cos(self.pendulum.theta), 0])

    def set_params(self, x):
        q = self.skel2.q
        q = x
        # q[6:] = x
        self.skel2.set_positions(q)

    def f(self, x):

        def rotationMatrixToAxisAngles(R):
            temp_r = np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]])
            angle = math.acos((R[0, 0]+R[1, 1]+R[2, 2] - 1)/2.)
            temp_r_norm = np.linalg.norm(temp_r)
            if temp_r_norm < 0.000001:
                return np.zeros(3)

            return temp_r/temp_r_norm * angle

        def rotationMatrixToEulerAngles(R):

            # assert (isRotationMatrix(R))

            sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

            singular = sy < 1e-6

            if not singular:
                x = math.atan2(R[2, 1], R[2, 2])
                y = math.atan2(-R[2, 0], sy)
                z = math.atan2(R[1, 0], R[0, 0])
            else:
                x = math.atan2(-R[1, 2], R[1, 1])
                y = math.atan2(-R[2, 0], sy)
                z = 0

            return np.array([x, y, z])

        self.set_params(x)
        pelvis_state = self.skel2.body("h_pelvis").to_world([0.0, 0.0, 0.0])
        # pelvis_state = self.skel2.body("h_abdomen").to_world([0.0, 0.0, 0.0])
        left_foot_state = self.skel2.body("h_blade_left").to_world([0.0, 0.0, 0.0])
        left_foot_ori = self.skel2.body("h_blade_left").world_transform()
        right_foot_state = self.skel2.body("h_blade_right").to_world([0.0, 0.0, 0.0])
        right_foot_ori = self.skel2.body("h_blade_right").world_transform()

        abdomen_ori = self.skel2.body("h_abdomen").world_transform()


        # print("foot_ori: ", left_foot_ori[:3, :3], type(left_foot_ori[:3, :3]))

        # todo : calculate the x_axis according to the spline

        # x_axis = np.array([0., 0., self.left_der[0][self.gtime]])
        x_axis = np.array([1.0, 0., 1.0])
        x_axis = x_axis / np.linalg.norm(x_axis)  # normalize
        y_axis = np.array([0., 1., 0.])
        z_axis = np.cross(x_axis, y_axis)
        z_axis = z_axis / np.linalg.norm(z_axis)  # normalize
        R_des = np.stack((x_axis, y_axis, z_axis), axis=-1)
        # print("x_axis: \n")
        # print(x_axis)
        # print("y_axis: \n")
        # print(y_axis)
        # print("z_axis: \n")
        # print(z_axis)
        # print("R: \n")
        # print(R_des)

        # left_axis_angle =rotationMatrixToAxisAngles(left_foot_ori[:3, :3])
        # right_axis_angle = rotationMatrixToAxisAngles(right_foot_ori[:3, :3])

        left_res_rot = np.transpose(R_des).dot(left_foot_ori[:3, :3])
        right_res_rot = np.transpose(R_des).dot(right_foot_ori[:3, :3])
        # print("vec res: ", left_res_rot)
        left_axis_angle = rotationMatrixToAxisAngles(left_res_rot)
        right_axis_angle = rotationMatrixToAxisAngles(right_res_rot)

        abdomen_axis_angle = rotationMatrixToAxisAngles(abdomen_ori[:3, :3])

        # print("axis_angle", axis_angle)

        # return 0.5 *  np.linalg.norm(left_foot_state - self.target_foot) ** 2
        return 0.5 * (np.linalg.norm(pelvis_state - self.target_pelvis) ** 2 + np.linalg.norm(
            left_foot_state - self.target_left_foot) ** 2 + np.linalg.norm(
            right_foot_state - self.target_right_foot) ** 2 + np.linalg.norm(
            left_axis_angle - np.array([0., 0., 0.])) ** 2 + np.linalg.norm(
            right_axis_angle - np.array([0., 0., 0.])) ** 2 + np.linalg.norm(
            abdomen_axis_angle - np.array([0., 0., 0.])) ** 2)

    def g(self, x):
        self.set_params(x)

        pelvis_state = self.skel2.body("h_pelvis").to_world([0.0, 0.0, 0.0])
        # pelvis_state = self.skel2.body("h_abdomen").to_world([0.0, 0.0, 0.0])
        left_foot_state = self.skel2.body("h_blade_left").to_world([0.0, 0.0, 0.0])
        right_foot_state = self.skel2.body("h_blade_right").to_world([0.0, 0.0, 0.0])

        J_pelvis = self.skel2.body("h_pelvis").linear_jacobian()
        J_left_foot = self.skel2.body("h_blade_left").linear_jacobian()
        J_right_foot = self.skel2.body("h_blade_right").linear_jacobian()

        J_temp = np.vstack((J_pelvis, J_left_foot))
        J = np.vstack((J_temp, J_right_foot))
        AA = np.append(pelvis_state - self.target_pelvis, left_foot_state - self.target_left_foot)
        AAA = np.append(AA, right_foot_state - self.target_right_foot)

        # print("pelvis", pelvis_state - self.target_pelvis)
        # print("J", J_pelvis)
        #
        # print("AA", AA)
        # print("J", J)

        # g = AA.dot(J)
        g = AAA.dot(J)
        # g = AA.dot(J)[6:]

        return g

    def solve(self, ):
        q_backup = self.skel2.q
        res = minimize(self.f, x0= self.skel2.q, jac=self.g, method="SLSQP")
        # res = minimize(self.f, x0=self.skeletons[2].q[6:], jac=self.g, method="SLSQP")
        # print(res)
        self.skel2.set_positions(q_backup)

        return res['x']