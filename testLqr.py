import numpy as np
import pydart2 as pydart
import control as ctrl
import math
from scipy.optimize import minimize
from scipy import interpolate

class Cart:
    def __init__(self, x, mass):
        self.x = x
        self.x_dot = 0.
        # self.y = int(0.6*world_size) 		# 0.6 was chosen for aesthetic reasons.
        self.mass = mass
        self.color = (0, 255, 0)


class Pendulum:
    def __init__(self, length, theta, ball_mass):
        self.length = length
        self.theta = theta
        self.theta_dot = 0.
        self.ball_mass = ball_mass
        self.color = (0,0,255)


def find_lqr_control_input(cart, pendulum, g):
    # Using LQR to find control inputs

    # The A and B matrices are derived in the report
    A = np.matrix([
        [0, 1, 0, 0],
        [0, 0, g * pendulum.ball_mass / cart.mass, 0],
        [0, 0, 0, 1],
        [0, 0, (cart.mass + pendulum.ball_mass) * g / (pendulum.length * cart.mass), 0]
    ])

    B = np.matrix([
        [0],
        [1 / cart.mass],
        [0],
        [1 / (pendulum.length * cart.mass)]
    ])

    # The Q and R matrices are emperically tuned. It is described further in the report
    Q = np.matrix([
        [1000, 0, 0, 0],
        [0, 0.01, 0, 0],
        [0, 0, 0.0000001, 0],
        [0, 0, 0, 100]
    ])

    R = np.matrix([10.])

    # The K matrix is calculated using the lqr function from the controls library
    K, S, E = ctrl.lqr(A, B, Q, R)
    np.matrix(K)

    x = np.matrix([
        [np.squeeze(np.asarray(cart.x))],
        [np.squeeze(np.asarray(cart.x_dot))],
        [np.squeeze(np.asarray(pendulum.theta))],
        [np.squeeze(np.asarray(pendulum.theta_dot))]
    ])

    desired = np.matrix([
        [2.],
        [0.05],
        [0.],
        [0]
    ])

    F = -np.dot(K, (x - desired))
    return np.squeeze(np.asarray(F))

def apply_control_input(cart, pendulum, F, time_delta, theta_dot, g):
    # Finding x and theta on considering the control inputs and the dynamics of the system
    theta_double_dot = (((cart.mass + pendulum.ball_mass) * g * math.sin(pendulum.theta)) + (F * math.cos(pendulum.theta)) - (pendulum.ball_mass * ((theta_dot)**2.0) * pendulum.length * math.sin(pendulum.theta) * math.cos(pendulum.theta))) / (pendulum.length * (cart.mass + (pendulum.ball_mass * (math.sin(pendulum.theta)**2.0))))
    x_double_dot = ((pendulum.ball_mass * g * math.sin(pendulum.theta) * math.cos(pendulum.theta)) - (pendulum.ball_mass * pendulum.length * math.sin(pendulum.theta) * (theta_dot**2)) + (F)) / (cart.mass + (pendulum.ball_mass * (math.sin(pendulum.theta)**2)))
    cart.x += cart.x_dot * time_delta
    pendulum.theta += pendulum.theta_dot * time_delta
    cart.x_dot += x_double_dot * time_delta
    pendulum.theta_dot = theta_double_dot * time_delta

    # cart.x += ((time_delta**2) * x_double_dot) + (((cart.x - x_tminus2) * time_delta) / previous_time_delta)
    # pendulum.theta += ((time_delta**2)*theta_double_dot) + (((pendulum.theta - theta_tminus2)*time_delta)/previous_time_delta)

from cvxopt import matrix, solvers

class Controller(object):
    def __init__(self, skel, h):
        self.h = h
        self.skel = skel
        ndofs = self.skel.ndofs
        self.target = None

        self.contact_num = 0

        self.V_c = []
        self.V_c_dot = []
        self.sol_tau = []
        self.sol_accel = []
        self.sol_lambda = []

        self.Kp = np.diagflat([0.0] * 6 + [400.0] * (ndofs - 6))
        self.Kd = np.diagflat([0.0] * 6 + [40.0] * (ndofs - 6))

        # coefficient of each term
        self.K_ef = 10.0
        self.K_tr = 100
        self.K_cf = 0.05

    def rotationMatrixToEulerAngles(self, R):

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

    def check_contact(self):
        skel = self.skel

        contact_name_list = ['h_blade_left', 'h_blade_left', 'h_blade_right', 'h_blade_right']
        offset_list = [[-0.1040+0.0216, +0.80354016-0.85354016, 0.0],
                       [0.1040+0.0216, +0.80354016-0.85354016, 0.0],
                       [-0.1040 + 0.0216, +0.80354016 - 0.85354016, 0.0],
                       [0.1040 + 0.0216, +0.80354016 - 0.85354016, 0.0]]

        contact_candi_list = [skel.body(contact_name_list[0]).to_world(offset_list[0]),
                        skel.body(contact_name_list[1]).to_world(offset_list[1]),
                        skel.body(contact_name_list[2]).to_world(offset_list[2]),
                        skel.body(contact_name_list[3]).to_world(offset_list[3])]
        contact_list = []
        # print("contact_list", contact_candi_list)
        cn = 0      # the number of contact point
        # print("len(contact_list): ", len(contact_list))
        for ii in range(len(contact_candi_list)):
            # print(ii, contact_candi_list[ii][1])
            if -0.92 + 0.025 > contact_candi_list[ii][1]:
                contact_list.append(contact_name_list[ii])
                contact_list.append(offset_list[ii])
                cn = cn + 1

        self.contact_num = cn
        return contact_list

    def compute(self):
        # goal : get acceleration and lambda to satisfy the objectives and constraints below

        skel = self.skel
        ndofs = self.skel.ndofs

        contact_list = self.check_contact()
        self.contact_list = contact_list
        contact_num = self.contact_num
        print("The number of contact is ", contact_num)

        # get desired acceleration
        invM = np.linalg.inv(skel.M + self.Kd * self.h)
        # p = -self.Kp.dot(skel.q + skel.dq * self.h - self.qhat)
        p = -self.Kp.dot(skel.q - self.target + skel.dq * self.h)
        d = -self.Kd.dot(skel.dq)
        qddot = invM.dot(-skel.c + p + d + skel.constraint_forces())
        des_accel = p + d + qddot

        K_ef = self.K_ef
        K_tr = self.K_tr
        K_cf = self.K_cf

        # ===========================
        # Objective function
        # ===========================
        P = 2 * matrix(np.diagflat([K_ef] * ndofs + [K_tr] * ndofs + [K_cf] * 4 * contact_num))
        # print("P: ")
        # print(P)
        qqv = np.append(np.zeros(ndofs), -2 * K_tr * des_accel)
        if contact_num != 0:
            qqv = np.append(qqv, np.zeros(4 * contact_num))

        qq = matrix(qqv)
        # print("qq: ", len(qq))

        # ===========================
        # inequality constraint
        # ===========================
        if contact_num != 0:
            J_c_t = np.zeros((ndofs, 3 * contact_num))
            J_c_t_der = np.zeros((ndofs, 3 * contact_num))
            # print("contact_list: \n", contact_list)
            for ii in range(int(len(contact_list)/2)):
                contact_body_name = contact_list[2*ii]
                contact_offset = contact_list[2*ii+1]
                jaco = skel.body(contact_body_name).linear_jacobian(contact_offset)
                jaco_der = skel.body(contact_body_name).linear_jacobian_deriv(contact_offset)
                J_c_t[0:, ii*3:(ii + 1) * 3] = np.transpose(jaco)
                J_c_t_der[0:, ii*3:(ii + 1) * 3] = np.transpose(jaco_der)

            # print("J_c_t :\n", J_c_t)
            # print("length o f J_c_t: ", len(J_c_t))

            # self.skeletons[0].body('ground').set_friction_coeff(0.02)
            myu = 0.02
            self.V_c = np.zeros((3 * self.contact_num, 4 * self.contact_num))

            v1 = np.array([myu, 1, 0])
            v1 = v1 / np.linalg.norm(v1)  # normalize
            v2 = np.array([0, 1, myu])
            v2 = v2 / np.linalg.norm(v2)  # normalize
            v3 = np.array([-myu, 1, 0])
            v3 = v3 / np.linalg.norm(v3)  # normalize
            v4 = np.array([0, 1, -myu])
            v4 = v4 / np.linalg.norm(v4)  # normalize
            #
            # v1 = np.array([0., 1., 0.])
            # v2 = np.array([0., 1., 0.])
            # v3 = np.array([0., 1., 0.])
            # v4 = np.array([0., 1., 0.])

            V_c_1 = np.array([v1, v2, v3, v4])

            # print("V_c_1 :\n", np.transpose(V_c_1))

            for n in range(self.contact_num):
                self.V_c[n*3:(n+1)*3, n*4:(n+1)*4] = np.transpose(V_c_1)

            # print("V_c :\n", self.V_c)

            V_c = self.V_c

            Gm = np.zeros((2*4*contact_num, 2*ndofs + 4*contact_num))
            Gm[0:4*contact_num, 2*ndofs:] = -1 * np.identity(4*contact_num)
            Gm[4*contact_num:, ndofs:2*ndofs] = -1 * np.transpose(V_c).dot(np.transpose(J_c_t))
            G = matrix(Gm)
            # print("G :\n", G)
            hv = np.zeros(2*4*contact_num)
            V_c_t_J_c_dot = np.transpose(V_c).dot(np.transpose(J_c_t_der))
            hv1 = V_c_t_J_c_dot.dot(skel.dq)

            self.V_c_dot = np.zeros((3 * self.contact_num, 4 * self.contact_num))

            omega = []
            V_c_dot_stack = []
            compensate_vel = []

            for ii in range(self.contact_num):
                contact_body_name = contact_list[2*ii]
                blade_ori = skel.body(contact_body_name).world_transform()
                angle = self.rotationMatrixToEulerAngles(blade_ori[:3, :3])
                omega.append(np.array([0., angle[1]/self.h, 0.]))

                v1_dot = np.cross(v1, omega[ii])
                v1_dot = v1_dot / np.linalg.norm(v1_dot)  # normalize
                v2_dot = np.cross(v2, omega[ii])
                v2_dot = v2_dot / np.linalg.norm(v2_dot)  # normalize
                v3_dot = np.cross(v3, omega[ii])
                v3_dot = v3_dot / np.linalg.norm(v3_dot)  # normalize
                v4_dot = np.cross(v4, omega[ii])
                v4_dot = v4_dot / np.linalg.norm(v4_dot)  # normalize

                V_c_dot_stack.append(np.array([v1_dot, v2_dot, v3_dot, v4_dot]))

                compensate_vel.append(skel.body(contact_body_name).com_linear_velocity()[1] * self.h)

            # print("omega: \n", omega)
            # print("V_c_dot :\n", V_c_dot_stack)

            compensate_vel_vec = np.zeros(4*contact_num)
            for n in range(self.contact_num):
                self.V_c_dot[n*3:(n+1)*3, n*4:(n+1)*4] = np.transpose(V_c_dot_stack[n])
                compensate_vel_vec[n*4:(n+1)*4] = compensate_vel[n]

            V_c_dot = self.V_c_dot
            V_c_t_dot_J_c = np.transpose(V_c_dot).dot(np.transpose(J_c_t))
            hv2 = V_c_t_dot_J_c.dot(skel.dq)

            # print("hv1", hv1)
            # print("hv2", hv2)
            # print("compensate_vel: \n", compensate_vel_vec)

            hv[4*contact_num:] = hv1 + hv2 + compensate_vel_vec
            h = matrix(hv)
            # print("h :\n", h)

        # =============================================
        # equality constraint  -> motion of equation
        # =============================================
        Am = np.zeros((ndofs, ndofs*2 + 4 * contact_num))
        Am[0:, 0:ndofs] = -1*np.identity(ndofs)
        Am[0:, ndofs:2*ndofs] = skel.M

        if contact_num != 0:
            J_c_t_V_c = J_c_t.dot(V_c)
            # print("J_c_t_V_c :\n", J_c_t_V_c)
            # print("size of J_c_t_V_c :", len(J_c_t_V_c))
            Am[0:, 2*ndofs:] = -1 * J_c_t_V_c

        A = matrix(Am)
        # print("A :\n", A)
        b = matrix(-skel.c)

        if contact_num == 0:
            sol = solvers.qp(P, qq, None, None, A, b)
        else:
            sol = solvers.qp(P, qq, G, h, A, b)

        #print(sol['x'])

        solution = np.array(sol['x'])

        self.sol_tau = solution[:skel.ndofs].flatten()
        self.sol_accel = solution[skel.ndofs:2 * skel.ndofs].flatten()
        self.sol_lambda = solution[2 * skel.ndofs:].flatten()

        # print("self.sol_tau: \n", self.sol_tau)
        return self.sol_tau


# class Controller(object):
#     def __init__(self, skel, h):
#         self.h = h
#         self.skel = skel
#         ndofs = self.skel.ndofs
#         # self.qhat = self.skel.q
#         self.target = None
#
#         self.Kp = np.diagflat([0.0] * 6 + [400.0] * (ndofs - 6))
#         self.Kd = np.diagflat([0.0] * 6 + [40.0] * (ndofs - 6))
#         self.preoffset = 0.0
#
#     def compute(self):
#         skel = self.skel
#
#         # Algorithm:
#         # Stable Proportional-Derivative Controllers.
#         # Jie Tan, Karen Liu, Greg Turk
#         # IEEE Computer Graphics and Applications, 31(4), 2011.
#
#         invM = np.linalg.inv(skel.M + self.Kd * self.h)
#         # p = -self.Kp.dot(skel.q + skel.dq * self.h - self.qhat)
#         p = -self.Kp.dot(skel.q - self.target + skel.dq * self.h)
#         d = -self.Kd.dot(skel.dq)
#         qddot = invM.dot(-skel.c + p + d + skel.constraint_forces())
#         tau = p + d - self.Kd.dot(qddot) * self.h
#         #
#         # # Check the balance
#         # COP = skel.body('h_heel_left').to_world([0.05, 0, 0])
#         # offset = skel.C[0] - COP[0] + 0.03
#         # preoffset = self.preoffset
#         #
#         # # Adjust the target pose -- translated from bipedStand app of DART
#         #
#         # # foot = skel.dof_indices(["j_heel_left_1", "j_toe_left",
#         # #                          "j_heel_right_1", "j_toe_right"])
#         # foot = skel.dof_indices(["j_heel_left_1", "j_heel_right_1"])
#         # # if 0.0 < offset < 0.1:
#         # #     k1, k2, kd = 200.0, 100.0, 10.0
#         # #     k = np.array([-k1, -k2, -k1, -k2])
#         # #     tau[foot] += k * offset + kd * (preoffset - offset) * np.ones(4)
#         # #     self.preoffset = offset
#         # # elif -0.2 < offset < -0.05:
#         # #     k1, k2, kd = 2000.0, 100.0, 100.0
#         # #     k = np.array([-k1, -k2, -k1, -k2])
#         # #     tau[foot] += k * offset + kd * (preoffset - offset) * np.ones(4)
#         # #     self.preoffset = offset
#         #
#         # # # High-stiffness
#         # # k1, k2, kd = 200.0, 10.0, 10.0
#         # # k = np.array([-k1, -k2, -k1, -k2])
#         # # tau[foot] += k * offset + kd * (preoffset - offset) * np.ones(4)
#         # # print("offset = %s" % offset)
#         # # self.preoffset = offset
#         #
#         # # Low-stiffness
#         # k1, k2, kd = 20.0, 1.0, 10.0
#         # k = np.array([-k1, -k1])
#         # tau[foot] += k * offset + kd * (preoffset - offset) * np.ones(2)
#         # if offset > 0.03:
#         #     tau[foot] += 0.3 * np.array([-100.0, -100.0])
#         #     # print("Discrete A")
#         # if offset < -0.02:
#         #     tau[foot] += -1.0 * np.array([-100.0, -100.0])
#         #     # print("Discrete B")
#         # # print("offset = %s" % offset)
#         # self.preoffset = offset
#
#         # Make sure the first six are zero
#         tau[:6] = 0
#         return tau


class MyWorld(pydart.World):
    def __init__(self, ):
        """
        """
        pydart.World.__init__(self, 1.0 / 2000.0, './data/skel/cart_pole_blade.skel')
        # pydart.World.__init__(self, 1.0 / 2000.0, './data/skel/cart_pole.skel')
        self.force = None
        self.duration = 0
        self.skeletons[0].body('ground').set_friction_coeff(0.02)
        self.state = [0., 0., 0., 0.]
        self.cart = Cart(0., 1.)
        self.pendulum = Pendulum(1.65*0.5, 0., 59.999663)

        # self.out_spline = self.generate_spline_trajectory()
        plist = [(0, 0), (0.2, 0), (0.5, -0.3), (1.0, -0.5), (1.5, -0.3), (1.8, 0), (2.0, 0.0)]
        self.left_foot_traj, self.left_der = self.generate_spline_trajectory(plist)
        plist = [(0, 0), (0.2, 0), (0.5, 0.3), (1.0, 0.5), (1.5, 0.3), (1.8, 0), (2.0, 0.0)]
        self.right_foot_traj, self.right_der = self.generate_spline_trajectory(plist)

        # print("out_spline", type(self.out_spline), type(self.out_spline[0]))

        skel = self.skeletons[2]
        # Set target pose
        # self.target = skel.positions()

        self.controller = Controller(skel, self.dt)

        self.gtime = 0
        self.update_target()

        #self.solve()
        self.controller.target = self.solve()
        skel.set_controller(self.controller)
        print('create controller OK')

        self.contact_force = []

    def generate_spline_trajectory(self, plist):
        ctr = np.array(plist)

        x = ctr[:, 0]
        y = ctr[:, 1]

        l = len(x)
        t = np.linspace(0, 1, l - 2, endpoint=True)
        t = np.append([0, 0, 0], t)
        t = np.append(t, [1, 1, 1])
        tck = [t, [x, y], 3]
        u3 = np.linspace(0, 1, (500), endpoint=True)

        # tck, u = interpolate.splprep([x, y], k=3, s=0)
        # u = np.linspace(0, 1, num=50, endpoint=True)
        out = interpolate.splev(u3, tck)
        # print("out: ", out)
        der = interpolate.splev(u3, tck, der = 1)
        # print("der: ", der)

        # print("x", out[0])
        # print("y", out[1])

        return out, der

    def update_target(self, ):
        skel1 = self.skeletons[1]
        # print("self.gtime:", self.gtime)
        # self.target_foot = np.array([skel1.q["j_cart_x"], -0.92, skel1.q["j_cart_z"]])

        # ground_height = -1.1
        ground_height = -0.92
        self.target_left_foot = np.array([skel1.q["j_cart_x"], ground_height, self.left_foot_traj[1][self.gtime]-0.05-0.3])
        self.target_right_foot = np.array([skel1.q["j_cart_x"], ground_height, self.right_foot_traj[1][self.gtime]+0.05+0.3])
        # l = self.skeletons[1].body('h_pole').local_com()[1]
        l = self.pendulum.length
        self.target_pelvis = np.array([skel1.q["j_cart_x"] - l*math.sin(self.pendulum.theta), l*math.cos(self.pendulum.theta), 0])

        # # Update PD-control target
        # self.target = self.skeletons[2].positions()
        # self.target[("j_bicep_left_x", "j_bicep_right_x")] = 1.5, -1.5
        # self.target[("j_heel_left_2")] = self.left_foot_traj[1][self.gtime] - 0.05
        # self.target[("j_heel_right_2")] = self.right_foot_traj[1][self.gtime] + 0.05

        # print("target", self.target_pelvis)

        # print("target", self.target_pelvis)
        # self.target_pelvis = np.array([0.5, 0.5, 0.])
        # print("target", self.target_pelvis)

    def set_params(self, x):
        q = self.skeletons[2].q
        q = x
        # q[6:] = x
        self.skeletons[2].set_positions(q)

    def f(self, x):

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
        pelvis_state = self.skeletons[2].body("h_pelvis").to_world([0.0, 0.0, 0.0])
        left_foot_state = self.skeletons[2].body("h_heel_left").to_world([0.0, 0.0, 0.0])
        left_foot_ori = self.skeletons[2].body("h_heel_left").world_transform()
        right_foot_state = self.skeletons[2].body("h_heel_right").to_world([0.0, 0.0, 0.0])
        right_foot_ori = self.skeletons[2].body("h_heel_right").world_transform()

        # print("foot_ori: ", left_foot_ori[:3, :3], type(left_foot_ori[:3, :3]))

        # todo : calculate the x_axis according to the spline

        # x_axis = np.array([0., 0., self.left_der[0][self.gtime]])
        x_axis = np.array([1.0, 0., 1.0])
        x_axis = x_axis / np.linalg.norm(x_axis)            #normalize
        y_axis = np.array([0., 1., 0.])
        z_axis = np.cross(x_axis, y_axis)
        z_axis = z_axis / np.linalg.norm(z_axis)            #normalize
        R_des = np.stack((x_axis, y_axis, z_axis), axis=-1)
        # print("x_axis: \n")
        # print(x_axis)
        # print("y_axis: \n")
        # print(y_axis)
        # print("z_axis: \n")
        # print(z_axis)
        # print("R: \n")
        # print(R_des)

        # left_axis_angle =rotationMatrixToEulerAngles(left_foot_ori[:3, :3])
        # right_axis_angle = rotationMatrixToEulerAngles(right_foot_ori[:3, :3])

        left_res_rot = np.transpose(R_des).dot(left_foot_ori[:3, :3])
        right_res_rot = np.transpose(R_des).dot(right_foot_ori[:3, :3])
        # print("vec res: ", left_res_rot)
        left_axis_angle =rotationMatrixToEulerAngles(left_res_rot)
        right_axis_angle = rotationMatrixToEulerAngles(right_res_rot)

        # print("axis_angle", axis_angle)

        # return 0.5 *  np.linalg.norm(left_foot_state - self.target_foot) ** 2
        return 0.5 * (np.linalg.norm(pelvis_state - self.target_pelvis) ** 2 + np.linalg.norm(left_foot_state - self.target_left_foot) ** 2 + np.linalg.norm(right_foot_state - self.target_right_foot) ** 2 + np.linalg.norm(left_axis_angle - np.array([0., 0., 0.])) ** 2 + np.linalg.norm(right_axis_angle - np.array([0., 0., 0.])) ** 2)

    def g(self, x):
        self.set_params(x)

        pelvis_state = self.skeletons[2].body("h_pelvis").to_world([0.0, 0.0, 0.0])
        left_foot_state = self.skeletons[2].body("h_heel_left").to_world([0.0, 0.0, 0.0])
        right_foot_state = self.skeletons[2].body("h_heel_right").to_world([0.0, 0.0, 0.0])

        J_pelvis = self.skeletons[2].body("h_pelvis").linear_jacobian()
        J_left_foot = self.skeletons[2].body("h_heel_left").linear_jacobian()
        J_right_foot = self.skeletons[2].body("h_heel_right").linear_jacobian()

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
        q_backup = self.skeletons[2].q
        res = minimize(self.f, x0=self.skeletons[2].q, jac=self.g, method="SLSQP")
        # res = minimize(self.f, x0=self.skeletons[2].q[6:], jac=self.g, method="SLSQP")
        # print(res)
        self.skeletons[2].set_positions(q_backup)

        return res['x']

    def step(self):

        # cf = pydart.world.CollisionResult.num_contacts()
        # print("CollisionResult : \n", cf)

        if self.force is not None and self.duration >= 0:
            self.duration -= 1
            self.skeletons[2].body('h_pelvis').add_ext_force(self.force)

        skel = world.skeletons[1]
        q = skel.q

        foot_center = .5 * (world.skeletons[2].body("h_heel_left").com() + world.skeletons[2].body("h_heel_right").com())
        # self.cart.x = world.skeletons[2].body("h_heel_left").to_world([0.0, 0.0, 0.0])[0]
        self.cart.x = foot_center[0]
        self.cart.x_dot = world.skeletons[2].body("h_heel_left").com_linear_velocity()[0]
        com_pos = world.skeletons[2].com()
        # v1 = com_pos - world.skeletons[2].body("h_heel_left").to_world([0.0, 0.0, 0.0])
        v1 = com_pos - foot_center
        v2 = np.array([0., 1., 0.])
        v1 = v1 / np.linalg.norm(v1)    # normalize
        self.pendulum.theta = np.sign(-v1[0]) * math.acos(v1.dot(v2))
        self.pendulum.theta_dot = self.pendulum.length * (world.skeletons[2].com_velocity()[0] - self.cart.x_dot) * math.cos(self.pendulum.theta)

        self.state[0] = self.cart.x
        self.state[1] = self.cart.x_dot
        self.state[2] = self.pendulum.theta
        self.state[3] = self.pendulum.theta_dot

        q["j_cart_x"] = self.state[0]
        q["j_cart_z"] = 0.
        # q["default"] = -0.92
        q["j_pole_x"] = 0.
        q["j_pole_y"] = 0.
        q["j_pole_z"] = self.state[2]

        skel.set_positions(q)

        F = find_lqr_control_input(self.cart, self.pendulum, -9.8)
        apply_control_input(self.cart, self.pendulum, F, 0.01, self.state[3], -9.8)

        self.state[0] = self.cart.x
        self.state[1] = self.cart.x_dot
        self.state[2] = self.pendulum.theta
        self.state[3] = self.pendulum.theta_dot

        q["j_cart_x"] = self.state[0]
        q["j_cart_z"] = 0.
        # q["default"] = 0.
        q["j_pole_x"] = 0.
        q["j_pole_y"] = 0.
        q["j_pole_z"] = self.state[2]

        # print("q", q)

        # if self.gtime < 250:
        #     world.skeletons[2].q["j_pelvis_rot_z"] = -0.2
        #     print(self.gtime)

        #solve ik
        if (len(self.left_foot_traj[0]) -1) > self.gtime:
            self.gtime = self.gtime + 1
        else:
            print("TRAJECTORY OVER!!!!")
            self.gtime = self.gtime

        self.update_target()
        self.controller.target = self.solve()
        # print(self.controller.target)

        # print("self.controller.sol_lambda: \n", self.controller.sol_lambda)

        # f= []
        # print("self.controller.V_c", type(self.controller.V_c), self.controller.V_c)
        # for n in range(len(self.controller.sol_lambda) // 4):
        #     f = self.controller.V_c.dot(self.controller.sol_lambda)

        if len(self.controller.sol_lambda) != 0:
            f_vec = self.controller.V_c.dot(self.controller.sol_lambda)
            # print("f", f_vec)

            f_vec = np.asarray(f_vec)

            # self.contact_force = np.zeros(self.controller.contact_num)
            for ii in range(self.controller.contact_num):
                self.contact_force.append(np.array([f_vec[3*ii], f_vec[3*ii+1], f_vec[3*ii+2]]))
                # self.contact_force[ii] = np.array([f_vec[3*ii], f_vec[3*ii+1], f_vec[3*ii+2]])
                # print("contact_force:", self.contact_force[ii])

            for ii in range(len(f_vec) // 3):
                self.skeletons[2].body(self.controller.contact_list[2 * ii])\
                    .add_ext_force(self.contact_force[ii], self.controller.contact_list[2 * ii+1])

        super(MyWorld, self).step()

        skel.set_positions(q)


    def on_key_press(self, key):
        if key == '1':
            self.force = np.array([500.0, 0.0, 0.0])
            self.duration = 1000
            print('push backward: f = %s' % self.force)
        elif key == '2':
            self.force = np.array([-50.0, 0.0, 0.0])
            self.duration = 100
            print('push backward: f = %s' % self.force)

    def render_with_ri(self, ri):
        if self.force is not None and self.duration >= 0:
            p0 = self.skeletons[2].body('h_pelvis').C
            p1 = p0 + 0.01 * self.force
            ri.set_color(1.0, 0.0, 0.0)
            ri.render_arrow(p0, p1, r_base=0.05, head_width=0.1, head_len=0.1)
        # ri.render_sphere(self.mPendulum1.calcCOMpos(), 0.05)        #mPendulum1 COM position

        # ri.render_sphere(np.array([2.0, 0.0, 0.0]), 0.05)
        # ri.render_sphere(np.array([-2.0, 0.0, 0.0]), 0.05)

        # for ii in range(len(self.desiredTraj1)):
        #     ri.render_sphere([self.desiredTraj1[ii, 0], 0.2, 0], 0.01)
        # ri.render_sphere([self.desiredTraj1[2000, 0], 0.2, 0], 0.02)
        ri.render_sphere([2.0, -0.90, 0], 0.05)
        ri.render_sphere([0.0, -0.90, 0], 0.05)

        # render contact force --yul

        # ri.set_color(1.0, 1.0, 1.0)
        # ri.render_sphere(skel.body('h_blade_right').to_world([-0.1040+0.0216, +0.80354016-0.85354016, 0.0]), 0.01)
        # ri.render_sphere(skel.body('h_blade_right').to_world([0.1040+0.0216, +0.80354016-0.85354016, 0.0]), 0.01)

        # ri.render_sphere(skel.body('h_blade_right').to_world([-0.1040 + 0.0216, -0.07, 0.0]), 0.01)
        # ri.render_sphere(skel.body('h_blade_right').to_world([0.1040 + 0.0216, -0.07, 0.0]), 0.01)

        # print("heel", self.skeletons[2].body("h_heel_left").to_world())
        contact_force = np.asarray(self.contact_force)

        if contact_force.size != 0:
            ri.set_color(1.0, 0.0, 0.0)
            for ii in range(self.controller.contact_num):
                print("contact force : ", contact_force[ii])
                ri.render_line(self.skeletons[2].body(self.controller.contact_list[2*ii]).
                               to_world(self.controller.contact_list[2*ii+1]), contact_force[ii])

        for ctr_n in range(0, len(self.left_foot_traj[0])-1, 10):
            ri.set_color(0., 1., 0.)
            ri.render_sphere(np.array([self.left_foot_traj[0][ctr_n], -0.9, self.left_foot_traj[1][ctr_n]]), 0.01)
            ri.set_color(0., 0., 1.)
            ri.render_sphere(np.array([self.right_foot_traj[0][ctr_n], -0.9, self.right_foot_traj[1][ctr_n]]), 0.01)
        # ri.render_sphere(np.array([self.out_spline[0][10], -0.9, self.out_spline[1][10]]), 0.03)
        # ri.render_sphere(np.array([self.out_spline[0][20], -0.9, self.out_spline[1][20]]), 0.03)
        # ri.render_sphere(np.array([self.out_spline[0][30], -0.9, self.out_spline[1][30]]), 0.03)
        # ri.render_sphere(np.array([self.out_spline[0][40], -0.9, self.out_spline[1][40]]), 0.03)

        # render axes
        ri.render_axes(np.array([0, 0, 0]), 0.5)
        # ri.set_color(0., 0., 0.)
        # ri.render_sphere(np.array([0, 0, 0]), 0.01)
        # ri.set_color(1.0, 0., 0.)
        # ri.render_line(np.array([0, 0, 0]), np.array([0.3, 0, 0]))
        # ri.set_color(0.0, 1., 0.)
        # ri.render_line(np.array([0, 0, 0]), np.array([0.0, 0.3, 0]))
        # ri.set_color(0.0, 0., 1.)
        # ri.render_line(np.array([0, 0, 0]), np.array([0.0, 0, 0.3])


if __name__ == '__main__':
    print('Example: test IPC in 3D')

    pydart.init()
    print('pydart initialization OK')

    world = MyWorld()
    print('MyWorld  OK')

    skel = world.skeletons[2]

    # enable the contact of skeleton
    # skel.set_self_collision_check(1)
    # world.skeletons[0].body('ground').set_collidable(False)

    q = skel.q

    # q["j_pelvis_pos_y"] = -0.05
    # q["j_pelvis_rot_y"] = -0.2
    # q["j_pelvis_rot_z"] = -0.1
    # q["j_thigh_left_z", "j_shin_left", "j_heel_left_1"] = 0.15, -0.4, 0.25
    # q["j_thigh_right_z", "j_shin_right", "j_heel_right_1"] = 0.15, -0.4, 0.25

    q["j_thigh_left_z", "j_shin_left", "j_heel_left_1"] = 0.30, -0.5, 0.25
    q["j_thigh_right_z", "j_shin_right", "j_heel_right_1"] = 0.30, -0.5, 0.25

    q["j_heel_left_2"] = 0.7
    q["j_heel_right_2"] = -0.7

    q["j_abdomen_2"] = 0.0
    # both arm T-pose
    # q["j_bicep_left_x", "j_bicep_left_y", "j_bicep_left_z"] = 1.5, 0.0, 0.0
    # q["j_bicep_right_x", "j_bicep_right_y", "j_bicep_right_z"] = -1.5, 0.0, 0.0

    q["j_pelvis_rot_z"] = -0.2

    skel.set_positions(q)
    print('skeleton position OK')

    # print('[Joint]')
    # for joint in skel.joints:
    #     print("\t" + str(joint))
    #     print("\t\tparent = " + str(joint.parent_bodynode))
    #     print("\t\tchild = " + str(joint.child_bodynode))
    #     print("\t\tdofs = " + str(joint.dofs))

    # Set joint damping
    # for dof in skel.dofs:
    #     dof.set_damping_coefficient(80.0)

    pydart.gui.viewer.launch_pyqt5(world)