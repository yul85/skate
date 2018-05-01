import numpy as np
import math
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
        print("+++++++++++++++++++++++++++++++++++")
        print("target:", self.target)
        print("+++++++++++++++++++++++++++++++++++")
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
