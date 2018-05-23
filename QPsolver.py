import numpy as np
import math
from cvxopt import matrix, solvers
from pydart2.skeleton import Skeleton

class Controller(object):
    def __init__(self, skel, h):
        self.h = h
        self.skel = skel  # type: Skeleton
        ndofs = self.skel.ndofs
        self.target = None

        self.contact_num = 0

        self.V_c = []
        self.V_c_dot = []
        self.sol_tau = []
        self.sol_accel = []
        self.sol_lambda = []

        self.pre_tau = np.zeros(skel.ndofs)

        self.Kp = np.diagflat([0.0] * 6 + [4000.0] * (ndofs - 6))
        # kp = 4000
        # kd <= 2 * math.sqrt(kp)
        self.Kd = np.diagflat([0.0] * 6 + [100.0] * (ndofs - 6))

        # coefficient of each term
        self.K_ef = 10.0
        self.K_tr = 1000.0
        self.K_cf = 10000.0
        # self.K_cf = 5000.0

        self.preoffset = 0.0
        self.preoffset_pelvis = np.zeros(2)

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
            if -0.92 +0.03 > contact_candi_list[ii][1]:
            # if -0.92 > contact_candi_list[ii][1]:
                contact_list.append(contact_name_list[ii])
                contact_list.append(offset_list[ii])
                cn = cn + 1

        self.contact_num = cn
        return contact_list

    def get_foot_info(self, body_name):
        skel = self.skel
        jaco = skel.body(body_name).linear_jacobian()
        jaco_der = skel.body(body_name).linear_jacobian_deriv()

        p1 = skel.body(body_name).to_world([-0.1040 + 0.0216, 0.0, 0.0])
        p2 = skel.body(body_name).to_world([0.1040 + 0.0216, 0.0, 0.0])

        if body_name == "h_blade_right":
            blade_direction_vec = p2 - p1
        else:
            blade_direction_vec = p1 - p2
        blade_direction_vec = blade_direction_vec / np.linalg.norm(blade_direction_vec)

        theta = math.acos(np.dot(np.array([1., 0., 0.]), blade_direction_vec))
        # print("theta: ", theta)
        # print("omega: ", skel.body("h_blade_left").world_angular_velocity()[1])
        next_step_angle = theta + skel.body(body_name).world_angular_velocity()[1] * self.h
        # print("next_step_angle: ", next_step_angle)
        sa = math.sin(next_step_angle)
        ca = math.cos(next_step_angle)

        return jaco, jaco_der, sa, ca

    def compute(self):
        # goal : get acceleration and lambda to satisfy the objectives and constraints below

        skel = self.skel
        ndofs = self.skel.ndofs

        contact_list = self.check_contact()
        self.contact_list = contact_list
        contact_num = self.contact_num
        # print("The number of contact is ", contact_num)

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
            # v1 = v1 / np.linalg.norm(v1)
            # v2 = np.array([0., 1., 0.])
            # v2 = v2 / np.linalg.norm(v2)
            # v3 = np.array([0., 1., 0.])
            # v3 = v3 / np.linalg.norm(v3)
            # v4 = np.array([0., 1., 0.])
            # v4 = v4 / np.linalg.norm(v4)

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

            # compensate the velocity at the moment colliding the ground

            omega = []
            V_c_dot_stack = []
            compensate_depth = []
            compensate_vel = []

            for ii in range(self.contact_num):
                contact_body_name = contact_list[2*ii]
                contact_offset = contact_list[2 * ii + 1]
                angular_vel = skel.body(contact_body_name).world_angular_velocity()
                omega.append(angular_vel)

                v1_dot = np.cross(v1, omega[ii])
                if np.linalg.norm(v1_dot) != 0:
                    v1_dot = v1_dot / np.linalg.norm(v1_dot)  # normalize
                v2_dot = np.cross(v2, omega[ii])
                if np.linalg.norm(v2_dot) != 0:
                    v2_dot = v2_dot / np.linalg.norm(v2_dot)  # normalize
                v3_dot = np.cross(v3, omega[ii])
                if np.linalg.norm(v3_dot) != 0:
                    v3_dot = v3_dot / np.linalg.norm(v3_dot)  # normalize
                v4_dot = np.cross(v4, omega[ii])
                if np.linalg.norm(v4_dot) != 0:
                    v4_dot = v4_dot / np.linalg.norm(v4_dot)  # normalize

                V_c_dot_stack.append(np.array([v1_dot, v2_dot, v3_dot, v4_dot]))

                #calculate the penetraction depth : compensate depth instead of velocity
                compensate_depth.append(skel.body(contact_body_name).to_world(contact_offset)[1]+0.92)

                compensate_vel.append(skel.body(contact_body_name).world_linear_velocity()[1] * self.h)

            # print("depth: ", compensate_depth)
            # print("velocity: ", compensate_vel)
            # print("omega: \n", omega)
            # print("V_c_dot :\n", V_c_dot_stack)

            compensate_depth_vec = np.zeros(4 * contact_num)
            compensate_vel_vec = np.zeros(4*contact_num)
            for n in range(self.contact_num):
                self.V_c_dot[n*3:(n+1)*3, n*4:(n+1)*4] = np.transpose(V_c_dot_stack[n])
                compensate_vel_vec[n*4:(n+1)*4] = compensate_vel[n]
                compensate_depth_vec[n*4:(n+1)*4] = compensate_depth[n]

            V_c_dot = self.V_c_dot
            V_c_t_dot_J_c = np.transpose(V_c_dot).dot(np.transpose(J_c_t))
            hv2 = V_c_t_dot_J_c.dot(skel.dq)

            # print("hv1", hv1)
            # print("hv2", hv2)
            # print("compensate_vel: \n", compensate_vel_vec)

            compensate_gain = 10
            # hv[4*contact_num:] = hv1 + hv2 + compensate_vel_vec
            hv[4 * contact_num:] = hv1 + hv2 + compensate_gain*compensate_depth_vec # + compensate_vel_vec
            h = matrix(hv)
            # print("h :\n", h)

        # ================================================================
        # Equality constraint
        # (1) motion of equation                       | ndofs
        # (2) tau[0:6] = 0                             | 6
        # (3) non-holonomic constraint : Knife-edge    | 2 (left, right)
        # comment (4) balancing                                | 2 (l_foot, r_foot)
        # ================================================================
        jaco_L, jaco_der_L, sa_L, ca_L = self.get_foot_info("h_blade_left")
        jaco_R, jaco_der_R, sa_R, ca_R = self.get_foot_info("h_blade_right")

        # Check the balance
        COP = skel.body('h_heel_left').to_world([0.05, 0, 0])
        offset = skel.C[0] - COP[0] + 0.03
        preoffset = self.preoffset
        preoffset_pelvis = self.preoffset_pelvis
        offset_x = skel.C[0] - COP[0]
        offset_z = skel.C[2] - COP[2]
        offset_pelvis = np.array([offset_x, offset_z])

        # Adjust the target pose -- translated from bipedStand app of DART
        foot = skel.dof_indices(["j_heel_left_1", "j_heel_right_1"])
        # pelvis = skel.dof_indices(["j_pelvis_rot_x", "j_pelvis_rot_z"])
        # print("index: ", foot)

        # Low-stiffness
        k1, kd = 20.0, 10.0

        is_non_holonomic = 1

        if is_non_holonomic == 1:
            # print("NON-HOLONOMIC!!!")
            if contact_num != 0:
                Am = np.zeros((ndofs+6+2, ndofs * 2 + 4 * contact_num))
                Am[0:ndofs, 0:ndofs] = -1 * np.identity(ndofs)
                Am[0:ndofs, ndofs:2 * ndofs] = skel.M
                J_c_t_V_c = J_c_t.dot(V_c)
                # print("J_c_t_V_c :\n", J_c_t_V_c)
                # print("size of J_c_t_V_c :", len(J_c_t_V_c))
                Am[0:ndofs, 2*ndofs:] = -1 * J_c_t_V_c
                Am[ndofs:ndofs+6, 0:6] = np.eye(6)
                Am[ndofs + 6:ndofs + 6 + 1, ndofs:2*ndofs] = np.dot(np.array([sa_L, 0., -1 * ca_L]), jaco_L)
                Am[ndofs + 6 + 1:, ndofs:2 * ndofs] = np.dot(np.array([sa_R, 0., -1 * ca_R]), jaco_R)
                # Am[ndofs + 6 + 2: ndofs + 6 + 3, skel.dof_indices(["j_heel_left_1"])] = np.ones(1)
                # Am[ndofs + 6 + 3:, skel.dof_indices(["j_heel_right_1"])] = np.ones(1)
            else:
                Am = np.zeros((ndofs+6+2, ndofs * 2))
                Am[0:ndofs, 0:ndofs] = -1 * np.identity(ndofs)
                Am[0:ndofs, ndofs:2 * ndofs] = skel.M
                Am[ndofs:ndofs+6, 0:6] = np.eye(6)
                Am[ndofs + 6:ndofs + 6+1, ndofs:] = np.dot(np.array([sa_L, 0., -1*ca_L]), jaco_L)
                Am[ndofs + 6 + 1:, ndofs:] = np.dot(np.array([sa_R, 0., -1 * ca_R]), jaco_R)
                # Am[ndofs + 6 + 2: ndofs + 6 + 3, skel.dof_indices(["j_heel_left_1"])] = np.ones(1)
                # Am[ndofs + 6 + 3:, skel.dof_indices(["j_heel_right_1"])] = np.ones(1)


            b_vec = np.zeros(ndofs + 6 + 2)
            b_vec[0:ndofs] = -1 * skel.c
            b_vec[ndofs+6:ndofs+6+1] = (np.dot(jaco_L, skel.dq) + np.dot(jaco_der_L, skel.dq))[2]*ca_L - \
                                       (np.dot(jaco_L, skel.dq) + np.dot(jaco_der_L, skel.dq))[0] * sa_L
            b_vec[ndofs + 6+1:] = (np.dot(jaco_R, skel.dq) + np.dot(jaco_der_R, skel.dq))[2] * ca_R - \
                                  (np.dot(jaco_R, skel.dq) + np.dot(jaco_der_R, skel.dq))[0] * sa_R
            # b_vec[ndofs + 6 + 2:ndofs + 6 + 3] = self.pre_tau[skel.dof_indices(["j_heel_left_1"])] + -k1 * offset + kd * (preoffset - offset)
            # b_vec[ndofs + 6 + 3:ndofs + 6 + 4] = self.pre_tau[skel.dof_indices(["j_heel_right_1"])] + -k1 * offset + kd * (preoffset - offset)
            # if offset > 0.03:
            #     b_vec[ndofs + 6 + 2:ndofs + 6 + 3] = b_vec[ndofs + 6 + 2:ndofs + 6 + 3] + 0.3 * np.array([-10.0])
            #     # b_vec[ndofs + 6 + 3:ndofs + 6 + 4] = b_vec[ndofs + 6 + 3:ndofs + 6 + 4] + 0.3 * np.array([-10.0])
            #     print("Discrete A:", offset)
            # if offset < -0.02:
            #     b_vec[ndofs + 6 + 2:ndofs + 6 + 3] = b_vec[ndofs + 6 + 2:ndofs + 6 + 3] - 1.0 * np.array([-10.0])
            #     # b_vec[ndofs + 6 + 3:ndofs + 6 + 4] = b_vec[ndofs + 6 + 3:ndofs + 6 + 4] - 1.0 * np.array([-10.0])
            #     print("Discrete B:", offset)
            # self.preoffset = offset

        else:
            # print("HOLONOMIC!!!")
            if contact_num != 0:
                Am = np.zeros((ndofs+6, ndofs * 2 + 4 * contact_num))
                Am[0:ndofs, 0:ndofs] = -1 * np.identity(ndofs)
                Am[0:ndofs, ndofs:2 * ndofs] = skel.M
                J_c_t_V_c = J_c_t.dot(V_c)
                # print("J_c_t_V_c :\n", J_c_t_V_c)
                # print("size of J_c_t_V_c :", len(J_c_t_V_c))
                Am[0:ndofs, 2*ndofs:] = -1 * J_c_t_V_c
                Am[ndofs:ndofs+6, 0:6] = np.eye(6)
            else:
                Am = np.zeros((ndofs+6, ndofs * 2))
                Am[0:ndofs, 0:ndofs] = -1 * np.identity(ndofs)
                Am[0:ndofs, ndofs:2 * ndofs] = skel.M
                Am[ndofs:ndofs+6, 0:6] = np.eye(6)

            b_vec = np.zeros(ndofs + 6)
            b_vec[0:ndofs] = -1 * skel.c

        A = matrix(Am)
        # print("A :\n", A)
        b = matrix(b_vec)
        # print("b: \n", b)

        solvers.options['show_progress'] = False
        if contact_num == 0:
            sol = solvers.qp(P, qq, None, None, A, b)
        else:
            sol = solvers.qp(P, qq, G, h, A, b)

        #print(sol['x'])

        solution = np.array(sol['x'])

        self.sol_tau = solution[:skel.ndofs].flatten()
        self.sol_accel = solution[skel.ndofs:2 * skel.ndofs].flatten()
        self.sol_lambda = solution[2 * skel.ndofs:].flatten()

        # self.pre_tau = self.sol_tau
        # # print("self.sol_tau: \n", self.sol_tau)
        #
        # Low-stiffness
        k1, k2, kd = 20.0, 1.0, 10.0
        k = np.array([-k1, -k1])
        self.sol_tau[foot] += k * offset + kd * (preoffset - offset) * np.ones(2)
        # self.sol_tau[pelvis] += k * offset_pelvis + kd * (preoffset_pelvis - offset_pelvis)

        if offset > 0.03:
            self.sol_tau[foot] += 0.3 * np.array([-100.0, -100.0])
            # self.sol_tau[pelvis] += 0.3 * np.array([-100.0, -100.0])
            print("Discrete A")
        if offset < -0.02:
            self.sol_tau[foot] += -1.0 * np.array([-100.0, -100.0])
            # self.sol_tau[pelvis] += -1.0 * np.array([-100.0, -100.0])
            print("Discrete B")
        print("offset = %s" % offset)
        # print("offset_pelvis = %s" % offset_pelvis)
        self.preoffset = offset
        # self.preoffset_pelvis = preoffset_pelvis


        return self.sol_tau
