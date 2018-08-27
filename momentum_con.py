import numpy as np
import math
import PyCommon.modules.Math.mmMath as mm

class momentum_control(object):
    def __init__(self, skel, ref_skel, h):
        self.skel = skel
        self.ref_skel = ref_skel
        self.target = None
        self.ndofs = self.skel.ndofs
        self.h = h

        self.contact_num = 0

        # self.Kp = np.diagflat([0.0] * 6 + [400.0] * (self.ndofs - 6))
        self.Kp = np.diagflat([0.0] * 6 + [25.0] * (self.ndofs - 6))
        # kp = 4000
        # kd <= 2 * math.sqrt(kp)
        # self.Kd = np.diagflat([0.0] * 6 + [10.0] * (self.ndofs - 6))

        self.Kd = np.diagflat([0.0] * 6 + [2 * np.sqrt(25.0)] * (self.ndofs - 6))

        self.pre_v = np.array([0.0, 0.0, 0.0])
        self.pre_p = np.array([0.0, 0.0, 0.0])

        # weight of each objective term
        # self.K_tr = 1000.0
        # self.K_lm = 400.0
        # self.K_am = 400.0

        self.K_tr = 1.
        self.K_lm = 0.1
        self.K_am = 0.13

        # self.K_tr = 100.
        # self.K_lm = 10
        # self.K_am = 1.

        #tracking weight map vector

        # self.weight_map_vec = np.diagflat([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.2, 0.2, 0.5, 0.5, 0.5, 0.5, 0.2, 0.2, 0.6, 0.6, 0.6, 0.6, 0.6, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2])

        # self.weight_map_vec = np.diagflat(
        #     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.1, 0.1, 0.1, 0.25, 0.2, 0.2, 0.6, 0.6, 0.6,
        #      0.6, 0.6, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2])

        # self.weight_map_vec = np.diagflat(
        #     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.1, 0.1, 0.25, 0.2, 0.2, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.5, 0.5, 0.5,
        #      1.0, 1.0, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2])

        self.weight_map_vec = np.diagflat(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
             0.1, 0.1, 0.1, 0.25, 0.25, 0.25, 0.2, 0.2, 0.2,
             0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8,
             0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8,
             0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2,
             0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2])

        # self.weight_map_vec = np.diagflat(
        #     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.1, 0.1, 0.5,
        #      0.8, 0.8, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

        #config['weightMap'] = {'j_scapula_left': .2, 'j_bicep_left': .2, 'j_forearm_left': .2, 'j_hand_left': .2,
                           # 'j_scapula_right': .2, 'j_bicep_right': .2, 'j_forearm_right': .2, 'j_hand_right': .2,
                           # 'j_abdomen': .6, 'j_spine': .6, 'j_head': .6, 'j_heel_right': .2, 'j_heel_left': .2,
                           # 'j_pelvis': 0.5,
                           # 'j_thigh_left': 5., 'j_shin_left': .5, 'j_thigh_right': 5., 'j_shin_right': .5}


    # def check_contact(self):
    #     skel = self.skel
    #
    #     contact_name_list = ['h_blade_left', 'h_blade_left', 'h_blade_right', 'h_blade_right']
    #     offset_list = [[-0.1040+0.0216, +0.80354016-0.85354016, 0.0],
    #                    [0.1040+0.0216, +0.80354016-0.85354016, 0.0],
    #                    [-0.1040 + 0.0216, +0.80354016 - 0.85354016, 0.0],
    #                    [0.1040 + 0.0216, +0.80354016 - 0.85354016, 0.0]]
    #
    #     contact_candi_list = [skel.body(contact_name_list[0]).to_world(offset_list[0]),
    #                     skel.body(contact_name_list[1]).to_world(offset_list[1]),
    #                     skel.body(contact_name_list[2]).to_world(offset_list[2]),
    #                     skel.body(contact_name_list[3]).to_world(offset_list[3])]
    #     contact_list = []
    #     # print("contact_list", contact_candi_list)
    #     cn = 0      # the number of contact point
    #     # print("len(contact_list): ", len(contact_list))
    #     for ii in range(len(contact_candi_list)):
    #         # print(ii, contact_candi_list[ii][1])
    #         if -0.98+0.05/2 + 0.07/2 > contact_candi_list[ii][1]:
    #         # if -0.99 > contact_candi_list[ii][1]:
    #             contact_list.append(contact_name_list[ii])
    #             contact_list.append(offset_list[ii])
    #             cn = cn + 1
    #
    #     self.contact_num = cn
    #     return contact_list

    def check_contact(self):
        skel = self.skel

        contact_name_list = ['h_blade_left', 'h_blade_left', 'h_blade_left', 'h_blade_left']
        offset_list = [[-0.1040+0.0216, +0.80354016-0.85354016, 0.054],
                       [-0.1040 + 0.0216, +0.80354016 - 0.85354016, -0.054],
                       [0.1040+0.0216, +0.80354016-0.85354016, 0.054],
                       [0.1040 + 0.0216, +0.80354016 - 0.85354016, -0.054]]

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
            if -0.98+0.05/2 + 0.07/2 > contact_candi_list[ii][1]:
            # if -0.99 > contact_candi_list[ii][1]:
                contact_list.append(contact_name_list[ii])
                contact_list.append(offset_list[ii])
                cn = cn + 1

        self.contact_num = cn
        return contact_list

    def compute(self, contact_list):
        skel = self.skel
        ndofs = self.ndofs

        # define output
        acc = np.zeros(skel.ndofs)

        def transformToSkewSymmetricMatrix(v):

            x = v[0]
            y = v[1]
            z = v[2]

            mat = np.zeros((3, 3))
            mat[0, 1] = -z
            mat[0, 2] = y
            mat[1, 2] = -x

            mat[1, 0] = z
            mat[2, 0] = -y
            mat[2, 1] = x

            return mat

        # get desired linear momentum (des_L_dot)

        # k_l = 400
        k_l = 100
        d_l = 2*np.sqrt(k_l)

        # CM_ref = footCenter.copy()
        # CM_ref[1] = dartMotionModel.getCOM()[1] - 0.1

        c_r = skel.body('h_blade_left').to_world([0.0, 0.0, 0.0])
        # c_r = np.array([-0.057, -0.9, -0.09])
        print("ref_cm, cm : ", c_r, skel.C)
        l_p = skel.m * k_l * (c_r - skel.C)
        # l_p[1] = 0.0
        l_d = skel.m * d_l * (skel.com_velocity())
        # l_d = skel.m * d_l * (skel.com_velocity() - np.array([0.0, 0., 0.0]))
        des_L_dot = l_p - l_d
        des_L_dot[1] = 0.


        # print("COM Vel: ", skel.com_velocity())
        # print("l_p: ", l_p)
        # print("l_d: ", l_d)
        # print("des_L_dot: ", des_L_dot)

        def world_cop():

            contacted_bodies = []
            for ii in range(int(len(contact_list) / 2)):
                contact_body_name = contact_list[2 * ii]
                contacted_bodies.append(skel.body(contact_body_name))
            #     print(contact_body_name, "\t")
            # print("======================================")
            # print("contacted_bodies: ", contacted_bodies)
            if len(contacted_bodies) == 0:
                # return None
                return np.array([0.0, 0.0, 0.0])
            pos_list = [b.C for b in contacted_bodies]
            norm_list = [np.linalg.norm(b.C) for b in contacted_bodies]
            numerator = 0
            for id in range(len(contacted_bodies)):
                numerator += norm_list[id]*pos_list[id]
            avg = numerator / sum(norm_list)
            # avg1 = sum(pos_list) / len(pos_list)
            return avg

        #  get desired angular momentum (des_H_dot)
        # k_h = 400
        k_h = 100
        d_h = 2*np.sqrt(k_h)

        # print("self.pre_v: ", self.pre_v)
        # print("self.pre_p: ", self.pre_p)

        pre_v = self.pre_v
        pre_p = self.pre_p

        pre_v = (world_cop() - pre_p)/self.h
        self.pre_p = world_cop()
        p_r_dot = np.zeros(3)
        # calculate p_dot by finite difference
        # p_dot = np.zeros((1, 3))
        p_dot = (skel.body("h_blade_left").to_world([0., 0., 0.]) - pre_p) / self.h
        # print("pre_p: ", pre_p)
        # print("body posi: ", skel.body("h_blade_right").to_world([0., 0., 0.]))
        # print("p_dot: ", p_dot)

        # print("skel.COP: ", world_cop())
        # p_r = skel.body("h_blade_left").to_world([0., 0., 0.])
        p_r = c_r
        d_des_two_dot = k_h * (p_r - world_cop()) + d_h * (p_r_dot - p_dot)
        # p_des = np.zeros(3)
        # integrate p_des
        # p_v = pre_v + d_des_two_dot * self.h
        # p_des = pre_p + p_v * self.h
        p_des = 1/2*d_des_two_dot*self.h*self.h + pre_v*self.h + pre_p
        des_H_dot = np.cross(p_des - skel.C, des_L_dot - skel.m * np.array([0.0, -9.81, 0.0]))
        des_H_dot[1] = 0.

        # self.pre_p = skel.body("h_blade_right").to_world([0., 0., 0.])
        # calculate momentum derivative matrices : R, S, r_bias, s_bias
        # [R S]_transpose = PJ

        Pmat = np.zeros((6, 6 * skel.num_bodynodes()))
        J = np.zeros((3 * 2 * skel.num_bodynodes(), skel.ndofs))
        Pmat_dot = np.zeros((6, 6 * skel.num_bodynodes()))
        J_dot = np.zeros((3 * 2 * skel.num_bodynodes(), skel.ndofs))
        # print("mm", skel.M)

        T = np.zeros((3, 3 * skel.num_bodynodes()))
        U = np.zeros((3, 3 * skel.num_bodynodes()))
        Udot = np.zeros((3, 3 * skel.num_bodynodes()))
        V = np.zeros((3, 3 * skel.num_bodynodes()))
        Vdot = np.zeros((3, 3 * skel.num_bodynodes()))

        #todo: replace jacobian
        for bi in range(skel.num_bodynodes()):
            r_v = skel.body(bi).to_world([0, 0, 0]) - skel.C
            r_m = transformToSkewSymmetricMatrix(r_v)
            T[:, 3 * bi:3 * (bi + 1)] = skel.body(bi).mass() * np.identity(3)
            U[:, 3 * bi:3 * (bi + 1)] = skel.body(bi).mass() * r_m
            v_v = skel.body(bi).dC - skel.dC
            v_m = transformToSkewSymmetricMatrix(v_v)
            Udot[:, 3 * bi:3 * (bi + 1)] = skel.body(bi).mass() * v_m
            V[:, 3 * bi:3 * (bi + 1)] = skel.body(bi).inertia()

            body_w_v = skel.body(bi).world_angular_velocity()
            body_w_m = transformToSkewSymmetricMatrix(body_w_v)
            # print("w: ", skel.body(bi).world_angular_velocity())
            Vdot[:, 3 * bi:3 * (bi + 1)] = np.dot(body_w_m, skel.body(bi).inertia())

            J[3 * bi:3 * (bi + 1), :] = skel.body(bi).linear_jacobian([0, 0, 0])
            J[3 * skel.num_bodynodes() + 3 * bi: 3 * skel.num_bodynodes() + 3 * (bi + 1), :] = skel.body(
                bi).angular_jacobian()

            J_dot[3 * bi:3 * (bi + 1), :] = skel.body(bi).linear_jacobian_deriv([0, 0, 0])
            J_dot[3 * skel.num_bodynodes() + 3 * bi: 3 * skel.num_bodynodes() + 3 * (bi + 1), :] = skel.body(
                bi).angular_jacobian_deriv()

        # print("r", r)
        Pmat[0:3, 0: 3 * skel.num_bodynodes()] = T
        Pmat[3:, 0:3 * skel.num_bodynodes()] = U
        Pmat[3:, 3 * skel.num_bodynodes():] = V

        Pmat_dot[3:, 0:3 * skel.num_bodynodes()] = Udot
        Pmat_dot[3:, 3 * skel.num_bodynodes():] = Vdot

        PJ = np.dot(Pmat, J)
        R = PJ[0:3, :]
        S = PJ[3:6, :]

        pdotJ_pJdot = (np.dot(Pmat_dot, J) + np.dot(Pmat, J_dot))
        bias = np.dot(pdotJ_pJdot, skel.dq)
        # print("bias", bias)

        r_bias = bias[0:3]
        s_bias = bias[3:6]

        # get desired acceleration
        invM = np.linalg.inv(skel.M + self.Kd * self.h)
        p = -self.Kp.dot(skel.q - self.target + skel.dq * self.h)
        d = -self.Kd.dot(skel.dq)
        qddot = invM.dot(-skel.c + p + d + skel.constraint_forces())
        des_accel = p + d + qddot

        K_tr = self.K_tr
        K_lm = self.K_lm
        K_am = self.K_am

        #=====================
        # Objective function
        # ====================

        A = np.zeros((2*ndofs+6, 2*ndofs))
        middle_p = np.zeros((ndofs, ndofs))
        middle_p[0:ndofs, 0:ndofs] = K_tr * self.weight_map_vec + K_lm * np.dot(R.transpose(), R) + K_am * np.dot(S.transpose(), S)
        P = 2 * middle_p
        q = np.zeros(ndofs)
        q = -2 * K_tr * self.weight_map_vec.dot(des_accel) + 2 * K_lm * (np.dot(r_bias, R) - np.dot(des_L_dot, R)) + 2 * K_am * (np.dot(s_bias, S) - np.dot(des_H_dot, S))
        # print("P size: ", P.size)
        # print("q size: ", q.size)

        # ====================
        # Equality constraint
        # ====================
        # np.dot(jaco_L, skel.ddq) + np.dot(jaco_der_L, skel.dq) = a_sup
        a_sup = np.zeros(6)
        a_sup_k = 40.
        a_sup_d = 2.*(a_sup_k**.5)
        # print(a_sup_k, a_sup_d)
        # print("1: ",np.array([0.0, 0.0, 0.0]) - skel.body('h_blade_left').to_world([0.0, 0, 0]))
        # print("2: ",skel.body('h_blade_left').world_linear_velocity())

        a_R = skel.body('h_blade_left').world_transform()
        a_R_mat = a_R[:3, :3]

        a_R_ref = self.ref_skel.body('h_blade_left').world_transform()
        a_R_ref_mat = a_R_ref[:3, :3]

        res_mat = np.dot(a_R_ref_mat.transpose(), a_R_mat).T

        # print("a_r:", a_R[:3,:3])
        # print("r_diff:", r_diff)

        def rotationMatrixToAxisAngles(R):
            temp_r = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
            angle = math.acos((R[0, 0] + R[1, 1] + R[2, 2] - 1) / 2.)
            temp_r_norm = np.linalg.norm(temp_r)
            if temp_r_norm < 0.000001:
                return np.zeros(3)

            return temp_r / temp_r_norm * angle

        r_diff = rotationMatrixToAxisAngles(res_mat)
        # r_diff = mm.logSO3(mm.getSO3FromVectors(np.dot(a_R_mat, np.array([0,1,0])), np.array([0,1,0])))

        a_sup[0:3] = a_sup_k * r_diff - a_sup_d * skel.body('h_blade_left').world_angular_velocity()

        # ref_pos = self.ref_skel.body('h_blade_left').to_world([0.0, 0.0, 1.8])
        # print("ref foot pos: ", ref_pos)
        # ref_pos = np.array([0.01649408,-0.98 + 0.025, -0.09072832])

        # ref_pos = c_r
        ref_pos = np.array([-0., -0.98, -0.1])

        a_sup[3:6] = a_sup_k * (
                ref_pos - skel.body('h_blade_left').to_world([0.0, 0, 0])) - a_sup_d * skel.body(
            'h_blade_left').world_linear_velocity()

        #solve linear equations

        n = ndofs+6     # matrix_dimension
        A = np.zeros((n, n))
        b = np.zeros(n)

        Am = skel.body("h_blade_left").world_jacobian()
        b_vec = a_sup - np.dot(skel.body("h_blade_left").world_jacobian_classic_deriv(), skel.dq)


        # print("a_sup:", a_sup)
        # print("Am: ", Am[ndofs + 6 + 2:, ndofs:2 * ndofs])
        # print("b_vec: ", b_vec[ndofs + 6 + 2:])

        A[0:ndofs, 0:ndofs] = P
        A[0:ndofs, ndofs:ndofs+6] = Am.transpose()
        A[ndofs:ndofs+6, 0:ndofs] = Am
        b[0:ndofs] = -q
        b[ndofs:ndofs+6] = b_vec

        acc = np.linalg.solve(A, b)
        # acc = np.linalg.svd()
        # print("Desired acc: ", acc[0:35])
        return acc[0:ndofs]