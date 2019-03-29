import numpy as np
from cvxopt import solvers, matrix
import pydart2 as pydart
import itertools
import math

from PyCommon.modules.Math import mmMath as mm
from PyCommon.modules.Optimization import csQPOASES

QP_MAX_ITER = 100
QP_EPS = 0.001
QP_PENETRATION_FACTOR = 0.01
QP_RESTITUTION = 1.

QP_CONE_DIM = 4
# MU_x = 1.
# MU_z = 1.
MU_x = 0.02
MU_z = 0.02

LAMBDA_CONTAIN_NORMAL = False
NON_HOLONOMIC = False
NON_HOLONOMIC = True

# FRICTION_CONE_ROTATION = False
FRICTION_CONE_ROTATION = True

QPOASES = False


def calc_QP(skel, ddq_des, ext_f, ddc, lf_tangent, rf_tangent, vc_list, inv_h):
    """
    :param skel:
    :type skel: pydart.Skeleton
    :param ddq_des:
    :type ddq_des: np.ndarray
    :return:
    """
    weight_map_vec = np.diagflat(
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
         0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
         0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
         10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
         10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
         10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0])

    solvers.options['show_progress'] = False

    num_dof = skel.num_dofs()

    bodies = list()  # type: list[pydart.BodyNode]
    position_globals = list()
    position_locals = list()
    vel_globals = list()

    vel_con_list = np.zeros(8)
    is_contact = np.zeros(8)

    for b in range(skel.num_bodynodes()):
        body = skel.body(b)  # type: pydart.BodyNode
        if 'blade' not in body.name:
            continue
        for i in range(body.num_shapenodes()):
            shapenode = body.shapenodes[i]  # type: pydart.ShapeNode
            if shapenode.has_collision_aspect():
                geom_local_T = shapenode.relative_transform()
                shape = shapenode.shape  # type: pydart.BoxShape
                data = shape.size()/2.  # type: np.ndarray

                for perm in itertools.product([1, -1], repeat=3):
                    # print("perm: ", perm)
                    position_local = np.dot(geom_local_T[:3, :3], np.multiply(np.array((data[0], data[1], data[2])), np.array(perm))) + geom_local_T[:3, 3]
                    # print(data)
                    # print(position_local)
                    position_global = body.to_world(position_local)

                    # if position_global[1] < -0.98 + 0.05:
                    if position_global[1] < -0.98 + 0.0251:
                        bodies.append(body)
                        position_locals.append(position_local)
                        position_globals.append(position_global)
                        vel_global = body.world_linear_velocity(position_local)
                        vel_globals.append(vel_global)

                        if 'left' in body.name:
                            if perm == (1, -1, 1):
                                vel_con_list[0] = 1
                                is_contact[0] = 1
                            if perm == (1, -1, -1):
                                vel_con_list[1] = 1
                                is_contact[1] = 1
                            if perm == (-1, -1, 1):
                            #     vel_con_list[2] = 0
                                is_contact[2] = 1
                            if perm == (-1, -1, -1):
                            #     vel_con_list[3] = 0
                                is_contact[3] = 1
                        if 'right' in body.name:
                            if perm == (1, -1, 1):
                                vel_con_list[4] = 1
                                is_contact[4] = 1
                            if perm == (1, -1, -1):
                                vel_con_list[5] = 1
                                is_contact[5] = 1
                            if perm == (-1, -1, 1):
                            #     vel_con_list[6] = 0
                                is_contact[6] = 1
                            if perm == (-1, -1, -1):
                            #     vel_con_list[7] = 0
                                is_contact[7] = 1
    num_contact = len(position_globals)

    num_lambda = (1 + QP_CONE_DIM) * num_contact if LAMBDA_CONTAIN_NORMAL else QP_CONE_DIM * num_contact

    num_tau = num_dof
    num_variable = num_dof + num_tau + num_lambda

    # print("velcon:", vel_con_list)
    # print("is contact: ", is_contact)

    #--------------------------yul------------------------------------
    # tracking COM
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

    # todo: replace jacobian
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
    PdotJ_PJdot = Pmat_dot.dot(J) + Pmat.dot(J_dot)

    #####################################################
    # objective
    #####################################################
    P = np.eye(num_variable)
    P[:num_dof, :num_dof] *= 100.
    # P[:num_dof, :num_dof] *= 100. + 1/skel.m * PJ.transpose().dot(PJ)
    # P[:num_dof, :num_dof] *= 100. + weight_map_vec
    # P[num_dof + num_tau:, num_dof + num_tau:] *= 0.01
    # P[num_dof:num_dof + num_tau, num_dof: num_dof + num_tau] *= 0.001
    q = np.zeros(num_variable)
    # q[:num_dof] = -100.*ddq_des - 30. * (ddc - 1/skel.m * PdotJ_PJdot.dot(skel.dq)).transpose().dot(PJ)
    # q[:num_dof] = -100. * np.dot(ddq_des, weight_map_vec)
    q[:num_dof] = -100. * ddq_des
    #####################################################
    # equality
    #####################################################
    num_equality = num_dof + 6
    if NON_HOLONOMIC:
        if num_contact > 0:
            num_equality = num_dof + 6 + 2

    A = np.zeros((num_equality, num_variable))
    b = np.zeros(num_equality)

    # equations of motion
    # ext_force = np.array((10., 0., 0.)) if False and self.time() > 1. else np.zeros(3)
    ext_force = np.zeros(3)
    if ext_f is not None:
        ext_force = ext_f
        # print("EXTERNAL FORCE ON!!!!!", ext_force)
    M = skel.mass_matrix()
    c = skel.coriolis_and_gravity_forces()

    A[:num_dof, :num_dof] = M
    A[:num_dof, num_dof:num_dof+num_tau] = -np.eye(num_tau)
    b[:num_dof] = -c + np.dot(skel.body('h_pelvis').linear_jacobian().T, ext_force)
    # b[:num_dof] = -c

    # floating base
    A[num_dof:num_dof+6, num_dof:num_dof+6] = np.eye(6)

    _h = 1 / inv_h

    if NON_HOLONOMIC:

        def get_foot_info(body_name):
            jaco = skel.body(body_name).linear_jacobian()
            jaco_der = skel.body(body_name).linear_jacobian_deriv()

            p1 = skel.body(body_name).to_world([-0.1040 + 0.0216, 0.0, 0.0])
            p2 = skel.body(body_name).to_world([0.1040 + 0.0216, 0.0, 0.0])

            blade_direction_vec = p2 - p1
            blade_direction_vec = np.array([1, 0, 1]) * blade_direction_vec
            if np.linalg.norm(blade_direction_vec) != 0:
                blade_direction_vec = blade_direction_vec / np.linalg.norm(blade_direction_vec)

            theta = math.atan2(np.dot(mm.unitX(), blade_direction_vec), np.dot(mm.unitZ(), blade_direction_vec))

            theta_x = math.acos(np.dot(np.array([1.0, 0.0, 0.0]), blade_direction_vec))

            # print("theta: ", body_name, ", ", theta)
            # print("omega: ", skel.body("h_blade_left").world_angular_velocity()[1])
            next_step_angle = theta + skel.body(body_name).world_angular_velocity()[1] * _h
            # print("next_step_angle: ", body_name, next_step_angle)

            sa = math.sin(next_step_angle)
            ca = math.cos(next_step_angle)

            return jaco, jaco_der, sa, ca, theta

            # FOR NON-HOLONOMIC CONSTRAINTS

        jaco_L, jaco_der_L, sa_L, ca_L, theta_L_x = get_foot_info("h_blade_left")
        jaco_R, jaco_der_R, sa_R, ca_R, theta_R_x = get_foot_info("h_blade_right")
        # print("Angular vel_y: ", des_w_y_L, des_w_y_R )
        if num_contact > 0:
            A[num_dof + 6:num_dof + 6 + 1, :num_dof] = np.dot(_h * np.array([ca_L, 0., -sa_L]), jaco_L)
            A[num_dof + 6 + 1:num_dof + 6 + 2, :num_dof] = np.dot(_h * np.array([ca_R, 0., -sa_R]), jaco_R)

            b[num_dof+6  :num_dof+6+1] = -(np.dot(jaco_L, skel.dq) + _h * np.dot(jaco_der_L, skel.dq))[0] * ca_L \
                                    + (np.dot(jaco_L, skel.dq) + _h * np.dot(jaco_der_L, skel.dq))[2] * sa_L
            b[num_dof+6+1:num_dof+6+2] = -(np.dot(jaco_R, skel.dq) + _h * np.dot(jaco_der_R, skel.dq))[0] * ca_R \
                                    + (np.dot(jaco_R, skel.dq) + _h * np.dot(jaco_der_R, skel.dq))[2] * sa_R


    #####################################################
    # inequality
    #####################################################
    num_inequality = 2*num_lambda + (QP_CONE_DIM + 1) * num_contact if LAMBDA_CONTAIN_NORMAL else 2*num_lambda
    G = np.zeros((num_inequality, num_variable))
    h = np.zeros(num_inequality)
    V = np.zeros((3*num_contact, num_lambda))

    if num_contact > 0:
        Jc = np.zeros((3*num_contact, num_dof))
        for i in range(num_contact):
            Jc[3*i:3*i+3, :] = bodies[i].linear_jacobian(position_locals[i])

        dJc = np.zeros((3*num_contact, num_dof))
        for i in range(num_contact):
            dJc[3*i:3*i+3, :] = bodies[i].linear_jacobian_deriv(position_locals[i])

        V = np.zeros((3*num_contact, num_lambda))

        for i in range(num_contact):
            if LAMBDA_CONTAIN_NORMAL:
                V[3*i:3*i+3, (QP_CONE_DIM + 1)*i+0] = np.array((1., 0., 0.))
                V[3*i:3*i+3, (QP_CONE_DIM + 1)*i+1] = np.array((0., 0., -1.))
                V[3*i:3*i+3, (QP_CONE_DIM + 1)*i+2] = np.array((-1., 0., 0.))
                V[3*i:3*i+3, (QP_CONE_DIM + 1)*i+3] = np.array((0., 0., 1.))
                V[3*i:3*i+3, (QP_CONE_DIM + 1)*i+4] = np.array((0., 1., 0.))
            else:
                V[3 * i:3 * i + 3, QP_CONE_DIM * i + 0] = mm.normalize(np.array((MU_x, 1., 0.)))
                V[3*i:3*i+3, QP_CONE_DIM*i+1] = mm.normalize(np.array((0., 0.2, -MU_z)))
                V[3 * i:3 * i + 3, QP_CONE_DIM * i + 2] = mm.normalize(np.array((-MU_x, 1., 0.)))
                V[3*i:3*i+3, QP_CONE_DIM*i+3] = mm.normalize(np.array((0., 0.2, MU_z)))

        # friction cone rotation matrix
        if FRICTION_CONE_ROTATION:
            # print("Friction cone rotation ON!!")
            R_c = np.zeros((3*num_contact, 3*num_contact))

            # p1 = skel.body("h_blade_left").to_world([-0.1040 + 0.0216, 0.0, 0.0])
            # p2 = skel.body("h_blade_left").to_world([0.1040 + 0.0216, 0.0, 0.0])
            #
            # blade_direction_vec = p2 - p1
            # blade_direction_vec = np.array([1, 0, 1]) * blade_direction_vec
            # if np.linalg.norm(blade_direction_vec) != 0:
            #     blade_direction_vec = blade_direction_vec / np.linalg.norm(blade_direction_vec)
            #
            # theta = math.acos(np.dot(np.array([1.0, 0.0, 0.0]), blade_direction_vec))

            contact_foot_list = []
            if is_contact[0] == 1:
                contact_foot_list.append('h_blade_left')
            if is_contact[1] == 1:
                contact_foot_list.append('h_blade_left')
            if is_contact[2] == 1:
                contact_foot_list.append('h_blade_left')
            if is_contact[3] == 1:
                contact_foot_list.append('h_blade_left')
            if is_contact[4] == 1:
                contact_foot_list.append('h_blade_right')
            if is_contact[5] == 1:
                contact_foot_list.append('h_blade_right')
            if is_contact[6] == 1:
                contact_foot_list.append('h_blade_right')
            if is_contact[7] == 1:
                contact_foot_list.append('h_blade_right')

            # print("contact_foot_list: ", contact_foot_list)

            # print("friction_rotation_theta: ", theta_L_x/math.pi * 180., theta_R_x/math.pi * 180.)
            for i in range(num_contact):
                if contact_foot_list[i] == 'h_blade_left':
                    _theta = theta_L_x
                else:
                    _theta = theta_R_x
                R_c[3 * i + 0, 3 * i:3 * i + 3] = np.array((-math.cos(_theta), 0, math.sin(_theta)))
                R_c[3 * i + 1, 3 * i:3 * i + 3] = np.array((0, 1, 0))
                R_c[3 * i + 2, 3 * i:3 * i + 3] = np.array((-math.sin(_theta), 0, math.cos(_theta)))

        #####################################################
        # equality
        #####################################################
        JcT_V = np.dot(Jc.T, V)

        # ext force
        A[:num_dof, num_dof+num_tau:] = -JcT_V

        if FRICTION_CONE_ROTATION:
            RV = np.dot(R_c, V)
            JcT_RV = np.dot(Jc.T, RV)
            A[:num_dof, num_dof + num_tau:] = -JcT_RV

        # print(des_w_y_L, des_w_y_R)
        # if num_contact > 0:
        #     # change the blade's direction
        #     Kw = 0.0001
        #
        #     lbody = "h_heel_left"
        #     rbody = "h_heel_right"
        #     A[num_dof + 6:num_dof + 6+1, :num_dof] = _h * skel.body(lbody).angular_jacobian()[1]
        #     A[num_dof + 6+1:num_dof + 6 + 2, :num_dof] = _h * skel.body(rbody).angular_jacobian()[1]
        #
        #     b[num_dof + 6:num_dof + 6 + 1] = Kw * des_w_y_L - np.dot(skel.body(lbody).angular_jacobian() + _h * skel.body(lbody).angular_jacobian_deriv(), skel.dq)[1]
        #     b[num_dof + 6+1:num_dof + 6 + 2] = Kw * des_w_y_R - np.dot(skel.body(rbody).angular_jacobian() + _h * skel.body(rbody).angular_jacobian_deriv(), skel.dq)[1]

        #####################################################
        # inequality
        #####################################################
        # lambda > 0
        G[:num_lambda, num_dof+num_tau:] = -np.eye(num_lambda)

        G[num_lambda:num_lambda+num_lambda, :num_dof] = -JcT_V.T
        # h[num_lambda:num_lambda+num_lambda] = np.dot(V.T, np.dot(dJc, self.control_skel.dq))
        h[num_lambda:num_lambda+num_lambda] = QP_RESTITUTION * np.dot(V.T, np.dot(dJc, skel.dq) + inv_h*np.dot(Jc, skel.dq))

        # friction cone rotation
        if FRICTION_CONE_ROTATION:
            G[num_lambda:num_lambda + num_lambda, :num_dof] = -JcT_RV.T
            # h[num_lambda:num_lambda+num_lambda] = np.dot(V.T, np.dot(dJc, self.control_skel.dq))

            #R_T * dR = [omega]
            #dR = R[omega]
            omega = np.zeros((3*num_contact, 3*num_contact))

            for i in range(num_contact):
                body_name = contact_foot_list[i]
                omega[3 * i:3 * i + 3, 3 * i:3 * i + 3] = transformToSkewSymmetricMatrix(skel.body(body_name).world_angular_velocity())

            dR = np.dot(R_c.T, omega)
            VT_dRT = np.dot(V.T, dR.T)
            h[num_lambda:num_lambda + num_lambda] = QP_RESTITUTION * \
                                                    (np.dot(RV.T, np.dot(dJc, skel.dq) + inv_h * np.dot(Jc,skel.dq)) \
                                                   + np.dot(VT_dRT, np.dot(Jc, skel.dq)))

        # friction coefficient
        if LAMBDA_CONTAIN_NORMAL:
            for i in range(num_contact):
                for j in range(QP_CONE_DIM):
                    g = np.zeros(QP_CONE_DIM+1)
                    g[j] = 1.
                    g[QP_CONE_DIM] = -MU_x if j % 2 == 0 else -MU_z

                    G[2*num_lambda+(QP_CONE_DIM+1)*i+j,
                    num_dof+num_tau+(QP_CONE_DIM+1)*i:num_dof+num_tau+(QP_CONE_DIM+1)*i + QP_CONE_DIM+1] = g

    forces = list()

    if num_contact > 0:
        zero_v = np.zeros(len(h))
        lb = np.stack(zero_v, b)
        ub = np.stack(h, b)
        A_stack = np.stack(G, A)
        result = dict()
        # result = solvers.qp(matrix(P), matrix(q), matrix(G), matrix(h), matrix(A), matrix(b))
        result['x'] = csQPOASES.qp(P, q, A_stack, None, None, lb, ub, 10, True, "HIGH")
        value_ddq = np.asarray(result['x']).flatten()[num_dof:]
        value_tau = np.asarray(result['x']).flatten()[num_dof:num_dof+num_tau]
        value_lambda = np.asarray(result['x']).flatten()[num_dof+num_tau:]
        force = np.dot(V, value_lambda)
        # print(force)
        for i in range(num_contact):
            forces.append(force[3*i:3*i+3])
    else:
        result = dict()
        # result = solvers.qp(matrix(P), matrix(q), None, None, matrix(A), matrix(b))
        # result['x'] = csQPOASES.qp(P, q, A, None, None, b, b, 10, True, "HIGH")
        result['x'] = csQPOASES.qp(P, q, A, None, None, b, b, 10, True, "HIGH")
        value_ddq = np.asarray(result['x']).flatten()[num_dof:]
        value_tau = np.asarray(result['x']).flatten()[num_dof:num_dof+num_tau]
    value_tau[:6] = np.zeros(6)

    return value_ddq, value_tau, [bodies[i].index_in_skeleton() for i in range(len(bodies))], position_globals, position_locals, forces, is_contact