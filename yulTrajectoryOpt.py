import numpy as np
import math
from scipy import optimize
from motionPlan import motionPlan
import pydart2 as pydart


from fltk import *
from PyCommon.modules.GUI import hpSimpleViewer as hsv
from PyCommon.modules.Renderer import ysRenderer as yr
from PyCommon.modules.Simulator import hpDartQpSimulator as hqp

np.set_printoptions(threshold=np.nan)

offset_list = [[-0.1040 + 0.0216, +0.80354016 - 0.85354016, 0.0],
               [0.1040 + 0.0216, +0.80354016 - 0.85354016, 0.0],
               [-0.1040 + 0.0216, +0.80354016 - 0.85354016, 0.0],
               [0.1040 + 0.0216, +0.80354016 - 0.85354016, 0.0]]


def solve_trajectory_optimization(skel, T, h):
    ndofs = skel.num_dofs()

    # print(skel.body('h_blade_right').to_world([-0.1040 + 0.0216, +0.80354016 - 0.85354016, 0.0]))

    #  todo: Given information
    target_COM = [0] * (3*T)
    target_l_foot = [0] * (3 * T)
    target_r_foot = [0] * (3 * T)
    for i in range(T):
        target_COM[3 * i] = 0.05*i
        target_COM[3 * i+1] = 0.
        target_COM[3 * i+2] = 0.

        target_l_foot[3 * i] = 0.05 * i
        target_l_foot[3 * i + 1] = -0.98
        target_l_foot[3 * i + 2] = -0.2

        target_r_foot[3 * i] = 0.05 * i
        target_r_foot[3 * i + 1] = -0.98
        target_r_foot[3 * i + 2] = 0.2

    # target_l_foot = np.array([0., -0.98, -0.2])
    # target_r_foot = np.array([0., -0.98, 0.2])

    contact_flag = [0] * (4 * T)
    for i in range(T):
        # if i > 2 and i < 7:
        #     contact_flag[4 * i] = 1
        #     contact_flag[4 * i + 1] = 1
        #     contact_flag[4 * i + 2] = 0
        #     contact_flag[4 * i + 3] = 0
        # else:
        #     contact_flag[4 * i] = 1
        #     contact_flag[4 * i + 1] = 1
        #     contact_flag[4 * i + 2] = 1
        #     contact_flag[4 * i + 3] = 1
        contact_flag[4 * i] = 1
        contact_flag[4 * i + 1] = 1
        contact_flag[4 * i + 2] = 0
        contact_flag[4 * i + 3] = 0

    print(contact_flag)

    # contact_flag = np.array([1, 1, 0, 0])

    contact_name_list = ['h_blade_left', 'h_blade_left', 'h_blade_right', 'h_blade_right']
    offset_list = [[-0.1040 + 0.0216, +0.80354016 - 0.85354016, 0.0],
                   [0.1040 + 0.0216, +0.80354016 - 0.85354016, 0.0],
                   [-0.1040 + 0.0216, +0.80354016 - 0.85354016, 0.0],
                   [0.1040 + 0.0216, +0.80354016 - 0.85354016, 0.0]]

    contact_list = []

    t = 1.0
    myu = 0.02

    cons_list = []


    # cn = 0  # the number of contact point
    cn = [0] * T
    total_cn = 0
    for i in range(T):
        for j in range(4):
            if contact_flag[i*4+j] == 1:
                # contact_list.append(contact_name_list[j])
                # contact_list.append(offset_list[j])
                cn[i] = cn[i] + 1
        # print(cn)
        total_cn = total_cn + cn[i]
    # print("con_num: ", cn)
    # print(contact_list)
    # print(total_cn)

    #####################################################
    # objective
    #####################################################

    # The objective function to be minimized.
    # walking speed + turning rate + regularizing term + target position of end-effoctor + target position of COM trajectory

    w_effort = 50
    w_speed = 50
    w_turn = 50
    w_end = 50
    w_com = 50
    w_regul = 0.1

    def func(x):
        mp = motionPlan(skel, T, x)
        # minimize joint torque
        effort_objective = 0.
        for i in range(T):
            effort_objective = effort_objective + np.linalg.norm(mp.get_tau(i-1))**2

        # regularizing term
        regularize_objective = 0.
        for i in range(1, T - 2):
            regularize_objective = regularize_objective + np.linalg.norm(mp.get_pose(i-1) - 2 * mp.get_pose(i) + mp.get_pose(i+1)) **2

        # target position for end_effectors
        end_effector_objective = 0.
        for i in range(T):
            skel.set_positions(mp.get_pose(i))
            end_effector_objective = end_effector_objective + np.linalg.norm(
                skel.body('h_blade_left').to_world([0.0, 0.0, 0.0]) - target_l_foot[3*i:3*(i+1)]) ** 2 + np.linalg.norm(
                skel.body('h_blade_right').to_world([0.0, 0.0, 0.0]) - target_r_foot[3*i:3*(i+1)]) ** 2

        # target position for COM
        com_objective = 0.
        for i in range(T):
            skel.set_positions(mp.get_pose(i))
            com_objective = com_objective + np.linalg.norm(skel.body('h_pelvis').to_world([0.0, 0.0, 0.0]) - target_COM[3*i:3*(i+1)])**2
        # speed
        # speed_objective = np.linalg.norm((mp.get_COM_position(T-1) - mp.get_COM_position(0)) - t)**2
        # turning rate
        # + w_turn*((mp[T] - mp[0]) - t)*((mp[T] - mp[0]) - t)
        # balance

        return w_effort * effort_objective + w_regul * regularize_objective + w_com * com_objective #+ w_end * end_effector_objective

    #####################################################
    # equality constraints
    #####################################################

    # equation of motion
    def eom_i(x):
        mp = motionPlan(skel, T, x)
        cons_value = [0] * (ndofs * (T-2))
        for i in range(1, T-1):
            contact_forces = mp.get_contact_force(i)

            skel.set_positions(mp.get_pose(i))
            J_c_t = np.zeros((ndofs, 3 * cn[i]))
            if contact_flag[4 * i + 0] == 1:
                jaco = skel.body('h_blade_left').linear_jacobian(offset_list[0])
                J_c_t[0:, 0:3] = np.transpose(jaco)
            if contact_flag[4 * i + 1] == 1:
                jaco = skel.body('h_blade_left').linear_jacobian(offset_list[1])
                J_c_t[0:, 3:6] = np.transpose(jaco)
            if contact_flag[4 * i + 2] == 1:
                jaco = skel.body('h_blade_right').linear_jacobian(offset_list[2])
                J_c_t[0:, 6:9] = np.transpose(jaco)
            if contact_flag[4 * i + 3] == 1:
                jaco = skel.body('h_blade_right').linear_jacobian(offset_list[3])
                J_c_t[0:, 9:12] = np.transpose(jaco)

            com_acc = mp.get_pose(i+1) - 2 * mp.get_pose(i) + mp.get_pose(i-1)
            # print(J_c_t.dot(contact_forces) + mp.get_tau(i) - skel.c - skel.M.dot(com_acc))
            cons_value[ndofs*i:ndofs*(i+1)] = J_c_t.dot(contact_forces) + mp.get_tau(i) - skel.c - skel.M.dot(com_acc)

        return cons_value

    #cons_list.append({'type': 'eq', 'fun': eom_i})

    # first six of tau=0
    def eq_tau(x):
        mp = motionPlan(skel, T, x)
        cons_value = [0]*(6*T)
        for i in range(T):
            cons_value[6*i:6*(i+1)] = mp.get_tau(i)[0:6]
        return cons_value

    cons_list.append({'type': 'eq', 'fun': eq_tau})


    # non_holonomic constraint
    def non_holonomic(x):
        mp = motionPlan(skel, T, x)
        cons_value = [0] * (2 * T)
        for i in range(T):
            skel.set_positions(mp.get_pose(i))
            p1 = skel.body('h_blade_left').to_world([-0.1040 + 0.0216, 0.0, 0.0])
            p2 = skel.body('h_blade_left').to_world([0.1040 + 0.0216, 0.0, 0.0])

            blade_direction_vec = p2 - p1

            theta_l = math.acos(np.dot(np.array([-1., 0., 0.]), blade_direction_vec))
            theta_r = math.acos(np.dot(np.array([1., 0., 0.]), blade_direction_vec))
            # print("theta: ", body_name, ", ", theta)
            # print("omega: ", skel.body("h_blade_left").world_angular_velocity()[1])

            next_step_angle_l = theta_l + skel.body('h_blade_left').world_angular_velocity()[1] * h
            next_step_angle_r = theta_r + skel.body('h_blade_right').world_angular_velocity()[1] * h
            # print("next_step_angle: ", next_step_angle)
            sa_l = math.sin(next_step_angle_l)
            ca_l = math.cos(next_step_angle_l)

            sa_r = math.sin(next_step_angle_r)
            ca_r = math.cos(next_step_angle_r)

            if i == 0:
                vel_l = np.zeros(3)
                vel_r = np.zeros(3)
            else:
                vel_l = (skel.body('h_blade_left').to_world([0.0, 0.0, 0.0]) - pre_pos_l) / h
                vel_r = (skel.body('h_blade_right').to_world([0.0, 0.0, 0.0]) - pre_pos_r) / h

            cons_value[2 * i + 0] = vel_l[0] * sa_l - vel_l[2] * ca_l
            cons_value[2 * i + 1] = vel_r[0] * sa_r - vel_r[2] * ca_r

            pre_pos_l = skel.body('h_blade_left').to_world([0.0, 0.0, 0.0])
            pre_pos_r = skel.body('h_blade_right').to_world([0.0, 0.0, 0.0])

        return cons_value

    #cons_list.append({'type': 'eq', 'fun': non_holonomic})
    # Contact_flag [ 0 or 1 ]

    def is_contact(x):
        mp = motionPlan(skel, T, x)

        cons_value = [0] * (total_cn * 3)
        cn_till = 0
        for i in range(T):
            contact_force = mp.get_contact_force(i)
            k = 0
            if contact_flag[4 * i + 0] == 0:
                cons_value[(cn_till+k)*3:(cn_till+(k+1))*3] = contact_force[0:3]
                k = k + 1
            if contact_flag[4 * i + 1] == 0:
                cons_value[(cn_till+k)*3:(cn_till+(k+1))*3] = contact_force[3:6]
                k = k + 1
            if contact_flag[4 * i + 2] == 0:
                cons_value[(cn_till+k)*3:(cn_till+(k+1))*3] = contact_force[6:9]
                k = k + 1
            if contact_flag[4 * i + 3] == 0:
                cons_value[(cn_till+k)*3:(cn_till+(k+1))*3] = contact_force[9:12]
                k = k + 1
            cn_till = cn_till + cn[i]
            # cons_value[cn * 3*i:cn * 3*(i+1)] = contact_force[0:6]
        return cons_value

    cons_list.append({'type': 'eq', 'fun': is_contact})

        #####################################################
        # inequality constraint
        #####################################################
        # coulomb friction model
    def friction_normal(x):
        mp = motionPlan(skel, T, x)
        cons_value = [0] * (total_cn)
        cn_till = 0
        for i in range(T):
            contact_force = mp.get_contact_force(i)
            k = 0
            if contact_flag[4 * i + 0] == 1:
                cons_value[cn_till+k] = contact_force[1]
                k = k+1
            if contact_flag[4 * i + 1] == 1:
                cons_value[cn_till+k] = contact_force[4]
                k = k + 1
            if contact_flag[4 * i + 2] == 1:
                cons_value[cn_till+k] = contact_force[7]
                k = k + 1
            if contact_flag[4 * i + 3] == 1:
                cons_value[cn_till+k] = contact_force[11]
                k = k + 1
            cn_till = cn_till + cn[i]
            # for j in range(4):
            #     cons_value[4*i + j] = contact_force[3*j + 1]
        return cons_value

    cons_list.append({'type': 'ineq', 'fun': friction_normal})

    def friction_tangent(x):
        mp = motionPlan(skel, T, x)
        cons_value = [0] * (total_cn)
        cn_till = 0
        for i in range(T):
            contact_force = mp.get_contact_force(i)
            k = 0
            if contact_flag[4 * i + 0] == 1:
                cons_value[cn_till+k] = -contact_force[0] ** 2 - contact_force[2] ** 2 + (myu ** 2) * (contact_force[1] ** 2)
                k = k+1
            if contact_flag[4 * i + 1] == 1:
                cons_value[cn_till+k] = -contact_force[3] ** 2 - contact_force[5] ** 2 + (myu ** 2) * (contact_force[4] ** 2)
                k = k + 1
            if contact_flag[4 * i + 2] == 1:
                cons_value[cn_till+k] = -contact_force[6] ** 2 - contact_force[8] ** 2 + (myu ** 2) * (contact_force[7] ** 2)
                k = k + 1
            if contact_flag[4 * i + 3] == 1:
                cons_value[cn_till+k] = -contact_force[9] ** 2 - contact_force[11] ** 2 + (myu ** 2) * (contact_force[10] ** 2)
                k = k + 1
            cn_till = cn_till + cn[i]
            # for j in range(4):
            #     cons_value[4*i + j] = -contact_force[3*j+0] ** 2 - contact_force[3*j+2] ** 2 + (myu ** 2) * (contact_force[3*j+1] ** 2)
        return cons_value

    cons_list.append({'type': 'ineq', 'fun': friction_tangent})

    def swing(x):
        mp = motionPlan(skel, T, x)
        cons_value = [0] * (total_cn)
        cn_till = 0
        for i in range(T):
            skel.set_positions(mp.get_pose(i))
            k = 0
            if contact_flag[4 * i + 0] == 0:
                cons_value[cn_till+k] = skel.body('h_blade_left').to_world([-0.1040 + 0.0216, +0.80354016 - 0.85354016, 0.0])[1] + 0.9
                k = k+1
            if contact_flag[4 * i + 1] == 0:
                cons_value[cn_till+k] = skel.body('h_blade_left').to_world([0.1040 + 0.0216, +0.80354016 - 0.85354016, 0.0])[1] + 0.9
                k = k + 1
            if contact_flag[4 * i + 2] == 0:
                cons_value[cn_till+k] = skel.body('h_blade_right').to_world([-0.1040 + 0.0216, +0.80354016 - 0.85354016, 0.0])[1] + 0.9
                k = k + 1
            if contact_flag[4 * i + 3] == 0:
                cons_value[cn_till+k] = skel.body('h_blade_right').to_world([0.1040 + 0.0216, +0.80354016 - 0.85354016, 0.0])[1] + 0.9
                k = k + 1
            cn_till = cn_till + cn[i]
        return cons_value

    cons_list.append({'type': 'ineq', 'fun': swing})
    # print(cons)

    #####################################################
    # solve
    #####################################################

    # bnds = ()
    x0 = [0.] * ((ndofs * 2 + 3 * 4) * T)  #  initial guess

    for i in range(T):
        x0[(ndofs * 2 + 3 * 4) * i + 6] = -0.1
        x0[(ndofs * 2 + 3 * 4) * i + 17] = 0.9
        x0[(ndofs * 2 + 3 * 4) * i + 20] = -1.5
        # x0[(ndofs * 2 + 3 * 4) * i + 36] = -1.5
        # x0[(ndofs * 2 + 3 * 4) * i + 48] = -1.5

    # x0 = None  # initial guess

    # res = optimize.minimize(func, x0, method='SLSQP', bounds=None, constraints=cons)
    # res = optimize.minimize(func, x0, method='SLSQP', bounds=None)
    res = optimize.minimize(func, x0, method='SLSQP', bounds=None, constraints=cons_list)
    return res

if __name__ == '__main__':
    pydart.init()
    world = pydart.World(1./1000., './data/skel/cart_pole_blade_3dof.skel')
    skel = world.skeletons[2]

    # print(solve_trajectory_optimization(skel, 10, world.time_step())['x'])
    # print(solve_trajectory_optimization(skel, 10, world.time_step()))
    # mp = motionPlan

    opt_res = solve_trajectory_optimization(skel, 10, world.time_step())
    print(opt_res)
    print("trajectory optimization finished!!")
    mp = motionPlan(skel, 10, opt_res['x'])
    # for i in range(10):
    #     skel.set_positions(mp.get_pose(i))

    viewer = hsv.hpSimpleViewer(viewForceWnd=False)
    viewer.setMaxFrame(1000)
    viewer.doc.addRenderer('controlModel', yr.DartRenderer(world, (255, 255, 255), yr.POLYGON_FILL))
    # viewer.doc.addRenderer('contactForce', yr.VectorsRenderer(render_vector, render_vector_origin, (255, 0, 0)))
    # viewer.doc.addRenderer('pushForce', yr.WideArrowRenderer(push_force, push_force_origin, (0, 255, 0)))
    # viewer.doc.addRenderer('rd_footCenter', yr.PointsRenderer(rd_footCenter))

    viewer.startTimer(1 / 25.)
    viewer.motionViewWnd.glWindow.pOnPlaneshadow = (0., -0.99 + 0.0251, 0.)

    # viewer.doc.addRenderer('bladeForce', yr.WideArrowRenderer(blade_force, blade_force_origin, (0, 0, 255)))

    def simulateCallback(frame):
        skel.set_positions(mp.get_pose(frame))

        #todo :
        skel.body('h_blade_left').add_ext_force(mp.get_contact_force(frame)[0:3], offset_list[0])
        skel.body('h_blade_left').add_ext_force(mp.get_contact_force(frame)[3:6], offset_list[1])
        skel.body('h_blade_right').add_ext_force(mp.get_contact_force(frame)[6:9], offset_list[2])
        skel.body('h_blade_right').add_ext_force(mp.get_contact_force(frame)[9:12], offset_list[3])

    viewer.setSimulateCallback(simulateCallback)
    viewer.show()

    Fl.run()