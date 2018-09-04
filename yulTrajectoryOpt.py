import numpy as np
import math
from scipy import optimize
from motionPlan import motionPlan
import pydart2 as pydart


def solve_trajectory_optimization(skel, T, h):
    ndofs = skel.num_dofs()

    #  todo: Given information
    target_COM = np.array([0., 0., 0.])
    target_l_foot = np.array([0., -0.98, 0.3])
    target_r_foot = np.array([0., -0.98, -0.3])
    contact_flag = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    t = 1.0
    myu = 0.02

    #####################################################
    # objective
    #####################################################

    # The objective function to be minimized.
    # walking speed + turning rate + regularizing term + target position of end-effoctor + target position of COM trajectory

    w_speed = 50
    w_turn = 50
    w_end = 50
    w_com = 50
    w_regul = 0.1

    def func(x):
        mp = motionPlan(skel, T, x)
        # regularizing term
        regularize_objective = 0.
        for i in range(1, T - 2):
            regularize_objective = regularize_objective + np.linalg.norm(mp.get_pose(i-1) - 2 * mp.get_pose(i) + mp.get_pose(i+1)) **2

        # todo : get com and end-effector positions from q
        # target position for end_effectors
        end_effector_objective = 0.
        for i in range(0, T-1):
            skel.set_positions(mp.get_pose(i))
            # end_effector_objective = end_effector_objective + np.linalg.norm(
            #             mp.get_end_effector_l_position(i) - target_l_foot) ** 2 + np.linalg.norm(
            #                                      mp.get_end_effector_r_position(i) - target_r_foot) ** 2

            end_effector_objective = end_effector_objective + np.linalg.norm(
                 ??? - target_l_foot) ** 2 + np.linalg.norm(
                 ??? - target_r_foot) ** 2

        # target position for COM
        com_objective = 0.
        for i in range(0, T-1):
            skel.set_positions(mp.get_pose(i))
            com_objective = com_objective + np.linalg.norm( ??? - target_COM)**2
        # speed
        # speed_objective = np.linalg.norm((mp.get_COM_position(T-1) - mp.get_COM_position(0)) - t)**2
        # turning rate
        # + w_turn*((mp[T] - mp[0]) - t)*((mp[T] - mp[0]) - t)
        # balance

        return w_regul * regularize_objective + w_com * com_objective + w_end * end_effector_objective

    #####################################################
    # equality constraints
    #####################################################

    offset_list = [[-0.1040+0.0216, +0.80354016-0.85354016, 0.0],
                   [0.1040+0.0216, +0.80354016-0.85354016, 0.0],
                   [-0.1040 + 0.0216, +0.80354016 - 0.85354016, 0.0],
                   [0.1040 + 0.0216, +0.80354016 - 0.85354016, 0.0]]

    cons_list = []

    # equation of motion

    def eom_i(x):
        mp = motionPlan(skel, T, x)
        cons_value = [0] * T
        for i in range(T):
            contact_forces = mp.get_contact_force(i)
            con_l1 = contact_forces[0:3]
            con_l2 = contact_forces[3:6]
            con_r1 = contact_forces[6:9]
            con_r2 = contact_forces[9:12]

            skel.set_positions(mp.get_pose(i))
            J_c_t = np.zeros((ndofs, 3 * 4))
            jaco = skel.body('h_blade_left').linear_jacobian(offset_list[0])
            J_c_t[0:, 0:3] = np.transpose(jaco)
            jaco = skel.body('h_blade_left').linear_jacobian(offset_list[0])
            J_c_t[0:, 3:6] = np.transpose(jaco)
            jaco = skel.body('h_blade_right').linear_jacobian(offset_list[0])
            J_c_t[0:, 6:9] = np.transpose(jaco)
            jaco = skel.body('h_blade_right').linear_jacobian(offset_list[0])
            J_c_t[0:, 9:12] = np.transpose(jaco)

            if i == 0:
                com_acc = np.zeros(ndofs)
            else:
                com_acc = mp.get_pose(i+1) - 2* mp.get_pose(i) + mp.get_pose(i-1)

            cons_value[i] = J_c_t.dot(contact_forces) + mp.get_tau() - skel.c - skel.M * com_acc

            return cons_value

    cons_list.append({'type': 'eq', 'fun': eom_i})

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
                #todo : check
                vel_l = (skel.body('h_blade_left').to_world([0.0, 0.0, 0.0]) - pre_pos_l) / h
                vel_r = (skel.body('h_blade_right').to_world([0.0, 0.0, 0.0]) - pre_pos_r) / h

            cons_value[2 * i + 0] = vel_l[0] * sa_l + vel_l[2] * ca_l
            cons_value[2 * i + 1] = vel_r[0] * sa_r + vel_r[2] * ca_r

            pre_pos_l = skel.body('h_blade_left').to_world([0.0, 0.0, 0.0])
            pre_pos_r = skel.body('h_blade_right').to_world([0.0, 0.0, 0.0])

        return cons_value

    cons_list.append({'type': 'eq', 'fun': non_holonomic})
    # Contact_flag [ 0 or 1 ]

    def is_contact(x):
        mp = motionPlan(skel, T, x)
        cons_value = [0] * (12 * T)
        for i in range(T):
            contact_force = mp.get_contact_force(i)
            cons_value[12*i:12*(i+1)] = contact_force*(np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]) - contact_flag)
        return cons_value

    cons_list.append({'type': 'eq', 'fun': is_contact})

        #####################################################
        # inequality constraint
        #####################################################
        # coulomb friction model
    def friction_normal(x):
        mp = motionPlan(skel, T, x)
        cons_value = [0] * (4 * T)
        for i in range(T):
            contact_force = mp.get_contact_force(i)
            for j in range(4):
                cons_value[4*i + j] = contact_force[3*j + 1]

        return cons_value

    cons_list.append({'type': 'ineq', 'fun': friction_normal})

    def friction_tangent(x):
        mp = motionPlan(skel, T, x)
        cons_value = [0] * (4 * T)
        for i in range(T):
            contact_force = mp.get_contact_force(i)
            for j in range(4):
                cons_value[4*i + j] = -contact_force[3*j+0] ** 2 - contact_force[3*j+2] ** 2 + (myu ** 2) * (contact_force[3*j+1] ** 2)

        return cons_value

    cons_list.append({'type': 'ineq', 'fun': friction_tangent})
    # print(cons)

    #####################################################
    # solve
    #####################################################

    # bnds = ()
    x0 = [-1.] * ((ndofs * 2 + 3 * 4) * T)  #  initial guess
    # x0 = None  # initial guess

    # res = optimize.minimize(func, x0, method='SLSQP', bounds=None, constraints=cons)
    # res = optimize.minimize(func, x0, method='SLSQP', bounds=None)
    res = optimize.minimize(func, x0, method='SLSQP', bounds=None, constraints=cons_list)
    return res

if __name__ == '__main__':
    pydart.init()
    world = pydart.World(1./1000., './data/skel/cart_pole_blade_3dof.skel')
    skel = world.skeletons[2]

    print(solve_trajectory_optimization(skel, 10, world.time_step())['x'])
    mp = motionPlan
