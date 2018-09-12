import numpy as np
import math
from scipy import optimize
from motionPlan_simple import motionPlan
import pydart2 as pydart
import time

import os
from datetime import datetime

from fltk import *
from PyCommon.modules.GUI import hpSimpleViewer as hsv
from PyCommon.modules.Renderer import ysRenderer as yr
from PyCommon.modules.Simulator import hpDartQpSimulator as hqp

np.set_printoptions(threshold=np.nan)

offset_list = [[-0.1040 + 0.0216, +0.80354016 - 0.85354016, 0.0],
               [0.1040 + 0.0216, +0.80354016 - 0.85354016, 0.0],
               [-0.1040 + 0.0216, +0.80354016 - 0.85354016, 0.0],
               [0.1040 + 0.0216, +0.80354016 - 0.85354016, 0.0]]

com_pos = []
l_footCenter = []
r_footCenter = []

def TicTocGenerator():
    ti = 0              # initial time
    tf = time.time()    # final time
    while True:
        ti = tf
        ti = time.time()
        yield ti - tf    # returns the time difference

TicToc = TicTocGenerator()      #create an instance of the TicToc generator

def toc(tempBool = True):
    #Prints the time difference yielded by generator instance TicToc
    tempTimeInterval = next(TicToc)
    if tempBool:
        print("Elapsed time: %f seconds.\n" %tempTimeInterval)

def tic():
    #Records a time in TicToc, marks the beginning
    toc(False)

def solve_trajectory_optimization(skel, T, h):
    # tic()
    ndofs = skel.num_dofs()

    # print(skel.body('h_blade_right').to_world([-0.1040 + 0.0216, +0.80354016 - 0.85354016, 0.0]))

    #  todo: Given information
    target_COM = [0] * (3*T)
    target_l_foot = [0] * (3 * T)
    target_r_foot = [0] * (3 * T)
    for i in range(T):
        target_COM[3 * i] = 0.1*i
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
        # if i > 5 and i < 15:
        #     contact_flag[4 * i] = 1
        #     contact_flag[4 * i + 1] = 1
        #     contact_flag[4 * i + 2] = 0
        #     contact_flag[4 * i + 3] = 0
        # else:
        #     contact_flag[4 * i] = 1
        #     contact_flag[4 * i + 1] = 1
        #     contact_flag[4 * i + 2] = 1
        #     contact_flag[4 * i + 3] = 1
        if i < 10:
            contact_flag[4 * i] = 1
            contact_flag[4 * i + 1] = 1
            contact_flag[4 * i + 2] = 0
            contact_flag[4 * i + 3] = 0
        else:
            contact_flag[4 * i] = 0
            contact_flag[4 * i + 1] = 0
            contact_flag[4 * i + 2] = 1
            contact_flag[4 * i + 3] = 1


    # print(contact_flag)

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

    w_effort = 50.
    w_speed = 50.
    w_turn = 50.
    w_end = 50.
    w_com = 50.
    w_dis = 50.
    w_regul = 0.1

    def func(x):
        mp = motionPlan(skel, T, x)

        # minimize force
        # effort_objective = 0.
        # for i in range(T):
        #     forces = mp.get_contact_force(i)
        #     effort_objective = effort_objective + np.linalg.norm(cf) ** 2

        # target position for end_effectors
        end_effector_objective = 0.
        for i in range(T):
            end_effector_objective = end_effector_objective + np.linalg.norm(
                skel.body('h_blade_left').to_world([0.0, 0.0, 0.0]) - target_l_foot[3*i:3*(i+1)]) ** 2 + np.linalg.norm(
                skel.body('h_blade_right').to_world([0.0, 0.0, 0.0]) - target_r_foot[3*i:3*(i+1)]) ** 2

        # target position for COM
        com_objective = 0.

        for i in range(T):
            com_objective = com_objective + np.linalg.norm(mp.get_com_position(i) - target_COM[3 * i:3 * (i + 1)]) ** 2
        # speed
        # speed_objective = np.linalg.norm((mp.get_COM_position(T-1) - mp.get_COM_position(0)) - t)**2
        # turning rate
        # + w_turn*((mp[T] - mp[0]) - t)*((mp[T] - mp[0]) - t)
        # balance

        #effort

        #distance btw COM and a foot
        distance_objective = 0.
        for i in range(T):
            d_l = np.linalg.norm(np.linalg.norm(mp.get_com_position(i) - mp.get_end_effector_l_position(i)))
            d_r = np.linalg.norm(np.linalg.norm(mp.get_com_position(i) - mp.get_end_effector_r_position(i)))
            ref_l_dis = np.linalg.norm(
                skel.body('h_pelvis').to_world([0., 0., 0.]) - skel.body('h_blade_left').to_world([0., 0., 0.]))
            ref_r_dis = np.linalg.norm(
                skel.body('h_pelvis').to_world([0., 0., 0.]) - skel.body('h_blade_right').to_world([0., 0., 0.]))
            distance_objective = distance_objective + (d_l - ref_l_dis) ** 2 + (d_r - ref_r_dis) ** 2

        return w_com * com_objective + w_dis * distance_objective #+ w_effort * effort_objective

    #####################################################
    # equality constraints
    #####################################################

    # equation of motion
    def eom_i(x):
        mp = motionPlan(skel, T, x)
        cons_value = [0] * (3 * (T-2))
        for i in range(1, T-1):
            com_vel1 = (mp.get_com_position(i) - mp.get_com_position(i-1))*(1/h)
            com_vel2 = (mp.get_com_position(i+1) - mp.get_com_position(i)) * (1 / h)

            com_acc = (1 / (h * h)) * (com_vel2 - com_vel1)

            contact_forces = mp.get_contact_force(i)

            contact_sum = np.zeros(3)
            if contact_flag[4 * i + 0] == 1:
                contact_sum = contact_sum + contact_forces[0:3]
            if contact_flag[4 * i + 1] == 1:
                contact_sum = contact_sum + contact_forces[3:6]
            if contact_flag[4 * i + 2] == 1:
                contact_sum = contact_sum + contact_forces[6:9]
            if contact_flag[4 * i + 3] == 1:
                contact_sum = contact_sum + contact_forces[9:12]

            # cons_value[ndofs * (i - 1):ndofs * i] = skel.mass() * skel.com_acceleration() - skel.mass() * np.array(
            #     [0., -9.8, 0.]) - contact_sum

            cons_value[ndofs * (i-1):ndofs * i] = skel.mass() * com_acc - skel.mass()*np.array([0., -9.8, 0.]) - contact_sum

        return cons_value

    # non_holonomic constraint
    def non_holonomic(x):
        mp = motionPlan(skel, T, x)
        cons_value = [0] * (2 * (T-1))

        pre_pos_l = mp.get_end_effector_l_position(0)
        pre_pos_r = mp.get_end_effector_r_position(0)

        for i in range(1, T):

            p1 = mp.get_end_effector_l_position(i) + np.array(offset_list[0])
            p2 = mp.get_end_effector_l_position(i) + np.array(offset_list[1])
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

            vel_l = (mp.get_end_effector_l_position(i) - pre_pos_l) / h
            vel_r = (mp.get_end_effector_r_position(i) - pre_pos_r) / h

            cons_value[2 * (i-1) + 0] = vel_l[0] * sa_l - vel_l[2] * ca_l
            cons_value[2 * (i-1) + 1] = vel_r[0] * sa_r - vel_r[2] * ca_r

            pre_pos_l = mp.get_end_effector_l_position(i)
            pre_pos_r = mp.get_end_effector_r_position(i)

        # print(cons_value)
        return cons_value

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

    # tangential friction.dot(tangential_velocity) <= 0
    def tangential_vel(x):
        mp = motionPlan(skel, T, x)
        cons_value = [0] * (total_cn)
        cn_till = 0
        for i in range(T-1):
            contact_force = mp.get_contact_force(i)
            k = 0
            if contact_flag[4 * i + 0] == 1:
                tangential_force = contact_force[0:3]
                tangential_force[1] = 0
                tangential_velocity = skel.body('h_blade_left').world_linear_velocity(offset_list[0])
                tangential_velocity[1] = 0

                cons_value[cn_till + k] = -tangential_force.dot(tangential_velocity)
                k = k + 1
            if contact_flag[4 * i + 1] == 1:
                tangential_force = contact_force[0:3]
                tangential_force[1] = 0
                tangential_velocity = skel.body('h_blade_left').world_linear_velocity(offset_list[1])
                tangential_velocity[1] = 0

                cons_value[cn_till + k] = -tangential_force.dot(tangential_velocity)
                k = k + 1
            if contact_flag[4 * i + 2] == 1:
                tangential_force = contact_force[0:3]
                tangential_force[1] = 0
                tangential_velocity = skel.body('h_blade_left').world_linear_velocity(offset_list[2])
                tangential_velocity[1] = 0

                cons_value[cn_till + k] = -tangential_force.dot(tangential_velocity)
                k = k + 1
            if contact_flag[4 * i + 3] == 1:
                tangential_force = contact_force[0:3]
                tangential_force[1] = 0
                tangential_velocity = skel.body('h_blade_left').world_linear_velocity(offset_list[3])
                tangential_velocity[1] = 0

                cons_value[cn_till + k] = -tangential_force.dot(tangential_velocity)
                k = k + 1
            cn_till = cn_till + cn[i]
            # for j in range(4):
            #     cons_value[4*i + j] = -contact_force[3*j+0] ** 2 - contact_force[3*j+2] ** 2 + (myu ** 2) * (contact_force[3*j+1] ** 2)
        return cons_value

    def swing(x):
        mp = motionPlan(skel, T, x)
        cons_value = [0] * (3*4*T - total_cn)
        cn_till = 0
        for i in range(T):
            k = 0
            if contact_flag[4 * i + 0] == 0:
                cons_value[cn_till+k] = mp.get_end_effector_l_position(i)[1] + 0.98
                k = k+1
            if contact_flag[4 * i + 1] == 0:
                cons_value[cn_till+k] = mp.get_end_effector_l_position(i)[1] + 0.98
                k = k + 1
            if contact_flag[4 * i + 2] == 0:
                cons_value[cn_till+k] = mp.get_end_effector_r_position(i)[1] + 0.98
                k = k + 1
            if contact_flag[4 * i + 3] == 0:
                cons_value[cn_till+k] = mp.get_end_effector_r_position(i)[1] + 0.98
                k = k + 1
            cn_till = cn_till + cn[i]
        return cons_value

    def stance(x):
        mp = motionPlan(skel, T, x)
        cons_value = [0] *(2 * total_cn)
        cn_till = 0

        threshold = 0.02
        for i in range(T):
            k = 0
            if contact_flag[4 * i + 0] == 1:
                y_pos = mp.get_end_effector_l_position(i)[1]
                cons_value[cn_till + k] = y_pos + 0.98 - threshold
                cons_value[cn_till + k+1] = -y_pos - 0.98 + threshold
                k = k + 2
            if contact_flag[4 * i + 1] == 1:
                y_pos = mp.get_end_effector_l_position(i)[1]
                cons_value[cn_till + k] = y_pos + 0.98 - threshold
                cons_value[cn_till + k+1] = -y_pos - 0.98 + threshold
                k = k + 2
            if contact_flag[4 * i + 2] == 1:
                y_pos = mp.get_end_effector_r_position(i)[1]
                cons_value[cn_till + k] = y_pos + 0.98 - threshold
                cons_value[cn_till + k+1] = -y_pos - 0.98 + threshold
                k = k + 2
            if contact_flag[4 * i + 3] == 1:
                y_pos = mp.get_end_effector_r_position(i)[1]
                cons_value[cn_till + k] = y_pos + 0.98 - threshold
                cons_value[cn_till + k+1] = -y_pos - 0.98 + threshold
                k = k + 2
            cn_till = cn_till + cn[i]
        return cons_value

    cons_list.append({'type': 'eq', 'fun': eom_i})
    cons_list.append({'type': 'eq', 'fun': non_holonomic})
    cons_list.append({'type': 'eq', 'fun': is_contact})
    cons_list.append({'type': 'ineq', 'fun': friction_normal})
    cons_list.append({'type': 'ineq', 'fun': friction_tangent})
    cons_list.append({'type': 'ineq', 'fun': tangential_vel})
    cons_list.append({'type': 'ineq', 'fun': swing})
    cons_list.append({'type': 'ineq', 'fun': stance})

    # toc()
    #####################################################
    # solve
    #####################################################

    tic()
    # bnds = ()
    x0 = [0.] * ((3 * 3 + 3 * 4) * T)  #  initial guess

    # x0 = None  # initial guess

    # res = optimize.minimize(func, x0, method='SLSQP', bounds=None, constraints=cons)
    # res = optimize.minimize(func, x0, method='SLSQP', bounds=None)
    res = optimize.minimize(func, x0, method='SLSQP', bounds=None, constraints=cons_list)
    toc()
    return res


class MyWorld(pydart.World):
    def __init__(self, ):
        # pydart.World.__init__(self, 1. / 1000., './data/skel/cart_pole_blade_3dof.skel')
        pydart.World.__init__(self, 1. / 50., './data/skel/cart_pole_blade_3dof.skel')

        self.force = None
        self.duration = 0
        self.skeletons[0].body('ground').set_friction_coeff(0.02)

        skel = self.skeletons[2]


    def step(self, i):
        # print("step")
        # cf = mp.get_contact_force(i)
        # # print(cf)
        #
        # tau = np.zeros(skel.num_dofs())
        #
        # if np.linalg.norm(cf[0:3]) > 0.001:
        #     print("foot1 contact", np.linalg.norm(cf[0:3]), cf[0:3])
        #     my_jaco = skel.body("h_blade_left").linear_jacobian(offset_list[0])
        #     my_jaco_t = my_jaco.transpose()
        #     tau = tau + np.dot(my_jaco_t, cf[0:3])
        #     skel.body("h_blade_left").add_ext_force(cf[0:3], offset_list[0])
        #
        # if np.linalg.norm(cf[3:6]) > 0.001:
        #     print("foot2 contact", np.linalg.norm(cf[3:6]), cf[3:6])
        #     my_jaco = skel.body("h_blade_left").linear_jacobian(offset_list[1])
        #     my_jaco_t = my_jaco.transpose()
        #     tau = tau + np.dot(my_jaco_t, cf[3:6])
        #
        #     skel.body("h_blade_left").add_ext_force(cf[3:6], offset_list[1])
        # if np.linalg.norm(cf[6:9]) > 0.001:
        #     print("foot3 contact", np.linalg.norm(cf[6:9]), cf[6:9])
        #     my_jaco = skel.body("h_blade_right").linear_jacobian(offset_list[2])
        #     my_jaco_t = my_jaco.transpose()
        #     tau = tau + np.dot(my_jaco_t, cf[6:9])
        #     skel.body("h_blade_right").add_ext_force(cf[6:9], offset_list[2])
        #
        # if np.linalg.norm(cf[9:12]) > 0.001:
        #     print("foot4 contact", np.linalg.norm(cf[9:12]), cf[9:12])
        #     my_jaco = skel.body("h_blade_right").linear_jacobian(offset_list[3])
        #     my_jaco_t = my_jaco.transpose()
        #     tau = tau + np.dot(my_jaco_t, cf[9:12])
        #     skel.body("h_blade_right").add_ext_force(cf[9:12], offset_list[3])
        #
        # tau[0:6] = np.zeros(6)
        # print("torque: ", tau)

        # print("contact_force: \n", cf)

        # skel.set_forces(tau)

        super(MyWorld, self).step()

    def render_with_ys(self, i):
        del com_pos[:]
        del l_footCenter[:]
        del r_footCenter[:]

        com_pos.append(mp.get_com_position(i))
        l_footCenter.append(mp.get_end_effector_l_position(i))
        r_footCenter.append(mp.get_end_effector_r_position(i))

if __name__ == '__main__':
    pydart.init()
    world = MyWorld()
    skel = world.skeletons[2]

    ground = pydart.World(1. / 50., './data/skel/ground.skel')

    # print(solve_trajectory_optimization(skel, 10, world.time_step())['x'])
    # print(solve_trajectory_optimization(skel, 10, world.time_step()))

    frame_num = 21

    opt_res = solve_trajectory_optimization(skel, frame_num, world.time_step())
    print(opt_res)
    # print("trajectory optimization finished!!")
    mp = motionPlan(skel, frame_num, opt_res['x'])

    #store q value(results of trajectory optimization) to the text file
    # newpath = 'OptRes'
    # if not os.path.exists(newpath):
    #     os.makedirs(newpath)
    # com_box = []
    # f_box = []
    #
    # for i in range(frame_num):
    #     com_box.append(mp.get_com_position(i))
    #     f_box.append(mp.get_contact_force(i))
    #
    # day = datetime.today().strftime("%Y%m%d%H%M")
    #
    # with open('OptRes/com_pos_' + day + '.txt', 'w') as f:
    #     for item in com_box:
    #         f.write("%s\n" % item)
    #
    # with open('OptRes/force_' + day + '.txt', 'w') as f:
    #     for item in f_box:
    #         f.write("%s\n" % item)

    viewer = hsv.hpSimpleViewer(viewForceWnd=False)
    viewer.setMaxFrame(1000)
    viewer.doc.addRenderer('controlModel', yr.DartRenderer(world, (255, 255, 255), yr.POLYGON_FILL), visible=False)
    viewer.doc.addRenderer('ground', yr.DartRenderer(ground, (255, 255, 255), yr.POLYGON_FILL))

    viewer.startTimer(1 / 25.)
    viewer.motionViewWnd.glWindow.pOnPlaneshadow = (0., -0.99 + 0.0251, 0.)
    viewer.doc.addRenderer('com_pos', yr.PointsRenderer(com_pos))
    viewer.doc.addRenderer('l_footCenter', yr.PointsRenderer(l_footCenter, (0, 255, 0)))
    viewer.doc.addRenderer('r_footCenter', yr.PointsRenderer(r_footCenter, (0, 0, 255)))

    def simulateCallback(frame):
        world.step(frame)
        world.render_with_ys(frame)

    viewer.setSimulateCallback(simulateCallback)
    viewer.show()

    Fl.run()