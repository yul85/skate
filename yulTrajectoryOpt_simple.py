import numpy as np
import math
from scipy import optimize
from motionPlan_simple import motionPlan
import pydart2 as pydart
import time
import yulQP

import os
from datetime import datetime

from fltk import *
from PyCommon.modules.GUI import hpSimpleViewer as hsv
from PyCommon.modules.Renderer import ysRenderer as yr
from PyCommon.modules.Simulator import hpDartQpSimulator as hqp

# readMode = True
readMode = False

np.set_printoptions(threshold=np.nan)

offset_list = [[-0.1040 + 0.0216, +0.80354016 - 0.85354016, 0.0],
               [0.1040 + 0.0216, +0.80354016 - 0.85354016, 0.0],
               [-0.1040 + 0.0216, +0.80354016 - 0.85354016, 0.0],
               [0.1040 + 0.0216, +0.80354016 - 0.85354016, 0.0]]

com_pos = []
l_footCenter = []
r_footCenter = []

# target_l_foot = np.array([0., -0.98, -0.2])
# target_r_foot = np.array([0., -0.98, 0.2])


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

    target_COM = world.target_COM
    target_l_foot = world.target_l_foot
    target_r_foot = world.target_r_foot

    # print(skel.body('h_blade_right').to_world([-0.1040 + 0.0216, +0.80354016 - 0.85354016, 0.0]))

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
        if i < int(T/2):
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
    w_track = 50.
    w_dis = 50.
    w_smooth = 0.1
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

        # tracking
        tracking_objective = 0.

        for i in range(T):
            x_ref = np.zeros(10)
            x_state = np.zeros(10)
            x_ref[0:3] = target_COM[3 * i:3 * (i + 1)]
            x_ref[3] = target_l_foot[3 * i]
            x_ref[4] = target_l_foot[3 * i + 2]
            x_ref[5] = target_r_foot[3 * i]
            x_ref[6] = target_r_foot[3 * i + 2]
            x_state[0:3] = mp.get_com_position(i)
            x_state[3:5] = mp.get_end_effector_l_position(i)
            x_state[5:7] = mp.get_end_effector_r_position(i)
            x_state[7:10] = mp.get_angular_momentum(i)

            tracking_objective = tracking_objective + np.linalg.norm(x_ref - x_state) ** 2
        # speed
        # speed_objective = np.linalg.norm((mp.get_COM_position(T-1) - mp.get_COM_position(0)) - t)**2
        # turning rate
        # + w_turn*((mp[T] - mp[0]) - t)*((mp[T] - mp[0]) - t)
        # balance

        #effort

        #distance btw COM and a foot
        distance_objective = 0.
        for i in range(T):
            l_f = np.zeros(3)
            l_f[0] = mp.get_end_effector_l_position(i)[0]
            l_f[1] = target_l_foot[3 * i + 1]
            l_f[2] = mp.get_end_effector_l_position(i)[1]

            r_f = np.zeros(3)
            r_f[0]= mp.get_end_effector_r_position(i)[0]
            r_f[1] = target_r_foot[3 * i + 1]
            r_f[2] = mp.get_end_effector_r_position(i)[1]

            d_l = np.linalg.norm(np.linalg.norm(mp.get_com_position(i) - l_f))
            d_r = np.linalg.norm(np.linalg.norm(mp.get_com_position(i) - r_f))
            ref_l_dis = np.linalg.norm(np.array(target_COM[3 * i: 3 * (i + 1)]) - np.array(target_l_foot[3 * i: 3 * (i + 1)]))
            ref_r_dis = np.linalg.norm(np.array(target_COM[3 * i: 3 * (i + 1)]) - np.array(target_r_foot[3 * i: 3 * (i + 1)]))
            distance_objective = distance_objective + (d_l - ref_l_dis) ** 2 + (d_r - ref_r_dis) ** 2

        # smooth term
        smooth_objective = 0.
        for i in range(1, T-1):
            com_vel1 = (mp.get_com_position(i) - mp.get_com_position(i - 1)) * (1 / h)
            com_vel2 = (mp.get_com_position(i + 1) - mp.get_com_position(i)) * (1 / h)
            com_acc = (1 / (h * h)) * (com_vel2 - com_vel1)
            am_val_der = (mp.get_angular_momentum(i) - mp.get_angular_momentum(i - 1)) * (1 / h)
            smooth_objective = smooth_objective + np.linalg.norm(np.vstack([com_acc, am_val_der]))**2

        return w_track * tracking_objective + w_dis * distance_objective #+ w_smooth * smooth_objective #+ w_effort * effort_objective

    #####################################################
    # equality constraints
    #####################################################

    # equation of motion
    def eom_i(x):
        mp = motionPlan(skel, T, x)
        cons_value = [0] * (3 * (T-2))
        for i in range(1, T-1):
            # print(i)
            # print(mp.get_com_position(i+1))
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
            cons_value[3 * (i - 1):3 * i] = skel.mass() * com_acc - skel.mass() * np.array([0., -9.8, 0.]) - contact_sum
            # print(cons_value[3 * (i - 1):3 * i])
            # cons_value[ndofs * (i-1):ndofs * i] = skel.mass() * com_acc - skel.mass()*np.array([0., -9.8, 0.]) - contact_sum

        return cons_value

    def am(x):
        mp = motionPlan(skel, T, x)
        cons_value = [0] * (3 * (T - 1))
        for i in range(1, T):
            contact_forces = mp.get_contact_force(i)
            # am_val = mp.get_angular_momentum(i)
            am_val_der = (mp.get_angular_momentum(i) - mp.get_angular_momentum(i-1))*(1/h)
            sum_diff_cop_com = 0

            lfp = np.zeros(3)
            lfp[0] = mp.get_end_effector_l_position(i)[0]
            lfp[1] = target_l_foot[1]
            lfp[2] = mp.get_end_effector_l_position(i)[1]
            rfp = np.zeros(3)
            rfp[0] = mp.get_end_effector_r_position(i)[0]
            rfp[1] = target_r_foot[1]
            rfp[2] = mp.get_end_effector_r_position(i)[1]

            if contact_flag[4 * i + 0] == 1:
                sum_diff_cop_com = sum_diff_cop_com + np.cross(lfp - mp.get_com_position(i), contact_forces[0:3])
            if contact_flag[4 * i + 1] == 1:
                sum_diff_cop_com = sum_diff_cop_com + np.cross(lfp - mp.get_com_position(i), contact_forces[3:6])
            if contact_flag[4 * i + 2] == 1:
                sum_diff_cop_com = sum_diff_cop_com + np.cross(rfp - mp.get_com_position(i), contact_forces[6:9])
            if contact_flag[4 * i + 3] == 1:
                sum_diff_cop_com = sum_diff_cop_com + np.cross(rfp - mp.get_com_position(i), contact_forces[9:12])

            cons_value[3 * (i - 1):3 * i] = am_val_der - sum_diff_cop_com

        return cons_value

    # non_holonomic constraint
    def non_holonomic(x):
        mp = motionPlan(skel, T, x)
        cons_value = [0] * (2 * (T-1))

        pre_pos_l = mp.get_end_effector_l_position(0)
        pre_pos_r = mp.get_end_effector_r_position(0)

        for i in range(1, T):
            lfp = np.zeros(3)
            lfp[0] = mp.get_end_effector_l_position(i)[0]
            lfp[1] = skel.body('h_blade_left').com()[1]
            lfp[2] = mp.get_end_effector_l_position(i)[1]

            rfp = np.zeros(3)
            rfp[0] = mp.get_end_effector_r_position(i)[0]
            rfp[1] = skel.body('h_blade_right').com()[1]
            rfp[2] = mp.get_end_effector_r_position(i)[1]

            p1 = lfp + np.array(offset_list[0])
            p2 = rfp + np.array(offset_list[1])

            blade_direction_vec = p2 - p1
            norm_vec = blade_direction_vec / np.linalg.norm(blade_direction_vec)
            # print("norm: ", norm_vec)
            # print('point: ', p1, p2, blade_direction_vec)
            # print('point: ', p1, p2, norm_vec)
            # print("dot product:", np.array([1., 0., 0.]).dot(norm_vec))
            theta_l = math.acos(np.array([-1., 0., 0.]).dot(norm_vec))
            theta_r = math.acos(np.array([1., 0., 0.]).dot(norm_vec))

            # print("theta_l:", theta_l, theta_r)
            #todo : angular velocity
            next_step_angle_l = theta_l + skel.body('h_blade_left').world_angular_velocity()[1] * h
            next_step_angle_r = theta_r + skel.body('h_blade_right').world_angular_velocity()[1] * h

            # print("angular_vel: ", skel.body('h_blade_left').world_angular_velocity()[1] * h, skel.body('h_blade_right').world_angular_velocity()[1] * h)
            # print("next_step_angle: ", next_step_angle_l, next_step_angle_r)
            sa_l = math.sin(next_step_angle_l)
            ca_l = math.cos(next_step_angle_l)

            sa_r = math.sin(next_step_angle_r)
            ca_r = math.cos(next_step_angle_r)

            vel_l = (mp.get_end_effector_l_position(i) - pre_pos_l) / h
            vel_r = (mp.get_end_effector_r_position(i) - pre_pos_r) / h

            cons_value[2 * (i-1) + 0] = vel_l[0] * sa_l - vel_l[1] * ca_l
            cons_value[2 * (i-1) + 1] = vel_r[0] * sa_r - vel_r[1] * ca_r

            pre_pos_l = mp.get_end_effector_l_position(i)
            pre_pos_r = mp.get_end_effector_r_position(i)

        # print(cons_value)
        return cons_value

    # Contact_flag [ 0 or 1 ]
    # def is_contact(x):
    #     mp = motionPlan(skel, T, x)
    #
    #     cons_value = [0] * (total_cn * 3)
    #     cn_till = 0
    #     for i in range(T):
    #         contact_force = mp.get_contact_force(i)
    #         k = 0
    #         if contact_flag[4 * i + 0] == 0:
    #             cons_value[(cn_till+k)*3:(cn_till+(k+1))*3] = contact_force[0:3]
    #             k = k + 1
    #         if contact_flag[4 * i + 1] == 0:
    #             cons_value[(cn_till+k)*3:(cn_till+(k+1))*3] = contact_force[3:6]
    #             k = k + 1
    #         if contact_flag[4 * i + 2] == 0:
    #             cons_value[(cn_till+k)*3:(cn_till+(k+1))*3] = contact_force[6:9]
    #             k = k + 1
    #         if contact_flag[4 * i + 3] == 0:
    #             cons_value[(cn_till+k)*3:(cn_till+(k+1))*3] = contact_force[9:12]
    #             k = k + 1
    #         cn_till = cn_till + cn[i]
    #         # cons_value[cn * 3*i:cn * 3*(i+1)] = contact_force[0:6]
    #     return cons_value

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

    # def swing(x):
    #     mp = motionPlan(skel, T, x)
    #     cons_value = [0] * (3*4*T - total_cn)
    #     cn_till = 0
    #     for i in range(T):
    #         k = 0
    #         if contact_flag[4 * i + 0] == 0:
    #             cons_value[cn_till+k] = mp.get_end_effector_l_position(i)[1] + 0.98
    #             k = k+1
    #         if contact_flag[4 * i + 1] == 0:
    #             cons_value[cn_till+k] = mp.get_end_effector_l_position(i)[1] + 0.98
    #             k = k + 1
    #         if contact_flag[4 * i + 2] == 0:
    #             cons_value[cn_till+k] = mp.get_end_effector_r_position(i)[1] + 0.98
    #             k = k + 1
    #         if contact_flag[4 * i + 3] == 0:
    #             cons_value[cn_till+k] = mp.get_end_effector_r_position(i)[1] + 0.98
    #             k = k + 1
    #         cn_till = cn_till + cn[i]
    #     return cons_value
    #
    # def stance(x):
    #     mp = motionPlan(skel, T, x)
    #     cons_value = [0] *(2 * total_cn)
    #     cn_till = 0
    #
    #     threshold = 0.035
    #     for i in range(T):
    #         k = 0
    #         if contact_flag[4 * i + 0] == 1:
    #             y_pos = mp.get_end_effector_l_position(i)[1]
    #             cons_value[cn_till + k] = y_pos + 0.98 - threshold
    #             cons_value[cn_till + k+1] = -y_pos - 0.98 + threshold
    #             k = k + 2
    #         if contact_flag[4 * i + 1] == 1:
    #             y_pos = mp.get_end_effector_l_position(i)[1]
    #             cons_value[cn_till + k] = y_pos + 0.98 - threshold
    #             cons_value[cn_till + k+1] = -y_pos - 0.98 + threshold
    #             k = k + 2
    #         if contact_flag[4 * i + 2] == 1:
    #             y_pos = mp.get_end_effector_r_position(i)[1]
    #             cons_value[cn_till + k] = y_pos + 0.98 - threshold
    #             cons_value[cn_till + k+1] = -y_pos - 0.98 + threshold
    #             k = k + 2
    #         if contact_flag[4 * i + 3] == 1:
    #             y_pos = mp.get_end_effector_r_position(i)[1]
    #             cons_value[cn_till + k] = y_pos + 0.98 - threshold
    #             cons_value[cn_till + k+1] = -y_pos - 0.98 + threshold
    #             k = k + 2
    #         cn_till = cn_till + cn[i]
    #     return cons_value


    def foot_collision(x):

        foot_threshold = 0.1
        mp = motionPlan(skel, T, x)
        cons_value = [0] * T
        cn_till = 0
        for i in range(T):
            lfp = mp.get_end_effector_l_position(i)
            rfp = mp.get_end_effector_r_position(i)
            dis = np.linalg.norm(lfp-rfp)
            cons_value[i] = dis - foot_threshold
        return cons_value

    cons_list.append({'type': 'eq', 'fun': eom_i})
    cons_list.append({'type': 'eq', 'fun': am})
    cons_list.append({'type': 'eq', 'fun': non_holonomic})

    cons_list.append({'type': 'ineq', 'fun': friction_normal})
    cons_list.append({'type': 'ineq', 'fun': friction_tangent})
    cons_list.append({'type': 'ineq', 'fun': tangential_vel})
    cons_list.append({'type': 'ineq', 'fun': foot_collision})

    # cons_list.append({'type': 'eq', 'fun': is_contact})
    # cons_list.append({'type': 'ineq', 'fun': swing})
    # cons_list.append({'type': 'ineq', 'fun': stance})

    # toc()
    #####################################################
    # solve
    #####################################################

    tic()
    # bnds = ()

    x0 = [0.] * ((3 + 2 * 2 + 3 * 4 + 3) * T)  #  initial guess

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

        pelvis_x = skel.dof_indices((["j_pelvis_rot_x"]))
        pelvis = skel.dof_indices((["j_pelvis_rot_y", "j_pelvis_rot_z"]))
        # upper_body = skel.dof_indices(["j_abdomen_1", "j_abdomen_2"])
        right_leg = skel.dof_indices(["j_thigh_right_x", "j_thigh_right_y", "j_thigh_right_z", "j_shin_right_z"])
        left_leg = skel.dof_indices(["j_thigh_left_x", "j_thigh_left_y", "j_thigh_left_z", "j_shin_left_z"])
        arms = skel.dof_indices(["j_bicep_left_x", "j_bicep_right_x"])
        # foot = skel.dof_indices(["j_heel_left_1", "j_heel_left_2", "j_heel_right_1", "j_heel_right_2"])
        leg_y = skel.dof_indices(["j_thigh_right_y", "j_thigh_left_y"])
        # blade = skel.dof_indices(["j_heel_right_2"])

        s0q = np.zeros(skel.ndofs)
        # s0q[pelvis] = 0., -0.
        # s0q[upper_body] = 0.3, -0.
        s0q[right_leg] = -0., -0., 0.9, -1.5
        s0q[left_leg] = -0.1, 0., 0.0, -0.0
        # s0q[leg_y] = -0.785, 0.785
        # s0q[arms] = 1.5, -1.5
        # s0q[foot] = 0., 0.1, 0., -0.0

        self.s0q = s0q

        s1q = np.zeros(skel.ndofs)
        # s1q[pelvis] = 0., -0.
        # s1q[upper_body] = 0.3, -0.
        s1q[right_leg] = -0.1, 0., 0.0, -0.0
        s1q[left_leg] = -0., -0., 0.9, -1.5
        # s1q[leg_y] = -0.785, 0.785
        # s1q[arms] = 1.5, -1.5
        # s1q[foot] = 0., 0.1, 0., -0.0

        self.s1q = s1q

        self.skeletons[3].set_positions(s0q)

        self.res_com_ff = []
        self.res_fl_ff = []
        self.res_fr_ff = []
        self.res_am_ff = []

        self.frame_num = 0

        self.target_COM = []
        self.target_l_foot = []
        self.target_r_foot = []

    def step(self, i):
        # print("step")
        h = self.time_step()
        # ======================================================
        #   todo : Full body Motion Synthesis
        #   Prioritized optimization scheme(Lasa et al.(2010))
        #      or JUST QP  ....
        # ======================================================
        if i < int(self.frame_num/2):
            state_q = self.s0q
        else:
            state_q = self.s1q
            self.skeletons[3].set_positions(state_q)
        ndofs = skel.num_dofs()
        h = self.time_step()
        Kp = np.diagflat([0.0] * 6 + [25.0] * (ndofs - 6))
        Kd = np.diagflat([0.0] * 6 + [2. * (25.0 ** .5)] * (ndofs - 6))
        invM = np.linalg.inv(skel.M + Kd * h)
        p = -Kp.dot(skel.q - state_q + skel.dq * h)
        d = -Kd.dot(skel.dq)
        qddot = invM.dot(-skel.c + p + d + skel.constraint_forces())
        des_accel = p + d + qddot
        # print("ddq: ", des_accel)

        # skel.set_positions(self.s0q)
        # print(skel.body('h_blade_left').to_world([0., 0., 0]), skel.body('h_blade_left').com())
        # print(skel.body('h_blade_right').to_world([0., 0., 0]), skel.body('h_blade_right').com())

        ddc = np.zeros(6)
        ddf_l = np.zeros(3)
        ddf_r = np.zeros(3)

        if i >= 1:
            if readMode == True:
                ddc[0:3] = 400. * (np.array(world.res_com_ff[i]) - skel.com()) - 10. * skel.dC
                # ddc[3:6] = 400. * 1/h * (np.array(world.res_com_ff[i]) - np.array(world.res_com_ff[i-1]))
                ddc[3:6] = 400. * 1/h * (np.array(world.res_am_ff[i]) - np.array(world.res_am_ff[i-1]))
                ddf_l = 400. * (np.array([world.res_fl_ff[i][0], world.target_l_foot[3*i + 1], world.res_fl_ff[i][1]]) - skel.body('h_blade_left').com()) - 10. * skel.body('h_blade_left').com_linear_velocity()
                ddf_r = 400. * (np.array([world.res_fr_ff[i][0], world.target_r_foot[3*i + 1], world.res_fr_ff[i][1]]) - skel.body('h_blade_right').com()) - 10. * skel.body('h_blade_right').com_linear_velocity()
            else:
                ddc[0:3] = 400. * (mp.get_com_position(i) - skel.com()) - 10. * skel.dC
                ddc[3:6] = 400. * 1/h * (mp.get_angular_momentum(i) - mp.get_angular_momentum(i-1))

                lfp = np.zeros(3)
                lfp[0] = mp.get_end_effector_l_position(i)[0]
                lfp[1] = world.target_l_foot[3*i + 1]
                lfp[2] = mp.get_end_effector_l_position(i)[1]

                rfp = np.zeros(3)
                rfp[0] = mp.get_end_effector_r_position(i)[0]
                rfp[1] = world.target_r_foot[3*i + 1]
                rfp[2] = mp.get_end_effector_r_position(i)[1]

                ddf_l = 400. * (lfp - skel.body('h_blade_left').com()) - 10. * skel.body('h_blade_left').com_linear_velocity()
                ddf_r = 400. * (rfp - skel.body('h_blade_right').com()) - 10. * skel.body('h_blade_right').com_linear_velocity()

        # ddf_l[1] = -0.87
        # ddf_r[1] = -0.66

        _ddq, _tau, _bodyIDs, _contactPositions, _contactPositionLocals, _contactForces = yulQP.calc_QP(skel, des_accel, ddc, ddf_l, ddf_r, 1. / h)

        # _ddq, _tau, _bodyIDs, _contactPositions, _contactPositionLocals, _contactForces = hqp.calc_QP(
        #     skel, des_accel, 1. / self.time_step())

        # print(_contactForces)
        for i in range(len(_bodyIDs)):
            skel.body(_bodyIDs[i]).add_ext_force(_contactForces[i], _contactPositionLocals[i])
        # dartModel.applyPenaltyForce(_bodyIDs, _contactPositionLocals, _contactForces)
        skel.set_forces(_tau)

        # cf = mp.get_contact_force(i)
        # # print(cf)
        # skel.body('h_blade_left').add_ext_force(cf[0:3], offset_list[0])
        # skel.body('h_blade_left').add_ext_force(cf[3:6], offset_list[1])
        # skel.body('h_blade_right').add_ext_force(cf[6:9], offset_list[2])
        # skel.body('h_blade_right').add_ext_force(cf[9:12], offset_list[3])

        skel.set_forces(_tau)

        super(MyWorld, self).step()

    def render_with_ys(self, i):
        del com_pos[:]
        del l_footCenter[:]
        del r_footCenter[:]

        if readMode == True:
            com_pos.append(world.res_com_ff[i])
            l_footCenter.append(np.array([world.res_fl_ff[i][0], world.target_l_foot[3*i + 1], world.res_fl_ff[i][1]]))
            r_footCenter.append(np.array([world.res_fr_ff[i][0], world.target_r_foot[3*i + 1], world.res_fr_ff[i][1]]))
        else:
            com_pos.append(mp.get_com_position(i))
            lfp = np.zeros(3)
            lfp[0] = mp.get_end_effector_l_position(i)[0]
            lfp[1] = world.target_l_foot[3*i + 1]
            lfp[2] = mp.get_end_effector_l_position(i)[1]

            rfp = np.zeros(3)
            rfp[0] = mp.get_end_effector_r_position(i)[0]
            rfp[1] = world.target_r_foot[3*i + 1]
            rfp[2] = mp.get_end_effector_r_position(i)[1]
            l_footCenter.append(lfp)
            r_footCenter.append(rfp)

if __name__ == '__main__':
    pydart.init()
    world = MyWorld()
    skel = world.skeletons[2]

    ground = pydart.World(1. / 50., './data/skel/ground.skel')

    # print(solve_trajectory_optimization(skel, 10, world.time_step())['x'])
    # print(solve_trajectory_optimization(skel, 10, world.time_step()))

    frame_num = 21
    # frame_num = 10
    world.frame_num = frame_num

    # Given information = Reference motion
    T = frame_num
    world.target_COM = [0] * (3 * T)
    world.target_l_foot = [0] * (3 * T)
    world.target_r_foot = [0] * (3 * T)

    for i in range(T):
        world.target_COM[3 * i] = 0.05 * i
        world.target_COM[3 * i + 1] = 0.
        world.target_COM[3 * i + 2] = 0.

        world.target_l_foot[3 * i] = 0.05 * i
        world.target_l_foot[3 * i + 2] = -0.09
        world.target_r_foot[3 * i] = 0.05 * i
        world.target_r_foot[3 * i + 2] = 0.09

        if i < int(T / 2):
            world.target_l_foot[3 * i + 1] = -0.87 - 0.07
            world.target_r_foot[3 * i + 1] = -0.66 - 0.07
        else:
            world.target_l_foot[3 * i + 1] = -0.66 - 0.07
            world.target_r_foot[3 * i + 1] = -0.87 - 0.07

    if readMode == False:
        opt_res = solve_trajectory_optimization(skel, frame_num, world.time_step())
        print(opt_res)
        # print("trajectory optimization finished!!")
        mp = motionPlan(skel, frame_num, opt_res['x'])

        # store q value(results of trajectory optimization) to the text file
        newpath = 'OptRes'
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        com_box = []
        fl_box = []
        fr_box = []
        am_box = []

        for i in range(frame_num):
            com_box.append(mp.get_com_position(i))
            fl_box.append(mp.get_end_effector_l_position(i))
            fr_box.append(mp.get_end_effector_r_position(i))
            am_box.append(mp.get_angular_momentum(i))

        day = datetime.today().strftime("%Y%m%d%H%M")

        with open('OptRes/com_pos_' + day + '.txt', 'w') as f:
            for item in com_box:
                # print("item: ", item[0], item[1], item[2])
                f.write("%s " % item[0])
                f.write("%s " % item[1])
                f.write("%s\n" % item[2])
        with open('OptRes/fl_' + day + '.txt', 'w') as f:
            for item in fl_box:
                f.write("%s " % item[0])
                f.write("%s\n" % item[1])
        with open('OptRes/fr_' + day + '.txt', 'w') as f:
            for item in fr_box:
                f.write("%s " % item[0])
                f.write("%s\n" % item[1])
        with open('OptRes/am_' + day + '.txt', 'w') as f:
            for item in am_box:
                # print("item: ", item[0], item[1], item[2])
                f.write("%s " % item[0])
                f.write("%s " % item[1])
                f.write("%s\n" % item[2])

    if readMode == True:
        # read file
        f_com = open("OptRes/com_pos_201810011530.txt", "r")
        f_fl = open("OptRes/fl_201810011530.txt", "r")
        f_fr = open("OptRes/fr_201810011530.txt", "r")
        f_am = open("OptRes/am_201810011530.txt", "r")

        # f_com = open("OptRes/com_pos_201809141443.txt", "r")
        # f_fl = open("OptRes/fl_201809141443.txt", "r")
        # f_fr = open("OptRes/fr_201809141443.txt", "r")

        # f_com = open("OptRes/com_pos_201809191214.txt", "r")
        # f_fl = open("OptRes/fl_201809191214.txt", "r")
        # f_fr = open("OptRes/fr_201809191214.txt", "r")

        res_com_ff = []
        res_fl_ff = []
        res_fr_ff = []
        res_am_ff = []

        for line in f_com:
            # print(line)
            value = line.split(" ")
            # print(value)
            # print("??: ", value[0], value[1], value[2], type(value[2]))
            vec = [float(value[0]), float(value[1]), float(value[2])]
            # print(vec)
            res_com_ff.append(vec)
        f_com.close()

        for line in f_fl:
            value = line.split(" ")
            vec = [float(value[0]), float(value[1])]
            res_fl_ff.append(vec)
        f_fl.close()

        for line in f_fr:
            value = line.split(" ")
            vec = [float(value[0]), float(value[1])]
            res_fr_ff.append(vec)
        f_fr.close()

        for line in f_am:
            value = line.split(" ")
            vec = [float(value[0]), float(value[1]), float(value[2])]
            res_am_ff.append(vec)
        f_am.close()

        world.res_com_ff = res_com_ff
        world.res_fl_ff = res_fl_ff
        world.res_fr_ff = res_fr_ff
        world.res_am_ff = res_am_ff

    viewer = hsv.hpSimpleViewer(viewForceWnd=False)
    viewer.setMaxFrame(1000)
    # viewer.doc.addRenderer('controlModel', yr.DartRenderer(world, (255, 255, 255), yr.POLYGON_FILL), visible=False)
    viewer.doc.addRenderer('controlModel', yr.DartRenderer(world, (255, 255, 255), yr.POLYGON_FILL))
    viewer.doc.addRenderer('ground', yr.DartRenderer(ground, (255, 255, 255), yr.POLYGON_FILL))

    viewer.startTimer(1 / 25.)
    viewer.motionViewWnd.glWindow.pOnPlaneshadow = (0., -0.99 + 0.0251, 0.)
    viewer.doc.addRenderer('com_pos', yr.PointsRenderer(com_pos))
    viewer.doc.addRenderer('l_footCenter', yr.PointsRenderer(l_footCenter, (0, 255, 0)))
    viewer.doc.addRenderer('r_footCenter', yr.PointsRenderer(r_footCenter, (0, 0, 255)))
    viewer.motionViewWnd.glWindow.planeHeight = -0.98+0.0251
    # yr.drawCross(np.array([0., 0., 0.]))
    def simulateCallback(frame):
        world.step(frame)
        world.render_with_ys(frame)

    viewer.setSimulateCallback(simulateCallback)
    viewer.show()

    Fl.run()