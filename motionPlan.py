import numpy as np
import math
import pydart2 as pydart
from scipy import optimize

class motionPlan(object):
    def __init__(self, skel, T, h):
        self.skel = skel
        self.ndofs = skel.num_dofs()
        self.mp = []
        self.T = T
        self.h = h
        m_length = self.ndofs + 3 + 3*2 + 3 * 4 + 3*2 + 2*2
        m = np.zeros(m_length)
        for ii in range(0, T):
            self.mp.append(m)


    def get_pose(self, i):
        return self.mp[i][0:self.ndofs]

    def get_COM_position(self, i):
        return self.mp[i][self.ndofs:self.ndofs+3]

    def get_end_effector_l_position(self, i):
        return self.mp[i][self.ndofs+3:self.ndofs+3+3]

    def get_end_effector_r_position(self, i):
        return self.mp[i][self.ndofs+3+3:self.ndofs+3+3*2]

    def get_contact_force(self, i):
        return self.mp[i][self.ndofs+3+3*2:self.ndofs+3+3*2 + 3 * 4]

    def get_end_effector_angular_speed(self, i):
        return self.mp[i][self.ndofs+3+3*2 + 3 * 4:self.ndofs+3+3*2 + 3 * 4 + 3*2]

    def get_end_effector_rotation_angle(self, i):
        return self.mp[i][self.ndofs+3+3*2 + 3 * 4 + 3*2:]

    # def get_end_effector_rotation_info(self, i):
    #
    #     angular_speed_l = self.mp[i][self.ndofs + 3 + 3 * 2+ 3 * 4:self.ndofs + 3 + 3 * 2 + 3 * 4 + 3]
    #     angular_speed_r = self.mp[i][self.ndofs + 3 + 3 * 2 + 3 * 4 + 3:self.ndofs + 3 + 3 * 2 + 3 * 4 + 3 * 2]
    #
    #     rotation_angle_l = self.mp[i][self.ndofs + 3 + 3 * 2 + 3 * 4 + 3 * 2:self.ndofs + 3 + 3 * 2 + 3 * 4 + 3 * 2 + 2]
    #     rotation_angle_r = self.mp[i][self.ndofs + 3 + 3 * 2 + 3 * 4 + 3 * 2 + 2:self.ndofs + 3 + 3 * 2 + 3 * 4 + 3 * 2 + 2 * 2]
    #
    #
    #     return self.mp[i][0:self.ndofs]


    def solve_trajectory_optimization(self):
        T = self.T
        mp = self.mp
        skel = self.skel

        #   Given information

        target_COM = np.array([0., 0., 0])
        target_l_foot = np.array([0., 0., 0])
        target_r_foot = np.array([0., 0., 0])
        contact_flag = np.array([1, 1])

        t= 1.0
        #####################################################
        # objective
        #####################################################

        #The objective function to be minimized.
        # walking speed + turning rate + regularizing term + target position of end-effoctor + target position of COM trajectory

        w_speed = 50
        w_turn = 50
        w_end = 50
        w_com = 50
        w_regul = 0.1

        def func(mp):
            # todo: quadratic term
            regularize_objective = 0.
            for i in range(1, T - 1):
                regularize_objective = regularize_objective + (
                            mp[i - 1].get_pose() - 2 * mp[i].get_pose() + mp[i + 1].get_pose()) * (
                                                   mp[i - 1].get_pose() - 2 * mp[i].get_pose() + mp[i + 1].get_pose())

            end_effector_objective = 0.
            for i in range(0, T-1):
                end_effector_objective = end_effector_objective + (
                            mp[i].get_end_effector_l_position() - target_l_foot) * (
                                                     mp[i].get_end_effector_l_position() - target_l_foot) + (
                                                     mp[i].get_end_effector_r_position() - target_r_foot) * (
                                                 mp[i].get_end_effector_r_position() - target_r_foot)

            com_objective = 0.
            for i in range(0, T-1):
                com_objective = com_objective + (mp[i].get_COM_position() - target_COM)*(mp[i].get_COM_position() - target_COM)

            return w_speed*((mp[T].get_COM_position() - mp[0].get_COM_position()) - t)*((mp[T].get_COM_position() - mp[0].get_COM_position()) - t)      # speed
            #+ w_turn*((mp[T] - mp[0]) - t)*((mp[T] - mp[0]) - t)                     # turning rate
            + w_regul * regularize_objective                                          # regularizing term
            #+ w_end * end_effector_objective                                          # target position for end_effectors
            + w_com * com_objective                                                   # target position for COM
            #+                                                                        # balance

        #####################################################
        # equality
        #####################################################

        # motion of equation

        contact_forces = mp[i].get_contact_force()


        con_l1 = contact_forces[0:3]
        con_l2 = contact_forces[3:6]
        con_r1 = contact_forces[6:9]
        con_r2 = contact_forces[9:12]

        sum_contact = con_l1 + con_l2 + +con_r1 + con_r2

        com_vel = (mp[i].get_COM_position() - mp[i - 1].get_COM_position()) / self.h

        com_acc = 2/(self.h*self.h) * (mp[i].get_COM_position() - mp[i-1].get_COM_position() - com_vel * self.h)

        ec_1 = {'type': 'eq', 'fun': lambda mp: sum_contact + skel.c - skel.M * com_acc}

        # non_holonomic constraint
        p1 = skel.body('h_blade_left').to_world([-0.1040 + 0.0216, 0.0, 0.0])
        p2 = skel.body('h_blade_left').to_world([0.1040 + 0.0216, 0.0, 0.0])

        blade_direction_vec = p2 - p1

        theta_l = math.acos(np.dot(np.array([-1., 0., 0.]), blade_direction_vec))
        theta_r = math.acos(np.dot(np.array([1., 0., 0.]), blade_direction_vec))
        # print("theta: ", body_name, ", ", theta)
        # print("omega: ", skel.body("h_blade_left").world_angular_velocity()[1])

        next_step_angle_l = theta_l + skel.body('h_blade_left').world_angular_velocity()[1] * self.h
        next_step_angle_r = theta_r + skel.body('h_blade_right').world_angular_velocity()[1] * self.h
        # print("next_step_angle: ", next_step_angle)
        sa_l = math.sin(next_step_angle_l)
        ca_l = math.cos(next_step_angle_l)

        sa_r = math.sin(next_step_angle_r)
        ca_r = math.cos(next_step_angle_r)

        vel_l = (mp[i].get_end_effector_l_position() - mp[i - 1].get_end_effector_l_position())/self.h
        vel_r = (mp[i].get_end_effector_r_position() - mp[i - 1].get_end_effector_r_position()) / self.h

        ec_2 = {'type': 'eq', 'fun': lambda mp: vel_l[0]*sa_l + vel_l*ca_l}
        ec_3 = {'type': 'eq', 'fun': lambda mp: vel_r[0] * sa_r + vel_r * ca_r}

        #Contact_flag [ 0 or 1 ]
        ec_4 = {'type': 'eq', 'fun': lambda mp: np.dot(mp[i].get_contact_force(), ([1, 1] - contact_flag[0]))}


        #####################################################
        # inequality
        #####################################################

        # coulomb friction model
        myu = 0.02

        normal_f = np.aray([0., con_l1[1], 0.])
        tangential_f = np.aray([con_l1[0], 0., con_l1[2]])

        tangential_f + myu * normal_f

        ec_5 = {'type': 'ineq', 'fun': lambda mp: normal_f}
        ec_6 = {'type': 'ineq', 'fun': lambda mp: tangential_f + myu * normal_f}
        ec_7 = {'type': 'ineq', 'fun': lambda mp: -tangential_f + myu * normal_f}

        cons = (ec_1, ec_2, ec_3, ec_4, ec_5, ec_6, ec_7)

        #####################################################
        # solve
        #####################################################

        # bnds = ()
        x0 = mp     #initial guess

        res = optimize.minimize(func, x0, method='SLSQP', bounds=None, constraints=cons)

