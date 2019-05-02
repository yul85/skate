import numpy as np
import pydart2 as pydart
import math
import IKsolve_double_stance
import copy

from fltk import *
from PyCommon.modules.GUI import hpSimpleViewer as hsv
from PyCommon.modules.Renderer import ysRenderer as yr

render_vector = []
render_vector_origin = []
push_force = []
push_force_origin = []
blade_force = []
blade_force_origin = []

rd_footCenter = []

ik_on = True
# ik_on = False


class State(object):
    def __init__(self, name, dt, c_d, c_v, angles):
        self.name = name
        self.dt = dt
        self.c_d = c_d
        self.c_v = c_v
        self.angles = angles


class MyWorld(pydart.World):
    def __init__(self, ):
        pydart.World.__init__(self, 1.0 / 1000.0, './data/skel/cart_pole_blade_3dof_with_ground.skel')
        # pydart.World.__init__(self, 1.0 / 1000.0, './data/skel/cart_pole_blade.skel')
        # pydart.World.__init__(self, 1.0 / 2000.0, './data/skel/cart_pole.skel')
        self.force = None
        self.force_r = None
        self.force_l = None
        self.force_hip_r = None
        self.duration = 0
        self.skeletons[0].body('ground').set_friction_coeff(0.02)
        self.ground_height = self.skeletons[0].body(0).to_world((0., 0.025, 0.))[1]

        skel = self.skeletons[2]
        # print("mass: ", skel.m, "kg")

        # print('[Joint]')
        # for joint in skel.joints:
        #     print("\t" + str(joint))
        #     print("\t\tparent = " + str(joint.parent_bodynode))
        #     print("\t\tchild = " + str(joint.child_bodynode))
        #     print("\t\tdofs = " + str(joint.dofs))

        # skel.joint("j_abdomen").set_position_upper_limit(10, 0.0)

        # skel.joint("j_heel_left").set_position_upper_limit(0, 0.0)
        # skel.joint("j_heel_left").set_position_lower_limit(0, -0.0)
        pelvis_pos_y = skel.dof_indices(["j_pelvis_pos_y"])
        pelvis_x = skel.dof_indices(["j_pelvis_rot_x"])
        pelvis = skel.dof_indices(["j_pelvis_rot_y", "j_pelvis_rot_z"])
        upper_body = skel.dof_indices(["j_abdomen_x", "j_abdomen_y", "j_abdomen_z"])
        spine = skel.dof_indices(["j_spine_x", "j_spine_y", "j_spine_z"])
        right_leg = skel.dof_indices(["j_thigh_right_x", "j_thigh_right_y", "j_thigh_right_z", "j_shin_right_z"])
        left_leg = skel.dof_indices(["j_thigh_left_x", "j_thigh_left_y", "j_thigh_left_z", "j_shin_left_z"])
        knee = skel.dof_indices(["j_shin_left_x", "j_shin_right_x"])
        arms = skel.dof_indices(["j_bicep_left_x", "j_bicep_right_x"])
        arms_y = skel.dof_indices(["j_bicep_left_y", "j_bicep_right_y"])
        arms_z = skel.dof_indices(["j_bicep_left_z", "j_bicep_right_z"])
        foot = skel.dof_indices(["j_heel_left_x", "j_heel_left_y", "j_heel_left_z", "j_heel_right_x", "j_heel_right_y", "j_heel_right_z"])
        leg_y = skel.dof_indices(["j_thigh_left_y", "j_thigh_right_y"])
        # blade = skel.dof_indices(["j_heel_right_2"])


        # ===========pushing side to side new===========

        s00q = np.zeros(skel.ndofs)

        s00q[upper_body] = 0.0, -0., -0.5
        # s00q[spine] = 0.0, 0.0, 0.2
        s00q[left_leg] = 0.05, 0., 0.5, -0.7
        s00q[right_leg] = -0.5, -0., 0.5, -0.9
        s00q[leg_y] = 0., -0.785
        # s00q[arms] = 1.5, -1.5
        # s00q[arms_y] = -1.5, 1.5
        # s00q[arms_z] = 3., 3.
        s00q[foot] = 0.03, 0., 0.2, -0., -0., 0.4
        state00 = State("state00", 2.0, 0.0, 0.2, s00q)

        s01q = np.zeros(skel.ndofs)
        s01q[upper_body] = 0., 0.2, -0.3
        # s01q[spine] = 0.0, 0., -0.2
        s01q[left_leg] = 0.3, 0., 1.5, -1.7
        s01q[right_leg] = -0.3, -0., 0.5, -0.9
        s01q[leg_y] = 0.4, -0.4
        # s01q[arms_y] = -0.7, -0.7
        # s01q[knee] = 0.05, -0.
        # s01q[arms] = 1.5, -1.5
        # # s01q[blade] = -0.3
        # s01q[foot] = -0., 0.785, 0.2, 0., -0.785, 0.2
        s01q[foot] = 0., 0.4, 0.5, -0.1, -0.4, 0.4
        state01 = State("state01", 0.5, 2.2, 0.0, s01q)

        s012q = np.zeros(skel.ndofs)
        s012q[upper_body] = 0., -0., -0.5
        # s012q[spine] = 0.0, 0., -0.2
        s012q[left_leg] = 0.1, 0., 0.5, -1.
        s012q[right_leg] = -0.5, -0., 1.0, -2.0
        s012q[leg_y] = 0.4, -0.
        # s012q[knee] = 0.2, -0.
        # s012q[arms] = 1.5, -1.5
        # # s012q[blade] = -0.3
        # s012q[foot] = -0., 0.785, 0.2, 0., -0.785, 0.2
        s012q[foot] = 0.1, 0.4, 0.5, -0., -0., 1.0
        state012 = State("state012", 0.5, 2.2, 0.0, s012q)

        s02q = np.zeros(skel.ndofs)
        s02q[upper_body] = 0., -0.2, -0.5
        # s02q[spine] = 0., 0., 0.5
        s02q[left_leg] = 0.5, 0., 1.0, -2.0
        s02q[right_leg] = -0.1, -0., 0.3, -1.0
        # s02q[right_leg] = -0.4, -0., 0.3, -0.7
        s02q[leg_y] = 0., -0.4
        # s02q[arms] = 1.5, -1.5
        # s02q[knee] = 0.4, -0.
        s02q[foot] = 0., 0., 1.0, -0.1, -0.4, 0.5
        state02 = State("state02", 0.5, 2.2, 0.0, s02q)

        s022q = np.zeros(skel.ndofs)
        s022q[upper_body] = 0., 0., -0.5
        s022q[left_leg] = 0.5, 0., 0.5, -0.7
        s022q[right_leg] = -0., -0., 1.0, -2.0
        # s022q[right_leg] = -0.4, -0., 0.3, -0.7
        s022q[leg_y] = 0.4, -0.
        s022q[foot] = 0.1, 0.4, 0.2, -0., -0., 1.0
        state022 = State("state022", 0.5, 2.2, 0.0, s022q)

        s_terminal_q = np.zeros(skel.ndofs)

        state_t = State("state_t", 50., 2.0, 2.0, s_terminal_q)

        #=======================END ===============================================

        s03q = np.zeros(skel.ndofs)
        # s03q[pelvis] = 0., -0.1
        s03q[upper_body] = 0., 0., -0.2
        # s03q[spine] = 0., 0., 0.5
        s03q[left_leg] = 0.3, 0., -0.1, -2.
        s03q[right_leg] = -0.3, -0., 0.5, -0.7
        s03q[leg_y] = 0., -0.4
        # s03q[knee] = 0.2, 0.
        # s03q[arms] = 1.5, -1.5
        s03q[foot] = 0.2, 0., 0.5, -0.2, -0., 0.2
        state03 = State("state03", 1., 2.2, 0.0, s03q)

        s04q = np.zeros(skel.ndofs)
        # s04q[pelvis] = -0.3, -0.0
        s04q[upper_body] = -0., 0, -0.5
        # s04q[spine] = -0.3, -0.3, -0.
        s04q[left_leg] = -0., 0., 0.1, -0.3
        s04q[right_leg] = -0.5, -0., -0.5, -0.
        s04q[arms] = 1.5, -1.5
        s04q[leg_y] = 0.785, -0.785
        s04q[foot] = -0., 0.0, 0.2, -0.2, -0.0, 0.
        state04 = State("state04", 3.0, 0.0, 0.2, s04q)

        s05q = np.zeros(skel.ndofs)
        s05q[upper_body] = 0.0, 0., -0.3
        s05q[spine] = 0.0, 0., 0.3
        s05q[left_leg] = 0., -0., 0.3, -0.5
        s05q[right_leg] = -0., -0., 0.3, -0.5
        s05q[arms] = 1.5, -1.5
        s05q[foot] = -0., 0., 0.2, 0., -0., 0.2
        state05 = State("state05", 1.0, 0.0, 0.2, s05q)

        # =====================LEFT TURN=============

        s10q = np.zeros(skel.ndofs)
        # s10q[pelvis] = 0., -0.3
        s10q[upper_body] = 0.0, 0., -0.5
        s10q[spine] = 0.0, 0., 0.5
        s10q[left_leg] = -0., 0., 0.3, -0.5
        s10q[right_leg] = -0., -0., 0.3, -0.5
        s10q[arms] = 1.5, -1.5
        # s10q[blade] = -0.3
        s10q[foot] = -0.0, 0.0, 0.2, 0.0, 0.0, 0.2
        state10 = State("state10", 0.5, 2.2, 0.0, s10q)

        roro_angle = 10.

        s11q = np.zeros(skel.ndofs)
        s11q[upper_body] = -0., 0., -0.5
        s11q[spine] = -0., 0., 0.5
        s11q[left_leg] = 0., 0., 0.5, -0.9
        s11q[right_leg] = 0., -0., 0.5, -0.9
        s11q[knee] = -roro_angle * math.pi / 180., -roro_angle * math.pi / 180.
        # s11q[arms] = 1.5, -1.5
        # s11q[arms] = -0.5, -0.5
        s11q[arms] = -roro_angle * math.pi / 180., -roro_angle * math.pi / 180.
        # s11q[foot] = 0., 0., 0.2, 0., 0., 0.2
        s11q[foot] = -0., 0., 0.4, 0., 0., 0.4
        state11 = State("state11", 2.0, 0.0, 0.2, s11q)

        # s12q = np.zeros(skel.ndofs)
        # # s12q[pelvis_x] = -10. * math.pi/180.
        # s12q[upper_body] = 0., 0., -0.5
        # s12q[spine] = 0.0, 0., 0.5
        # s12q[left_leg] = -0., 0., 0.5, -0.9
        # s12q[right_leg] = 0., -0., 0.3, -0.5
        # s12q[knee] = -roro_angle * math.pi / 180., -roro_angle * math.pi / 180.
        # # s12q[arms] = 1.5, -1.5
        # s12q[arms] = -roro_angle * math.pi/180., -roro_angle * math.pi/180.
        # s12q[foot] = -0., 0., 0.4, 0., -0., 0.2
        # # s12q[foot] = -roro_angle * math.pi / 180., 0., 0., -roro_angle * math.pi / 180., -0., 0.
        # state12 = State("state12", 1.0, 0.0, 0.2, s12q)
        self.state_list = [state00, state01, state012, state02, state012, state02, state022, state02, state_t] #state022, state03, state04, state05, state11]
        # self.state_list = [state00, state01, state02, state03, state04, state05, state11]
        # self.state_list = [state00, state01, state012, state02, state03, state04, state05, state11]

        state_num = len(self.state_list)
        self.state_num = state_num
        # print("state_num: ", state_num)

        self.curr_state = self.state_list[0]
        self.elapsedTime = 0.0
        self.curr_state_index = 0
        # print("backup angle: ", backup_q)
        # print("cur angle: ", self.curr_state.angles)

        if ik_on:
            self.ik = IKsolve_double_stance.IKsolver(skel, self.skeletons[3], self.dt)
            print('IK ON')

        # print("after:", state00.angles)

        self.revise_pose(state00)
        # self.revise_pose(state01)
        # self.revise_pose(state012)
        # self.revise_pose(state02)

        # debug_rf1 = self.skeletons[3].body("h_blade_right").to_world([-0.1040 + 0.0216, -0.027 - 0.0216, 0.0])[1]
        # debug_rf2 = self.skeletons[3].body("h_blade_right").to_world([-0.1040 - 0.0216, -0.027 - 0.0216, 0.0])[1]
        # print("check: ", debug_rf1, debug_rf2)
        # print('create controller OK')

        self.skeletons[3].set_positions(self.curr_state.angles)

        self.rd_contact_forces = []
        self.rd_contact_positions = []

        # print("dof: ", skel.ndofs)

        # nonholonomic constraint initial setting
        _th = 5. * math.pi / 180.

        self.nh0 = pydart.constraints.NonHolonomicContactConstraint(skel.body('h_blade_right'), np.array((0.0216+0.104, -0.0216-0.027, 0.)))
        self.nh1 = pydart.constraints.NonHolonomicContactConstraint(skel.body('h_blade_right'), np.array((0.0216-0.104, -0.0216-0.027, 0.)))
        self.nh2 = pydart.constraints.NonHolonomicContactConstraint(skel.body('h_blade_left'), np.array((0.0216+0.104, -0.0216-0.027, 0.)))
        self.nh3 = pydart.constraints.NonHolonomicContactConstraint(skel.body('h_blade_left'), np.array((0.0216-0.104, -0.0216-0.027, 0.)))

        self.nh1.set_violation_angle_ignore_threshold(_th)
        self.nh1.set_length_for_violation_ignore(0.208)

        self.nh3.set_violation_angle_ignore_threshold(_th)
        self.nh3.set_length_for_violation_ignore(0.208)

        self.nh0.add_to_world()
        self.nh1.add_to_world()
        self.nh2.add_to_world()
        self.nh3.add_to_world()

        self.step_count = 0

    def revise_pose(self, pose):

        self.skeletons[3].set_positions(pose.angles)
        ground_height = max(self.skeletons[3].body('h_blade_right').to_world((-0.104 + 0.0216, -0.027 - 0.0216, 0.))[1],
                            self.skeletons[3].body('h_blade_right').to_world((+0.104 + 0.0216, -0.027 - 0.0216, 0.))[1],
                            self.skeletons[3].body('h_blade_left').to_world((-0.104 + 0.0216, -0.027 - 0.0216, 0.))[1],
                            self.skeletons[3].body('h_blade_left').to_world((+0.104 + 0.0216, -0.027 - 0.0216, 0.))[1]
                            )

        ik_res = copy.deepcopy(pose.angles)
        self.ik.update_target(ground_height)
        ik_res[6:] = self.ik.solve()
        pose.angles = ik_res

    def step(self):

        # print("self.curr_state: ", self.curr_state.name)

        # if self.curr_state.name == "state1":
        #     self.force = np.array([50.0, 0.0, 0.0])
        # # elif self.curr_state.name == "state2":
        # #     self.force = np.array([0.0, 0.0, -10.0])
        # else:
        #     self.force = None
        if self.force is not None:
            self.skeletons[2].body('h_pelvis').add_ext_force(self.force)

        self.skeletons[3].set_positions(self.curr_state.angles)

        if self.curr_state.dt < self.time() - self.elapsedTime:
            # print("change the state!!!", self.curr_state_index)
            self.curr_state_index = self.curr_state_index + 1
            self.curr_state_index = self.curr_state_index % self.state_num
            self.elapsedTime = self.time()
            self.curr_state = self.state_list[self.curr_state_index]
            # print("state_", self.curr_state_index)
            # print(self.curr_state.angles)

        # ground_height = -0.98 + 0.0251
        # ground_height = 0.
        ground_height = self.skeletons[0].body("ground").to_world()[1] #0.0251
        # print("ground h: ", ground_height)
        # ground_height = -0.98 + 0.05
        # ground_height = -0.98
        # if ik_on:
        #     ik_res = copy.deepcopy(self.curr_state.angles)
        #     if self.curr_state.name != "state02":
        #         # print("ik solve!!-------------------")
        #         self.ik.update_target(ground_height)
        #         ik_res[6:] = self.ik.solve()
        #         # self.curr_state.angles = ik_res
        #         self.skeletons[3].set_positions(ik_res)

        # HP QP solve
        lf_tangent_vec = np.array([1.0, 0.0, .0])
        rf_tangent_vec = np.array([1.0, 0.0, .0])

        # character_dir = copy.deepcopy(skel.com_velocity())
        character_dir = skel.com_velocity()
        # print(character_dir, skel.com_velocity())
        character_dir[1] = 0
        if np.linalg.norm(character_dir) != 0:
            character_dir = character_dir / np.linalg.norm(character_dir)

        centripetal_force_dir = np.cross([0.0, 1.0, 0.0], character_dir)

        empty_list = []

        # nonholonomic constraint

        right_blade_front_point = self.skeletons[2].body("h_blade_right").to_world((0.0216+0.104, -0.0216-0.027, 0.))
        right_blade_rear_point = self.skeletons[2].body("h_blade_right").to_world((0.0216-0.104, -0.0216-0.027, 0.))
        if right_blade_front_point[1] < 0.005+self.ground_height and right_blade_rear_point[1] < 0.005+self.ground_height:
            self.nh0.activate(True)
            self.nh1.activate(True)
            self.nh0.set_joint_pos(right_blade_front_point)
            self.nh0.set_projected_vector(right_blade_front_point - right_blade_rear_point)
            self.nh1.set_joint_pos(right_blade_rear_point)
            self.nh1.set_projected_vector(right_blade_front_point - right_blade_rear_point)
        else:
            self.nh0.activate(False)
            self.nh1.activate(False)

        left_blade_front_point = self.skeletons[2].body("h_blade_left").to_world((0.0216+0.104, -0.0216-0.027, 0.))
        left_blade_rear_point = self.skeletons[2].body("h_blade_left").to_world((0.0216-0.104, -0.0216-0.027, 0.))
        if left_blade_front_point[1] < 0.005 +self.ground_height and left_blade_rear_point[1] < 0.005+self.ground_height:
            self.nh2.activate(True)
            self.nh3.activate(True)
            self.nh2.set_joint_pos(left_blade_front_point)
            self.nh2.set_projected_vector(left_blade_front_point - left_blade_rear_point)
            self.nh3.set_joint_pos(left_blade_rear_point)
            self.nh3.set_projected_vector(left_blade_front_point - left_blade_rear_point)
        else:
            self.nh2.activate(False)
            self.nh3.activate(False)

        _tau = skel.get_spd(self.curr_state.angles, self.time_step(), 400., 40.)
        # _tau = skel.get_spd(self.curr_state.angles, self.time_step(), 100., 10.)
        # print("friction: ", world.collision_result.contacts)

        contacts = world.collision_result.contacts
        del self.rd_contact_forces[:]
        del self.rd_contact_positions[:]
        for contact in contacts:
            if contact.skel_id1 == 0:
                self.rd_contact_forces.append(-contact.f / 1000.)
            else:
                self.rd_contact_forces.append(contact.f / 1000.)
            self.rd_contact_positions.append(contact.p)

        # print("contact num: ", len(self.rd_contact_forces))
        #Jacobian transpose control

        jaco_r = skel.body("h_blade_right").linear_jacobian()
        jaco_l = skel.body("h_blade_left").linear_jacobian()
        jaco_hip_r = skel.body("h_thigh_right").linear_jacobian()
        # jaco_pelvis = skel.body("h_pelvis").linear_jacobian()

        if self.curr_state.name == "state01":
            self.force_r = 0. * np.array([-1.0, -0., 1.])
            self.force_l = 0. * np.array([1.0, -0.5, -1.0])
            t_r = self.add_JTC_force(jaco_r, self.force_r)
            t_l = self.add_JTC_force(jaco_l, self.force_l)
            _tau += t_r + t_l

        if self.curr_state.name == "state02":
            self.force_r = 0. * np.array([0.0, 5.0, 5.])
            # self.force_r = 2. * np.array([-1.0, -5.0, 1.])
            self.force_l = 0. * np.array([1.0, -0., -1.])
            t_r = self.add_JTC_force(jaco_r, self.force_r)
            t_l = self.add_JTC_force(jaco_l, self.force_l)
            _tau += t_r + t_l

        if self.curr_state.name == "state03":
            self.force_r = 0. * np.array([1.0, 3.0, -1.])
            self.force_l = 0. * np.array([1.0, -0., -1.])
            t_r = self.add_JTC_force(jaco_r, self.force_r)
            t_l = self.add_JTC_force(jaco_l, self.force_l)
            _tau += t_r + t_l

        if self.curr_state.name == "state04":
            self.force_r = 0. * np.array([-1.0, 0., 1.])
            self.force_l = 0. * np.array([1.0, 0., -1.0])
            t_r = self.add_JTC_force(jaco_r, self.force_r)
            t_l = self.add_JTC_force(jaco_l, self.force_l)
            _tau += t_r + t_l

        _tau[0:6] = np.zeros(6)

        skel.set_forces(_tau)

        super(MyWorld, self).step()


    def add_JTC_force(self, jaco, force):
        jaco_t = jaco.transpose()
        tau = np.dot(jaco_t, force)
        return tau

    def on_key_press(self, key):
        if key == '1':
            self.force = np.array([100.0, 0.0, 0.0])
            self.duration = 1000
            print('push backward: f = %s' % self.force)
        elif key == '2':
            self.force = np.array([-100.0, 0.0, 0.0])
            self.duration = 100
            print('push backward: f = %s' % self.force)

    def render_with_ys(self):
        # render contact force --yul
        del render_vector[:]
        del render_vector_origin[:]
        del push_force[:]
        del push_force_origin[:]
        del blade_force[:]
        del blade_force_origin[:]
        del rd_footCenter[:]

        com = self.skeletons[2].C
        com[1] = 0.
        # com[1] = -0.99 +0.05

        # com = self.skeletons[2].body('h_blade_left').to_world(np.array([0.1040 + 0.0216, +0.80354016 - 0.85354016, -0.054]))

        # if self.curr_state.name == "state2":
        #     com = self.ik.target_foot_left1

        rd_footCenter.append(com)

        if self.force_r is not None:
            blade_force.append(self.force_r*0.05)
            blade_force_origin.append(self.skeletons[2].body('h_heel_right').to_world())
        if self.force_l is not None:
            blade_force.append(self.force_l*0.05)
            blade_force_origin.append(self.skeletons[2].body('h_heel_left').to_world())

        if self.force is not None:
            push_force.append(self.force*0.05)
            push_force_origin.append(self.skeletons[2].body('h_pelvis').to_world())

        if len(self.rd_contact_forces) != 0:
            for ii in range(len(self.rd_contact_forces)):
                render_vector.append(self.rd_contact_forces[ii] * 10.)
                render_vector_origin.append(self.rd_contact_positions[ii])

if __name__ == '__main__':
    print('Example: Skating -- pushing side to side')

    pydart.init()
    print('pydart initialization OK')

    world = MyWorld()
    print('MyWorld  OK')
    ground = pydart.World(1. / 1000., './data/skel/ground.skel')

    skel = world.skeletons[2]

    q = world.curr_state.angles
    # q = skel.q
    #
    # q["j_pelvis_pos_y"] = -0.2
    #
    # q["j_thigh_left_x", "j_thigh_left_y", "j_thigh_left_z"] = 0., 0., 0.3
    # q["j_thigh_right_x", "j_thigh_right_y", "j_thigh_right_z"] = -0.2, -0., 0.
    # q["j_shin_left_z", "j_shin_right_z"] = -0.5, -0.35
    # q["j_heel_left_z", "j_heel_right_z"] = 0.2, 0.35

    # q["j_thigh_left_y", "j_thigh_right_y"] = 0., -0.785

    # # both arm T-pose
    # q["j_bicep_left_x", "j_bicep_left_y", "j_bicep_left_z"] = 1.5, 0.0, 0.0
    # q["j_bicep_right_x", "j_bicep_right_y", "j_bicep_right_z"] = -1.5, 0.0, 0.0

    skel.set_positions(q)
    print('skeleton position OK')

    # print('[Joint]')
    # for joint in skel.joints:
    #     print("\t" + str(joint))
    #     print("\t\tparent = " + str(joint.parent_bodynode))
    #     print("\t\tchild = " + str(joint.child_bodynode))
    #     print("\t\tdofs = " + str(joint.dofs))

    # pydart.gui.viewer.launch_pyqt5(world)
    viewer = hsv.hpSimpleViewer(viewForceWnd=False)
    viewer.setMaxFrame(1000)
    viewer.doc.addRenderer('controlModel', yr.DartRenderer(world, (255,255,255), yr.POLYGON_FILL))
    viewer.doc.addRenderer('ground', yr.DartRenderer(ground, (255, 255, 255), yr.POLYGON_FILL),visible=False)
    viewer.doc.addRenderer('contactForce', yr.VectorsRenderer(render_vector, render_vector_origin, (255, 0, 0)))
    viewer.doc.addRenderer('pushForce', yr.WideArrowRenderer(push_force, push_force_origin, (0, 255,0)))
    viewer.doc.addRenderer('rd_footCenter', yr.PointsRenderer(rd_footCenter))

    viewer.startTimer(1/25.)

    viewer.doc.addRenderer('bladeForce', yr.WideArrowRenderer(blade_force, blade_force_origin, (0, 0, 255)))
    viewer.motionViewWnd.glWindow.planeHeight = 0.

    def simulateCallback(frame):
        for i in range(40):
            world.step()
        world.render_with_ys()

    viewer.setSimulateCallback(simulateCallback)
    viewer.show()

    Fl.run()
