from SkateUtils.KeyPoseState import State
import numpy as np
import pydart2 as pydart
import math
import IKsolve_double_stance
import IKsolveGlobal
import dart_ik
import copy
import pickle

from fltk import *
from PyCommon.modules.GUI import hpSimpleViewer as hsv
from PyCommon.modules.Renderer import ysRenderer as yr

from os.path import join

render_vector = []
render_vector_origin = []
push_force = []
push_force_origin = []
blade_force = []
blade_force_origin = []

rd_footCenter = []

ik_on = True
# ik_on = False

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
        # self.ground_height = -0.5
        # print("self.ground_height: ", self.ground_height)

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
        # pelvis_pos_y = skel.dof_indices(["j_pelvis_pos_y"])
        # pelvis_x = skel.dof_indices(["j_pelvis_rot_x"])
        # pelvis = skel.dof_indices(["j_pelvis_rot_y", "j_pelvis_rot_z"])
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
        elbows = skel.dof_indices(["j_forearm_left_x", "j_forearm_right_x"])
        # blade = skel.dof_indices(["j_heel_right_2"])

        # import pose angle info. from txt file

        self.q_list = []
        # self.load_poses(skel, 'data/mocap/jump_poses.txt')
        # self.fn = 5
        # self.load_poses(skel, 'data/mocap/skate_joint_3d_pos.txt')
        # self.load_poses(skel, 'data/mocap/spin_joint_3d_pos.txt')
        # self.load_poses(skel, 'data/mocap/crossover1_joint_3d_pos.txt')
        # self.load_poses(skel, 'data/mocap/3turn_joint_3d_pos.txt')
        # self.load_poses(skel, 'data/mocap/crossover_new_joint_3d_pos.txt')
        # self.load_poses(skel, 'data/mocap/back_3turn_joint_3d_pos.txt')
        # self.load_poses(skel, 'data/mocap/back_crossover_joint_3d_pos.txt')
        # self.load_poses(skel, 'data/mocap/forward_gliding_joint_3d_pos.txt')

        # Test Moving Camera result!
        file_dir = 'data/mocap/movingcam/'
        # fn = 'bc_small'
        # fn = 'backflip_a'
        # self.load_poses(skel, join(file_dir, fn+'_result_joint_3d_pos.txt'))
        file_dir = 'data/mocap/'
        fn = 'spiral1'
        self.load_poses(skel, join(file_dir, fn+'_joint_3d_pos.txt'))

        state_q = []
        self.state_list = []
        for i in range(self.fn):
            state_q.append(self.q_list[i])
            state_name = "pos"+str(i)
            # print("name: ", state_name)
            state_q[i][4] += 0.5
            self.state_list.append(State(state_name, 0.3, 0.0, 0.0, state_q[i]))

        for i in range(self.fn-1):
            self.state_list[i].set_next(self.state_list[i+1])

        # filename = 'hmr_skating_3turn.skkey'
        # filename = 'hmr_skating_gliding.skkey'

        # filename = 'sfv_'+fn + '.skkey'
        filename = 'sfv_' + fn + '_global.skkey'
        print("filename: ", filename)

        with open(filename, 'wb') as f:
            pickle.dump(self.state_list, f)

        # self.load_poses(skel, 'data/mocap/jump_poses.txt')
        #
        # s00q = np.zeros(skel.ndofs)
        # s00q[upper_body] = 0.0, -0., -0.1
        # state00 = State("state00", 2.0, 0.0, 0.2, s00q)
        #
        # # s0 = self.q_list[0]
        # # state0 = State("state0", 5.0, 0.0, 0.0, s0)
        #
        # s1 = self.q_list[1]
        # state1 = State("state1", 1.0, 0.0, 0.0, s1)
        #
        # s2 = self.q_list[2]
        # state2 = State("state2", 0.5, 0.0, 0.0, s2)
        #
        # s3 = self.q_list[3]
        # state3 = State("state3", 0.5, 0.0, 0.0, s3)
        #
        # s4 = np.zeros(skel.ndofs)
        # s4[left_leg] = -0.2, 0.0, -0.1, 0.
        # s4[right_leg] = 0.2, 0.0, 0.2, -0.4
        # # s4[arms] = -0.5, -0.5
        # s4[arms_z] = 0.7, 0.7
        # s4[elbows] = -2.0, 2.0
        # s4[foot] = 0.0, 0.0, -0.5, 0.0, 0.0, 0.2
        # state4 = State("state4", 0.5, 0.0, 0.0, s4)
        #
        # s5 = self.q_list[4]
        # state5 = State("state5", 2.0, 0.0, 0.0, s5)
        #
        # s_terminal_q = np.zeros(skel.ndofs)
        # state_t = State("state_t", 50., 2.0, 2.0, s_terminal_q)
        #
        # state00.set_next(state1)
        # state1.set_next(state2)
        # state2.set_next(state3)
        # state3.set_next(state4)
        # state4.set_next(state5)
        # state5.set_next(s_terminal_q)
        #
        # self.state_list = [state00, state1, state2, state3, state4, state5, state_t]

        # filename = 'hmr_skating_jump_test.skkey'
        #
        # with open(filename, 'wb') as f:
        #     pickle.dump(self.state_list, f)

        state_num = len(self.state_list)
        self.state_num = state_num
        # print("state_num: ", state_num)

        self.curr_state = self.state_list[0]
        self.elapsedTime = 0.0
        self.curr_state_index = 0
        # print("backup angle: ", backup_q)
        # print("cur angle: ", self.curr_state.angles)

        if ik_on:
            # self.ik = IKsolve_double_stance.IKsolver(skel, self.skeletons[3], self.dt)
            self.ik = IKsolveGlobal.IKsolver(skel, self.skeletons[3], self.dt)

            print('IK ON')

        # print("after:", state00.angles)

        # self.revise_pose(state00)
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

    def load_poses(self, skel, file_path):
        fn = 0

        # read 3d positions of each key pose frame from txt file
        self.position_storage = []
        # with open('data/mocap/skate_spin.txt') as f:
        with open(file_path) as f:
            # lines = f.read().splitlines()
            for line in f:
                q_temp = []
                line = line.replace(' \n', '')
                values = line.split(" ")
                temp_pos = np.asarray([float(values[0]), -float(values[1]), -float(values[2])])
                # temp_pos = temp_pos * 0.9  # scaling
                # for a in values:
                #     q_temp.append(float(a))
                self.position_storage.append(temp_pos)
                fn = fn + 1

        fn = int(fn / 19)
        self.fn = fn
        # print("frame num: ", fn)
        self.ik = dart_ik.DartIk(skel)

        fi = 0
        for f in range(fn):
            self.ik.clean_constraints()


            # # self.ik.add_orientation_const('h_pelvis', np.asarray([0., 0., 0.]))

            self.ik.add_joint_pos_const('j_heel_right', self.position_storage[fi])
            self.ik.add_joint_pos_const('j_shin_right', self.position_storage[fi + 1])
            self.ik.add_joint_pos_const('j_thigh_right', self.position_storage[fi + 2])

            self.ik.add_joint_pos_const('j_thigh_left', self.position_storage[fi + 3])
            self.ik.add_joint_pos_const('j_shin_left', self.position_storage[fi + 4])
            self.ik.add_joint_pos_const('j_heel_left', self.position_storage[fi + 5])

            self.ik.add_joint_pos_const('j_hand_right', self.position_storage[fi + 6])
            self.ik.add_joint_pos_const('j_forearm_right', self.position_storage[fi + 7])
            # self.ik.add_joint_pos_const('j_scapula_right', self.position_storage[fi + 8])
            # self.ik.add_position_const('h_scapula_right', self.position_storage[fi + 8], [0.0, 0.397146, 0.169809])
            self.ik.add_joint_pos_const('j_bicep_right', self.position_storage[fi + 8])

            # self.ik.add_joint_pos_const('j_scapula_left', self.position_storage[fi + 9])
            # self.ik.add_position_const('h_scapula_left', self.position_storage[fi+9], [0.0, 0.397146, -0.169809])
            self.ik.add_joint_pos_const('j_bicep_left', self.position_storage[fi + 9])
            self.ik.add_joint_pos_const('j_forearm_left', self.position_storage[fi + 10])
            self.ik.add_joint_pos_const('j_hand_left', self.position_storage[fi + 11])

            # self.ik.add_joint_pos_const('j_forearm_right', self.position_storage[fi + 6])
            # self.ik.add_joint_pos_const('j_bicep_right', self.position_storage[fi + 7])
            # self.ik.add_joint_pos_const('j_scapula_right', self.position_storage[fi + 8])
            #
            # self.ik.add_joint_pos_const('j_scapula_left', self.position_storage[fi + 9])
            # self.ik.add_joint_pos_const('j_bicep_left', self.position_storage[fi + 10])
            # self.ik.add_joint_pos_const('j_forearm_left', self.position_storage[fi + 11])

            # self.ik.add_joint_pos_const('j_spine', self.position_storage[fi + 12])
            # self.ik.add_joint_pos_const('j_head', self.position_storage[fi + 13])
            self.ik.add_position_const('h_spine', self.position_storage[fi + 12], [0.0, -0.0908, 0.0])
            self.ik.add_position_const('h_head', self.position_storage[fi + 13], [0.0, 0.07825, 0.0])     #head_up
            self.ik.add_position_const('h_head', self.position_storage[fi + 14], [0.0508625, 0.0, 0.0])       #nose
            self.ik.add_position_const('h_head', self.position_storage[fi + 17], [0., 0.0, -0.0626])      # left ear
            self.ik.add_position_const('h_head', self.position_storage[fi + 18], [0., 0.0, 0.0626])      # right ear

            # self.ik.add_position_const('h_blade_left', left_toe_pos, [-0.1040 + 0.0216, +0.80354016 - 0.85354016, 0.0])

            # right_blade_pos = self.position_storage[fi] + np.array([0., -0.054, 0.])
            # left_blade_pos = self.position_storage[fi+5] + np.array([0., -0.054, 0.])
            #
            # right_blade_pos = self.position_storage[fi] + np.array([0., -0.1, 0.])
            # left_blade_pos = self.position_storage[fi + 5] + np.array([0., -0.2, 0.])

            # self.ik.add_position_const('h_blade_right', right_blade_pos,
            #                            [-0.1040 + 0.0216, 0.80354016 - 0.85354016, 0.0])
            # self.ik.add_position_const('h_blade_right', right_blade_pos,
            #                            [0.1040 + 0.0216, 0.80354016 - 0.85354016, 0.0])
            #
            # self.ik.add_position_const('h_blade_left', left_blade_pos,
            #                            [0.1040 + 0.0216, 0.80354016 - 0.85354016, 0.0])
            # self.ik.add_position_const('h_blade_left', left_blade_pos,
            #                            [-0.1040 + 0.0216, 0.80354016 - 0.85354016, 0.0])

            self.ik.solve()

            q = skel.q
            # q[0:3] = np.asarray([0., -0.785, 0.5])
            # q[3:6] = np.asarray([0., 0.01, 0.])
            # q[0:2] = np.asarray([0.0, 0.0])
            # q[3:6] = np.asarray([0.0, 0.0, 0.0])
            skel.set_positions(q)

            self.q_list.append(skel.q)
            fi += 19

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

        if self.curr_state.name == "state_force":
            self.force = np.array([-20.0, 0.0, 0.0])
        # elif self.curr_state.name == "state2":
        #     self.force = np.array([0.0, 0.0, -10.0])
        else:
            self.force = None
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
    print('Solve IK ')

    pydart.init()
    print('pydart initialization OK')

    world = MyWorld()
    print('MyWorld  OK')

    skel = world.skeletons[2]

    q = world.curr_state.angles
    skel.set_positions(q)
    print('skeleton position OK')
    print("COM height: ", skel.com()[1])
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
    viewer.doc.addRenderer('contactForce', yr.VectorsRenderer(render_vector, render_vector_origin, (255, 0, 0)))
    viewer.doc.addRenderer('pushForce', yr.WideArrowRenderer(push_force, push_force_origin, (0, 255,0)))
    viewer.doc.addRenderer('rd_footCenter', yr.PointsRenderer(rd_footCenter))

    viewer.startTimer(1/25.)

    viewer.doc.addRenderer('bladeForce', yr.WideArrowRenderer(blade_force, blade_force_origin, (0, 0, 255)))
    viewer.motionViewWnd.glWindow.planeHeight = 0.
    # viewer.motionViewWnd.glWindow.planeHeight = -0.5

    def simulateCallback(frame):
        for i in range(40):
            world.step()
        world.render_with_ys()

    viewer.setSimulateCallback(simulateCallback)
    viewer.show()

    Fl.run()