import numpy as np
import pydart2 as pydart
import QPsolver
import IKsolve_one

from fltk import *
from PyCommon.modules.GUI import hpSimpleViewer as hsv
from PyCommon.modules.Renderer import ysRenderer as yr


render_vector = []
render_vector_origin = []
push_force = []
push_force_origin = []
blade_force = []
blade_force_origin = []


class State(object):
    def __init__(self, name, dt, c_d, c_v, angles):
        self.name = name
        self.dt = dt
        self.c_d = c_d
        self.c_v = c_v
        self.angles = angles

class MyWorld(pydart.World):
    def __init__(self, ):
        pydart.World.__init__(self, 1.0 / 1000.0, './data/skel/cart_pole_blade.skel')
        # pydart.World.__init__(self, 1.0 / 2000.0, './data/skel/cart_pole.skel')
        self.force = None
        self.duration = 0
        self.skeletons[0].body('ground').set_friction_coeff(0.02)

        skel = self.skeletons[2]
        # print("mass: ", skel.m, "kg")

        skel.joint("j_heel_left").set_position_upper_limit(0, 0.0)
        skel.joint("j_heel_left").set_position_lower_limit(0, -0.0)
        pelvis_x = skel.dof_indices((["j_pelvis_rot_x"]))
        pelvis = skel.dof_indices((["j_pelvis_rot_y", "j_pelvis_rot_z"]))
        upper_body = skel.dof_indices(["j_abdomen_1", "j_abdomen_2"])
        right_leg = skel.dof_indices(["j_thigh_right_x", "j_thigh_right_y", "j_thigh_right_z", "j_shin_right"])
        left_leg = skel.dof_indices(["j_thigh_left_x", "j_thigh_left_y", "j_thigh_left_z", "j_shin_left"])
        arms = skel.dof_indices(["j_bicep_left_x", "j_bicep_right_x"])
        foot = skel.dof_indices(["j_heel_left_1", "j_heel_left_2", "j_heel_right_1", "j_heel_right_2"])
        leg_y = skel.dof_indices(["j_thigh_right_y", "j_thigh_left_y"])
        # blade = skel.dof_indices(["j_heel_right_2"])

        #----------------------------------
        # one foot glide
        #----------------------------------
        # s0q = np.zeros(skel.ndofs)
        # # s0q[pelvis] = 0., -0.5
        # # s0q[right_leg] = -0.0, -0., 0., -0.17
        # # s0q[left_leg] = 0.0, 0., 0., -0.17
        # # s0q[leg_y] = -0.785, 0.785
        # s0q[arms] = 1.5, -1.5
        # # s0q[foot] = 0.1, 0.0, 0.1, -0.0
        # state0 = State("state0", 2.0, 0.0, 0.2, s0q)
        #
        # s1q = np.zeros(skel.ndofs)
        # # s1q[pelvis_x] = -0.5
        # # s1q[pelvis] = 0., -0.3
        # s1q[upper_body] = -0.5, -0.
        # s1q[left_leg] = 0., 0., 0., -0.17
        # # s1q[right_leg] = -0.0, -0.785, -0.66, -0.0
        # s1q[right_leg] = -0., -0., 0.9, -1.5
        # s1q[arms] = 0.8, -2.2
        # # s1q[blade] = -0.3
        # # s1q[foot] = -0.0, 0.0, 0.4,  0.0
        # state1 = State("state1", 0.5, 2.2, 0.0, s1q)
        #
        # self.state_list = [state0, state1]

        # #----------------------------------
        # # pushing side to side
        # #----------------------------------
        # s0q = np.zeros(skel.ndofs)
        # s0q[pelvis] = 0., -0.5
        # s0q[right_leg] = -0.02, -0.785, 0.5, -0.17
        # s0q[left_leg] = 0.02, 0.785, 0.5, -0.17
        # # s0q[leg_y] = -0.785, 0.785
        # s0q[arms] = 1.5, -1.5
        # # s0q[foot] = 0.1, 0.0, 0.1, -0.0
        # state0 = State("state0", 1.0, 0.0, 0.2, s0q)
        #
        # s01q = np.zeros(skel.ndofs)
        # s01q[pelvis] = 0., -0.3
        # # s01q[upper_body] = 0.0, -0.5
        # s01q[left_leg] = 0., 0.785, 0., -0.17
        # # s01q[right_leg] = -0.0, -0.785, -0.66, -0.0
        # s01q[right_leg] = -0.3, -0.785, 0.3, -0.3
        # s01q[arms] = 1.5, -1.5
        # # s01q[blade] = -0.3
        # # s01q[foot] = -0.0, 0.0, 0.4,  0.0
        # state01 = State("state01", 0.5, 2.2, 0.0, s01q)
        #
        # s1q = np.zeros(skel.ndofs)
        # s1q[pelvis] = 0., -0.1
        # # s1q[upper_body] = 0.0, -0.5
        # s1q[left_leg] = 0., 0.785, 0., -0.17
        # # s1q[right_leg] = -0.0, -0.785, -0.66, -0.0
        # s1q[right_leg] = -0.5, -0.785, -0., -0.17
        # s1q[arms] = 1.5, -1.5
        # # s1q[blade] = -0.3
        # # s1q[foot] = -0.0, 0.0, 0.4,  0.0
        # state1 = State("state1", 1.0, 2.2, 0.0, s1q)
        #
        # s11q = np.zeros(skel.ndofs)
        # s11q[pelvis] = -0.0, 0.3
        # # s1q[upper_body] = 0.0, -0.2
        # s11q[right_leg] = 0.0, -0.9, -0.2, -0.
        # s11q[left_leg] = 0.0, 0.785, 0.35, -0.3
        # s11q[arms] = 1.5, -1.5
        # # s11q[foot] = -0.0, 0.0, -0.4, 0.0
        # state11 = State("state11", 0.5, 2.2, 0.0, s11q)
        #
        # # s1q = np.zeros(skel.ndofs)
        # # # s1q[pelvis] = -0., -0.3
        # # s1q[upper_body] = 0.0, -0.3
        # # s1q[right_leg] = 0., -0.785, -0.3, -0.17
        # # s1q[left_leg] = 0., 0.2, 0.2, -0.4
        # # s1q[arms] = 1.5, -1.5
        # # # s1q[foot] = 0.1, 0.0, -0.6,  0.0
        # # state1 = State("state1", 1.0, 2.2, 0.0, s1q)
        #
        # s2q = np.zeros(skel.ndofs)
        # # s2q[pelvis] = -0.3, -0.0
        # # s2q[upper_body] = 0.0, -0.4
        # s2q[right_leg] = -0.0, -0., 0.8, -1.5
        # # s2q[left_leg] = 0.0, -0.0, -0.2, -0.17
        # s2q[left_leg] = 0.0, 0.785, 0.0, -0.17
        # s2q[arms] = 1.5, -1.5
        # s2q[foot] = 0.2, 0.0, -0.6, -0.0
        # state2 = State("state2", 5.0, 0.0, 0.2, s2q)
        #
        # s3q = np.zeros(skel.ndofs)
        # # s1q[pelvis] = -0.3, -0.0
        # # s1q[upper_body] = 0.0, -0.2
        # s3q[right_leg] = 0.9, -0.785, 0.1, -0.17
        # s3q[left_leg] = -0., 0.785, 0.1, -0.17
        # s3q[arms] = 1.5, -1.5
        # s3q[foot] = 0.2, 0.0, 0.2, 0.0
        # state3 = State("state3", 3.0, 2.2, 0.0, s3q)

        #----------------------------------
        # pushing side to side new (180718)
        #----------------------------------
        s0q = np.zeros(skel.ndofs)
        # s0q[pelvis] = 0., -0.
        # s0q[upper_body] = 0.0, -0.5
        # s0q[right_leg] = -0., -0., -0.05, -0.05
        # s0q[left_leg] = 0., 0., 0.05, -0.05
        # s0q[leg_y] = -0.785, 0.785
        s0q[arms] = 1.5, -1.5
        # s0q[foot] = 0.1, 0.0, 0.1, -0.0
        state0 = State("state0", 1.0, 0.0, 0.2, s0q)

        s01q = np.zeros(skel.ndofs)
        # s01q[pelvis] = 0., -0.3
        # s01q[upper_body] = 0.0, -0.4
        s01q[left_leg] = 0., 0., 0.4, -0.4
        # s01q[right_leg] = -0.0, -0.785, -0.66, -0.0
        s01q[right_leg] = -0., -0., 0.4, -0.4
        s01q[arms] = 1.5, -1.5
        # s01q[blade] = -0.3
        # s01q[foot] = -0.0, 0.0, 0.4,  0.0
        state01 = State("state01", 0.5, 2.2, 0.0, s01q)

        s1q = np.zeros(skel.ndofs)
        # s1q[pelvis] = 0., -0.1
        # s1q[upper_body] = 0.0, 0.2
        s1q[left_leg] = 0., 0.0, 0.4, -0.4
        # s1q[right_leg] = -0.0, -0.785, -0.66, -0.0
        s1q[right_leg] = -0.0, -0.0, 0.2, -0.3
        s1q[arms] = 1.5, -1.5
        # s1q[blade] = -0.3
        # s1q[foot] = -0.0, 0.0, -0.4,  0.0
        state1 = State("state1", 0.1, 2.2, 0.0, s1q)

        s11q = np.zeros(skel.ndofs)
        # s1q[pelvis] = 0., -0.1
        # s1q[upper_body] = 0.0, -0.
        s11q[left_leg] = 0., 0.0, 0.3, -0.3
        # s1q[right_leg] = -0.0, -0.785, -0.66, -0.0
        s11q[right_leg] = -0.0, -0.0, 0., -0.3
        s11q[arms] = 1.5, -1.5
        # s1q[blade] = -0.3
        s11q[foot] = -0.0, 0.0, -0.1,  0.0
        state11 = State("state11", 0.1, 2.2, 0.0, s11q)

        s12q = np.zeros(skel.ndofs)
        # s12q[pelvis] = 0., -0.1
        # s12q[upper_body] = 0.0, -0.2
        s12q[left_leg] = 0., 0.0, 0.3, -0.2
        # s12q[right_leg] = -0.0, -0.785, -0.66, -0.0
        s12q[right_leg] = -0.0, -0.0, -0.0, -0.3
        s12q[arms] = 1.5, -1.5
        # s12q[blade] = -0.3
        s12q[foot] = -0.0, 0.0, -0.3,  0.0
        state12 = State("state12", 0.1, 2.2, 0.0, s12q)

        s13q = np.zeros(skel.ndofs)
        # s13q[pelvis] = 0., -0.1
        # s13q[upper_body] = 0.0, -0.3
        s13q[left_leg] = 0., 0.0, 0.3, -0.17
        # s13q[right_leg] = -0.0, -0.785, -0.66, -0.0
        s13q[right_leg] = -0.0, -0.0, -0.3, -0.3
        s13q[arms] = 1.5, -1.5
        # s1q[blade] = -0.3
        s13q[foot] = -0.0, 0.0, -0.9,  0.0
        state13 = State("state13", 0.2, 2.2, 0.0, s13q)

        s2q = np.zeros(skel.ndofs)
        # s2q[pelvis] = -0.3, -0.0
        # s2q[upper_body] = 0.0, -0.4
        s2q[left_leg] = -0., -0., 0.3, -0.17
        # s2q[left_leg] = 0.0, -0.0, -0.2, -0.17
        s2q[right_leg] = 0.0, 0., -0.5, -0.9
        s2q[arms] = 1.5, -1.5
        # s2q[foot] = 0.2, 0.0, -0.6, -0.0
        state2 = State("state2", 1.0, 0.0, 0.2, s2q)

        # s14q = np.zeros(skel.ndofs)
        # # s14q[pelvis] = 0., -0.1
        # # s14q[upper_body] = 0.0, -0.1
        # s14q[left_leg] = 0., 0.0, 0.0, -0.17
        # # s14q[right_leg] = -0.0, -0.785, -0.66, -0.0
        # s14q[right_leg] = -0.0, -0.0, -0.5, -0.3
        # s14q[arms] = 1.5, -1.5
        # # s14q[blade] = -0.3
        # s14q[foot] = -0.0, 0.0, -0.9,  0.0
        # state14 = State("state14", 0.9, 2.2, 0.0, s14q)

        # s1q = np.zeros(skel.ndofs)
        # # s1q[pelvis] = 0., -0.1
        # s1q[upper_body] = 0.0, -0.4
        # s1q[left_leg] = 0.3, 0.785, 0.3, -0.3
        # # s1q[right_leg] = -0.0, -0.785, -0.66, -0.0
        # s1q[right_leg] = -0.5, -0.785, -0., -0.17
        # s1q[arms] = 1.5, -1.5
        # # s1q[blade] = -0.3
        # # s1q[foot] = -0.0, 0.0, 0.4,  0.0
        # state1 = State("state1", 0.9, 2.2, 0.0, s1q)
        #
        # s11q = np.zeros(skel.ndofs)
        # # s1q[pelvis] = 0., -0.1
        # s11q[upper_body] = 0.0, -0.4
        # s11q[left_leg] = 0., 0.785, 0., -0.17
        # # s1q[right_leg] = -0.0, -0.785, -0.66, -0.0
        # s11q[right_leg] = -0.5, -0.785, -0., -0.17
        # s11q[arms] = 1.5, -1.5
        # # s1q[blade] = -0.3
        # # s1q[foot] = -0.0, 0.0, 0.4,  0.0
        # state11 = State("state11", 0.2, 2.2, 0.0, s11q)
        #
        # s2q = np.zeros(skel.ndofs)
        # # s2q[pelvis] = -0.3, -0.0
        # # s2q[upper_body] = 0.0, -0.4
        # s2q[right_leg] = -0.3, -0., 0., -0.1
        # # s2q[left_leg] = 0.0, -0.0, -0.2, -0.17
        # s2q[left_leg] = 0.0, 0.785, 0.0, -0.17
        # s2q[arms] = 1.5, -1.5
        # # s2q[foot] = 0.2, 0.0, -0.6, -0.0
        # # state2 = State("state2", 1.0, 0.0, 0.2, s2q)
        # state2 = State("state2", 0.2, 2.2, 0.0, s1q)
        #
        # s3q = np.zeros(skel.ndofs)
        # # s1q[pelvis] = -0.3, -0.0
        # # s1q[upper_body] = 0.0, -0.2
        # s3q[right_leg] = -0.9, -0.785, 0.1, -0.17
        # s3q[left_leg] = -0., 0.785, 0.1, -0.17
        # s3q[arms] = 1.5, -1.5
        # # s3q[foot] = 0.2, 0.0, 0.2, 0.0
        # state3 = State("state3", 3.0, 2.2, 0.0, s3q)

        # self.state_list = [state0, state2, state3]
        # self.state_list = [state0, state1, state11, state2, state3]
        self.state_list = [state0, state01, state1, state11, state12, state13, state2]
        # self.state_list = [state0,  state1, state2, state3]
        # self.state_list = [state2, state0, state1]
        # self.state_list = [state2, state1]
        # self.state_list = [state1, state2]
        # self.state_list = [state1, state2, state1, state3]
        # self.state_list = [state1, state2, state22, state3, state33]
        state_num = len(self.state_list)
        self.state_num = state_num
        # print("state_num: ", state_num)

        self.curr_state = self.state_list[0]
        self.elapsedTime = 0.0
        self.curr_state_index = 0
        # print("backup angle: ", backup_q)
        # print("cur angle: ", self.curr_state.angles)
        self.controller = QPsolver.Controller(skel, self.dt, self.curr_state.name)

        self.skeletons[3].set_positions(self.curr_state.angles)
        # self.skeletons[3].set_positions(np.zeros(skel.ndofs))

        self.ik = IKsolve_one.IKsolver(self.skeletons[2], self.dt)

        # merged_target = self.curr_state.angles
        self.ik.update_target(self.curr_state.name)
        merged_target = np.zeros(skel.ndofs)
        merged_target[:6] = self.curr_state.angles[:6]
        merged_target[6:18] = self.ik.solve()
        merged_target[18:] = self.curr_state.angles[18:]
        # print("ik res: ", self.ik.solve())
        # print("merged_target: ", merged_target)
        self.controller.target = merged_target

        # self.controller.target = self.curr_state.angles
        # self.controller.target = skel.q
        skel.set_controller(self.controller)
        print('create controller OK')

        self.contact_force = []
        # print("dof: ", skel.ndofs)

    def step(self):
        # print("self.curr_state: ", self.curr_state.name)
        # if self.curr_state.name == "state2" or self.curr_state.name == "state3":
        # if self.curr_state.name == "state1":
        # if self.time() > 1.0 and self.time() < 2.0:
        #     self.force = np.array([20.0, 0.0, 0.0])
        # else:
        #     self.force = None

        self.controller.cur_state = self.curr_state.name
        if self.force is not None:
            self.skeletons[2].body('h_pelvis').add_ext_force(self.force)
            # self.skeletons[2].body('h_spine').add_ext_force(self.force)
            # if self.curr_state.name == "state2":
            #     self.skeletons[2].body('h_pelvis').add_ext_force(self.force)
            # if self.curr_state.name == "state3":
            #     self.skeletons[2].body('h_pelvis').add_ext_force(self.force)
        # if self.force is not None and self.duration >= 0:
        #     self.duration -= 1
        #     self.skeletons[2].body('h_spine').add_ext_force(self.force)

        self.skeletons[3].set_positions(self.curr_state.angles)
        # self.skeletons[3].set_positions(np.zeros(skel.ndofs))
        if self.curr_state.dt < self.time() - self.elapsedTime:
            # print("change the state!!!", self.curr_state_index)
            self.curr_state_index = self.curr_state_index + 1
            self.curr_state_index = self.curr_state_index % self.state_num
            self.elapsedTime = self.time()
            self.curr_state = self.state_list[self.curr_state_index]
            # print("state_", self.curr_state_index)
            # print(self.curr_state.angles)

        # self.controller.target = skel.q
        # self.controller.target = self.curr_state.angles

        # if self.curr_state.name == "state2":
        #     self.ik.update_target(self.curr_state.name)
        #     merged_target = np.zeros(skel.ndofs)
        #     merged_target[:12] = self.curr_state.angles[:12]
        #     merged_target[12:18] = self.ik.solve()
        #     merged_target[18:] = self.curr_state.angles[18:]
        #     # print("ik res: ", self.ik.solve())
        #     # print("merged_target: ", merged_target)
        #     self.controller.target = merged_target
        #     # self.controller.target = self.curr_state.angles
        if self.curr_state.name == "state2":
            self.ik.update_target(self.curr_state.name)
            merged_target = np.zeros(skel.ndofs)
            merged_target[:6] = self.curr_state.angles[:6]
            merged_target[6:12] = self.ik.solve()
            merged_target[12:] = self.curr_state.angles[12:]
            # print("ik res: ", self.ik.solve())
            # print("merged_target: ", merged_target)
            self.controller.target = merged_target
            # self.controller.target = self.curr_state.angles
        else:
            # self.controller.target = self.curr_state.angles
            self.ik.update_target(self.curr_state.name)
            merged_target = np.zeros(skel.ndofs)
            merged_target[:6] = self.curr_state.angles[:6]
            merged_target[6:18] = self.ik.solve()
            merged_target[18:] = self.curr_state.angles[18:]
            # print("ik res: ", self.ik.solve())
            # print("merged_target: ", merged_target)
            # self.controller.target = merged_target
            self.controller.target = self.curr_state.angles

        del self.contact_force[:]
        if len(self.controller.sol_lambda) != 0:
            f_vec = self.controller.V_c.dot(self.controller.sol_lambda)
            # print("f", f_vec)
            f_vec = np.asarray(f_vec)
            # print("contact num ?? : ", self.controller.contact_num)
            # self.contact_force = np.zeros(self.controller.contact_num)
            for ii in range(self.controller.contact_num):
                self.contact_force.append(np.array([f_vec[3*ii], f_vec[3*ii+1], f_vec[3*ii+2]]))
                # self.contact_force[ii] = np.array([f_vec[3*ii], f_vec[3*ii+1], f_vec[3*ii+2]])
                # print("contact_force:", ii, self.contact_force[ii])

            # print("contact_force:\n", self.contact_force)
            for ii in range(self.controller.contact_num):
                self.skeletons[2].body(self.controller.contact_list[2 * ii])\
                    .add_ext_force(self.contact_force[ii], self.controller.contact_list[2 * ii+1])

        super(MyWorld, self).step()

        # skel.set_positions(q)

    def on_key_press(self, key):
        if key == '1':
            self.force = np.array([100.0, 0.0, 0.0])
            self.duration = 1000
            print('push backward: f = %s' % self.force)
        elif key == '2':
            self.force = np.array([-100.0, 0.0, 0.0])
            self.duration = 100
            print('push backward: f = %s' % self.force)

    def render_with_ri(self, ri):
        # if self.force is not None and self.duration >= 0:
        if self.force is not None:
            # if self.curr_state.name == "state2":
            #     p0 = self.skeletons[2].body('h_heel_right').C
            #     p1 = p0 + 0.01 * self.force
            #     ri.set_color(1.0, 0.0, 0.0)
            #     ri.render_arrow(p0, p1, r_base=0.03, head_width=0.1, head_len=0.1)
            # if self.curr_state.name == "state3":
            #     p0 = self.skeletons[2].body('h_heel_left').C
            #     p1 = p0 + 0.01 * self.force
            #     ri.set_color(1.0, 0.0, 0.0)
            #     ri.render_arrow(p0, p1, r_base=0.03, head_width=0.1, head_len=0.1)
            p0 = self.skeletons[2].body('h_spine').C
            p1 = p0 + 0.05 * self.force
            ri.set_color(1.0, 0.0, 0.0)
            ri.render_arrow(p0, p1, r_base=0.03, head_width=0.1, head_len=0.1)


        # render contact force --yul
        contact_force = self.contact_force

        if len(contact_force) != 0:
            # print(len(contact_force), len(self.controller.contact_list))
            # print("contact_force.size?", contact_force.size, len(contact_force))
            ri.set_color(1.0, 0.0, 0.0)
            for ii in range(len(contact_force)):
                if 2 * len(contact_force) == len(self.controller.contact_list):
                    body = self.skeletons[2].body(self.controller.contact_list[2*ii])
                    contact_offset = self.controller.contact_list[2*ii+1]
                    # print("contact force : ", contact_force[ii])
                    ri.render_line(body.to_world(contact_offset), contact_force[ii]/100.)
        ri.set_color(1, 0, 0)
        ri.render_sphere(np.array([self.skeletons[2].C[0], -0.99, self.skeletons[2].C[2]]), 0.05)
        ri.set_color(1, 0, 1)
        ri.render_sphere(self.ik.target_foot + np.array([0.0, 0.0, -0.1]), 0.05)

        # COP = self.skeletons[2].body('h_heel_right').to_world([0.05, 0, 0])
        # ri.set_color(0, 0, 1)
        # ri.render_sphere(COP, 0.05)

        # todo : ground rendering
        # ri.render_chessboard(5)
        # render axes
        ri.render_axes(np.array([0, 0, 0]), 0.5)

        #Height
        # ri.render_sphere(np.array([0.0, 1.56-0.92, 0.0]), 0.01)

    def render_with_ys(self):
        # render contact force --yul
        contact_force = self.contact_force
        del render_vector[:]
        del render_vector_origin[:]
        del push_force[:]
        del push_force_origin[:]
        del blade_force[:]
        del blade_force_origin[:]

        if self.curr_state.name == "state1" or self.curr_state.name == "state11" :
            blade_force.append(np.array([1.0, 0.0, 0.0]))
            blade_force_origin.append(self.skeletons[2].body('h_heel_left').to_world())
            # blade_force.append(-self.controller.blade_direction_L)
            # blade_force_origin.append(self.skeletons[2].body('h_heel_right').to_world())
        if self.curr_state.name == "state12":
            blade_force.append(np.array([-0.7, 1.0, 0.0]))
            blade_force_origin.append(self.skeletons[2].body('h_heel_right').to_world())
        if self.curr_state.name == "state13":
            blade_force.append(np.array([-1.0, 1.5, 0.0]))
            blade_force_origin.append(self.skeletons[2].body('h_heel_right').to_world())
        if self.force is not None:
            push_force.append(self.force*0.05)
            push_force_origin.append(self.skeletons[2].body('h_spine').to_world())
        if len(contact_force) != 0:
            # print(len(contact_force), len(self.controller.contact_list))
            # print("contact_force.size?", contact_force.size, len(contact_force))
            # ri.set_color(1.0, 0.0, 0.0)
            for ii in range(len(contact_force)):
                if 2 * len(contact_force) == len(self.controller.contact_list):
                    body = self.skeletons[2].body(self.controller.contact_list[2*ii])
                    contact_offset = self.controller.contact_list[2*ii+1]
                    # print("contact force : ", contact_force[ii])
                    # ri.render_line(body.to_world(contact_offset), contact_force[ii]/100.)
                    render_vector.append(contact_force[ii]/100.)
                    render_vector_origin.append(body.to_world(contact_offset))


if __name__ == '__main__':
    print('Example: Skating -- pushing side to side')

    pydart.init()
    print('pydart initialization OK')


    world = MyWorld()
    print('MyWorld  OK')

    skel = world.skeletons[2]
    q = skel.q

    # q["j_abdomen_1"] = -0.2
    # q["j_abdomen_2"] = -0.2

    # q["j_thigh_right_y", "j_thigh_left_y"] = -0.785, 0.785
    # q["j_shin_right", "j_shin_left"] = 0., 0.
    # q["j_thigh_right_x", "j_thigh_right_y", "j_thigh_right_z"] = -0.1, -0.5, 0.2
    # q["j_thigh_left_x", "j_thigh_left_y"] = 0.2, 0.5
    # q["j_thigh_left_z", "j_shin_left"] = 0.2, -0.2
    # q["j_thigh_right_z", "j_shin_right"] = 0.2, -0.2
    # q["j_heel_left_1"] = 0.2
    # q["j_heel_right_1"] = 0.2
    #
    # # both arm T-pose
    q["j_bicep_left_x", "j_bicep_left_y", "j_bicep_left_z"] = 1.5, 0.0, 0.0
    q["j_bicep_right_x", "j_bicep_right_y", "j_bicep_right_z"] = -1.5, 0.0, 0.0

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
    viewer.doc.addRenderer('contactForce', yr.VectorsRenderer(render_vector, render_vector_origin, (255, 0, 0)))
    viewer.doc.addRenderer('pushForce', yr.WideArrowRenderer(push_force, push_force_origin, (0, 255,0)))
    viewer.startTimer(1/25.)
    viewer.motionViewWnd.glWindow.pOnPlaneshadow = (0., -0.99+0.0251, 0.)

    viewer.doc.addRenderer('bladeForce', yr.WideArrowRenderer(blade_force, blade_force_origin, (0, 0, 255)))

    def simulateCallback(frame):
        for i in range(40):
            world.step()
        world.render_with_ys()

    viewer.setSimulateCallback(simulateCallback)
    viewer.show()

    Fl.run()
