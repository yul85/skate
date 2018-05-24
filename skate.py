import numpy as np
import pydart2 as pydart
import QPsolver

class State(object):
    def __init__(self, dt, c_d, c_v, angles):
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

        s2q = np.zeros(skel.ndofs)
        pelvis = skel.dof_indices((["j_pelvis_rot_y", "j_pelvis_rot_z"]))
        upper_body = skel.dof_indices(["j_abdomen_1", "j_abdomen_2"])
        right_leg = skel.dof_indices(["j_thigh_right_x", "j_thigh_right_y", "j_thigh_right_z", "j_shin_right"])
        left_leg = skel.dof_indices(["j_thigh_left_x", "j_thigh_left_y", "j_thigh_left_z", "j_shin_left"])
        arms = skel.dof_indices(["j_bicep_left_x", "j_bicep_right_x"])
        foot = skel.dof_indices(["j_heel_left_1", "j_heel_right_1"])

        s2q[pelvis] = -0.3, -0.0
        # s2q[upper_body] = 0.0, -0.5
        s2q[right_leg] = 0.3, -0.0, -0.2, -0.1
        s2q[left_leg] = 0.0, 0.5, -0.5, -0.5
        s2q[arms] = 1.5, -1.5
        s2q[foot] = 0.2, 0.4

        # state2 = State(0.1, 0.0, 0.2, s2q)
        state2 = State(0.5, 0.0, 0.2, s2q)

        s0q = np.zeros(skel.ndofs)
        right_leg = skel.dof_indices(["j_thigh_right_x", "j_thigh_right_y", "j_thigh_right_z", "j_shin_right"])
        left_leg = skel.dof_indices(["j_thigh_left_x", "j_thigh_left_y", "j_thigh_left_z", "j_shin_left"])
        arms = skel.dof_indices(["j_bicep_left_x", "j_bicep_right_x"])
        s0q[pelvis] = -0.3, -0.0
        # s0q[upper_body] = 0.0, -0.2
        # s0q[right_leg] = 0., -0.5, 0.2, -0.3
        s0q[right_leg] = 0., -0.5, 0.0, -0.17
        s0q[left_leg] = 0.5, 0.8, -0.5, -0.17
        s0q[arms] = 1.5, -1.5
        s0q[foot] = 0.2, 0.2
        state0 = State(0.07, 0.0, 0.2, s0q)

        s1q = np.zeros(skel.ndofs)
        s1q[pelvis] = -0.3, -0.0
        # s1q[upper_body] = 0.0, -0.2
        s1q[right_leg] = 0., -0.0, -0.0, -0.17
        s1q[left_leg] = -0., -0.0, 0.0, -0.17
        s1q[arms] = 1.5, -1.5
        state1 = State(0.3, 2.2, 0.0, s1q)
        # self.state_list = [state2, state0, state1]
        self.state_list = [state2, state1]
        state_num = len(self.state_list)
        self.state_num = state_num
        # print("state_num: ", state_num)

        self.curr_state = self.state_list[0]
        self.elapsedTime = 0.0
        self.curr_state_index = 0
        # print("backup angle: ", backup_q)
        print("cur angle: ", self.curr_state.angles)
        self.controller = QPsolver.Controller(skel, self.dt)

        self.skeletons[3].set_positions(self.curr_state.angles)
        # self.skeletons[3].set_positions(np.zeros(skel.ndofs))
        self.controller.target = self.curr_state.angles
        # self.controller.target = skel.q
        skel.set_controller(self.controller)
        print('create controller OK')

        self.contact_force = []

    def step(self):
        self.skeletons[3].set_positions(self.curr_state.angles)
        # self.skeletons[3].set_positions(np.zeros(skel.ndofs))
        if self.curr_state.dt < self.time() - self.elapsedTime:
            print("change the state!!!", self.curr_state_index)
            self.curr_state_index = self.curr_state_index + 1
            self.curr_state_index = self.curr_state_index % self.state_num
            self.elapsedTime = self.time()
            self.curr_state = self.state_list[self.curr_state_index]
            print("state_", self.curr_state_index)
            print(self.curr_state.angles)

        # self.controller.target = skel.q
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
            self.force = np.array([500.0, 0.0, 0.0])
            self.duration = 1000
            print('push backward: f = %s' % self.force)
        elif key == '2':
            self.force = np.array([-50.0, 0.0, 0.0])
            self.duration = 100
            print('push backward: f = %s' % self.force)

    def render_with_ri(self, ri):
        if self.force is not None and self.duration >= 0:
            p0 = self.skeletons[2].body('h_pelvis').C
            p1 = p0 + 0.01 * self.force
            ri.set_color(1.0, 0.0, 0.0)
            ri.render_arrow(p0, p1, r_base=0.05, head_width=0.1, head_len=0.1)

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
        ri.set_color(1, 0,0)
        ri.render_sphere(np.array([self.skeletons[2].C[0], -0.92, self.skeletons[2].C[2]]), 0.05)

        COP = self.skeletons[2].body('h_heel_right').to_world([0.05, 0, 0])
        ri.set_color(0, 0,1)
        ri.render_sphere(COP, 0.05)

        # todo : ground rendering
        # ri.render_chessboard(5)
        # render axes
        ri.render_axes(np.array([0, 0, 0]), 0.5)

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

    # q["j_thigh_right_x", "j_thigh_right_y"] = -0.2, -0.5
    # q["j_thigh_left_x", "j_thigh_left_y"] = 0.2, 0.5
    # q["j_thigh_left_z", "j_shin_left"] = 0.30, -0.5
    # q["j_thigh_right_z", "j_shin_right"] = 0.30, -0.5
    #
    q["j_heel_left_1"] = 0.2
    q["j_heel_right_1"] = 0.2
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

    pydart.gui.viewer.launch_pyqt5(world)
