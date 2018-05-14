import numpy as np
import pydart2 as pydart
import math
import IKsolver
import QPsolver
import IPC_1D
from scipy import interpolate

class MyWorld(pydart.World):
    def __init__(self, ):
        pydart.World.__init__(self, 1.0 / 1000.0, './data/skel/cart_pole_blade.skel')
        # pydart.World.__init__(self, 1.0 / 2000.0, './data/skel/cart_pole.skel')
        self.force = None
        self.duration = 0
        self.skeletons[0].body('ground').set_friction_coeff(0.02)
        self.state = [0., 0., 0., 0.]
        self.cart = IPC_1D.Cart(0., 1.)
        # self.pendulum = IPC_1D.Pendulum(1.65*0.5, 0., 59.999663)
        self.pendulum = IPC_1D.Pendulum(1.65 * 0.5, 0., 59.999663)
        # self.out_spline = self.generate_spline_trajectory()
        plist = [(0, 0), (0.2, 0), (0.5, -0.3), (1.0, -0.5), (1.5, -0.3), (1.8, 0), (2.0, 0.0)]
        self.left_foot_traj, self.left_der = self.generate_spline_trajectory(plist)
        plist = [(0, 0), (0.2, 0), (0.5, 0.3), (1.0, 0.5), (1.5, 0.3), (1.8, 0), (2.0, 0.0)]
        self.right_foot_traj, self.right_der = self.generate_spline_trajectory(plist)

        # print("out_spline", type(self.out_spline), type(self.out_spline[0]))

        skel = self.skeletons[2]
        # Set target pose
        # self.target = skel.positions()

        self.controller = QPsolver.Controller(skel, self.dt)

        self.gtime = 0
        self.ik = IKsolver.IKsolver(self.skeletons[1], self.skeletons[2], self.left_foot_traj, self.right_foot_traj, self.gtime, self.pendulum)
        self.ik.update_target()
        self.controller.target = self.ik.solve()
        # self.controller.target = np.zeros(self.skeletons[2].ndofs)
        # self.controller.target = np.zeros(self.skeletons[2].ndofs)  #self.skeletons[2].q
        skel.set_controller(self.controller)
        print('create controller OK')

        self.contact_force = []

    def generate_spline_trajectory(self, plist):
        ctr = np.array(plist)

        x = ctr[:, 0]
        y = ctr[:, 1]

        l = len(x)
        t = np.linspace(0, 1, l - 2, endpoint=True)
        t = np.append([0, 0, 0], t)
        t = np.append(t, [1, 1, 1])
        tck = [t, [x, y], 3]
        u3 = np.linspace(0, 1, (500), endpoint=True)

        # tck, u = interpolate.splprep([x, y], k=3, s=0)
        # u = np.linspace(0, 1, num=50, endpoint=True)
        out = interpolate.splev(u3, tck)
        # print("out: ", out)
        der = interpolate.splev(u3, tck, der = 1)
        # print("der: ", der)

        # print("x", out[0])
        # print("y", out[1])

        return out, der

    def step(self):

        # cf = pydart.world.CollisionResult.num_contacts()
        # print("CollisionResult : \n", cf)

        if self.force is not None and self.duration >= 0:
            self.duration -= 1
            self.skeletons[2].body('h_pelvis').add_ext_force(self.force)

        skel = world.skeletons[1]
        q = skel.q

        foot_center = .5 * (world.skeletons[2].body("h_heel_left").com() + world.skeletons[2].body("h_heel_right").com())
        # self.cart.x = world.skeletons[2].body("h_heel_left").to_world([0.0, 0.0, 0.0])[0]
        self.cart.x = foot_center[0]
        self.cart.x_dot = world.skeletons[2].body("h_heel_left").com_linear_velocity()[0]
        com_pos = world.skeletons[2].com()
        # v1 = com_pos - world.skeletons[2].body("h_heel_left").to_world([0.0, 0.0, 0.0])
        v1 = com_pos - foot_center
        v2 = np.array([0., 1., 0.])
        v1 = v1 / np.linalg.norm(v1)    # normalize
        self.pendulum.theta = np.sign(-v1[0]) * math.acos(v1.dot(v2))
        self.pendulum.theta_dot = self.pendulum.length * (world.skeletons[2].com_velocity()[0] - self.cart.x_dot) * math.cos(self.pendulum.theta)

        self.state[0] = self.cart.x
        self.state[1] = self.cart.x_dot
        self.state[2] = self.pendulum.theta
        self.state[3] = self.pendulum.theta_dot

        q["j_cart_x"] = self.state[0]
        q["j_cart_z"] = 0.
        # q["default"] = -0.92
        q["j_pole_x"] = 0.
        q["j_pole_y"] = 0.
        q["j_pole_z"] = self.state[2]

        skel.set_positions(q)

        F = IPC_1D.find_lqr_control_input(self.cart, self.pendulum, -9.8)
        IPC_1D.apply_control_input(self.cart, self.pendulum, F, 0.01, self.state[3], -9.8)

        self.state[0] = self.cart.x
        self.state[1] = self.cart.x_dot
        self.state[2] = self.pendulum.theta
        self.state[3] = self.pendulum.theta_dot

        q["j_cart_x"] = self.state[0]
        q["j_cart_z"] = 0.
        # q["default"] = 0.
        q["j_pole_x"] = 0.
        q["j_pole_y"] = 0.
        q["j_pole_z"] = self.state[2]

        # print("q", q)

        # if self.gtime < 250:
        #     world.skeletons[2].q["j_pelvis_rot_z"] = -0.2
        #     print(self.gtime)

        if (len(self.left_foot_traj[0]) -1) > self.gtime:
            print("UPDATE TARGET: ", self.gtime)
            self.gtime = self.gtime + 1
        else:
            print("TRAJECTORY OVER!!!!")
            self.gtime = self.gtime

        #solve ik
        self.ik.update_target()
        self.controller.target = self.ik.solve()
        # self.controller.target = np.zeros(self.skeletons[2].ndofs)
        # self.controller.target = self.skeletons[2].q
        # print("qqqqqq", self.controller.target)

        # print("self.controller.sol_lambda: \n", self.controller.sol_lambda)

        # f= []
        # print("self.controller.V_c", type(self.controller.V_c), self.controller.V_c)
        # for n in range(len(self.controller.sol_lambda) // 4):
        #     f = self.controller.V_c.dot(self.controller.sol_lambda)
        del self.contact_force[:]
        if len(self.controller.sol_lambda) != 0:
        #     for ii in range(len(self.controller.sol_lambda)):
                # if ii % 2 != 0.0:
                #     self.controller.sol_lambda[ii] = 5*self.controller.sol_lambda[ii]
            # print("self.controller.sol_lambda: \n", self.controller.sol_lambda)

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

        skel.set_positions(q)

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

        ri.render_sphere([2.0, -0.90, 0], 0.05)
        ri.render_sphere([0.0, -0.90, 0], 0.05)

        # ri.set_color(0.0, 0.0, 1.0)
        # ri.render_sphere(self.skeletons[2].body("h_pelvis").to_world([0.0, 0.0, 0.0]), 0.1)

        # ri.render_sphere([0.0, -0.92+ self.pendulum.length, 0], 0.05)
        # ri.render_sphere([0.0, -0.92+ 0.5*self.pendulum.length, 0], 0.05)
        # ri.set_color(0.0, 1.0, 0.0)
        # ri.render_sphere([self.skeletons[1].q["j_cart_x"] - self.pendulum.length *
        #                   math.sin(self.pendulum.theta), -0.92 + self.pendulum.length * math.cos(self.pendulum.theta), 0.], 0.05)

        # ri.render_sphere([self.skeletons[1].q["j_cart_x"] - self.pendulum.length *
        #                   math.sin(self.pendulum.theta), 0., 0.], 0.05)

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

        for ctr_n in range(0, len(self.left_foot_traj[0])-1, 10):
            ri.set_color(0., 1., 0.)
            ri.render_sphere(np.array([self.left_foot_traj[0][ctr_n], -0.9, self.left_foot_traj[1][ctr_n]]), 0.01)
            ri.set_color(0., 0., 1.)
            ri.render_sphere(np.array([self.right_foot_traj[0][ctr_n], -0.9, self.right_foot_traj[1][ctr_n]]), 0.01)

        # render axes
        ri.render_axes(np.array([0, 0, 0]), 0.5)

if __name__ == '__main__':
    print('Example: Skating -- Swizzles')

    pydart.init()
    print('pydart initialization OK')

    world = MyWorld()
    print('MyWorld  OK')

    skel = world.skeletons[2]

    # enable the contact of skeleton
    # skel.set_self_collision_check(1)
    # world.skeletons[0].body('ground').set_collidable(False)

    q = skel.q

    q["j_thigh_left_z", "j_shin_left", "j_heel_left_1"] = 0.30, -0.5, 0.25
    q["j_thigh_right_z", "j_shin_right", "j_heel_right_1"] = 0.30, -0.5, 0.25

    # q["j_thigh_left_z", "j_shin_left", "j_heel_left_1"] = 0.30, -0.0, 0.0
    # q["j_thigh_right_z", "j_shin_right", "j_heel_right_1"] = 0.30, -0.0, 0.0

    q["j_heel_left_2"] = 0.7
    q["j_heel_right_2"] = -0.7

    # q["j_heel_left_2"] = 0.01
    # q["j_heel_right_2"] = -0.01

    # q["j_abdomen_2"] = -0.5
    # both arm T-pose
    # q["j_bicep_left_x", "j_bicep_left_y", "j_bicep_left_z"] = 1.5, 0.0, 0.0
    # q["j_bicep_right_x", "j_bicep_right_y", "j_bicep_right_z"] = -1.5, 0.0, 0.0

    # todo : make the character move forward !!!!
    q["j_pelvis_rot_z"] = -0.2
    # q["j_thigh_left_z"] = 0.30
    # q["j_thigh_right_z"] = 0.30
    # q["j_shin_left"] = -0.3
    # q["j_shin_right"] = -0.3
    # q["j_heel_left_1"] = 0.25
    # q["j_heel_right_1"] = 0.25

    print('[Joint]')
    for joint in skel.joints:
        print("\t" + str(joint))
        print("\t\tparent = " + str(joint.parent_bodynode))
        print("\t\tchild = " + str(joint.child_bodynode))
        print("\t\tdofs = " + str(joint.dofs))


    skel.set_positions(q)
    print('skeleton position OK')

    pydart.gui.viewer.launch_pyqt5(world)