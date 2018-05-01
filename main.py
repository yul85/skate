import numpy as np
import pydart2 as pydart
import math
import IKsolver
import QPsolver
import IPC_1D
from scipy import interpolate

class MyWorld(pydart.World):
    def __init__(self, ):
        pydart.World.__init__(self, 1.0 / 2000.0, './data/skel/cart_pole_blade.skel')
        # pydart.World.__init__(self, 1.0 / 2000.0, './data/skel/cart_pole.skel')
        self.force = None
        self.duration = 0
        self.skeletons[0].body('ground').set_friction_coeff(0.02)
        self.state = [0., 0., 0., 0.]
        self.cart = IPC_1D.Cart(0., 1.)
        self.pendulum = IPC_1D.Pendulum(1.65*0.5, 0., 59.999663)

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

        #self.solve()
        self.controller.target = self.ik.solve()
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

        #solve ik
        if (len(self.left_foot_traj[0]) -1) > self.gtime:
            self.gtime = self.gtime + 1
        else:
            print("TRAJECTORY OVER!!!!")
            self.gtime = self.gtime

        self.ik.update_target()
        self.controller.target = self.ik.solve()
        # print(self.controller.target)

        # print("self.controller.sol_lambda: \n", self.controller.sol_lambda)

        # f= []
        # print("self.controller.V_c", type(self.controller.V_c), self.controller.V_c)
        # for n in range(len(self.controller.sol_lambda) // 4):
        #     f = self.controller.V_c.dot(self.controller.sol_lambda)

        if len(self.controller.sol_lambda) != 0:
            f_vec = self.controller.V_c.dot(self.controller.sol_lambda)
            # print("f", f_vec)

            f_vec = np.asarray(f_vec)

            # self.contact_force = np.zeros(self.controller.contact_num)
            for ii in range(self.controller.contact_num):
                self.contact_force.append(np.array([f_vec[3*ii], f_vec[3*ii+1], f_vec[3*ii+2]]))
                # self.contact_force[ii] = np.array([f_vec[3*ii], f_vec[3*ii+1], f_vec[3*ii+2]])
                # print("contact_force:", self.contact_force[ii])

            for ii in range(len(f_vec) // 3):
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

        # render contact force --yul
        contact_force = np.asarray(self.contact_force)

        if contact_force.size != 0:
            ri.set_color(1.0, 0.0, 0.0)
            for ii in range(self.controller.contact_num):
                print("contact force : ", contact_force[ii])
                ri.render_line(self.skeletons[2].body(self.controller.contact_list[2*ii]).
                               to_world(self.controller.contact_list[2*ii+1]), contact_force[ii])

        for ctr_n in range(0, len(self.left_foot_traj[0])-1, 10):
            ri.set_color(0., 1., 0.)
            ri.render_sphere(np.array([self.left_foot_traj[0][ctr_n], -0.9, self.left_foot_traj[1][ctr_n]]), 0.01)
            ri.set_color(0., 0., 1.)
            ri.render_sphere(np.array([self.right_foot_traj[0][ctr_n], -0.9, self.right_foot_traj[1][ctr_n]]), 0.01)

        # render axes
        ri.render_axes(np.array([0, 0, 0]), 0.5)

if __name__ == '__main__':
    print('Example: test IPC in 3D')

    pydart.init()
    print('pydart initialization OK')

    world = MyWorld()
    print('MyWorld  OK')

    skel = world.skeletons[2]

    # enable the contact of skeleton
    # skel.set_self_collision_check(1)
    # world.skeletons[0].body('ground').set_collidable(False)

    q = skel.q

    # q["j_pelvis_pos_y"] = -0.05
    # q["j_pelvis_rot_y"] = -0.2
    # q["j_pelvis_rot_z"] = -0.1
    # q["j_thigh_left_z", "j_shin_left", "j_heel_left_1"] = 0.15, -0.4, 0.25
    # q["j_thigh_right_z", "j_shin_right", "j_heel_right_1"] = 0.15, -0.4, 0.25

    q["j_thigh_left_z", "j_shin_left", "j_heel_left_1"] = 0.30, -0.5, 0.25
    q["j_thigh_right_z", "j_shin_right", "j_heel_right_1"] = 0.30, -0.5, 0.25

    q["j_heel_left_2"] = 0.7
    q["j_heel_right_2"] = -0.7

    q["j_abdomen_2"] = 0.0
    # both arm T-pose
    # q["j_bicep_left_x", "j_bicep_left_y", "j_bicep_left_z"] = 1.5, 0.0, 0.0
    # q["j_bicep_right_x", "j_bicep_right_y", "j_bicep_right_z"] = -1.5, 0.0, 0.0

    q["j_pelvis_rot_z"] = -0.2

    skel.set_positions(q)
    print('skeleton position OK')

    # print('[Joint]')
    # for joint in skel.joints:
    #     print("\t" + str(joint))
    #     print("\t\tparent = " + str(joint.parent_bodynode))
    #     print("\t\tchild = " + str(joint.child_bodynode))
    #     print("\t\tdofs = " + str(joint.dofs))

    # Set joint damping
    # for dof in skel.dofs:
    #     dof.set_damping_coefficient(80.0)

    pydart.gui.viewer.launch_pyqt5(world)