import numpy as np
import pydart2 as pydart
import control as ctrl
import math
from scipy.optimize import minimize
from scipy import interpolate

class Cart:
    def __init__(self, x, mass):
        self.x = x
        self.x_dot = 0.
        # self.y = int(0.6*world_size) 		# 0.6 was chosen for aesthetic reasons.
        self.mass = mass
        self.color = (0, 255, 0)


class Pendulum:
    def __init__(self, length, theta, ball_mass):
        self.length = length
        self.theta = theta
        self.theta_dot = 0.
        self.ball_mass = ball_mass
        self.color = (0,0,255)


def find_lqr_control_input(cart, pendulum, g):
    # Using LQR to find control inputs

    # The A and B matrices are derived in the report
    A = np.matrix([
        [0, 1, 0, 0],
        [0, 0, g * pendulum.ball_mass / cart.mass, 0],
        [0, 0, 0, 1],
        [0, 0, (cart.mass + pendulum.ball_mass) * g / (pendulum.length * cart.mass), 0]
    ])

    B = np.matrix([
        [0],
        [1 / cart.mass],
        [0],
        [1 / (pendulum.length * cart.mass)]
    ])

    # The Q and R matrices are emperically tuned. It is described further in the report
    Q = np.matrix([
        [1000, 0, 0, 0],
        [0, 0.01, 0, 0],
        [0, 0, 0.0000001, 0],
        [0, 0, 0, 100]
    ])

    R = np.matrix([10.])

    # The K matrix is calculated using the lqr function from the controls library
    K, S, E = ctrl.lqr(A, B, Q, R)
    np.matrix(K)

    x = np.matrix([
        [np.squeeze(np.asarray(cart.x))],
        [np.squeeze(np.asarray(cart.x_dot))],
        [np.squeeze(np.asarray(pendulum.theta))],
        [np.squeeze(np.asarray(pendulum.theta_dot))]
    ])

    desired = np.matrix([
        [2.],
        [0.05],
        [0.],
        [0]
    ])

    F = -np.dot(K, (x - desired))
    return np.squeeze(np.asarray(F))

def apply_control_input(cart, pendulum, F, time_delta, theta_dot, g):
    # Finding x and theta on considering the control inputs and the dynamics of the system
    theta_double_dot = (((cart.mass + pendulum.ball_mass) * g * math.sin(pendulum.theta)) + (F * math.cos(pendulum.theta)) - (pendulum.ball_mass * ((theta_dot)**2.0) * pendulum.length * math.sin(pendulum.theta) * math.cos(pendulum.theta))) / (pendulum.length * (cart.mass + (pendulum.ball_mass * (math.sin(pendulum.theta)**2.0))))
    x_double_dot = ((pendulum.ball_mass * g * math.sin(pendulum.theta) * math.cos(pendulum.theta)) - (pendulum.ball_mass * pendulum.length * math.sin(pendulum.theta) * (theta_dot**2)) + (F)) / (cart.mass + (pendulum.ball_mass * (math.sin(pendulum.theta)**2)))
    cart.x += cart.x_dot * time_delta
    pendulum.theta += pendulum.theta_dot * time_delta
    cart.x_dot += x_double_dot * time_delta
    pendulum.theta_dot = theta_double_dot * time_delta

    # cart.x += ((time_delta**2) * x_double_dot) + (((cart.x - x_tminus2) * time_delta) / previous_time_delta)
    # pendulum.theta += ((time_delta**2)*theta_double_dot) + (((pendulum.theta - theta_tminus2)*time_delta)/previous_time_delta)


class Controller(object):
    def __init__(self, skel, h):
        self.h = h
        self.skel = skel
        ndofs = self.skel.ndofs
        # self.qhat = self.skel.q
        self.target = None

        self.Kp = np.diagflat([0.0] * 6 + [400.0] * (ndofs - 6))
        self.Kd = np.diagflat([0.0] * 6 + [40.0] * (ndofs - 6))
        self.preoffset = 0.0

    def compute(self):
        skel = self.skel

        # Algorithm:
        # Stable Proportional-Derivative Controllers.
        # Jie Tan, Karen Liu, Greg Turk
        # IEEE Computer Graphics and Applications, 31(4), 2011.

        invM = np.linalg.inv(skel.M + self.Kd * self.h)
        # p = -self.Kp.dot(skel.q + skel.dq * self.h - self.qhat)
        p = -self.Kp.dot(skel.q - self.target + skel.dq * self.h)
        d = -self.Kd.dot(skel.dq)
        qddot = invM.dot(-skel.c + p + d + skel.constraint_forces())
        tau = p + d - self.Kd.dot(qddot) * self.h
        #
        # # Check the balance
        # COP = skel.body('h_heel_left').to_world([0.05, 0, 0])
        # offset = skel.C[0] - COP[0] + 0.03
        # preoffset = self.preoffset
        #
        # # Adjust the target pose -- translated from bipedStand app of DART
        #
        # # foot = skel.dof_indices(["j_heel_left_1", "j_toe_left",
        # #                          "j_heel_right_1", "j_toe_right"])
        # foot = skel.dof_indices(["j_heel_left_1", "j_heel_right_1"])
        # # if 0.0 < offset < 0.1:
        # #     k1, k2, kd = 200.0, 100.0, 10.0
        # #     k = np.array([-k1, -k2, -k1, -k2])
        # #     tau[foot] += k * offset + kd * (preoffset - offset) * np.ones(4)
        # #     self.preoffset = offset
        # # elif -0.2 < offset < -0.05:
        # #     k1, k2, kd = 2000.0, 100.0, 100.0
        # #     k = np.array([-k1, -k2, -k1, -k2])
        # #     tau[foot] += k * offset + kd * (preoffset - offset) * np.ones(4)
        # #     self.preoffset = offset
        #
        # # # High-stiffness
        # # k1, k2, kd = 200.0, 10.0, 10.0
        # # k = np.array([-k1, -k2, -k1, -k2])
        # # tau[foot] += k * offset + kd * (preoffset - offset) * np.ones(4)
        # # print("offset = %s" % offset)
        # # self.preoffset = offset
        #
        # # Low-stiffness
        # k1, k2, kd = 20.0, 1.0, 10.0
        # k = np.array([-k1, -k1])
        # tau[foot] += k * offset + kd * (preoffset - offset) * np.ones(2)
        # if offset > 0.03:
        #     tau[foot] += 0.3 * np.array([-100.0, -100.0])
        #     # print("Discrete A")
        # if offset < -0.02:
        #     tau[foot] += -1.0 * np.array([-100.0, -100.0])
        #     # print("Discrete B")
        # # print("offset = %s" % offset)
        # self.preoffset = offset

        # Make sure the first six are zero
        tau[:6] = 0
        return tau



class MyWorld(pydart.World):
    def __init__(self, ):
        """
        """
        pydart.World.__init__(self, 1.0 / 2000.0, './data/skel/cart_pole.skel')

        self.force = None
        self.duration = 0
        # self.skeletons[0].body('ground').set_friction_coeff(0.02)
        self.state = [0., 0., 0., 0.]
        self.cart = Cart(0., 1.)
        self.pendulum = Pendulum(1.65*0.5, 0., 59.999663)

        # self.out_spline = self.generate_spline_trajectory()
        plist = [(0, 0), (0.2, 0), (0.5, -0.3), (1.0, -0.5), (1.5, -0.3), (1.8, 0), (2.0, 0.0)]
        self.left_foot_traj, self.left_der = self.generate_spline_trajectory(plist)
        plist = [(0, 0), (0.2, 0), (0.5, 0.3), (1.0, 0.5), (1.5, 0.3), (1.8, 0), (2.0, 0.0)]
        self.right_foot_traj, self.right_der = self.generate_spline_trajectory(plist)

        # print("out_spline", type(self.out_spline), type(self.out_spline[0]))

        skel = self.skeletons[2]
        # Set target pose
        # self.target = skel.positions()
        self.controller = Controller(skel, self.dt)

        self.gtime = 0
        self.update_target()

        # self.controller.target = self.target

        #self.solve()
        self.controller.target = self.solve()
        skel.set_controller(self.controller)
        print('create controller OK')

    def generate_spline_trajectory(self, plist):
        ctr = np.array(plist)

        x = ctr[:, 0]
        y = ctr[:, 1]

        l = len(x)
        t = np.linspace(0, 1, l - 2, endpoint=True)
        t = np.append([0, 0, 0], t)
        t = np.append(t, [1, 1, 1])
        tck = [t, [x, y], 3]
        u3 = np.linspace(0, 1, (200), endpoint=True)

        # tck, u = interpolate.splprep([x, y], k=3, s=0)
        # u = np.linspace(0, 1, num=50, endpoint=True)
        out = interpolate.splev(u3, tck)
        print("out: ", out)
        der = interpolate.splev(u3, tck, der = 1)
        print("der: ", der)

        # print("x", out[0])
        # print("y", out[1])

        return out, der

    def update_target(self, ):
        skel1 = self.skeletons[1]
        # print("self.gtime:", self.gtime)
        # self.target_foot = np.array([skel1.q["j_cart_x"], -0.92, skel1.q["j_cart_z"]])

        # ground_height = -1.1
        ground_height = -0.92
        self.target_left_foot = np.array([skel1.q["j_cart_x"], ground_height, self.left_foot_traj[1][self.gtime]-0.05-0.3])
        self.target_right_foot = np.array([skel1.q["j_cart_x"], ground_height, self.right_foot_traj[1][self.gtime]+0.05+0.3])
        # l = self.skeletons[1].body('h_pole').local_com()[1]
        l = self.pendulum.length
        self.target_pelvis = np.array([skel1.q["j_cart_x"] - l*math.sin(self.pendulum.theta), l*math.cos(self.pendulum.theta), 0])

        # # Update PD-control target
        # self.target = self.skeletons[2].positions()
        # self.target[("j_bicep_left_x", "j_bicep_right_x")] = 1.5, -1.5
        # self.target[("j_heel_left_2")] = self.left_foot_traj[1][self.gtime] - 0.05
        # self.target[("j_heel_right_2")] = self.right_foot_traj[1][self.gtime] + 0.05

        # print("target", self.target_pelvis)

        # print("target", self.target_pelvis)
        # self.target_pelvis = np.array([0.5, 0.5, 0.])
        # print("target", self.target_pelvis)

    def set_params(self, x):
        q = self.skeletons[2].q
        q = x
        # q[6:] = x
        self.skeletons[2].set_positions(q)

    def f(self, x):

        def rotationMatrixToEulerAngles(R):

            # assert (isRotationMatrix(R))

            sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

            singular = sy < 1e-6

            if not singular:
                x = math.atan2(R[2, 1], R[2, 2])
                y = math.atan2(-R[2, 0], sy)
                z = math.atan2(R[1, 0], R[0, 0])
            else:
                x = math.atan2(-R[1, 2], R[1, 1])
                y = math.atan2(-R[2, 0], sy)
                z = 0

            return np.array([x, y, z])

        self.set_params(x)
        pelvis_state = self.skeletons[2].body("h_pelvis").to_world([0.0, 0.0, 0.0])
        left_foot_state = self.skeletons[2].body("h_heel_left").to_world([0.0, 0.0, 0.0])
        left_foot_ori = self.skeletons[2].body("h_heel_left").world_transform()
        right_foot_state = self.skeletons[2].body("h_heel_right").to_world([0.0, 0.0, 0.0])
        right_foot_ori = self.skeletons[2].body("h_heel_right").world_transform()

        # print("foot_ori: ", left_foot_ori[:3, :3], type(left_foot_ori[:3, :3]))

        # todo : calculate the x_axis according to the spline

        x_axis = np.array([0., 0., self.left_der[0][self.gtime]])
        x_axis = x_axis / np.linalg.norm(x_axis)            #normalize
        y_axis = np.array([0., 1., 0.])
        z_axis = np.cross(x_axis, y_axis)
        z_axis = z_axis / np.linalg.norm(z_axis)            #normalize
        R_des = np.stack((x_axis, y_axis, z_axis), axis=-1)
        # print("x_axis: \n")
        # print(x_axis)
        # print("y_axis: \n")
        # print(y_axis)
        # print("z_axis: \n")
        # print(z_axis)
        # print("R: \n")
        # print(R_des)

        # left_axis_angle =rotationMatrixToEulerAngles(left_foot_ori[:3, :3])
        # right_axis_angle = rotationMatrixToEulerAngles(right_foot_ori[:3, :3])

        left_res_rot = np.transpose(R_des).dot(left_foot_ori[:3, :3])
        right_res_rot = np.transpose(R_des).dot(right_foot_ori[:3, :3])
        # print("vec res: ", left_res_rot)
        left_axis_angle =rotationMatrixToEulerAngles(left_res_rot)
        right_axis_angle = rotationMatrixToEulerAngles(right_res_rot)

        # print("axis_angle", axis_angle)

        # return 0.5 *  np.linalg.norm(left_foot_state - self.target_foot) ** 2
        return 0.5 * (np.linalg.norm(pelvis_state - self.target_pelvis) ** 2 + np.linalg.norm(left_foot_state - self.target_left_foot) ** 2 + np.linalg.norm(right_foot_state - self.target_right_foot) ** 2 + np.linalg.norm(left_axis_angle - np.array([0., 0., 0.])) ** 2 + np.linalg.norm(right_axis_angle - np.array([0., 0., 0.])) ** 2)

    def g(self, x):
        self.set_params(x)

        pelvis_state = self.skeletons[2].body("h_pelvis").to_world([0.0, 0.0, 0.0])
        left_foot_state = self.skeletons[2].body("h_heel_left").to_world([0.0, 0.0, 0.0])
        right_foot_state = self.skeletons[2].body("h_heel_right").to_world([0.0, 0.0, 0.0])

        J_pelvis = self.skeletons[2].body("h_pelvis").linear_jacobian()
        J_left_foot = self.skeletons[2].body("h_heel_left").linear_jacobian()
        J_right_foot = self.skeletons[2].body("h_heel_right").linear_jacobian()

        J_temp = np.vstack((J_pelvis, J_left_foot))
        J = np.vstack((J_temp, J_right_foot))
        AA = np.append(pelvis_state - self.target_pelvis, left_foot_state - self.target_left_foot)
        AAA = np.append(AA, right_foot_state - self.target_right_foot)

        # print("pelvis", pelvis_state - self.target_pelvis)
        # print("J", J_pelvis)
        #
        # print("AA", AA)
        # print("J", J)

        # g = AA.dot(J)
        g = AAA.dot(J)
        # g = AA.dot(J)[6:]

        return g

    def solve(self, ):
        q_backup = self.skeletons[2].q
        res = minimize(self.f, x0=self.skeletons[2].q, jac=self.g, method="SLSQP")
        # res = minimize(self.f, x0=self.skeletons[2].q[6:], jac=self.g, method="SLSQP")
        # print(res)
        self.skeletons[2].set_positions(q_backup)

        return res['x']

    def step(self):
        # todo : Update pendulum state according to present character state
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

        F = find_lqr_control_input(self.cart, self.pendulum, -9.8)
        apply_control_input(self.cart, self.pendulum, F, 0.01, self.state[3], -9.8)

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

        #solve ik
        if (len(self.left_foot_traj[0]) -1) > self.gtime:
            self.gtime = self.gtime + 1
        else:
            self.gtime = self.gtime

        self.update_target()
        self.controller.target = self.solve()
        # print(self.controller.target)

        super(MyWorld, self).step()
        skel.set_positions(q)

    def on_step_event(self, ):
         if self.force is not None and self.duration >= 0:
             self.duration -= 1
             self.skeletons[2].body('h_spine').add_ext_force(self.force)

    def on_key_press(self, key):
        if key == '1':
            self.force = np.array([50.0, 0.0, 0.0])
            self.duration = 100
            print('push backward: f = %s' % self.force)
        elif key == '2':
            self.force = np.array([-50.0, 0.0, 0.0])
            self.duration = 100
            print('push backward: f = %s' % self.force)

    def render_with_ri(self, ri):
        if self.force is not None and self.duration >= 0:
            p0 = self.skeletons[2].body('h_spine').C
            p1 = p0 + 0.01 * self.force
            ri.set_color(1.0, 0.0, 0.0)
            ri.render_arrow(p0, p1, r_base=0.05, head_width=0.1, head_len=0.1)
        # ri.render_sphere(self.mPendulum1.calcCOMpos(), 0.05)        #mPendulum1 COM position

        # ri.render_sphere(np.array([2.0, 0.0, 0.0]), 0.05)
        # ri.render_sphere(np.array([-2.0, 0.0, 0.0]), 0.05)

        # for ii in range(len(self.desiredTraj1)):
        #     ri.render_sphere([self.desiredTraj1[ii, 0], 0.2, 0], 0.01)
        # ri.render_sphere([self.desiredTraj1[2000, 0], 0.2, 0], 0.02)
        ri.render_sphere([2.0, -0.90, 0], 0.05)
        ri.render_sphere([0.0, -0.90, 0], 0.05)

        for ctr_n in range(0, len(self.left_foot_traj[0])-1, 10):
            ri.set_color(0., 1., 0.)
            ri.render_sphere(np.array([self.left_foot_traj[0][ctr_n], -0.9, self.left_foot_traj[1][ctr_n]]), 0.01)
            ri.set_color(0., 0., 1.)
            ri.render_sphere(np.array([self.right_foot_traj[0][ctr_n], -0.9, self.right_foot_traj[1][ctr_n]]), 0.01)
        # ri.render_sphere(np.array([self.out_spline[0][10], -0.9, self.out_spline[1][10]]), 0.03)
        # ri.render_sphere(np.array([self.out_spline[0][20], -0.9, self.out_spline[1][20]]), 0.03)
        # ri.render_sphere(np.array([self.out_spline[0][30], -0.9, self.out_spline[1][30]]), 0.03)
        # ri.render_sphere(np.array([self.out_spline[0][40], -0.9, self.out_spline[1][40]]), 0.03)

        # render axes
        ri.render_axes(np.array([0, 0, 0]), 0.5)
        # ri.set_color(0., 0., 0.)
        # ri.render_sphere(np.array([0, 0, 0]), 0.01)
        # ri.set_color(1.0, 0., 0.)
        # ri.render_line(np.array([0, 0, 0]), np.array([0.3, 0, 0]))
        # ri.set_color(0.0, 1., 0.)
        # ri.render_line(np.array([0, 0, 0]), np.array([0.0, 0.3, 0]))
        # ri.set_color(0.0, 0., 1.)
        # ri.render_line(np.array([0, 0, 0]), np.array([0.0, 0, 0.3])


if __name__ == '__main__':
    print('Example: test IPC in 3D')

    pydart.init()
    print('pydart initialization OK')

    world = MyWorld()
    print('MyWorld  OK')

    skel = world.skeletons[2]
    q = skel.q

    q["j_pelvis_pos_y"] = -0.05
    q["j_pelvis_rot_y"] = -0.2
    q["j_thigh_left_z", "j_shin_left", "j_heel_left_1"] = 0.15, -0.4, 0.25
    q["j_thigh_right_z", "j_shin_right", "j_heel_right_1"] = 0.15, -0.4, 0.25
    q["j_abdomen_2"] = 0.0
    # both arm T-pose
    q["j_bicep_left_x", "j_bicep_left_y", "j_bicep_left_z"] = 1.5, 0.0, 0.0
    q["j_bicep_right_x", "j_bicep_right_y", "j_bicep_right_z"] = -1.5, 0.0, 0.0

    skel.set_positions(q)
    print('skeleton position OK')

    print('[Joint]')
    for joint in skel.joints:
        print("\t" + str(joint))
        print("\t\tparent = " + str(joint.parent_bodynode))
        print("\t\tchild = " + str(joint.child_bodynode))
        print("\t\tdofs = " + str(joint.dofs))

    # Set joint damping
    # for dof in skel.dofs:
    #     dof.set_damping_coefficient(80.0)

    pydart.gui.viewer.launch_pyqt5(world)