import numpy as np
import pydart2 as pydart
import math
import IKsolve_double_stance
import copy
from scipy import interpolate
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

lf_trajectory = []
rf_trajectory = []

lft = []
lft_origin = []
rft = []
rft_origin = []

e_t_vec = []
e_n_vec = []
e_b_vec = []
coordinate_origin = []

ik_on = True
ik_on = False


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

        self.radius = 1.0
        x = [0, 0.2, 0.5, 0.8, 1.0]
        y = []
        plist = []
        for i in range(len(x)):
            y.append(math.sqrt(self.radius**2 - x[i]**2))
            plist.append((x[i], -y[i]))

        # print(plist)
        # print(y)

        # plist = [(0, 0), (0.2, 0), (0.5, -0.15), (1.0, -0.25), (1.5, -0.15), (1.8, 0), (2.0, 0.0)]
        # plist = [(0, 0), (0.2, 0.), (0.5, 0.4), (1.0, 0.7), (1.5, 1.0), (1.8, 1.2), (2.0, 1.5)]
        self.left_foot_traj, self.left_der = self.generate_spline_trajectory(plist)
        plist = [(0, 0), (0.2, 0), (0.5, 0.15), (1.0, 0.25), (1.5, 0.15), (1.8, 0), (2.0, 0.0)]
        self.right_foot_traj, self.right_der = self.generate_spline_trajectory(plist)

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
        # s00q[upper_body] = 0.0, -0., -0.2
        # s00q[spine] = 0.0, 0.0, 0.2
        # s00q[left_leg] = 0., 0., 0.3, -0.5
        # s00q[right_leg] = -0.3, -0., 0., -0.5
        # s00q[leg_y] = 0., -0.785
        # s00q[arms] = 1.5, -1.5
        # s00q[arms_y] = -1.5, 1.5
        # s00q[arms_z] = 3., 3.
        # s00q[foot] = 0., 0., 0.2, -0.3, -0., 0.3
        state00 = State("state00", 1.0, 0.0, 0.2, s00q)

        s01q = np.zeros(skel.ndofs)
        s01q[upper_body] = 0., 0., -0.4
        s01q[spine] = 0.0, 0., 0.4
        s01q[left_leg] = 0., 0., 0.3, -0.5
        s01q[right_leg] = -0., -0., 0.3, -0.5
        s01q[foot] = 0., 0., 0.2, -0., -0., 0.2
        state01 = State("state01", 3.0, 2.2, 0.0, s01q)

        s2q = np.zeros(skel.ndofs)
        s2q[upper_body] = 0., 0., -0.4
        s2q[spine] = 0.0, 0., 0.4
        s2q[left_leg] = 0., 0., 0.3, -0.5
        s2q[right_leg] = -0., -0., 0.3, -0.5
        s2q[foot] = 0., 0., 0.2, -0., -0., 0.2
        state2 = State("state2", 5., 2.2, 0.0, s2q)

        s3q = np.zeros(skel.ndofs)
        s3q[upper_body] = 0., 0., -0.4
        s3q[spine] = 0.0, 0., 0.4
        s3q[left_leg] = 0., 0., 0.3, -0.5
        s3q[right_leg] = -0., -0., 0.3, -0.5
        s3q[foot] = 0., 0., 0.2, -0., -0., 0.2
        state3 = State("state3", 100.0, 2.2, 0.0, s3q)

        self.state_list = [state00, state01, state2, state3]

        state_num = len(self.state_list)
        self.state_num = state_num
        # print("state_num: ", state_num)

        self.curr_state = self.state_list[0]
        self.elapsedTime = 0.0
        self.curr_state_index = 0
        # print("backup angle: ", backup_q)
        # print("cur angle: ", self.curr_state.angles)

        self.skeletons[3].set_positions(self.curr_state.angles)

        if ik_on:
            self.ik = IKsolve_double_stance.IKsolver(skel, self.skeletons[3], self.dt)
            print('IK ON')

        print('create controller OK')

        self.rd_contact_forces = []
        self.rd_contact_positions = []

        # print("dof: ", skel.ndofs)

        # nonholonomic constraint initial setting
        _th = 10. * math.pi / 180.

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

        self.trajUpdateTime = 0
        self.tangent_index = 0

        self.e_t = np.array([1.0, 0.0, 0.0])
        self.e_n = np.array([0.0, 1.0, 0.0])
        self.e_b = np.array([0.0, 0.0, 1.0])

        self.old_lf_tangent_vec = np.array([1.0, 0.0, 0.0])


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

        # print("self.curr_state: ", self.curr_state.name)

        # if self.curr_state.name == "state01":
        #     self.force = np.array([0.0, 0.0, 20.0])
        # else:
        #     self.force = None

        if self.curr_state.name == "state2":
            self.force = np.array([20.0, 0.0, 0.0])
        # elif self.curr_state.name == "state2":
        #     self.force = np.array([0.0, 0.0, -10.0])
        else:
            self.force = None
        if self.force is not None:
            self.skeletons[2].body('h_pelvis').add_ext_force(self.force)

        # turning
        # lf_tangent_vec = np.array([1.0, 0.0, .0])
        # rf_tangent_vec = np.array([1.0, 0.0, .0])

        skel = self.skeletons[2]
        # print(skel.com_velocity()[1])
        if self.curr_state.name == "state3":
            skel = self.skeletons[2]
            m = skel.mass()
            r = self.radius  # radius = 2.0

            # make ski track

            # lf_tangent_vec_normal = np.array([0.0, 0.0, -1.0])
            # rf_tangent_vec_normal = np.array([0.0, 0.0, 1.0])

            # calculate tangent vector

            # print("time: ", self.time())
            if 0.01 > self.time() - self.trajUpdateTime:
                # print("in loop", self.tangent_index)
                lf_tangent_vec = np.asarray([self.left_der[0][self.tangent_index], 0.0, self.left_der[1][self.tangent_index]])

                if np.linalg.norm(lf_tangent_vec) != 0:
                    lf_tangent_vec = lf_tangent_vec / np.linalg.norm(lf_tangent_vec)
            else:
                lf_tangent_vec = self.old_lf_tangent_vec
                if self.tangent_index < len(self.left_foot_traj[0])-1:
                    self.tangent_index += 1
                else:
                    print("over?")
                    self.tangent_index = 0
                self.trajUpdateTime = self.time()
            # print("left foot traj: ", lf_tangent_vec)
            # print("right_foot_traj: ", rf_tangent_vec)

            # e_t = np.array([1, 0, 0])
            # e_n = np.array([0, 1, 0])
            # e_b = np.array([0, 0, 1])

            self.old_lf_tangent_vec = lf_tangent_vec

            e_t = lf_tangent_vec
            e_n = np.array([0, 1, 0])
            e_b = np.cross(e_t, e_n)
            # print(e_b, np.linalg.norm(e_b))

            self.e_t = e_t
            self.e_n = e_n
            self.e_b = e_b

            f_n = m * np.array([0, -9.8, 0])
            f_b = -m * skel.com_acceleration()[0] * e_t
            f_centrifugal = 1 * m * skel.com_velocity()[0] * skel.com_velocity()[0] * e_b / r
            f_load = f_n + f_b + f_centrifugal

            val = np.dot(f_load, e_b) / np.dot(f_load, e_n)
            inclined_angle = math.atan(val)

            # print(e_t)
            # print(inclined_angle * 180 / math.pi)

            # q = skel.q
            # q["j_heel_left_x", "j_heel_right_x"] = inclined_angle, inclined_angle
            # self.curr_state.angles = q

            self.curr_state.angles[12] = inclined_angle
            self.curr_state.angles[21] = inclined_angle

            # get foot y rotation angle
            # rfpoint = skel.body("h_blade_right").to_world((0.0216 + 0.104, -0.0216 - 0.027, 0.))
            # rrpoint = skel.body("h_blade_right").to_world((0.0216 - 0.104, -0.0216 - 0.027, 0.))
            #
            # blade_vec = rfpoint-rrpoint
            # blade_vec = blade_vec/np.linalg.norm(blade_vec)
            # y_angle = math.acos(blade_vec.dot(e_t))
            # print(y_angle*180/math.pi)
            # self.curr_state.angles[13] = y_angle
            # self.curr_state.angles[22] = y_angle

        # print(skel.dof_indices(["j_heel_left_x", "j_heel_right_x"]))

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
        ground_height = 0.
        # ground_height = -0.98 + 0.05
        # ground_height = -0.98
        if ik_on:
            ik_res = copy.deepcopy(self.curr_state.angles)
            if self.curr_state.name == "state2":
                # print("ik solve!!-------------------")
                self.ik.update_target(ground_height)
                ik_res[6:] = self.ik.solve()

            if self.curr_state.name == "state2":
                self.skeletons[3].set_positions(ik_res)

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

        del lf_trajectory[:]
        del rf_trajectory[:]

        del lft[:]
        del lft_origin[:]
        del rft[:]
        del rft_origin[:]

        del e_t_vec[:]
        del e_n_vec[:]
        del e_b_vec[:]
        del coordinate_origin[:]

        e_t_vec.append(self.e_t)
        e_n_vec.append(self.e_n)
        e_b_vec.append(self.e_b)
        coordinate_origin.append(self.skeletons[2].C)

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

        for ctr_n in range(0, len(self.left_foot_traj[0])-1, 10):
            lf_trajectory.append(np.array([self.left_foot_traj[0][ctr_n], 0., self.left_foot_traj[1][ctr_n]]))
            rf_trajectory.append(np.array([self.right_foot_traj[0][ctr_n], 0., self.right_foot_traj[1][ctr_n]]))
            lft.append(np.array([self.left_der[0][ctr_n+1], -0., self.left_der[1][ctr_n+1]]))
            lft_origin.append(np.array([self.left_foot_traj[0][ctr_n], -0., self.left_foot_traj[1][ctr_n]]))
            rft.append(np.array([self.right_foot_traj[0][ctr_n + 1], -0., self.right_foot_traj[1][ctr_n + 1]]))
            rft_origin.append(np.array([self.right_foot_traj[0][ctr_n], -0., self.right_foot_traj[1][ctr_n]]))

if __name__ == '__main__':
    print('Example: Skating -- pushing side to side')

    pydart.init()
    print('pydart initialization OK')

    world = MyWorld()
    print('MyWorld  OK')
    ground = pydart.World(1. / 1000., './data/skel/ground.skel')

    skel = world.skeletons[2]
    q = skel.q

    # q["j_pelvis_pos_y"] = -0.07

    # q["j_thigh_left_x", "j_thigh_left_y", "j_thigh_left_z"] = 0., 0., 0.1
    # q["j_thigh_right_x", "j_thigh_right_y", "j_thigh_right_z"] = -0.2, 0., -0.1
    # q["j_shin_left_z", "j_shin_right_z"] = -0.1, -0.1
    # q["j_heel_left_z", "j_heel_right_z"] = 0.2, 0.2

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
    viewer.doc.addRenderer('ground', yr.DartRenderer(ground, (255, 255, 255), yr.POLYGON_FILL), visible=False)
    viewer.doc.addRenderer('contactForce', yr.VectorsRenderer(render_vector, render_vector_origin, (255, 0, 0)))
    viewer.doc.addRenderer('pushForce', yr.WideArrowRenderer(push_force, push_force_origin, (0, 255,0)))
    viewer.doc.addRenderer('rd_footCenter', yr.PointsRenderer(rd_footCenter))
    viewer.doc.addRenderer('lf_traj', yr.PointsRenderer(lf_trajectory, (200, 0, 0)), visible=False)
    viewer.doc.addRenderer('rf_traj', yr.PointsRenderer(rf_trajectory, (200, 200, 200)), visible=False)
    viewer.doc.addRenderer('lf_tangent', yr.VectorsRenderer(lft, lft_origin, (0, 255, 0)), visible=False)
    viewer.doc.addRenderer('rf_tangent', yr.VectorsRenderer(rft, rft_origin, (0, 0, 255)), visible=False)

    viewer.doc.addRenderer('e_t', yr.VectorsRenderer(e_t_vec, coordinate_origin, (255, 0, 0)))
    viewer.doc.addRenderer('e_n', yr.VectorsRenderer(e_n_vec, coordinate_origin, (0, 255, 0)))
    viewer.doc.addRenderer('e_b', yr.VectorsRenderer(e_b_vec, coordinate_origin, (0, 0, 255)))

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
