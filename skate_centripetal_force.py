import numpy as np
import pydart2 as pydart
import QPsolver
import math
import IKsolve_double_stance
import momentum_con
import motionPlan
from scipy import optimize
import yulTrajectoryOpt
import copy

from fltk import *
from PyCommon.modules.GUI import hpSimpleViewer as hsv
from PyCommon.modules.Renderer import ysRenderer as yr
from PyCommon.modules.Simulator import yulQpSimulator_equality_blade_centripetal_force as hqp

render_vector = []
render_vector_origin = []
push_force = []
push_force_origin = []
blade_force = []
blade_force_origin = []

rd_footCenter = []

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
        # pydart.World.__init__(self, 1.0 / 1000.0, './data/skel/cart_pole_blade_3dof_same.skel')
        pydart.World.__init__(self, 1.0 / 1000.0, './data/skel/cart_pole_blade_3dof.skel')
        self.force = None
        self.duration = 0
        self.skeletons[0].body('ground').set_friction_coeff(0.02)

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
        pelvis_x = skel.dof_indices(["j_pelvis_rot_x"])
        pelvis = skel.dof_indices(["j_pelvis_rot_y", "j_pelvis_rot_z"])
        upper_body = skel.dof_indices(["j_abdomen_x", "j_abdomen_y", "j_abdomen_z"])
        spine = skel.dof_indices(["j_spine_x", "j_spine_y", "j_spine_z"])
        right_leg = skel.dof_indices(["j_thigh_right_x", "j_thigh_right_y", "j_thigh_right_z", "j_shin_right_z"])
        left_leg = skel.dof_indices(["j_thigh_left_x", "j_thigh_left_y", "j_thigh_left_z", "j_shin_left_z"])
        knee = skel.dof_indices(["j_shin_left_x", "j_shin_right_x"])
        # arms = skel.dof_indices(["j_bicep_left_x", "j_bicep_right_x"])
        foot = skel.dof_indices(["j_heel_left_x", "j_heel_left_y", "j_heel_left_z", "j_heel_right_x", "j_heel_right_y", "j_heel_right_z"])
        leg_y = skel.dof_indices(["j_thigh_right_y", "j_thigh_left_y"])
        # blade = skel.dof_indices(["j_heel_right_2"])


        # #----------------------------------
        # # pushing side to side new (180718)
        # #----------------------------------

        s0q = np.zeros(skel.ndofs)
        # s0q[pelvis] = 0., -0.
        # s0q[upper_body] = 0.0, -0.5
        # s0q[left_leg] = 0., 0., 0.4, -1.1
        # s0q[right_leg] = -0., -0., -0., -0.05
        # s0q[leg_y] = -0.785, 0.785
        # s0q[arms] = 1.5, -1.5
        # s0q[foot] = -0.0, 0.0, 0., 0.0, -0., 0.
        state0 = State("state0", 0.3, 0.0, 0.2, s0q)

        s01q = np.zeros(skel.ndofs)
        # s01q[pelvis] = 0., -0.3
        s01q[upper_body] = 0.0, 0., -0.5
        s01q[spine] = 0.0, 0., 0.5
        s01q[left_leg] = -0., 0., 0.3, -0.5
        s01q[right_leg] = -0., -0., 0.3, -0.5
        # s01q[arms] = 1.5, -1.5
        # s01q[blade] = -0.3
        s01q[foot] = -0.0, 0.0, 0.2, 0.0, 0.0, 0.2
        state01 = State("state01", 0.5, 2.2, 0.0, s01q)

        s1q = np.zeros(skel.ndofs)
        # s1q[pelvis] = 0., -0.1
        s1q[upper_body] = 0.0, 0., -0.5
        s1q[spine] = 0.0, 0., 0.5
        s1q[left_leg] = -0., 0., 0.3, -0.5
        s1q[right_leg] = -0.2, -0., 0.3, -0.5
        # s1q[arms] = 1.5, -1.5
        # s1q[blade] = -0.3
        s1q[foot] = -0.0, 0.0, 0.2, 0.0, -0., 0.2
        state1 = State("state1", 0.5, 2.2, 0.0, s1q)

        roro_angle = 10.

        s2q = np.zeros(skel.ndofs)
        s2q[upper_body] = -0., 0., -0.5
        s2q[spine] = -0., 0., 0.5
        s2q[left_leg] = -0., 0., 0.3, -0.5
        s2q[right_leg] = -0.2, -0., 0.3, -0.5
        # s2q[knee] = roro_angle * math.pi/180., roro_angle * math.pi/180.
        # s2q[arms] = 1.5, -1.5
        # s2q[arms] = -0.5, -0.5
        # s2q[arms] = roro_angle * math.pi/180., roro_angle * math.pi/180.
        s2q[foot] = -0.2, 0., 0.2, -0.2, 0., 0.2
        # s2q[foot] = -0., 0., 0.4, 0., 0., 0.4
        # s2q[foot] = -roro_angle * math.pi/180., 0., 0.4, -roro_angle * math.pi/180., 0., 0.2
        state2 = State("state2", 2.0, 0.0, 0.2, s2q)

        s02q = np.zeros(skel.ndofs)
        # s02q[pelvis_x] = -10. * math.pi/180.
        s02q[upper_body] = 0., 0., -0.5
        s02q[spine] = 0.0, 0., 0.5
        s02q[left_leg] = 0.2, 0., 0.3, -0.5
        s02q[right_leg] = -0.2, -0., 0.3, -0.5
        # s02q[knee] = -roro_angle * math.pi / 180., -roro_angle * math.pi / 180.
        # s02q[arms] = 1.5, -1.5
        # s02q[arms] = -roro_angle * math.pi/180., -roro_angle * math.pi/180.
        s02q[foot] = -0.2, 0., 0.2, -0.2, 0., 0.2
        # s02q[foot] = roro_angle * math.pi / 180., 0., 0.4, roro_angle * math.pi / 180., -0., 0.2
        state02 = State("state02", 2.0, 0.0, 0.2, s02q)

        s3q = np.zeros(skel.ndofs)
        # s3q[pelvis_x] = 10. * math.pi / 180.
        s3q[upper_body] = 0.0, 0., -0.5
        s3q[spine] = 0.0, 0., 0.5
        s3q[left_leg] = -0.3, 0., 0.5, -0.7
        s3q[right_leg] = -0.2, -0., 0.3, -0.5
        # s3q[knee] = -roro_angle * math.pi / 180., -roro_angle * math.pi / 180.
        # s3q[arms] = 1.5, -1.5
        s3q[foot] = -0.2, 0.2, 0.2, -0.2, -0., 0.2
        state3 = State("state3", 3.0, 2.2, 0.0, s3q)

        s03q = np.zeros(skel.ndofs)
        # s3q[pelvis_x] = 10. * math.pi / 180.
        # s03q[upper_body] = 0.0, 0., -0.5
        # s03q[spine] = 0.0, 0., 0.5
        s03q[left_leg] = -0., 0., 0.5, -0.5
        s03q[right_leg] = -0., -0., 0.3, -0.5
        # s3q[arms] = 1.5, -1.5
        s03q[foot] = 0., 0.0, 0.2, 0., 0.0, 0.2
        state03 = State("state03", 1.0, 2.2, 0.0, s03q)

        # self.state_list = [state0, state01, state1, state2, state02, state3, state03]
        self.state_list = [state0, state01, state1, state2, state02, state01, state01, state01, state01, state01, state01, state01]

        # self.state_list = [state0, state01, state1, state2, state3, state03]
        # self.state_list = [state0, state1, state2, state02, state01, state3, state03]

        state_num = len(self.state_list)
        self.state_num = state_num
        # print("state_num: ", state_num)

        self.curr_state = self.state_list[0]
        self.elapsedTime = 0.0
        self.curr_state_index = 0
        # print("backup angle: ", backup_q)
        # print("cur angle: ", self.curr_state.angles)

        self.controller = QPsolver.Controller(skel, self.skeletons[3], self.dt, self.curr_state.name)

        self.skeletons[3].set_positions(self.curr_state.angles)

        self.controller.target = self.curr_state.angles
        # self.controller.target = skel.q
        # skel.set_controller(self.controller)

        if ik_on:
            self.ik = IKsolve_double_stance.IKsolver(skel, self.skeletons[3], self.dt)
            print('IK ON')

        print('create controller OK')

        self.contact_force = []
        self.contactPositionLocals = []
        self.bodyIDs = []
        # print("dof: ", skel.ndofs)

        self.last_vec = np.zeros(3)

    def step(self):

        if self.curr_state.name == "state1":
            self.force = np.array([70.0, 0.0, 0.0])
        # elif self.curr_state.name == "state2":
        #     self.force = np.array([0.0, 0.0, -10.0])
        else:
            self.force = None

        self.controller.cur_state = self.curr_state.name
        if self.force is not None:
            self.skeletons[2].body('h_pelvis').add_ext_force(self.force)

        self.skeletons[3].set_positions(self.curr_state.angles)

        if self.curr_state.dt < self.time() - self.elapsedTime:
            self.curr_state_index = self.curr_state_index + 1
            self.curr_state_index = self.curr_state_index % self.state_num
            self.elapsedTime = self.time()
            self.curr_state = self.state_list[self.curr_state_index]
            # print("state_", self.curr_state_index)
            # print(self.curr_state.angles)

        # ground_height = -0.98 + 0.0251
        ground_height = -0.98 + 0.05
        # ground_height = -0.98
        if ik_on:
            ik_res = copy.deepcopy(self.curr_state.angles)
            if self.curr_state.name == "state2":
                # print("ik solve!!-------------------")
                self.ik.update_target(ground_height)
                ik_res[6:] = self.ik.solve()
            if self.curr_state.name == "state2":
                self.skeletons[3].set_positions(ik_res)

        if self.curr_state.name == "state3":
            q = skel.q
            pelvis = skel.dof_indices(["j_pelvis_rot_x", "j_pelvis_rot_y", "j_pelvis_rot_z"])
            foot_x = skel.dof_indices(["j_heel_left_x", "j_heel_right_x"])
            foot_y = skel.dof_indices(["j_heel_left_y", "j_heel_right_y"])
            thigh_x = skel.dof_indices(["j_thigh_left_x", "j_thigh_right_x"])
            thigh_y = skel.dof_indices(["j_thigh_left_y", "j_thigh_right_y"])

            ti = self.time() - self.elapsedTime
            small_ti = ti * 0.1
            # if ti <= 1.57:
                # self.curr_state.angles[foot_y] = ti, ti
                # self.curr_state.angles[thigh_y] = ti, ti
                # self.curr_state.angles[pelvis] = 0.0, ti, 0.0
            if ti <= 1.57:
                print("time: ", round(ti, 2))
                # self.curr_state.angles[foot_x] = -ti, -ti
                # self.curr_state.angles[thigh_x] = -ti, -ti
                self.curr_state.angles[foot_y] = 0.2 + small_ti, small_ti
                # self.curr_state.angles[foot_y] = 0.2 + small_ti, 0.2 + small_ti
                self.curr_state.angles[foot_x] = -0.2 -small_ti, -0.2-small_ti
        #========================
        # PD control
        # ========================
        # gain_value = 50.0
        gain_value = 100.0

        ndofs = skel.num_dofs()
        h = self.time_step()

        Kp = np.diagflat([0.0] * 6 + [gain_value] * (ndofs - 6))
        Kd = np.diagflat([0.0] * 6 + [2. * (gain_value ** .5)] * (ndofs - 6))
        invM = np.linalg.inv(skel.M + Kd * h)
        # p = -Kp.dot(skel.q - target_angle + skel.dq * h)
        # p = -Kp.dot(skel.q - self.curr_state.angles + skel.dq * h)
        if ik_on:
            p = -Kp.dot(skel.position_differences(skel.q, ik_res) + skel.dq * h)
        else:
            p = -Kp.dot(skel.position_differences(skel.q, self.curr_state.angles) + skel.dq * h)
        d = -Kd.dot(skel.dq)
        qddot = invM.dot(-skel.c + p + d + skel.constraint_forces())
        # tau = p + d - Kd.dot(qddot) * h
        # des_accel = p + d + qddot
        des_accel = qddot

        ddc = np.zeros(6)

        # HP QP solve
        lf_tangent_vec = np.array([1.0, 0.0, .0])
        rf_tangent_vec = np.array([1.0, 0.0, .0])

        # character_dir = copy.deepcopy(skel.com_velocity())
        character_dir = skel.com_velocity()
        # print(character_dir, skel.com_velocity())
        character_dir[1] = 0
        if np.linalg.norm(character_dir) != 0:
            character_dir = character_dir / np.linalg.norm(character_dir)

        empty_list = []
        _ddq, _tau, _bodyIDs, _contactPositions, _contactPositionLocals, _contactForces, _is_contact_list = hqp.calc_QP(
            skel, des_accel, self.force, ddc, lf_tangent_vec, rf_tangent_vec, empty_list, 1./self.time_step())

        del self.contact_force[:]
        del self.bodyIDs[:]
        del self.contactPositionLocals[:]

        self.bodyIDs = _bodyIDs

        for i in range(len(_bodyIDs)):
            skel.body(_bodyIDs[i]).add_ext_force(_contactForces[i], _contactPositionLocals[i])
            self.contact_force.append(_contactForces[i])
            self.contactPositionLocals.append(_contactPositionLocals[i])
        # dartModel.applyPenaltyForce(_bodyIDs, _contactPositionLocals, _contactForces)

        #Jacobian transpose control

        # jaco_r = skel.body("h_blade_right").linear_jacobian()
        # jaco_l = skel.body("h_blade_left").linear_jacobian()
        # jaco_hip_r = skel.body("h_thigh_right").linear_jacobian()
        jaco_pelvis = skel.body("h_pelvis").linear_jacobian()

        # if self.curr_state.name == "state011":
        #     force_r = 10. * np.array([-1.0, -8., 1.0])
        #     force_l = 10. * np.array([1.0, -.0, -1.0])
        #     t_r = self.add_JTC_force(jaco_r, force_r)
        #     t_l = self.add_JTC_force(jaco_l, force_l)
        #     _tau += t_r + t_l

        # if self.curr_state.name == "state1":
        #     force_r = 30. * np.array([-0., 0., 1.0])
        #     force_l = 70. * np.array([1.0, 0., -.0])
        #     t_r = self.add_JTC_force(jaco_r, force_r)
        #     t_l = self.add_JTC_force(jaco_l, force_l)
        #     _tau += t_r + t_l
        #

        # print("centripetal_force_dir: ", centripetal_force_dir)

        # if self.curr_state.name == "state2":
        #     # centripetal_force = math.tan(10*math.pi/180) * skel.m * 9.8 * np.array([0.0, 0., -1.0])
        #     centripetal_force = math.tan(10 * math.pi / 180) * skel.m * 9.8 * centripetal_force_dir
        #     tau = self.add_JTC_force(jaco_pelvis, centripetal_force)
        #     _tau += tau
        #
        # if self.curr_state.name == "state3":
        #     # centripetal_force = math.tan(10*math.pi/180) * skel.m * 9.8 * np.array([0.0, 0., 1.0])
        #     centripetal_force = math.tan(10 * math.pi / 180) * skel.m * 9.8 * centripetal_force_dir
        #     tau = self.add_JTC_force(jaco_pelvis, centripetal_force)
        #     _tau += tau
        #
        # if self.curr_state.name == "state03":
        #     force_r = 10. * np.array([-1.0, -8., 1.0])
        #     force_l = 10. * np.array([1.0, -8.0, -5.0])
        #     t_r = self.add_JTC_force(jaco_r, force_r)
        #     t_l = self.add_JTC_force(jaco_l, force_l)
        #     _tau += t_r + t_l

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
        contact_force = self.contact_force
        del render_vector[:]
        del render_vector_origin[:]
        del push_force[:]
        del push_force_origin[:]
        del blade_force[:]
        del blade_force_origin[:]
        del rd_footCenter[:]

        com = self.skeletons[2].C
        com[1] = -0.99 +0.05

        # com = self.skeletons[2].body('h_blade_left').to_world(np.array([0.1040 + 0.0216, +0.80354016 - 0.85354016, -0.054]))

        # if self.curr_state.name == "state2":
        #     com = self.ik.target_foot_left1

        rd_footCenter.append(com)

        # if self.curr_state.name == "state3":
        #     blade_force.append(np.array([1.0, -1.0, 1.0]))
        #     blade_force_origin.append(self.skeletons[2].body('h_heel_right').to_world())

        # if self.curr_state.name == "state1" or self.curr_state.name == "state11" :
        #     blade_force.append(np.array([1.0, 0.0, 0.0]))
        #     blade_force_origin.append(self.skeletons[2].body('h_heel_left').to_world())
        #     # blade_force.append(-self.controller.blade_direction_L)
        #     # blade_force_origin.append(self.skeletons[2].body('h_heel_right').to_world())
        # if self.curr_state.name == "state12":
        #     blade_force.append(np.array([-0.7, 1.0, 0.0]))
        #     blade_force_origin.append(self.skeletons[2].body('h_heel_right').to_world())
        # if self.curr_state.name == "state011":
        #     blade_force.append(np.array([1.0, -7., 1.0]))
        #     blade_force_origin.append(self.skeletons[2].body('h_heel_right').to_world())
        #
        # if self.curr_state.name == "state2":
        #     blade_force.append(np.array([-0., -7.0, -1.0]))
        #     blade_force_origin.append(self.skeletons[2].body('h_heel_left').to_world())

        # if self.curr_state.name == "state2":
        #     # blade_force.append(np.array([-1.0, 0., 1.0]))
        #     # blade_force_origin.append(self.skeletons[2].body('h_heel_right').to_world())
        #     blade_force.append(np.array([1., .0, -1.0]))
        #     blade_force_origin.append(self.skeletons[2].body('h_heel_left').to_world())
        if self.force is not None:
            push_force.append(self.force*0.05)
            push_force_origin.append(self.skeletons[2].body('h_pelvis').to_world())
        if len(self.bodyIDs) != 0:
            print(len(self.bodyIDs))
            for ii in range(len(self.contact_force)):
                if self.bodyIDs[ii] == 4:
                    body = self.skeletons[2].body('h_blade_left')
                else:
                    body = self.skeletons[2].body('h_blade_right')

                render_vector.append(contact_force[ii] / 100.)
                render_vector_origin.append(body.to_world(self.contactPositionLocals[ii]))
            #             render_vector_origin.append(body.to_world(contact_offset))
        # if len(contact_force) != 0:
        #     # print(len(contact_force), len(self.controller.contact_list))
        #     # print("contact_force.size?", contact_force.size, len(contact_force))
        #     # ri.set_color(1.0, 0.0, 0.0)
        #     for ii in range(len(contact_force)):
        #         if 2 * len(contact_force) == len(self.controller.contact_list):
        #             body = self.skeletons[2].body(self.controller.contact_list[2*ii])
        #             contact_offset = self.controller.contact_list[2*ii+1]
        #             # print("contact force : ", contact_force[ii])
        #             # ri.render_line(body.to_world(contact_offset), contact_force[ii]/100.)
        #             render_vector.append(contact_force[ii]/100.)
        #             render_vector_origin.append(body.to_world(contact_offset))


if __name__ == '__main__':
    print('Example: Skating -- pushing side to side')

    pydart.init()
    print('pydart initialization OK')

    world = MyWorld()
    print('MyWorld  OK')
    ground = pydart.World(1. / 1000., './data/skel/ground.skel')

    skel = world.skeletons[2]
    q = skel.q

    # q["j_thigh_left_x", "j_thigh_left_y", "j_thigh_left_z"] = 0., 0., 0.4
    # q["j_thigh_right_x", "j_thigh_right_y", "j_thigh_right_z"] = 0., 0., 0.
    # q["j_shin_left_z", "j_shin_right_z"] = -1.1, -0.05
    # q["j_heel_left_z", "j_heel_right_z"] = 0.2, 0.2

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
    viewer.motionViewWnd.glWindow.pOnPlaneshadow = (0., -0.99+0.0251, 0.)

    viewer.doc.addRenderer('bladeForce', yr.WideArrowRenderer(blade_force, blade_force_origin, (0, 0, 255)))
    viewer.motionViewWnd.glWindow.planeHeight = -0.98 + 0.0251
    def simulateCallback(frame):
        for i in range(10):
            world.step()
        world.render_with_ys()

    viewer.setSimulateCallback(simulateCallback)
    viewer.show()

    Fl.run()
