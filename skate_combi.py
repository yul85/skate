import numpy as np
import pydart2 as pydart
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
        pydart.World.__init__(self, 1.0 / 1000.0, './data/skel/cart_pole_blade_3dof.skel')
        # pydart.World.__init__(self, 1.0 / 1000.0, './data/skel/cart_pole_blade.skel')
        # pydart.World.__init__(self, 1.0 / 2000.0, './data/skel/cart_pole.skel')
        self.force = None
        self.force_r = None
        self.force_l = None
        self.force_hip_r = None
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
        arms = skel.dof_indices(["j_bicep_left_x", "j_bicep_right_x"])
        foot = skel.dof_indices(["j_heel_left_x", "j_heel_left_y", "j_heel_left_z", "j_heel_right_x", "j_heel_right_y", "j_heel_right_z"])
        leg_y = skel.dof_indices(["j_thigh_left_y", "j_thigh_right_y"])
        # blade = skel.dof_indices(["j_heel_right_2"])


        # ===========pushing side to side new===========

        s00q = np.zeros(skel.ndofs)
        # s00q[pelvis] = 0., -0.
        # s00q[upper_body] = 0.0, 0.0, -0.5
        # s00q[right_leg] = -0., -0., -0.0, -0.0
        # s00q[left_leg] = 0., 0., 0.0, -0.0
        s00q[leg_y] = 0.785, -0.785
        s00q[arms] = 1.5, -1.5
        # s00q[foot] = -0., 0.785, 0., 0., -0.785, 0.
        state00 = State("state00", 0.3, 0.0, 0.2, s00q)

        s01q = np.zeros(skel.ndofs)
        # s01q[upper_body] = 0., 0., -0.2
        # s01q[spine] = 0.0, 0., 0.5
        s01q[left_leg] = 0., 0., 0.3, -0.5
        s01q[right_leg] = -0.1, -0., 0.3, -0.5
        s01q[leg_y] = 0.785, -0.785
        # s01q[knee] = 0.2, 0.
        s01q[arms] = 1.5, -1.5
        # # s01q[blade] = -0.3
        # s01q[foot] = -0., 0.785, 0.2, 0., -0.785, 0.2
        s01q[foot] = -0., 0., 0.2, -0., -0., 0.2
        state01 = State("state01", 0.5, 2.2, 0.0, s01q)

        s02q = np.zeros(skel.ndofs)
        # s02q[pelvis] = 0., -0.3
        s02q[upper_body] = 0., 0., -0.5
        # s02q[spine] = 0.5, 0.5, 0.
        s02q[left_leg] = 0., 0., 0.3, -0.5
        s02q[right_leg] = -0., -0., 0.1, -0.5
        # s02q[right_leg] = -0.4, -0., 0.3, -0.7
        s02q[leg_y] = 0.785, -0.785
        s02q[arms] = 1.5, -1.5
        # s02q[knee] = 0., -0.2
        # s02q[foot] = -0., 0.785, 0.2, 0., -0.785, 0.2
        s02q[foot] = 0., 0., 0.2, -0., -0., 0.3
        state02 = State("state02", 1.0, 2.2, 0.0, s02q)

        s03q = np.zeros(skel.ndofs)
        # s03q[pelvis] = 0., -0.1
        s03q[upper_body] = 0., 0., -0.5
        # s03q[spine] = 0.5, 0.5, 0.
        s03q[left_leg] = 0., 0., 0.3, -0.5
        s03q[right_leg] = -0., -0., 0.1, -0.5
        s03q[leg_y] = 0.785, -0.785
        # s03q[knee] = 0.2, 0.
        s03q[arms] = 1.5, -1.5
        s03q[foot] = -0., 0., 0.2, 0., -0., 0.4
        state03 = State("state03", 2.0, 2.2, 0.0, s03q)

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


        #=====================LEFT TURN=============

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
        s11q[knee] = -roro_angle * math.pi/180., -roro_angle * math.pi/180.
        # s11q[arms] = 1.5, -1.5
        # s11q[arms] = -0.5, -0.5
        s11q[arms] = -roro_angle * math.pi/180., -roro_angle * math.pi/180.
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


        self.state_list = [state00, state01, state02, state03, state04, state05, state11]

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

        self.contact_force = []
        self.contactPositionLocals = []
        self.bodyIDs = []
        # print("dof: ", skel.ndofs)

        self.last_vec = np.zeros(3)

    def step(self):
        # print("self.curr_state: ", self.curr_state.name)

        # apply external force

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
        # ground_height = -0.98 + 0.05
        # ground_height = -0.98
        ground_height = 0.05
        if ik_on:
            ik_res = copy.deepcopy(self.curr_state.angles)
            if self.curr_state.name == "state2":
                # print("ik solve!!-------------------")
                self.ik.update_target(ground_height)
                ik_res[6:] = self.ik.solve()

            if self.curr_state.name == "state2":
                self.skeletons[3].set_positions(ik_res)

        gain_value = 50.0
        # gain_value = 300.0

        ndofs = skel.num_dofs()
        h = self.time_step()

        Kp = np.diagflat([0.0] * 6 + [gain_value] * (ndofs - 6))
        Kd = np.diagflat([0.0] * 6 + [2. * (gain_value ** .5)] * (ndofs - 6))
        invM = np.linalg.inv(skel.M + Kd * h)
        # p = -Kp.dot(skel.q - target_angle + skel.dq * h)
        # p = -Kp.dot(skel.q - self.curr_state.angles + skel.dq * h)
        if ik_on:
            p = -Kp.dot(skel.q - ik_res + skel.dq * h)
        else:
            p = -Kp.dot(skel.q - self.curr_state.angles + skel.dq * h)
        d = -Kd.dot(skel.dq)
        qddot = invM.dot(-skel.c + p + d + skel.constraint_forces())
        # tau = p + d - Kd.dot(qddot) * h
        # des_accel = p + d + qddot
        des_accel = qddot

        ddc = np.zeros(6)

        # HP QP solve
        lf_tangent_vec = np.array([1.0, 0.0, .0])
        rf_tangent_vec = np.array([1.0, 0.0, .0])
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

        jaco_r = skel.body("h_blade_right").linear_jacobian()
        jaco_l = skel.body("h_blade_left").linear_jacobian()
        jaco_hip_r = skel.body("h_thigh_right").linear_jacobian()
        # jaco_pelvis = skel.body("h_pelvis").linear_jacobian()


        if self.curr_state.name == "state01":
            self.force_r = 0. * np.array([1.0, -2., -1.])
            self.force_l = 0. * np.array([1.0, -0.5, -1.0])
            t_r = self.add_JTC_force(jaco_r, self.force_r)
            t_l = self.add_JTC_force(jaco_l, self.force_l)
            _tau += t_r + t_l

        if self.curr_state.name == "state02":
            self.force_r = 10. * np.array([-1.0, -.0, 1.])
            self.force_l = 0. * np.array([1.0, -0., -1.])
            t_r = self.add_JTC_force(jaco_r, self.force_r)
            t_l = self.add_JTC_force(jaco_l, self.force_l)
            _tau += t_r + t_l

        if self.curr_state.name == "state03":
            self.force_r = 20. * np.array([-1.0, -.0, 1.])
            self.force_l = 0. * np.array([1.0, -0., -1.])
            t_r = self.add_JTC_force(jaco_r, self.force_r)
            t_l = self.add_JTC_force(jaco_l, self.force_l)
            _tau += t_r + t_l

        if self.curr_state.name == "state04":
            self.force_r = 0. * np.array([-1.0, 0., 1.])
            self.force_l = 10. * np.array([1.0, 0., -1.0])
            t_r = self.add_JTC_force(jaco_r, self.force_r)
            t_l = self.add_JTC_force(jaco_l, self.force_l)
            _tau += t_r + t_l

        # if self.curr_state.name == "state04":
        #     self.force_r = 0. * np.array([2.0, -0., -1.0])
        #     self.force_l = 0. * np.array([-1.0, -0., -1.0])
        #     t_r = self.add_JTC_force(jaco_r, self.force_r)
        #     t_l = self.add_JTC_force(jaco_l, self.force_l)
        #     _tau += t_r + t_l

        # if self.curr_state.name == "state05":
        #     self.force_r = 10. * np.array([1.0, -1., 1.0])
        #     t_r = self.add_JTC_force(jaco_r, self.force_r)
        #     self.force_hip_r = 3. * np.array([-.0, 0., -1.0])
        #     t_hip_r = self.add_JTC_force(jaco_hip_r, self.force_hip_r)
        #     _tau += t_r + t_hip_r

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

        if self.force_r is not None:
            blade_force.append(self.force_r*0.05)
            blade_force_origin.append(self.skeletons[2].body('h_heel_right').to_world())
        if self.force_l is not None:
            blade_force.append(self.force_l*0.05)
            blade_force_origin.append(self.skeletons[2].body('h_heel_left').to_world())

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

    q["j_heel_left_y", "j_heel_right_y"] = 0.785, -0.785

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
    viewer.doc.addRenderer('ground', yr.DartRenderer(ground, (255, 255, 255), yr.POLYGON_FILL),visible=False)
    viewer.doc.addRenderer('contactForce', yr.VectorsRenderer(render_vector, render_vector_origin, (255, 0, 0)))
    viewer.doc.addRenderer('pushForce', yr.WideArrowRenderer(push_force, push_force_origin, (0, 255,0)))
    viewer.doc.addRenderer('rd_footCenter', yr.PointsRenderer(rd_footCenter))

    viewer.startTimer(1/25.)
    viewer.motionViewWnd.glWindow.pOnPlaneshadow = (0., -0.99+0.0251, 0.)

    viewer.doc.addRenderer('bladeForce', yr.WideArrowRenderer(blade_force, blade_force_origin, (0, 0, 255)))
    # viewer.motionViewWnd.glWindow.planeHeight = -0.98 + 0.0251
    viewer.motionViewWnd.glWindow.planeHeight = 0.
    def simulateCallback(frame):
        for i in range(10):
            world.step()
        world.render_with_ys()

    viewer.setSimulateCallback(simulateCallback)
    viewer.show()

    Fl.run()
