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
from PyCommon.modules.Simulator import yulQpSimulator_stick as hqp

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
        pydart.World.__init__(self, 1.0 / 1000.0, './data/skel/stick_model.skel')

        self.force = None
        self.jtc_force = None

        self.duration = 0
        self.skeletons[0].body('ground').set_friction_coeff(0.02)

        skel = self.skeletons[1]
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

        pelvis = skel.dof_indices(["j_pelvis_rot_x", "j_pelvis_rot_y", "j_pelvis_rot_z"])
        ankle = skel.dof_indices(["j_heel_x", "j_heel_y", "j_heel_z"])


        # ===========pushing side to side new===========

        s00q = np.zeros(skel.ndofs)
        s00q[ankle] = 0.0, 0.0, 0.0
        state00 = State("state00", 0.5, 0.0, 0.2, s00q)

        s01q = np.zeros(skel.ndofs)
        s01q[ankle] = 0., 0.0, 0.0
        state01 = State("state01", 0.5, 0.0, 0.2, s01q)

        s02q = np.zeros(skel.ndofs)
        s02q[ankle] = 0., -0.3, -0.
        # s02q[ankle] = -0.2, -0.0, -0.
        # s02q[ankle] = -0.2, -0.2, -0.
        state02 = State("state02", 1.0, 0.0, 0.2, s02q)

        s03q = np.zeros(skel.ndofs)
        s03q[ankle] = -0., 0., -0.
        state03 = State("state02", 5.0, 0.0, 0.2, s03q)

        self.state_list = [state00, state01, state02, state03]

        state_num = len(self.state_list)
        self.state_num = state_num
        # print("state_num: ", state_num)

        self.curr_state = self.state_list[0]
        self.elapsedTime = 0.0
        self.curr_state_index = 0
        # print("backup angle: ", backup_q)
        # print("cur angle: ", self.curr_state.angles)

        self.controller = QPsolver.Controller(skel, self.skeletons[2], self.dt, self.curr_state.name)

        self.skeletons[2].set_positions(self.curr_state.angles)

        self.controller.target = self.curr_state.angles
        # self.controller.target = skel.q
        # skel.set_controller(self.controller)

        if ik_on:
            self.ik = IKsolve_double_stance.IKsolver(skel, self.skeletons[2], self.dt)
            print('IK ON')

        print('create controller OK')

        self.contact_force = []
        self.contactPositionLocals = []
        self.bodyIDs = []
        # print("dof: ", skel.ndofs)

        self.last_vec = np.zeros(3)

    def step(self):
        # print("self.curr_state: ", self.curr_state.name)

        # if self.curr_state.name == "state1":
        #     self.force = np.array([50.0, 0.0, 0.0])
        # # elif self.curr_state.name == "state2":
        # #     self.force = np.array([0.0, 0.0, -10.0])
        # else:
        #     self.force = None

        if self.curr_state.name =="state01":
            self.force = np.array([20.0, 0.0, 0.0])
            # self.force = np.array([0.0, 0.0, 50.0])
        else:
            self.force = None

        self.controller.cur_state = self.curr_state.name
        if self.force is not None:
            self.skeletons[1].body('h_pelvis').add_ext_force(self.force)

        self.skeletons[2].set_positions(self.curr_state.angles)

        if self.curr_state.dt < self.time() - self.elapsedTime:
            # print("change the state!!!", self.curr_state_index)
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
                self.skeletons[2].set_positions(ik_res)

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

        # character_dir = copy.deepcopy(skel.com_velocity())
        character_dir = skel.com_velocity()
        # print(character_dir, skel.com_velocity())
        character_dir[1] = 0
        if np.linalg.norm(character_dir) != 0:
            character_dir = character_dir / np.linalg.norm(character_dir)

        centripetal_force_dir = np.cross([0.0, 1.0, 0.0], character_dir)

        # if self.curr_state.name == "state2" or self.curr_state.name == "state02":
        #
        #     vec = np.array([1.0, 0.0, -0.1])
        #     # vec[0] = 1.0 -0.1
        #     # vec[2] = 0.0 +0.1
        #
        #     # ti = self.time() - self.elapsedTime
        #     #
        #     # # print("time: ", round(ti, 2))
        #     # vec = np.array([1.0, 0.0, -round(ti, 2)])
        #     #
        #     # vec_norm = np.linalg.norm(vec)
        #     # if vec_norm != 0:
        #     #     vec = vec / vec_norm
        #     #
        #     # lf_tangent_vec = vec
        #     # rf_tangent_vec = vec
        #
        #     # self.last_vec = vec
        #
        #     lf_tangent_vec = character_dir
        #     rf_tangent_vec = character_dir
        #
        # # if self.curr_state.name == "state02":
        # #     print("last_vec: ", self.last_vec)
        # #     lf_tangent_vec = self.last_vec
        # #     rf_tangent_vec = self.last_vec


        # if self.curr_state.name == "state1" or self.curr_state.name == "state1" or self.curr_state.name == "state2":
        #     p1 = skel.body("h_blade_left").to_world([-0.1040 + 0.0216, 0.0, 0.0])
        #     p2 = skel.body("h_blade_left").to_world([0.1040 + 0.0216, 0.0, 0.0])
        #
        #     blade_direction_vec = p2 - p1
        #     blade_direction_vec = np.array([1, 0, 1]) * blade_direction_vec
        #
        #     if np.linalg.norm(blade_direction_vec) != 0:
        #         blade_direction_vec = blade_direction_vec / np.linalg.norm(blade_direction_vec)
        #
        #     lf_tangent_vec = blade_direction_vec
        #
        #     p1 = skel.body("h_blade_right").to_world([-0.1040 + 0.0216, 0.0, 0.0])
        #     p2 = skel.body("h_blade_right").to_world([0.1040 + 0.0216, 0.0, 0.0])
        #
        #     blade_direction_vec = p2 - p1
        #     blade_direction_vec = np.array([1, 0, 1]) * blade_direction_vec
        #
        #     if np.linalg.norm(blade_direction_vec) != 0:
        #         blade_direction_vec = blade_direction_vec / np.linalg.norm(blade_direction_vec)
        #
        #     rf_tangent_vec = blade_direction_vec

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

        jaco_blade = skel.body("h_blade").linear_jacobian()

        if self.curr_state.name == "state01":
            self.jtc_force = 0. * np.array([1.0, -2., -1.])
            tq = self.add_JTC_force(jaco_blade, self.jtc_force)
            _tau += tq

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

        com = self.skeletons[1].C
        com[1] = -0.99 +0.05

        # com = self.skeletons[1].body('h_blade_left').to_world(np.array([0.1040 + 0.0216, +0.80354016 - 0.85354016, -0.054]))

        # if self.curr_state.name == "state2":
        #     com = self.ik.target_foot_left1

        rd_footCenter.append(com)

        if self.jtc_force is not None:
            blade_force.append(self.jtc_force*0.05)
            blade_force_origin.append(self.skeletons[1].body('h_blade').to_world())

        if self.force is not None:
            push_force.append(self.force*0.05)
            push_force_origin.append(self.skeletons[1].body('h_pelvis').to_world())
        if len(self.bodyIDs) != 0:
            print("contact num: ", len(self.bodyIDs))
            for ii in range(len(self.contact_force)):

                body = self.skeletons[1].body('h_blade')
                render_vector.append(contact_force[ii] / 100.)
                render_vector_origin.append(body.to_world(self.contactPositionLocals[ii]))

if __name__ == '__main__':
    print('Example: Skating -- pushing side to side')

    pydart.init()
    print('pydart initialization OK')

    world = MyWorld()
    print('MyWorld  OK')
    ground = pydart.World(1. / 1000., './data/skel/ground.skel')

    skel = world.skeletons[1]
    q = skel.q


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
    viewer.doc.addRenderer('rd_footCenter', yr.PointsRenderer(rd_footCenter, (0, 255, 0)))

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
