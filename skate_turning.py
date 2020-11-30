import numpy as np
import pydart2 as pydart
import QPsolver
from scipy import interpolate
import IKsolve_one
import momentum_con
import motionPlan
from scipy import optimize
import yulTrajectoryOpt

from fltk import *
from PyCommon.modules.GUI import hpSimpleViewer as hsv
from PyCommon.modules.Renderer import ysRenderer as yr
# from PyCommon.modules.Simulator import hpDartQpSimulator_turning as hqp
from PyCommon.modules.Simulator import yulQpSimulator_equality_blade_turning as hqp
# from PyCommon.modules.Simulator import yulQpSimulator_inequality_blade_turning as hqp
# from PyCommon.modules.Simulator import hpDartQpSimulator_turning_penalty as hqp

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
        self.duration = 0
        self.skeletons[0].body('ground').set_friction_coeff(0.02)

        plist = [(0, 0), (0.2, 0), (0.5, -0.15), (1.0, -0.25), (1.5, -0.15), (1.8, 0), (2.0, 0.0)]
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
        pelvis_x = skel.dof_indices((["j_pelvis_rot_x"]))
        pelvis = skel.dof_indices((["j_pelvis_rot_y", "j_pelvis_rot_z"]))
        upper_body = skel.dof_indices(["j_abdomen_x", "j_abdomen_y", "j_abdomen_z"])
        spine = skel.dof_indices(["j_spine_x", "j_spine_y", "j_spine_z"])
        right_leg = skel.dof_indices(["j_thigh_right_x", "j_thigh_right_y", "j_thigh_right_z", "j_shin_right_z"])
        left_leg = skel.dof_indices(["j_thigh_left_x", "j_thigh_left_y", "j_thigh_left_z", "j_shin_left_z"])
        knee = skel.dof_indices(["j_shin_left_x", "j_shin_right_x"])
        arms = skel.dof_indices(["j_bicep_left_x", "j_bicep_right_x"])
        arms_y = skel.dof_indices(["j_bicep_left_y", "j_bicep_right_y"])
        foot = skel.dof_indices(["j_heel_left_x", "j_heel_left_y", "j_heel_left_z", "j_heel_right_x", "j_heel_right_y", "j_heel_right_z"])
        leg_y = skel.dof_indices(["j_thigh_right_y", "j_thigh_left_y"])
        # blade = skel.dof_indices(["j_heel_right_2"])

        # #----------------------------------
        # # pushing side to side new (180718)
        # #----------------------------------
        s0q = np.zeros(skel.ndofs)
        # s0q[pelvis] = 0., -0.
        # s0q[upper_body] = 0.0, 0.0, -0.5
        # s0q[right_leg] = -0., -0., -0.0, -0.0
        # s0q[left_leg] = 0., 0., 0.0, -0.0
        # s0q[leg_y] = -0.785, 0.785
        s0q[arms] = 1.5, -1.5
        s0q[foot] = -0., 0.785, 0., 0., -0.785, 0.
        state0 = State("state0", 0.2, 0.0, 0.2, s0q)

        s001q = np.zeros(skel.ndofs)
        # s01q[pelvis] = 0., -0.3
        s001q[upper_body] = 0.0, 0., -0.5
        # s001q[spine] = 0.0, 0., 0.5
        s001q[left_leg] = -0., 0., 0., -0.5
        s001q[right_leg] = -0.0, -0., 0., -0.5
        s001q[arms] = 1.5, -1.5
        # # s01q[blade] = -0.3
        s001q[foot] = -0., 0.785, 0.2, 0., -0.785, 0.2
        state001 = State("state001", 0.5, 2.2, 0.0, s001q)

        s01q = np.zeros(skel.ndofs)
        # s01q[pelvis] = 0., -0.3
        s01q[upper_body] = 0.0, 0., -0.5
        # s01q[spine] = 0.0, 0., 0.5
        s01q[left_leg] = 0., 0., 0., -0.3
        s01q[right_leg] = -0., -0., 0., -0.3
        s01q[arms] = 1.5, -1.5
        # s01q[blade] = -0.3
        s01q[foot] = -0., 0.785, 0.1, 0., -0.785, 0.1
        state01 = State("state01", 0.5, 2.2, 0.0, s01q)

        s011q = np.zeros(skel.ndofs)
        # s1q[pelvis] = 0., -0.1
        s011q[upper_body] = 0.0, 0., -0.5
        # s1q[spine] = 0.0, 0., 0.5
        s011q[left_leg] = 0.2, -0., -0., -0.3
        s011q[right_leg] = -0.2, 0., -0., -0.3
        # s1q[knee] = 0.1, -0.1
        s011q[arms] = 1.5, -1.5
        # s1q[blade] = -0.3
        # s1q[foot] = -0.0, 0.4, 0.2, 0.0, -0.4, 0.2
        s011q[foot] = -0.0, -0., 0.3, 0.0, 0., 0.3
        state011 = State("state011", 0.3, 2.2, 0.0, s011q)

        s1q = np.zeros(skel.ndofs)
        # s1q[pelvis] = 0., -0.1
        s1q[upper_body] = 0.0, 0., -0.3
        # s1q[spine] = 0.0, 0., 0.5
        s1q[left_leg] = -0., -0., -0., -0.5
        s1q[right_leg] = 0., 0., -0., -0.5
        # s1q[knee] = 0.1, -0.1
        s1q[arms] = 1.5, -1.5
        # s1q[blade] = -0.3
        # s1q[foot] = -0.0, 0.4, 0.2, 0.0, -0.4, 0.2
        s1q[foot] = -0.0, -0.785, 0.3, 0.0, 0.785, 0.3
        state1 = State("state1", 0.5, 2.2, 0.0, s1q)

        s12q = np.zeros(skel.ndofs)
        s12q[upper_body] = 0.0, 0., -0.
        s12q[left_leg] = -0., -0., 0.1, -0.1
        s12q[right_leg] = 0., 0., 0.1, -0.1
        s12q[arms] = 1.5, -1.5
        s12q[foot] = -0.0, -0., 0.2, 0.0, 0., 0.2
        state12 = State("state12", 0.3, 2.2, 0.0, s12q)

        self.state_list = [state0, state001, state01, state011, state1,  state12, state01, state011, state1, state12]

        state_num = len(self.state_list)
        self.state_num = state_num
        # print("state_num: ", state_num)

        self.curr_state = self.state_list[0]
        self.elapsedTime = 0.0
        self.curr_state_index = 0
        # print("backup angle: ", backup_q)
        # print("cur angle: ", self.curr_state.angles)

        self.controller = QPsolver.Controller(skel, self.skeletons[3], self.dt, self.curr_state.name)

        self.mo_con = momentum_con.momentum_control(self.skeletons[2], self.skeletons[3], self.time_step())

        self.skeletons[3].set_positions(self.curr_state.angles)
        # self.skeletons[3].set_positions(np.zeros(skel.ndofs))

        # self.ik = IKsolve_one.IKsolver(self.skeletons[2], self.dt)

        # merged_target = self.curr_state.angles
        # self.ik.update_target(self.curr_state.name)
        # merged_target = np.zeros(skel.ndofs)
        # merged_target[:6] = self.curr_state.angles[:6]
        # merged_target[6:18] = self.ik.solve()
        # merged_target[18:] = self.curr_state.angles[18:]
        # print("ik res: ", self.ik.solve())
        # print("merged_target: ", merged_target)
        # self.controller.target = merged_target
        self.controller.target = self.curr_state.angles
        # self.controller.target = skel.q
        # skel.set_controller(self.controller)
        print('create controller OK')

        self.contact_force = []
        self.contactPositionLocals = []
        self.bodyIDs = []
        # print("dof: ", skel.ndofs)

        self.trajUpdateTime = 0
        self.tangent_index = 0

        self.lf_ = None
        self.rf_ = None

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
        # if self.curr_state.name == "state2" or self.curr_state.name == "state3":
        # if self.curr_state.name == "state1":
        # if self.time() > 1.0 and self.time() < 2.0:
        #     self.force = np.array([20.0, 0.0, 0.0])
        # else:
        #     self.force = None

        # print("left foot pos:",  self.skeletons[2].body('h_blade_left').to_world([0.0, 0.0, 0.0]))
        # self.force = np.array([20.0, 0.0, 0.0])
        # self.skeletons[2].body('h_pelvis').add_ext_force(self.force)

        # if self.curr_state.name == "state1":
        #     self.force = np.array([10.0, 0.0, 0.0])
        # else:
        #     self.force = None

        # if self.curr_state.name == "state01":
        #     self.force = 2. * np.array([10.0, 0.0, 0.0])
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
        #a = self.skeletons[2].get_positions()

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

        # print("Current state name: ", self.curr_state.name)

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
        # if self.curr_state.name == "state2":
        #     self.ik.update_target(self.curr_state.name)
        #     merged_target = np.zeros(skel.ndofs)
        #     merged_target[:6] = self.curr_state.angles[:6]
        #     merged_target[6:12] = self.ik.solve()
        #     merged_target[12:] = self.curr_state.angles[12:]
        #     # print("ik res: ", self.ik.solve())
        #     # print("merged_target: ", merged_target)
        #     # self.controller.target = merged_target
        #     self.controller.target = self.curr_state.angles
        # else:
        #     # self.controller.target = self.curr_state.angles
        #     self.ik.update_target(self.curr_state.name)
        #     merged_target = np.zeros(skel.ndofs)
        #     merged_target[:6] = self.curr_state.angles[:6]
        #     merged_target[6:18] = self.ik.solve()
        #     merged_target[18:] = self.curr_state.angles[18:]
        #     # print("ik res: ", self.ik.solve())
        #     # print("merged_target: ", merged_target)
        #     # self.controller.target = merged_target
        #     self.controller.target = self.curr_state.angles

        self.controller.target = self.curr_state.angles

        # self.controller.target = self.curr_state.angles
        # print(self.curr_state.angles)

        contact_list = self.mo_con.check_contact()

        # gain_value = 25.0
        gain_value = 50.0

        # if self.mo_con.contact_num == 0:
        #     ndofs = skel.num_dofs()
        #     h = self.time_step()
        #     Kp = np.diagflat([0.0] * 6 + [gain_value] * (ndofs - 6))
        #     Kd = np.diagflat([0.0] * 6 + [2.*(gain_value**.5)] * (ndofs - 6))
        #     invM = np.linalg.inv(skel.M + Kd * h)
        #     p = -Kp.dot(skel.q - self.curr_state.angles + skel.dq * h)
        #     d = -Kd.dot(skel.dq)
        #     qddot = invM.dot(-skel.c + p + d + skel.constraint_forces())
        #     des_accel = p + d + qddot
        # else:
        #     # print("contact num: ", self.mo_con.contact_num )
        #     self.mo_con.target = self.curr_state.angles
        #     des_accel = self.mo_con.compute(contact_list)

        ndofs = skel.num_dofs()
        h = self.time_step()

        Kp = np.diagflat([0.0] * 6 + [gain_value] * (ndofs - 6))
        Kd = np.diagflat([0.0] * 6 + [2. * (gain_value ** .5)] * (ndofs - 6))
        invM = np.linalg.inv(skel.M + Kd * h)
        p = -Kp.dot(skel.q - self.curr_state.angles + skel.dq * h)
        d = -Kd.dot(skel.dq)
        qddot = invM.dot(-skel.c + p + d + skel.constraint_forces())
        des_accel = p + d + qddot

        ddc = np.zeros(6)
        # if self.curr_state.name == "state3":
        #     # print("com control : state3!!", skel.body('h_blade_left').to_world([0., 0.98, 0.]), skel.com())
        #     # ddc[0:3] = 400. * (skel.body('h_blade_left').to_world([0., 0.98, 0.]) - skel.com()) - 10. * skel.dC
        #     ddc[0:3] = 400. * (np.array([0.52, 0., -0.09]) - skel.com()) - 10. * skel.dC

        # print(skel.body('h_blade_left').to_world([0., 0, 0.]), skel.com())
        # HP QP solve
        # lf_tangent_vec = np.array([0.0, 0.0, -1.0])
        # rf_tangent_vec = np.array([0.0, 0.0, 1.0])
        lf_tangent_vec = np.array([1.0, 0.0, .0])
        rf_tangent_vec = np.array([1.0, 0.0, .0])

        # lf_tangent_vec_normal = np.array([0.0, 0.0, -1.0])
        # rf_tangent_vec_normal = np.array([0.0, 0.0, 1.0])

        # calculate tangent vector

        # if self.curr_state.name == "state1" or self.curr_state.name == "state2":
        # print("time: ", self.time())
        # if self.time() >= 0.2:
        #     if 0.005 > self.time() - self.trajUpdateTime:
        #         # print("in loop", self.tangent_index)
        #         lf_tangent_vec = np.asarray([self.left_der[0][self.tangent_index], 0.0, self.left_der[1][self.tangent_index]])
        #
        #         lf_tangent_vec_normal = np.cross(np.array([0.0, -1.0, 0.0]), lf_tangent_vec)
        #         if np.linalg.norm(lf_tangent_vec) != 0:
        #             lf_tangent_vec = lf_tangent_vec / np.linalg.norm(lf_tangent_vec)
        #
        #         if np.linalg.norm(lf_tangent_vec_normal) != 0:
        #             lf_tangent_vec_normal = lf_tangent_vec_normal / np.linalg.norm(lf_tangent_vec_normal)
        #         rf_tangent_vec = np.asarray([self.right_der[0][self.tangent_index], 0.0, self.right_der[1][self.tangent_index]])
        #         if np.linalg.norm(rf_tangent_vec) != 0:
        #             rf_tangent_vec = rf_tangent_vec / np.linalg.norm(rf_tangent_vec)
        #
        #         rf_tangent_vec_normal = np.cross(rf_tangent_vec, np.array([0.0, -1.0, 0.0]))
        #         if np.linalg.norm(rf_tangent_vec_normal) != 0:
        #             rf_tangent_vec_normal = rf_tangent_vec_normal / np.linalg.norm(rf_tangent_vec_normal)
        #
        #         # print("left foot traj: ", lf_tangent_vec)
        #         # print("right_foot_traj: ", rf_tangent_vec)
        #
        #     else:
        #         if self.tangent_index < len(self.left_foot_traj[0])-1:
        #             self.tangent_index += 1
        #         else:
        #             self.tangent_index = 0
        #         self.trajUpdateTime = self.time()
            # print("left foot traj: ", lf_tangent_vec)
            # print("right_foot_traj: ", rf_tangent_vec)

        # if self.time() > 0.2:
        #     lf_tangent_vec = np.array([1.0, 0.0, -1.0])
        #     rf_tangent_vec = np.array([1.0, 0.0, 1.0])
        #     lf_tangent_vec = lf_tangent_vec / np.linalg.norm(lf_tangent_vec)
        #     rf_tangent_vec = rf_tangent_vec / np.linalg.norm(rf_tangent_vec)

        vel_cone_list = np.zeros(8)

        if self.curr_state.name == "state01":
            lf_tangent_vec = np.array([1.0, 0.0, -0.5])
            rf_tangent_vec = np.array([1.0, 0.0, 0.5])
            lf_tangent_vec = lf_tangent_vec / np.linalg.norm(lf_tangent_vec)
            rf_tangent_vec = rf_tangent_vec / np.linalg.norm(rf_tangent_vec)
            vel_cone_list[0] = 1
            vel_cone_list[1] = 1
            vel_cone_list[4] = 1
            vel_cone_list[5] = 1
        if self.curr_state.name == "state1":
            lf_tangent_vec = np.array([1.0, 0.0, 0.5])
            rf_tangent_vec = np.array([1.0, 0.0, -0.5])
            lf_tangent_vec = lf_tangent_vec / np.linalg.norm(lf_tangent_vec)
            rf_tangent_vec = rf_tangent_vec / np.linalg.norm(rf_tangent_vec)
            vel_cone_list[2] = 1
            vel_cone_list[3] = 1
            vel_cone_list[6] = 1
            vel_cone_list[7] = 1

        self.lf_ = lf_tangent_vec
        self.rf_ = rf_tangent_vec

        _ddq, _tau, _bodyIDs, _contactPositions, _contactPositionLocals, _contactForces = hqp.calc_QP(skel, des_accel, ddc, lf_tangent_vec, rf_tangent_vec, vel_cone_list, 1./self.time_step())

        # print("lf_normal: ", lf_tangent_vec_normal)
        # print("rf_normal: ", rf_tangent_vec_normal)
        # _ddq, _tau, _bodyIDs, _contactPositions, _contactPositionLocals, _contactForces = hqp.calc_QP(
        #     skel, des_accel, ddc, lf_tangent_vec_normal, rf_tangent_vec_normal, 1. / self.time_step())

        offset_list = [[-0.1040 + 0.0216, +0.80354016 - 0.85354016, 0.0],
                       [0.1040+0.0216, +0.80354016-0.85354016, 0.0]]

        lp1 = skel.body("h_blade_left").to_world(offset_list[0])
        lp2 = skel.body("h_blade_left").to_world(offset_list[1])
        rp1 = skel.body("h_blade_right").to_world(offset_list[0])
        rp2 = skel.body("h_blade_right").to_world(offset_list[1])

        debug = False
        if debug:
            if (lp1[1] - lp2[1]) > 0.001:
                print("l: ", lp1[1] - lp2[1], lp1, lp2)
            if (rp1[1] - rp2[1]) > 0.001:
                print("r: ", rp1[1] - rp2[1], rp1, rp2)

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
        # todo : make function

        if self.curr_state.name == "state001":
            l_jaco = skel.body("h_blade_left").linear_jacobian()
            l_jaco_t = l_jaco.transpose()
            l_force = 1. * np.array([10.0, 0., 0.])
            l_tau = np.dot(l_jaco_t, l_force)

            r_jaco = skel.body("h_blade_right").linear_jacobian()
            r_jaco_t = r_jaco.transpose()
            r_force = 1. * np.array([10.0, 0., 0.])
            r_tau = np.dot(r_jaco_t, r_force)

            _tau += l_tau + r_tau

        # if self.curr_state.name == "state01":
        #     l_jaco = skel.body("h_blade_left").linear_jacobian()
        #     l_jaco_t = l_jaco.transpose()
        #     l_force = 3. * np.array([10.0, 0., -10.])
        #     l_tau = np.dot(l_jaco_t, l_force)
        #
        #     r_jaco = skel.body("h_blade_right").linear_jacobian()
        #     r_jaco_t = r_jaco.transpose()
        #     r_force = 3. * np.array([10.0, 0., 10.])
        #     r_tau = np.dot(r_jaco_t, r_force)
        #
        #     _tau += l_tau + r_tau
        #
        # if self.curr_state.name == "state1":
        #     l_jaco = skel.body("h_blade_left").linear_jacobian()
        #     l_jaco_t = l_jaco.transpose()
        #     l_force = 3. * np.array([10.0, 0., 10.])
        #     l_tau = np.dot(l_jaco_t, l_force)
        #
        #     r_jaco = skel.body("h_blade_right").linear_jacobian()
        #     r_jaco_t = r_jaco.transpose()
        #     r_force = 3. * np.array([10.0, 0., -10.])
        #     r_tau = np.dot(r_jaco_t, r_force)
        #
        #     _tau += l_tau + r_tau

        skel.set_forces(_tau)

        '''
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
        '''

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

        # ground rendering
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
        del rd_footCenter[:]

        del lf_trajectory[:]
        del rf_trajectory[:]

        del lft[:]
        del lft_origin[:]
        del rft[:]
        del rft_origin[:]

        com = self.skeletons[2].C
        com[1] = -0.99 +0.05

        # com = self.skeletons[2].body('h_blade_left').to_world(np.array([0.1040 + 0.0216, +0.80354016 - 0.85354016, -0.054]))
        rd_footCenter.append(com)

        # lft.append(np.array([self.left_der[0][self.tangent_index + 1], -0., self.left_der[1][self.tangent_index + 1]]))
        # lft_origin.append(np.array([self.left_foot_traj[0][self.tangent_index], -0.9, self.left_foot_traj[1][self.tangent_index]]))
        # rft.append(np.array([self.right_der[0][self.tangent_index + 1], -0., self.right_der[1][self.tangent_index + 1]]))
        # rft_origin.append(np.array([self.right_foot_traj[0][self.tangent_index], -0.9, self.right_foot_traj[1][self.tangent_index]]))

        lft.append(self.lf_)
        lft_origin.append(np.array([0., -0.9, 0.]))
        rft.append(self.rf_)
        rft_origin.append(np.array([0., -0.9, 0.]))

        for ctr_n in range(0, len(self.left_foot_traj[0])-1, 10):
            lf_trajectory.append(np.array([self.left_foot_traj[0][ctr_n], -0.9, self.left_foot_traj[1][ctr_n]]))
            rf_trajectory.append(np.array([self.right_foot_traj[0][ctr_n], -0.9, self.right_foot_traj[1][ctr_n]]))
            # lft.append(np.array([self.left_der[0][ctr_n+1], -0., self.left_der[1][ctr_n+1]]))
            # lft_origin.append(np.array([self.left_foot_traj[0][ctr_n], -0.9, self.left_foot_traj[1][ctr_n]]))
            # rft.append(np.array([self.right_foot_traj[0][ctr_n + 1], -0., self.right_foot_traj[1][ctr_n + 1]]))
            # rft_origin.append(np.array([self.right_foot_traj[0][ctr_n], -0.9, self.right_foot_traj[1][ctr_n]]))

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

    # q["j_abdomen_1"] = -0.2
    # q["j_abdomen_2"] = -0.2

    # q["j_thigh_right_x", "j_thigh_right_y", "j_thigh_right_z", "j_shin_right_z"] = -0., -0., 0.9, -1.5
    # q["j_heel_left_1", "j_heel_left_2", "j_heel_right_1", "j_heel_right_2"] = 0., 0.1, 0., 0.

    # q["j_thigh_right_y", "j_thigh_left_y"] = -0.785, 0.785
    # q["j_shin_right", "j_shin_left"] = 0., 0.
    # q["j_thigh_right_x", "j_thigh_right_y", "j_thigh_right_z"] = -0.1, -0.5, 0.2
    # q["j_thigh_left_x", "j_thigh_left_y"] = 0.2, 0.5
    # q["j_thigh_left_z", "j_shin_left"] = 0.2, -0.2
    # q["j_thigh_right_z", "j_shin_right"] = 0.2, -0.2
    # q["j_heel_left_1"] = 0.2
    # q["j_heel_right_1"] = 0.2
    #
    # q["j_heel_left_y", "j_heel_right_y"] = 0.4, -0.4
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
    viewer.doc.addRenderer('ground', yr.DartRenderer(ground, (255, 255, 255), yr.POLYGON_FILL))
    viewer.doc.addRenderer('contactForce', yr.VectorsRenderer(render_vector, render_vector_origin, (255, 0, 0)))
    viewer.doc.addRenderer('pushForce', yr.WideArrowRenderer(push_force, push_force_origin, (0, 255,0)))
    viewer.doc.addRenderer('rd_footCenter', yr.PointsRenderer(rd_footCenter))
    viewer.doc.addRenderer('lf_traj', yr.PointsRenderer(lf_trajectory, (200, 200, 200)), visible=False)
    viewer.doc.addRenderer('rf_traj', yr.PointsRenderer(rf_trajectory, (200, 200, 200)), visible=False)
    viewer.doc.addRenderer('lf_tangent', yr.VectorsRenderer(lft, lft_origin, (0, 255, 0)))
    viewer.doc.addRenderer('rf_tangent', yr.VectorsRenderer(rft, rft_origin, (0, 0, 255)))

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
