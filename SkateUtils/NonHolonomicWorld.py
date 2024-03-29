import pydart2 as pydart
import numpy as np
import math
from PyCommon.modules.Math import mmMath as mm


class NHWorld(pydart.World):
    def __init__(self, step, skel_path=None, xmlstr=False):
        self.has_nh = False

        super(NHWorld, self).__init__(step, skel_path, xmlstr)

        # nonholonomic constraint initial setting
        _th = 5. * math.pi / 180.
        skel = self.skeletons[1]
        self.skeletons[0].body(0).set_friction_coeff(0.02)

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

        self.has_nh = True

        self.ground_height = 0.

    def reset(self):
        super(NHWorld, self).reset()
        self.skeletons[0].body(0).set_friction_coeff(0.02)
        if self.has_nh:
            self.nh0.add_to_world()
            self.nh1.add_to_world()
            self.nh2.add_to_world()
            self.nh3.add_to_world()

    def step(self):
        # nonholonomic constraint
        right_blade_front_point = self.skeletons[1].body("h_blade_right").to_world((0.0216+0.104, -0.0216-0.027, 0.))
        right_blade_rear_point = self.skeletons[1].body("h_blade_right").to_world((0.0216-0.104, -0.0216-0.027, 0.))
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

        left_blade_front_point = self.skeletons[1].body("h_blade_left").to_world((0.0216+0.104, -0.0216-0.027, 0.))
        left_blade_rear_point = self.skeletons[1].body("h_blade_left").to_world((0.0216-0.104, -0.0216-0.027, 0.))
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

        super(NHWorld, self).step()


class NHWorldV2(pydart.World):
    def __init__(self, step, skel_path=None, xmlstr=False):
        self.has_nh = False

        super(NHWorldV2, self).__init__(step, skel_path, xmlstr)

        # nonholonomic constraint initial setting
        skel = self.skeletons[1]
        self.skeletons[0].body(0).set_friction_coeff(0.02)

        self.nh0 = pydart.constraints.NonHolonomicContactConstraint(skel.body('h_blade_right'), np.array((0.0216+0.104, -0.0216-0.027, 0.)))
        self.nh1 = pydart.constraints.NonHolonomicContactConstraint(skel.body('h_blade_right'), np.array((0.0216-0.104, -0.0216-0.027, 0.)))
        self.nh2 = pydart.constraints.NonHolonomicContactConstraint(skel.body('h_blade_left'), np.array((0.0216+0.104, -0.0216-0.027, 0.)))
        self.nh3 = pydart.constraints.NonHolonomicContactConstraint(skel.body('h_blade_left'), np.array((0.0216-0.104, -0.0216-0.027, 0.)))

        self.nh0.add_to_world()
        self.nh1.add_to_world()
        self.nh2.add_to_world()
        self.nh3.add_to_world()

        self.has_nh = True

        self.ground_height = 0.

    def reset(self):
        super(NHWorldV2, self).reset()
        self.skeletons[0].body(0).set_friction_coeff(0.02)
        if self.has_nh:
            self.nh0.add_to_world()
            self.nh1.add_to_world()
            self.nh2.add_to_world()
            self.nh3.add_to_world()

    def step(self):
        # nonholonomic constraint

        right_blade_front_point = self.skeletons[1].body("h_blade_right").to_world((0.0216+0.104, -0.0216-0.027, 0.))
        right_blade_rear_point = self.skeletons[1].body("h_blade_right").to_world((0.0216-0.104, -0.0216-0.027, 0.))

        if right_blade_front_point[1] < 0.005 + self.ground_height and right_blade_rear_point[1] < 0.005+self.ground_height:
            self.nh0.activate(True)
            self.nh1.activate(True)

            body_pos = self.skeletons[1].body('h_blade_right').to_world((0., -0.0216-0.027, 0.))
            body_vec = self.skeletons[1].body('h_blade_right').to_world() - body_pos  # type: np.ndarray
            projected_body_vec = body_vec.copy()
            projected_body_vec[1] = 0.
            projected_body_vel = self.skeletons[1].body('h_blade_right').world_linear_velocity()
            projected_body_vel[1] = 0.
            inclined_angle = math.pi/2. - math.acos(np.dot(mm.normalize(body_vec), mm.normalize(projected_body_vec)))
            #calculate inclined angle theta
            # print("vel: ", self.skeletons[1].body('h_blade_right').com_linear_velocity()[1])
            # print(self.time())

            # if 9. < self.time() and self.time() < 9.5:
            # if 8. < self.time() and self.time() < 8.5:
            #     radi = 1.0
            #     g = 9.8
            #     inclined_angle = math.atan(self.skeletons[1].body('h_pelvis').com_linear_velocity()[0]**2 / (radi * g) )
            #     print("right_angle: ", inclined_angle)
            #     q_temp = self.skeletons[1].q
            #     q_temp[self.skeletons[1].dof_indices(["j_heel_right_x"])] = inclined_angle
            #     self.skeletons[1].set_positions(q_temp)

            # turning_angle = np.copysign(inclined_angle / 4. ,
            #                             np.dot(projected_body_vel, mm.cross(projected_body_vec, mm.unitY())))
            turning_angle = np.copysign(max(inclined_angle-15./180.*math.pi, 0.)/4., np.dot(projected_body_vel, mm.cross(projected_body_vec, mm.unitY())))

            R = mm.exp(mm.unitY(), turning_angle)

            self.nh0.set_joint_pos(np.dot(R, right_blade_front_point - body_pos) + body_pos)
            self.nh1.set_joint_pos(np.dot(R, right_blade_rear_point - body_pos) + body_pos)
            self.nh0.set_projected_vector(np.dot(R, right_blade_front_point - right_blade_rear_point))
            self.nh1.set_projected_vector(np.dot(R, right_blade_front_point - right_blade_rear_point))
        else:
            self.nh0.activate(False)
            self.nh1.activate(False)

        left_blade_front_point = self.skeletons[1].body("h_blade_left").to_world((0.0216+0.104, -0.0216-0.027, 0.))
        left_blade_rear_point = self.skeletons[1].body("h_blade_left").to_world((0.0216-0.104, -0.0216-0.027, 0.))

        if left_blade_front_point[1] < 0.005 +self.ground_height and left_blade_rear_point[1] < 0.005+self.ground_height:
            self.nh2.activate(True)
            self.nh3.activate(True)

            body_pos = self.skeletons[1].body('h_blade_left').to_world((0., -0.0216-0.027, 0.))
            body_vec = self.skeletons[1].body('h_blade_left').to_world() - body_pos  # type: np.ndarray
            projected_body_vec = body_vec.copy()
            projected_body_vec[1] = 0.
            projected_body_vel = self.skeletons[1].body('h_blade_left').world_linear_velocity()
            projected_body_vel[1] = 0.
            inclined_angle = math.pi/2. - math.acos(np.dot(mm.normalize(body_vec), mm.normalize(projected_body_vec)))

            # calculate inclined angle theta
            # if 9.0 < self.time() and self.time() < 9.5:
            # if 8. < self.time() and self.time() < 8.5:
            #     radi = 1.
            #     g = 9.8
            #     inclined_angle = math.atan(
            #         self.skeletons[1].body('h_pelvis').com_linear_velocity()[0] ** 2 / (radi * g))
            #     print("left_angle: ", inclined_angle)
            #     q_temp = self.skeletons[1].q
            #     q_temp[self.skeletons[1].dof_indices(["j_heel_left_x"])] = inclined_angle
            #     self.skeletons[1].set_positions(q_temp)
            #
            # turning_angle = np.copysign(inclined_angle / 4. , np.dot(projected_body_vel, mm.cross(projected_body_vec, mm.unitY())))

            turning_angle = np.copysign(max(inclined_angle-15./180.*math.pi, 0.)/4., np.dot(projected_body_vel, mm.cross(projected_body_vec, mm.unitY())))
            R = mm.exp(mm.unitY(), turning_angle)

            self.nh2.set_joint_pos(np.dot(R, left_blade_front_point - body_pos) + body_pos)
            self.nh3.set_joint_pos(np.dot(R, left_blade_rear_point - body_pos) + body_pos)
            self.nh2.set_projected_vector(np.dot(R, left_blade_front_point - left_blade_rear_point))
            self.nh3.set_projected_vector(np.dot(R, left_blade_front_point - left_blade_rear_point))
        else:
            self.nh2.activate(False)
            self.nh3.activate(False)

        super(NHWorldV2, self).step()


class NHWorldV3(pydart.World):
    def __init__(self, step, skel_path=None, xmlstr=False):
        self.has_nh = False

        super(NHWorldV3, self).__init__(step, skel_path, xmlstr)

        # nonholonomic constraint initial setting
        skel = self.skeletons[1]
        self.skeletons[0].body(0).set_friction_coeff(0.02)
        # self.skeletons[0].body(0).set_friction_coeff(0.32)

        self.nh0 = pydart.constraints.NonHolonomicContactConstraint(skel.body('h_blade_right'), np.array((0.0216+0.104, -0.0216-0.027, 0.)))
        self.nh1 = pydart.constraints.NonHolonomicContactConstraint(skel.body('h_blade_right'), np.array((0.0216-0.104, -0.0216-0.027, 0.)))
        self.nh2 = pydart.constraints.NonHolonomicContactConstraint(skel.body('h_blade_left'), np.array((0.0216+0.104, -0.0216-0.027, 0.)))
        self.nh3 = pydart.constraints.NonHolonomicContactConstraint(skel.body('h_blade_left'), np.array((0.0216-0.104, -0.0216-0.027, 0.)))

        self.nh0.add_to_world()
        self.nh1.add_to_world()
        self.nh2.add_to_world()
        self.nh3.add_to_world()

        self.has_nh = True

        self.ground_height = 0.

    def reset(self):
        super(NHWorldV3, self).reset()
        self.skeletons[0].body(0).set_friction_coeff(0.02)
        # self.skeletons[0].body(0).set_friction_coeff(0.32)
        if self.has_nh:
            self.nh0.add_to_world()
            self.nh1.add_to_world()
            self.nh2.add_to_world()
            self.nh3.add_to_world()

    def step(self):
        # nonholonomic constraint

        right_blade_front_point = self.skeletons[1].body("h_blade_right").to_world((0.0216+0.104, -0.0216-0.027, 0.))
        right_blade_rear_point = self.skeletons[1].body("h_blade_right").to_world((0.0216-0.104, -0.0216-0.027, 0.))

        if right_blade_front_point[1] < 0.005 + self.ground_height and right_blade_rear_point[1] > 0.05 + self.ground_height:
            self.skeletons[0].body(0).set_friction_coeff(1.0)
            # print("TOE PICK CONTACT!!! RIGHT")
        else:
            self.skeletons[0].body(0).set_friction_coeff(0.02)

        if right_blade_front_point[1] < 0.005+self.ground_height and right_blade_rear_point[1] < 0.005+self.ground_height:
            self.nh0.activate(True)
            self.nh1.activate(True)

            body_pos = self.skeletons[1].body('h_blade_right').to_world((0., -0.0216-0.027, 0.))
            body_vec = self.skeletons[1].body('h_blade_right').to_world() - body_pos  # type: np.ndarray
            projected_body_vec = body_vec.copy()
            projected_body_vec[1] = 0.
            projected_body_vel = self.skeletons[1].body('h_blade_right').world_linear_velocity()
            projected_body_vel[1] = 0.
            inclined_angle = math.pi/2. - math.acos(np.dot(mm.normalize(body_vec), mm.normalize(projected_body_vec)))
            #calculate inclined angle theta
            # print("vel: ", self.skeletons[1].body('h_blade_right').com_linear_velocity()[1])
            # print(self.time())

            # if 9. < self.time() and self.time() < 9.5:
            # if 8. < self.time() and self.time() < 8.5:
            #     radi = 1.0
            #     g = 9.8
            #     inclined_angle = math.atan(self.skeletons[1].body('h_pelvis').com_linear_velocity()[0]**2 / (radi * g) )
            #     print("right_angle: ", inclined_angle)
            #     q_temp = self.skeletons[1].q
            #     q_temp[self.skeletons[1].dof_indices(["j_heel_right_x"])] = inclined_angle
            #     self.skeletons[1].set_positions(q_temp)

            # turning_angle = np.copysign(inclined_angle / 4. ,
            #                             np.dot(projected_body_vel, mm.cross(projected_body_vec, mm.unitY())))
            turning_angle = np.copysign(max(inclined_angle-15./180.*math.pi, 0.)/4., np.dot(projected_body_vel, mm.cross(projected_body_vec, mm.unitY())))
            # turning_angle = np.copysign(max(inclined_angle-5./180.*math.pi, 0.) / 4.,
            #                             np.dot(projected_body_vel, mm.cross(projected_body_vec, mm.unitY())))
            R = mm.exp(mm.unitY(), turning_angle)

            self.nh0.set_joint_pos(np.dot(R, right_blade_front_point - body_pos) + body_pos)
            self.nh1.set_joint_pos(np.dot(R, right_blade_rear_point - body_pos) + body_pos)
            self.nh0.set_projected_vector(np.dot(R, right_blade_front_point - right_blade_rear_point))
            self.nh1.set_projected_vector(np.dot(R, right_blade_front_point - right_blade_rear_point))
        else:
            self.nh0.activate(False)
            self.nh1.activate(False)

        left_blade_front_point = self.skeletons[1].body("h_blade_left").to_world((0.0216+0.104, -0.0216-0.027, 0.))
        left_blade_rear_point = self.skeletons[1].body("h_blade_left").to_world((0.0216-0.104, -0.0216-0.027, 0.))

        # if left_blade_front_point[1] < 0.005 + self.ground_height and left_blade_rear_point[1] > 0.1 + self.ground_height:
        #     self.skeletons[0].body(0).set_friction_coeff(1.0)
        #     print("TOE PICK CONTACT!!! LEFT")
        # else:
        #     self.skeletons[0].body(0).set_friction_coeff(0.02)

        if left_blade_front_point[1] < 0.005 +self.ground_height and left_blade_rear_point[1] < 0.005+self.ground_height:
            self.nh2.activate(True)
            self.nh3.activate(True)

            body_pos = self.skeletons[1].body('h_blade_left').to_world((0., -0.0216-0.027, 0.))
            body_vec = self.skeletons[1].body('h_blade_left').to_world() - body_pos  # type: np.ndarray
            projected_body_vec = body_vec.copy()
            projected_body_vec[1] = 0.
            projected_body_vel = self.skeletons[1].body('h_blade_left').world_linear_velocity()
            projected_body_vel[1] = 0.
            inclined_angle = math.pi/2. - math.acos(np.dot(mm.normalize(body_vec), mm.normalize(projected_body_vec)))

            # calculate inclined angle theta
            # if 9.0 < self.time() and self.time() < 9.5:
            # if 8. < self.time() and self.time() < 8.5:
            #     radi = 1.
            #     g = 9.8
            #     inclined_angle = math.atan(
            #         self.skeletons[1].body('h_pelvis').com_linear_velocity()[0] ** 2 / (radi * g))
            #     print("left_angle: ", inclined_angle)
            #     q_temp = self.skeletons[1].q
            #     q_temp[self.skeletons[1].dof_indices(["j_heel_left_x"])] = inclined_angle
            #     self.skeletons[1].set_positions(q_temp)
            #
            # turning_angle = np.copysign(inclined_angle / 4. , np.dot(projected_body_vel, mm.cross(projected_body_vec, mm.unitY())))

            turning_angle = np.copysign(max(inclined_angle-15./180.*math.pi, 0.)/4., np.dot(projected_body_vel, mm.cross(projected_body_vec, mm.unitY())))
            # turning_angle = np.copysign(max(inclined_angle-5./180.*math.pi, 0.) / 4.,
            #                             np.dot(projected_body_vel, mm.cross(projected_body_vec, mm.unitY())))
            R = mm.exp(mm.unitY(), turning_angle)

            self.nh2.set_joint_pos(np.dot(R, left_blade_front_point - body_pos) + body_pos)
            self.nh3.set_joint_pos(np.dot(R, left_blade_rear_point - body_pos) + body_pos)
            self.nh2.set_projected_vector(np.dot(R, left_blade_front_point - left_blade_rear_point))
            self.nh3.set_projected_vector(np.dot(R, left_blade_front_point - left_blade_rear_point))
        else:
            self.nh2.activate(False)
            self.nh3.activate(False)

        super(NHWorldV3, self).step()
