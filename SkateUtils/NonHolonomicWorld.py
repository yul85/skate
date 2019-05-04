import pydart2 as pydart
import numpy as np
from math import pi


class NHWorld(pydart.World):
    def __init__(self, step, skel_path=None, xmlstr=False):
        self.has_nh = False

        super(NHWorld, self).__init__(step, skel_path, xmlstr)

        # nonholonomic constraint initial setting
        _th = 5. * pi / 180.
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
