import pydart2 as pydart
from SkateUtils.NonHolonomicWorld import NHWorldV2
import numpy as np
import gym
import gym.spaces
import copy
from skate_cma.BSpline import BSpline
from skate_cma.PenaltyType import PenaltyType
from PyCommon.modules.Math import mmMath as mm


class SkateDartEnv(gym.Env):
    def __init__(self, env_name='speed_skating'):
        self.world = NHWorldV2(1./1200., '../data/skel/skater_3dof_with_ground.skel')
        self.world.control_skel = self.world.skeletons[1]
        self.skel = self.world.skeletons[1]
        # self.Kp, self.Kd = 400., 40.
        self.Kp, self.Kd = 1000., 60.

        self.torque_max = 1000.

        self.total_time = 1.

        self.env_name = env_name

        self.ref_motion = None  # type: ym.Motion

        self.ref_world = NHWorldV2(1./1200., '../data/skel/skater_3dof_with_ground.skel')
        self.ref_skel = self.ref_world.skeletons[1]
        # revise_pose(self.world.skeletons[1], self.ref_root_state)

        self.bs = BSpline()
        self.bs_ori = BSpline()

        self.step_per_frame = 40

        self.w_cv = 1.
        self.exp_cv = 2.

        self.body_num = self.skel.num_bodynodes()

        self.phase_frame = 0

        self.penalty_type_on = [None] * len(PenaltyType)
        self.penalty_type_weight = [1.] * len(PenaltyType)

        self.com_init = np.zeros(3)

        self.state_duration = 0

        self.recent_torque_sum = 0.
        self.recent_ori_q = np.zeros_like(self.skel.q)

    def set_penalty(self, penalty_option, penalty_weight):
        self.penalty_type_on = copy.deepcopy(penalty_option)
        self.penalty_type_weight = copy.deepcopy(penalty_weight)

    def save_com_init(self):
        self.com_init = self.skel.com()

    def calc_angular_momentum(self, pos):
        am = np.zeros(3)
        for body in self.skel.bodynodes:
            am += body.mass() * mm.cross(body.com() - pos, body.com_linear_velocity())
            am += np.dot(body.inertia(), body.world_angular_velocity())
        return am

    def penalty_instant(self):
        E = 0.
        E += self.penalty_type_weight[PenaltyType.TORQUE] * self.recent_torque_sum
        # E += 1e-4 * np.sum(np.square(self.skel.position_differences(self.skel.q, self.recent_ori_q)[6:]))
        if self.penalty_type_on[PenaltyType.COM_HEIGHT] is not None:
            E += self.penalty_type_weight[PenaltyType.COM_HEIGHT] \
                * np.square(self.penalty_type_on[PenaltyType.COM_HEIGHT] - self.skel.com()[1])

        if self.penalty_type_on[PenaltyType.LEFT_FOOT_CONTACT] is not None:
            E += self.penalty_type_weight[PenaltyType.LEFT_FOOT_CONTACT] \
                * np.square(self.skel.body('h_blade_left').to_world([0., -0.0486, 0.])[1])

        if self.penalty_type_on[PenaltyType.RIGHT_FOOT_CONTACT] is not None:
            E += self.penalty_type_weight[PenaltyType.RIGHT_FOOT_CONTACT] \
                * np.square(self.skel.body('h_blade_right').to_world([0., -0.0486, 0.])[1])

        if self.penalty_type_on[PenaltyType.LEFT_FOOT_TOE_CONTACT] is not None:
            E += self.penalty_type_weight[PenaltyType.LEFT_FOOT_TOE_CONTACT] \
                 * np.square(self.skel.body('h_blade_left').to_world([0.1256, -0.0486, 0.])[1])

        if self.penalty_type_on[PenaltyType.RIGHT_FOOT_TOE_CONTACT] is not None:
            E += self.penalty_type_weight[PenaltyType.RIGHT_FOOT_TOE_CONTACT] \
                 * np.square(self.skel.body('h_blade_right').to_world([0.1256, -0.0486, 0.])[1])

        if self.penalty_type_on[PenaltyType.COM_IN_LEFT_RIGHT_CENTER] is not None:
            com_projected = self.skel.com()
            com_projected[1] = 0.
            centroid = .5 * (self.skel.body('h_blade_left').to_world([0.0216, -0.0486, 0.]) + self.skel.body('h_blade_right').to_world([0.0216, -0.0486, 0.]))
            centroid[1] = 0.
            E += self.penalty_type_weight[PenaltyType.COM_IN_LEFT_RIGHT_CENTER] \
                * np.sum(np.square(com_projected - centroid))

        if self.penalty_type_on[PenaltyType.COM_IN_LEFT_FOOT] is not None:
            com_projected = self.skel.com()
            com_projected[1] = 0.
            centroid = self.skel.body('h_blade_left').to_world([0.0216, -0.0486, 0.])
            centroid[1] = 0.
            E += self.penalty_type_weight[PenaltyType.COM_IN_LEFT_FOOT] \
                * np.sum(np.square(com_projected - centroid))

        if self.penalty_type_on[PenaltyType.COM_IN_RIGHT_FOOT] is not None:
            com_projected = self.skel.com()
            com_projected[1] = 0.
            centroid = self.skel.body('h_blade_right').to_world([0.0216, -0.0486, 0.])
            centroid[1] = 0.
            E += self.penalty_type_weight[PenaltyType.COM_IN_RIGHT_FOOT] \
                * np.sum(np.square(com_projected - centroid))

        if self.penalty_type_on[PenaltyType.COM_IN_RIGHT_FOOT_TOE] is not None:
            com_projected = self.skel.com()
            com_projected[1] = 0.
            centroid = self.skel.body('h_blade_right').to_world([0.1256, -0.0486, 0.])
            centroid[1] = 0.
            E += self.penalty_type_weight[PenaltyType.COM_IN_RIGHT_FOOT_TOE] \
                * np.sum(np.square(com_projected - centroid))

        if self.penalty_type_on[PenaltyType.COM_IN_LEFT_FOOT_TOE] is not None:
            com_projected = self.skel.com()
            com_projected[1] = 0.
            centroid = self.skel.body('h_blade_left').to_world([0.1256, -0.0486, 0.])
            centroid[1] = 0.
            E += self.penalty_type_weight[PenaltyType.COM_IN_LEFT_FOOT_TOE] \
                * np.sum(np.square(com_projected - centroid))

        if self.penalty_type_on[PenaltyType.COM_IN_HEAD] is not None:
            com_projected = self.skel.com()
            com_projected[1] = 0.
            centroid = self.skel.body('h_head').to_world([0., 0.07825, 0.])
            centroid[1] = 0.
            E += self.penalty_type_weight[PenaltyType.COM_IN_HEAD] \
                * np.sum(np.square(com_projected - centroid))

        if self.penalty_type_on[PenaltyType.MAX_Y_ANGULAR_MOMENTUM] is not None:
            E -= self.penalty_type_weight[PenaltyType.MAX_Y_ANGULAR_MOMENTUM] \
                * np.square(self.calc_angular_momentum(self.skel.com())[1]) / (self.skel.mass() * self.skel.mass())

        if self.penalty_type_on[PenaltyType.COM_VEL] is not None:
            E += self.penalty_type_weight[PenaltyType.COM_VEL] \
                * np.sum(np.square(self.skel.com_velocity() - self.penalty_type_on[PenaltyType.COM_VEL]))

        if self.penalty_type_on[PenaltyType.PELVIS_HEADING] is not None:
            heading_des = mm.normalize2(self.penalty_type_on[PenaltyType.PELVIS_HEADING])
            heading_cur = np.dot(self.skel.body('h_pelvis').world_transform()[:3, :3], mm.unitX())
            E += self.penalty_type_weight[PenaltyType.PELVIS_HEADING] \
                * np.sum(np.square(heading_cur - heading_des))

        if self.penalty_type_on[PenaltyType.COM_VEL_DIR_WITH_PELVIS_LOCAL] is not None:
            R_pelvis = self.skel.body('h_pelvis').world_transform()[:3, :3]
            com_vel_dir_des = np.asarray(self.penalty_type_on[PenaltyType.COM_VEL_DIR_WITH_PELVIS_LOCAL])
            com_vel_dir_des_in_world = mm.normalize2(np.multiply(np.array([1., 0., 1.]), np.dot(R_pelvis, com_vel_dir_des)))
            com_vel_dir = mm.normalize2(np.multiply(np.array([1., 0., 1.]), self.skel.com_velocity()))

            E += self.penalty_type_weight[PenaltyType.COM_VEL_DIR_WITH_PELVIS_LOCAL] \
                * np.sum(np.square(com_vel_dir_des_in_world - com_vel_dir))

        if self.penalty_type_on[PenaltyType.MAX_Y_LEFT_FOOT_HEEL] is not None:
            E -= self.penalty_type_weight[PenaltyType.MAX_Y_LEFT_FOOT_HEEL] \
                 * np.square(self.skel.body('h_blade_left').to_world([-0.0824, -0.0486, 0.])[1])

        if self.penalty_type_on[PenaltyType.MAX_Y_RIGHT_FOOT_HEEL] is not None:
            E -= self.penalty_type_weight[PenaltyType.MAX_Y_RIGHT_FOOT_HEEL] \
                 * np.square(self.skel.body('h_blade_right').to_world([-0.0824, -0.0486, 0.])[1])

        if self.penalty_type_on[PenaltyType.MAX_Y_RIGHT_TOE_HEAD_DIFF] is not None:
            E -= self.penalty_type_weight[PenaltyType.MAX_Y_RIGHT_TOE_HEAD_DIFF] \
                 * np.square(self.skel.body('h_head').com()[1] - self.skel.body('h_blade_right').to_world([0.1256, -0.0486, 0.])[1])

        if self.penalty_type_on[PenaltyType.MAX_Y_ANGULAR_MOMENTUM_RIGHT_TOE] is not None:
            E -= self.penalty_type_weight[PenaltyType.MAX_Y_ANGULAR_MOMENTUM_RIGHT_TOE] \
                 * np.square(self.calc_angular_momentum(self.skel.body('h_blade_right').to_world([0.1256, -0.0486, 0.]))[1] / (self.skel.mass() * self.skel.mass()))


        return E

    def penalty_final(self):
        E = 0.
        if self.penalty_type_on[PenaltyType.COM_HEIGHT_END] is not None:
            E += self.penalty_type_weight[PenaltyType.COM_HEIGHT_END] \
                * np.square(self.penalty_type_on[PenaltyType.COM_HEIGHT_END] - self.skel.com()[1])

        if self.penalty_type_on[PenaltyType.COM_VEL_END] is not None:
            E += self.penalty_type_weight[PenaltyType.COM_VEL_END] \
                * np.sum(np.square(self.penalty_type_on[PenaltyType.COM_VEL_END] - (self.skel.com()[1] - self.com_init)))

        if self.penalty_type_on[PenaltyType.LEFT_FOOT_CONTACT_END] is not None:
            E += self.penalty_type_weight[PenaltyType.LEFT_FOOT_CONTACT_END] \
                * np.square(self.skel.body('h_blade_left').to_world([0., -0.0486, 0.])[1])

        if self.penalty_type_on[PenaltyType.RIGHT_FOOT_CONTACT_END] is not None:
            E += self.penalty_type_weight[PenaltyType.RIGHT_FOOT_CONTACT_END] \
                * np.square(self.skel.body('h_blade_right').to_world([0., -0.0486, 0.])[1])

        if self.penalty_type_on[PenaltyType.LEFT_FOOT_TOE_CONTACT_END] is not None:
            E += self.penalty_type_weight[PenaltyType.LEFT_FOOT_TOE_CONTACT_END] \
                 * np.square(self.skel.body('h_blade_left').to_world([0.1256, -0.0486, 0.])[1])

        if self.penalty_type_on[PenaltyType.RIGHT_FOOT_TOE_CONTACT_END] is not None:
            E += self.penalty_type_weight[PenaltyType.RIGHT_FOOT_TOE_CONTACT_END] \
                 * np.square(self.skel.body('h_blade_right').to_world([0.1256, -0.0486, 0.])[1])

        if self.penalty_type_on[PenaltyType.COM_IN_LEFT_RIGHT_CENTER_END] is not None:
            com_projected = self.skel.com()
            com_projected[1] = 0.
            centroid = .5 * (self.skel.body('h_blade_left').to_world([0.0216, -0.0486, 0.]) + self.skel.body('h_blade_right').to_world([0.0216, -0.0486, 0.]))
            centroid[1] = 0.
            E += self.penalty_type_weight[PenaltyType.COM_IN_LEFT_RIGHT_CENTER_END] \
                 * np.sum(np.square(com_projected - centroid))

        if self.penalty_type_on[PenaltyType.COM_IN_LEFT_FOOT_END] is not None:
            com_projected = self.skel.com()
            com_projected[1] = 0.
            centroid = self.skel.body('h_blade_left').to_world([0.0216, -0.0486, 0.])
            centroid[1] = 0.
            E += self.penalty_type_weight[PenaltyType.COM_IN_LEFT_FOOT_END]\
                * np.sum(np.square(com_projected - centroid))

        if self.penalty_type_on[PenaltyType.COM_IN_RIGHT_FOOT_END] is not None:
            com_projected = self.skel.com()
            com_projected[1] = 0.
            centroid = self.skel.body('h_blade_right').to_world([0.0216, -0.0486, 0.])
            centroid[1] = 0.
            E += self.penalty_type_weight[PenaltyType.COM_IN_RIGHT_FOOT_END] \
                * np.sum(np.square(com_projected - centroid))

        if self.penalty_type_on[PenaltyType.COM_IN_RIGHT_FOOT_TOE_END] is not None:
            com_projected = self.skel.com()
            com_projected[1] = 0.
            centroid = self.skel.body('h_blade_right').to_world([0.1256, -0.0486, 0.])
            centroid[1] = 0.
            E += self.penalty_type_weight[PenaltyType.COM_IN_RIGHT_FOOT_TOE_END] \
                 * np.sum(np.square(com_projected - centroid))

        if self.penalty_type_on[PenaltyType.COM_IN_LEFT_FOOT_TOE_END] is not None:
            com_projected = self.skel.com()
            com_projected[1] = 0.
            centroid = self.skel.body('h_blade_left').to_world([0.1256, -0.0486, 0.])
            centroid[1] = 0.
            E += self.penalty_type_weight[PenaltyType.COM_IN_LEFT_FOOT_TOE_END] \
                 * np.sum(np.square(com_projected - centroid))

        if self.penalty_type_on[PenaltyType.MAX_Y_VELOCITY_END] is not None:
            E -= self.penalty_type_weight[PenaltyType.MAX_Y_VELOCITY_END] \
                * np.sign(self.skel.com_velocity()[1]) * np.square(self.skel.com_velocity()[1])

        if self.penalty_type_on[PenaltyType.MAX_Y_ANGULAR_MOMENTUM_END] is not None:
            E -= self.penalty_type_weight[PenaltyType.MAX_Y_ANGULAR_MOMENTUM_END] \
                * np.sum(np.square(self.calc_angular_momentum(self.skel.com()))) / (self.skel.mass() * self.skel.mass())

        if self.penalty_type_on[PenaltyType.MAX_Y_POS_DIFF_END] is not None:
            E -= self.penalty_type_weight[PenaltyType.MAX_Y_POS_DIFF_END] \
                * np.square(self.skel.body('h_head').com()[1] - (self.skel.body('h_blade_left').com()[1]+self.skel.body('h_blade_right').com()[1]))/2.

        if self.penalty_type_on[PenaltyType.PELVIS_HEADING_END] is not None:
            heading_des = mm.normalize2(self.penalty_type_on[PenaltyType.PELVIS_HEADING_END])
            heading_cur = np.dot(self.skel.body('h_pelvis').world_transform()[:3, :3], mm.unitX())
            E += self.penalty_type_weight[PenaltyType.PELVIS_HEADING_END] \
                * np.sum(np.square(heading_cur - heading_des))

        if self.penalty_type_on[PenaltyType.MAX_Y_LEFT_FOOT_HEEL_END] is not None:
            E -= self.penalty_type_weight[PenaltyType.MAX_Y_LEFT_FOOT_HEEL_END] \
                 * np.square(self.skel.body('h_blade_left').to_world([-0.0824, -0.0486, 0.])[1])

        if self.penalty_type_on[PenaltyType.MAX_Y_RIGHT_FOOT_HEEL_END] is not None:
            E -= self.penalty_type_weight[PenaltyType.MAX_Y_RIGHT_FOOT_HEEL_END] \
                 * np.square(self.skel.body('h_blade_right').to_world([-0.0824, -0.0486, 0.])[1])

        if self.penalty_type_on[PenaltyType.MAX_Y_ANGULAR_MOMENTUM_RIGHT_TOE_END] is not None:
            E -= self.penalty_type_weight[PenaltyType.MAX_Y_ANGULAR_MOMENTUM_RIGHT_TOE_END] \
                 * np.square(self.calc_angular_momentum(self.skel.body('h_blade_right').to_world([0.1256, -0.0486, 0.]))[1] / (self.skel.mass() * self.skel.mass()))

        return E

    def is_done(self):
        if self.skel.com()[1] < 0.2:
            # print('fallen')
            return True
        elif True in np.isnan(np.asarray(self.skel.q)) or True in np.isnan(np.asarray(self.skel.dq)):
            # print('nan')
            return True
        elif self.world.time() >= self.total_time:
            # print('timeout')
            return True
        return False

    def _step(self, _action):
        action = np.hstack((np.zeros(6), _action))
        self.ref_skel.set_positions(action)

        self.recent_torque_sum = 0.
        for i in range(self.step_per_frame):
            torque = np.clip(self.skel.get_spd(self.ref_skel.q, self.world.time_step(), self.Kp, self.Kd), -self.torque_max, self.torque_max)

            self.skel.set_forces(torque)
            self.world.step()
            self.recent_torque_sum += np.sum(np.square(torque))

    def step(self, bs_t):
        _action = self.bs.get_value(bs_t)
        self._step(_action)
        self.recent_ori_q = np.hstack((np.zeros(6), self.bs_ori.get_value(bs_t)))

    def update_ref_states(self, x, x0, q0, dq0, duration):
        assert(duration+1 == len(x0))
        self.bs.clear()
        self.state_duration = duration
        self.bs.generate_control_point(x, x0)

        self.bs_ori.clear()
        self.bs_ori.generate_control_point(x0)

        self.skel.set_positions(q0)
        self.skel.set_velocities(dq0)

    def reset(self):
        """
        Resets the state of the environment and returns an initial observation.

        # Returns
            observation (object): The initial observation of the space. Initial reward is assumed to be 0.
        """
        self.world.reset()
        self.penalty_type_on = [False] * len(PenaltyType)
        return None

    def hard_reset(self):
        self.world = NHWorldV2(1./1200., '../data/skel/skater_3dof_with_ground.skel')
        self.world.control_skel = self.world.skeletons[1]
        self.skel = self.world.skeletons[1]
        return self.reset()
