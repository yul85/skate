# from rl.core import Env
import pydart2 as pydart
import numpy as np
from math import exp, pi
from PyCommon.modules.Math import mmMath as mm
from random import randrange, random
import gym
import gym.spaces
from gym.utils import seeding
from speed_skating.make_skate_keyframe import make_keyframe, revise_pose

import PyCommon.modules.Resource.ysMotionLoader as yf

from multiprocessing import Pool


def exp_reward_term(w, exp_w, v0, v1):
    norm = np.linalg.norm(v0 - v1)
    return w * exp(-exp_w * norm * norm)


class NHWorld(pydart.World):
    def __init__(self, step, skel_path=None, xmlstr=False):
        self.has_nh = False

        super(NHWorld, self).__init__(step, skel_path, xmlstr)

        # nonholonomic constraint initial setting
        _th = 5. * pi / 180.
        skel = self.skeletons[2]
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

        super(NHWorld, self).step()


class SkateDartEnv(gym.Env):
    def __init__(self, env_name='walk', env_slaves=1):
        self.world = NHWorld(1./1200., '../data/skel/cart_pole_blade_3dof_with_ground.skel')
        self.world.control_skel = self.world.skeletons[2]
        self.skel = self.world.skeletons[2]
        self.Kp, self.Kd = 400., 40.

        self.total_time = 10.

        self.env_name = env_name

        self.ref_motion = None  # type: ym.Motion

        self.ref_world = self.world
        self.ref_skel = self.ref_world.skeletons[3]
        self.ref_root_state = make_keyframe(self.skel)
        revise_pose(self.world.skeletons[3], self.ref_root_state)
        self.ref_state = self.ref_root_state
        self.ref_state_time = 0.
        self.ref_state_num = 4

        # self.step_per_frame = round((1./self.world.time_step()) / self.ref_motion.fps)
        self.step_per_frame = 40

        self.rsi = False

        self.w_cv = 1.

        self.exp_cv = 2.

        self.ref_com_vel = np.array([0.9, 0., 0.])

        self.body_num = self.skel.num_bodynodes()

        self.time_offset = 0.

        state_num = 2 + (3*3 + 4) * self.body_num
        action_num = self.skel.num_dofs() - 6

        state_high = np.array([np.finfo(np.float32).max] * state_num)
        action_high = np.array([pi*10./2.] * action_num)

        self.action_space = gym.spaces.Box(-action_high, action_high, dtype=np.float32)
        self.observation_space = gym.spaces.Box(-state_high, state_high, dtype=np.float32)

        self.viewer = None

        self.phase_frame = 0

    def state(self):
        pelvis = self.skel.body(0)
        p_pelvis = pelvis.world_transform()[:3, 3]
        R_pelvis = pelvis.world_transform()[:3, :3]

        phase = min(1., self.ref_state_time/self.ref_state.dt)
        state = [self.ref_state.get_number()/self.ref_state_num, phase]

        p = np.array([np.dot(R_pelvis.T, body.to_world() - p_pelvis) for body in self.skel.bodynodes]).flatten()
        R = np.array([mm.rot2quat(np.dot(R_pelvis.T, body.world_transform()[:3, :3])) for body in self.skel.bodynodes]).flatten()
        v = np.array([np.dot(R_pelvis.T, body.world_linear_velocity()) for body in self.skel.bodynodes]).flatten()
        w = np.array([np.dot(R_pelvis.T, body.world_angular_velocity())/20. for body in self.skel.bodynodes]).flatten()

        state.extend(p)
        state.extend(R)
        state.extend(v)
        state.extend(w)

        return np.asarray(state).flatten()

    def reward(self):
        com_vel = self.skel.com_velocity()
        com_vel[1] = 0.
        r_cv = 0.
        if self.world.time() > 2.3:
            r_cv = exp_reward_term(self.w_cv, self.exp_cv, com_vel, self.ref_com_vel)
        elif self.world.time() < 2.:
            r_cv = exp_reward_term(self.w_cv, self.exp_cv, com_vel, np.zeros(3))

        return r_cv

    def is_done(self):
        if self.skel.com()[1] < 0.4:
            # print('fallen')
            return True
        elif True in np.isnan(np.asarray(self.skel.q)) or True in np.isnan(np.asarray(self.skel.dq)):
            # print('nan')
            return True
        elif self.world.time() + self.time_offset > self.total_time:
            # print('timeout')
            return True
        return False

    def step(self, _action):
        """Run one timestep of the environment's dynamics.
        Accepts an action and returns a tuple (observation, reward, done, info).

        # Arguments
            action (object): An action provided by the environment.

        # Returns
            observation (object): Agent's observation of the current environment.
            reward (float) : Amount of reward returned after previous action.
            done (boolean): Whether the episode has ended, in which case further step() calls will return undefined results.
            info (dict): Contains auxiliary diagnostic information (helpful for debugging, and sometimes learning).
        """
        action = np.hstack((np.zeros(6), _action/10.))
        self.ref_skel.set_positions(self.ref_state.angles)
        for i in range(self.step_per_frame):
            # self.skel.set_forces(self.skel.get_spd(self.ref_skel.q + action, self.world.time_step(), self.Kp, self.Kd))
            self.skel.set_forces(self.skel.get_spd(self.ref_state.angles + action, self.world.time_step(), self.Kp, self.Kd))
            self.world.step()

        self.ref_state_time += self.step_per_frame * self.world.time_step()
        if self.ref_state_time >= self.ref_state.dt:
            self.ref_state_time -= self.ref_state.dt
            self.ref_state = self.ref_state.get_next()

        return tuple([self.state(), self.reward(), self.is_done(), dict()])

    def continue_from_now_by_phase(self, phase):
        self.phase_frame = round(phase * (self.motion_len-1))
        skel_pelvis_offset = self.skel.joint(0).position_in_world_frame() - self.ref_motion[self.phase_frame].getJointPositionGlobal(0)
        skel_pelvis_offset[1] = 0.
        self.ref_motion.translateByOffset(skel_pelvis_offset)
        self.time_offset = - self.world.time() + (self.phase_frame / self.ref_motion.fps)

    def reset(self):
        """
        Resets the state of the environment and returns an initial observation.

        # Returns
            observation (object): The initial observation of the space. Initial reward is assumed to be 0.
        """
        self.world.reset()
        self.ref_state = self.ref_root_state
        # self.continue_from_now_by_phase(random() if self.rsi else 0.)
        self.skel.set_positions(self.ref_state.angles)
        # self.skel.set_positions(self.ref_motion.get_q(self.phase_frame))
        # dq = self.ref_motion.get_dq_dart(self.phase_frame)
        # self.skel.set_velocities(dq)
        self.skel.set_velocities(np.zeros(self.skel.ndofs))

        return self.state()

    def render(self, mode='human', close=False):
        """Renders the environment.
        The set of supported modes varies per environment. (And some
        environments do not support rendering at all.)

        # Arguments
            mode (str): The mode to render with.
            close (bool): Close all open renderings.
        """
        return None

    def close(self):
        """Override in your subclass to perform any necessary cleanup.
        Environments will automatically close() themselves when
        garbage collected or when the program exits.
        """
        pass

    def seed(self, seed=None):
        """Sets the seed for this env's random number generator(s).

        # Returns
            Returns the list of seeds used in this env's random number generators
        """
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def GetState(self, param):
        return self.state()

    def GetStates(self):
        return [self.state()]

    def Resets(self, rsi):
        self.rsi = rsi
        self.reset()

    def Reset(self, rsi, idx):
        self.rsi = rsi
        self.reset()

    def Steps(self, actions):
        return self.step(actions[0])

    def IsTerminalState(self, j):
        return self.is_done()

    def GetReward(self, j):
        return self.reward()

    def IsTerminalStates(self):
        return [self.is_done()]
