from SkateUtils.NonHolonomicWorld import NHWorld, NHWorldV2
from SkateUtils.DartMotionEdit import DartSkelMotion
import numpy as np
from math import exp, pi, log
from PyCommon.modules.Math import mmMath as mm
from random import random, randrange
import gym
import gym.spaces
from gym.utils import seeding
import pydart2 as pydart


def exp_reward_term(w, exp_w, v):
    norm = np.linalg.norm(v)
    return w * exp(-exp_w * norm * norm)


class SkateDartEnv(gym.Env):
    def __init__(self, ):
        pydart.init()
        cur_path = '/'.join(__file__.split('/')[:-1])
        self.world = NHWorldV2(1./1200., cur_path+'/../../data/skel/skater_3dof_with_ground.skel')
        self.world.control_skel = self.world.skeletons[1]
        self.skel = self.world.skeletons[1]
        self.Kp, self.Kd = 1000., 60.

        self.ref_world = NHWorldV2(1./1200., cur_path+'/../../data/skel/skater_3dof_with_ground.skel')
        self.ref_skel = self.ref_world.skeletons[1]
        self.ref_motion = DartSkelMotion()
        self.ref_motion.load(cur_path+'/skate_ref.skmo')
        self.ref_motion.refine_dqs(self.ref_skel)
        self.step_per_frame = 40

        self.rsi = True

        self.w_p = 0.65
        self.w_v = 0.1
        self.w_e = 0.15
        self.w_c = 0.1

        self.exp_p = 2.
        self.exp_v = 0.1
        self.exp_e = 40.
        self.exp_c = 10.

        self.body_num = self.skel.num_bodynodes() - 2
        self.reward_bodies = [body for body in self.skel.bodynodes]
        self.reward_bodies.pop(self.reward_bodies.index(self.skel.body('h_blade_left')))
        self.reward_bodies.pop(self.reward_bodies.index(self.skel.body('h_blade_right')))
        self.idx_e = [self.skel.bodynode_index('h_hand_left'), self.skel.bodynode_index('h_hand_right'),
                      self.skel.bodynode_index('h_blade_left'), self.skel.bodynode_index('h_blade_right')]
        self.body_e = list(map(self.skel.body, self.idx_e))
        self.ref_body_e = list(map(self.ref_skel.body, self.idx_e))
        self.motion_len = len(self.ref_motion)
        self.motion_time = len(self.ref_motion) / self.ref_motion.fps

        state_num = 1 + (3*3 + 4) * self.body_num
        action_num = self.skel.num_dofs() - 6

        state_high = np.array([np.finfo(np.float32).max] * state_num)
        action_high = np.array([pi*10./2.] * action_num)

        self.action_space = gym.spaces.Box(-action_high, action_high, dtype=np.float32)
        self.observation_space = gym.spaces.Box(-state_high, state_high, dtype=np.float32)

        self.viewer = None

        self.current_frame = 0
        self.count_frame = 0
        self.max_frame = 30*10

    def state(self):
        pelvis = self.skel.body(0)
        p_pelvis = pelvis.world_transform()[:3, 3]
        R_pelvis = pelvis.world_transform()[:3, :3]

        # phase = min(1., (self.world.time() + self.time_offset)/self.motion_time)
        phase = self.ref_motion.get_frame_looped(self.current_frame)
        state = [phase]

        p = np.array([np.dot(R_pelvis.T, body.to_world() - p_pelvis) for body in self.reward_bodies]).flatten()
        R = np.array([mm.rot2quat(np.dot(R_pelvis.T, body.world_transform()[:3, :3])) for body in self.reward_bodies]).flatten()
        v = np.array([np.dot(R_pelvis.T, body.world_linear_velocity()) for body in self.reward_bodies]).flatten()
        w = np.array([np.dot(R_pelvis.T, body.world_angular_velocity())/20. for body in self.reward_bodies]).flatten()

        state.extend(p)
        state.extend(R)
        state.extend(v)
        state.extend(w)

        return np.asarray(state).flatten()

    def reward(self):
        self.ref_skel.set_positions(self.ref_motion.qs[self.current_frame])
        self.ref_skel.set_velocities(self.ref_motion.dqs[self.current_frame])

        p_e_hat = np.asarray([body.world_transform()[:3, 3] for body in self.ref_body_e]).flatten()
        p_e = np.asarray([body.world_transform()[:3, 3] for body in self.body_e]).flatten()

        r_p = exp_reward_term(self.w_p, self.exp_p, self.skel.position_differences(self.skel.q, self.ref_skel.q))
        r_v = exp_reward_term(self.w_v, self.exp_v, self.skel.velocity_differences(self.skel.dq, self.ref_skel.dq))
        r_e = exp_reward_term(self.w_e, self.exp_e, p_e - p_e_hat)
        r_com = exp_reward_term(self.w_c, self.exp_c, self.skel.com() - self.ref_skel.com())

        return r_p + r_v + r_e + r_com

    def is_done(self):
        if self.skel.com()[1] < 0.2:
            # print('fallen')
            return True
        elif self.skel.body('h_head') in self.world.collision_result.contacted_bodies:
            return True
        elif True in np.isnan(np.asarray(self.skel.q)) or True in np.isnan(np.asarray(self.skel.dq)):
            # print('nan')
            return True
        elif self.ref_motion.has_loop and self.count_frame >= self.max_frame:
            # print('timeout')
            return True
        elif self.current_frame == self.motion_len - 1 or self.count_frame >= self.max_frame:
            # print('timeout')
            return True
        return False

    def step(self, _action):
        action = np.hstack((np.zeros(6), _action/10.))

        next_frame = self.current_frame + 1
        self.ref_skel.set_positions(self.ref_motion.qs[next_frame])
        self.ref_skel.set_velocities(self.ref_motion.dqs[next_frame])

        h = self.world.time_step()
        q_des = self.ref_skel.q + action

        for i in range(self.step_per_frame):
            self.skel.set_forces(self.skel.get_spd(q_des, h, self.Kp, self.Kd))
            self.world.step()

        self.current_frame = next_frame
        self.count_frame += 1

        return tuple([self.state(), self.reward(), self.is_done(), dict()])

    def continue_from_frame(self, frame):
        self.current_frame = frame
        self.ref_skel.set_positions(self.ref_motion.qs[self.current_frame])
        skel_pelvis_offset = self.skel.joint(0).position_in_world_frame() - self.ref_skel.joint(0).position_in_world_frame()
        skel_pelvis_offset[1] = 0.
        self.ref_motion.translate_by_offset(skel_pelvis_offset)

    def reset(self):
        self.world.reset()
        self.continue_from_frame(randrange(self.motion_len-1))
        self.skel.set_positions(self.ref_motion.qs[self.current_frame])
        self.skel.set_velocities(np.asarray(self.ref_motion.dqs[self.current_frame]))
        self.count_frame = 0

        return self.state()

    def render(self, mode='human', close=False):
        return None

    def close(self):
        pass

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def flag_rsi(self, rsi=True):
        self.rsi = rsi
