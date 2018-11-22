# from rl.core import Env
import pydart2 as pydart
import numpy as np
from math import exp, pi
from PyCommon.modules.Math import mmMath as mm
from random import randrange, random
import gym
import gym.spaces
from gym.utils import seeding

from rl.pd_controller import PDController
from PyCommon.modules.Simulator import yulQpSimulator_penalty as yulqp
# from PyCommon.modules.Simulator import yulQpSimulator as yulqp
# from PyCommon.modules.Simulator import yulQpSimulator_noknifecon as yulqp
from multiprocessing import Pool

debug = False
# debug = True

def exp_reward_term(w, exp_w, v0, v1):
    norm = np.linalg.norm(v0 - v1)
    return w * exp(-exp_w * norm * norm)

class YulDartEnv(gym.Env):
    def __init__(self, env_name='spin', env_slaves=1):
        self.world = pydart.World(1. / 1200., "../data/skel/skate_model.skel")
        self.ground = pydart.World(1. / 1200., "../data/skel/ground.skel")
        # self.world.control_skel = self.world.skeletons[0]
        self.skel = self.world.skeletons[0]
        self.Kp, self.Kd = 400., 40.
        self.pdc = PDController(self.skel, self.world.time_step(), 400., 40.)

        self.env_name = env_name

        self.ref_motion = []

        if env_name == 'spin':
            with open('../data/mocap/skate_spin.txt') as f:
            # with open('data/mocap/skate_spin_cut.txt') as f:
                # lines = f.read().splitlines()
                for line in f:
                    q_temp = []
                    line = line.replace(' \n', '')
                    values = line.split(" ")
                    for a in values:
                        q_temp.append(float(a))
                    self.ref_motion.append(q_temp)

        self.ref_motion_ft = 0.0333333
        self.ref_world = pydart.World(1./1200., "../data/skel/skate_ref.skel")
        self.ref_skel = self.ref_world.skeletons[0]
        # self.step_per_frame = round((1./self.world.time_step()) / self.ref_motion.fps)
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

        self.body_num = self.skel.num_bodynodes()
        self.idx_e = [self.skel.bodynode_index('h_blade_left'), self.skel.bodynode_index('h_blade_right'),
                      self.skel.bodynode_index('h_hand_left'), self.skel.bodynode_index('h_hand_right')]
        self.body_e = list(map(self.skel.body, self.idx_e))
        self.ref_body_e = list(map(self.ref_skel.body, self.idx_e))
        self.motion_len = len(self.ref_motion)
        self.motion_time = len(self.ref_motion) / self.ref_motion_ft

        self.time_offset = 0.

        state_num = 1 + (3*3 + 4) * self.body_num
        action_num = self.skel.num_dofs() - 6

        state_high = np.array([np.finfo(np.float32).max] * state_num)
        action_high = np.array([pi*10./2.] * action_num)

        self.action_space = gym.spaces.Box(-action_high, action_high, dtype=np.float32)
        self.observation_space = gym.spaces.Box(-state_high, state_high, dtype=np.float32)

        self.viewer = None

        self.phase_frame = 0

        self.contact_force = []
        self.contactPositionLocals = []
        self.bodyIDs = []

    def get_dq(self, frame):

        prevFrame = frame-1 if frame > 0 else frame
        nextFrame = frame+1 if frame < len(self.ref_motion)-1 else frame

        return self.ref_skel.position_differences(self.ref_motion[nextFrame], self.ref_motion[prevFrame])/(2.*self.ref_motion_ft)

    def state(self):
        pelvis = self.skel.body(0)
        p_pelvis = pelvis.world_transform()[:3, 3]
        R_pelvis = pelvis.world_transform()[:3, :3]

        phase = min(1., (self.world.time() + self.time_offset)/self.motion_time)
        state = [phase]

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
        # current_frame = min(len(self.ref_motion)-1, int((self.world.time() + self.time_offset) * self.ref_motion.fps))
        current_time = self.world.time() + self.time_offset
        current_frame = int(current_time / self.ref_motion_ft)

        self.ref_skel.set_positions(self.ref_motion[current_frame])
        self.ref_skel.set_velocities(self.get_dq(current_frame))

        # p_e_hat = np.asarray([body.world_transform()[:3, 3] for body in self.ref_body_e]).flatten()
        p_e_hat_list = []
        for body in self.ref_body_e:
            if 'blade' in body.name:
                p_e_hat_list.append(body.world_transform()[:3, 3])
                p_e_hat_list[-1][1] = 0.
            else:
                p_e_hat_list.append(body.world_transform()[:3, 3])

        p_e_hat = np.asarray(p_e_hat_list).flatten()
        p_e = np.asarray([body.world_transform()[:3, 3] for body in self.body_e]).flatten()

        return exp_reward_term(self.w_p, self.exp_p, self.skel.q, self.ref_skel.q) \
              + exp_reward_term(self.w_v, self.exp_v, self.skel.dq, self.ref_skel.dq) \
              + exp_reward_term(self.w_e, self.exp_e, p_e, p_e_hat) \
              + exp_reward_term(self.w_c, self.exp_c, self.skel.com(), self.ref_skel.com())

    def is_done(self):

        frame_time = self.world.time() + self.time_offset
        frame = int(frame_time / self.ref_motion_ft)
        if self.skel.com()[1] < 0.4:
            if debug:
                print('fallen')
            return True
        elif True in np.isnan(np.asarray(self.skel.q)) or True in np.isnan(np.asarray(self.skel.dq)):
            if debug:
                print('nan')
            return True
        elif frame > len(self.ref_motion)-2:
        # elif self.world.time() + self.time_offset > self.motion_time:
            if debug:
                print('timeout')
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

        next_frame_time = self.world.time() + self.time_offset + self.world.time_step() * self.step_per_frame
        next_frame = int(next_frame_time / self.ref_motion_ft)
        if debug:
            # print(next_frame_time, next_frame, self.ref_motion_ft)
            print(next_frame)
        self.ref_skel.set_positions(self.ref_motion[next_frame])
        self.ref_skel.set_velocities(self.get_dq(next_frame))
        ndofs = self.ref_skel.ndofs
        try:
            for i in range(self.step_per_frame):
                # PD CONTROL
                h = self.world.time_step()
                # gain_value = 25.0
                # gain_value = 50.0
                gain_value = self.Kp
                Kp = np.diagflat([0.0] * 6 + [gain_value] * (ndofs - 6))
                Kd = np.diagflat([0.0] * 6 + [2. * (gain_value ** .5)] * (ndofs - 6))
                invM = np.linalg.inv(self.skel.M + Kd * h)
                p = -Kp.dot(self.skel.q - self.ref_skel.q - action + self.skel.dq * h)
                d = -Kd.dot(self.skel.dq)
                qddot = invM.dot(-self.skel.c + p + d + self.skel.constraint_forces())
                des_accel = p + d + qddot

                # YUL QP solve
                # _ddq, _tau, _bodyIDs, _contactPositions, _contactPositionLocals, _contactForces = yulqp.calc_QP(
                #     self.skel, des_accel, None, 1. / self.world.time_step())

                _ddq, _tau, _bodyIDs, _contactPositions, _contactPositionLocals, _contactForces = yulqp.calc_QP(
                    self.skel, des_accel, None, None, None, None, None, 1. / self.world.time_step())

                del self.contact_force[:]
                del self.bodyIDs[:]
                del self.contactPositionLocals[:]

                self.bodyIDs = _bodyIDs

                for i in range(len(_bodyIDs)):
                    self.skel.body(_bodyIDs[i]).add_ext_force(_contactForces[i], _contactPositionLocals[i])
                    self.contact_force.append(_contactForces[i])
                    self.contactPositionLocals.append(_contactPositionLocals[i])

                self.skel.set_forces(_tau)

                self.world.step()
        # except ZeroDivisionError:
        except ArithmeticError:
            if debug:
                print('qp error!')
            return tuple([self.state(), self.reward(), True, dict()])
        return tuple([self.state(), self.reward(), self.is_done(), dict()])

    def continue_from_now_by_phase(self, phase):
        self.phase_frame = min(round(phase * (self.motion_len-1)), len(self.ref_motion)-2)
        skel_pelvis_offset = self.skel.joint(0).position_in_world_frame() - self.ref_motion[self.phase_frame][3:6]
        skel_pelvis_offset[1] = 0.
        for i in range(len(self.ref_motion)):
            self.ref_motion[i][3:6] += skel_pelvis_offset
        self.time_offset = - self.world.time() + (self.phase_frame * self.ref_motion_ft)

    def reset(self):
        """
        Resets the state of the environment and returns an initial observation.
        # Returns
            observation (object): The initial observation of the space. Initial reward is assumed to be 0.
        """
        self.world.reset()
        self.continue_from_now_by_phase(random() if self.rsi else 0.)
        self.skel.set_positions(self.ref_motion[self.phase_frame])
        dq = self.get_dq(self.phase_frame)
        self.skel.set_velocities(dq)

        return self.state()

    def render(self, mode='human', close=False):
        """Renders the environment.
        The set of supported modes varies per environment. (And some
        environments do not support rendering at all.)
        # Arguments
            mode (str): The mode to render with.
            close (bool): Close all open renderings.
        """
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(500,500)
            self.viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)
            # rod = rendering.make_capsule(1, .2)
            # rod.set_color(.8, .3, .3)
            # rod.add_attr(self.pole_transform)
            # self.viewer.add_geom(rod)
            self.body_transform = list()
            self.ref_body_transform = list()
            for i in range(self.body_num):
                axle = rendering.make_circle(.05)
                axle.set_color(0, 0, 0)
                self.body_transform.append(rendering.Transform())
                axle.add_attr(self.body_transform[i])
                self.viewer.add_geom(axle)

            for i in range(self.body_num):
                axle = rendering.make_circle(.05)
                axle.set_color(1, 0, 0)
                self.ref_body_transform.append(rendering.Transform())
                axle.add_attr(self.ref_body_transform[i])
                self.viewer.add_geom(axle)

        for i in range(self.body_num):
            self.body_transform[i].set_translation(self.skel.body(i).world_transform()[:3, 3][0]-1., self.skel.body(i).world_transform()[:3, 3][1])
            self.ref_body_transform[i].set_translation(self.ref_skel.body(i).world_transform()[:3, 3][0]-1., self.ref_skel.body(i).world_transform()[:3, 3][1])

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

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