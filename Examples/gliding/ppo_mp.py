import numpy as np

import pydart2 as pydart
from Examples.gliding.skate_env_using_cma_result import SkateDartEnv

from collections import namedtuple
from collections import deque
import random
import time
import os

from multiprocessing import Process, Pipe

import torch
from torch import nn, optim
import torch.nn.functional as F

MultiVariateNormal = torch.distributions.Normal
temp = MultiVariateNormal.log_prob
MultiVariateNormal.log_prob = lambda self, val: temp(self, val).sum(-1, keepdim=True)

temp2 = MultiVariateNormal.entropy
MultiVariateNormal.entropy = lambda self: temp2(self).sum(-1)
MultiVariateNormal.mode = lambda self: self.mean


class Model(nn.Module):
    def __init__(self, num_states, num_actions):
        super(Model, self).__init__()

        hidden_layer_size1 = 128
        hidden_layer_size2 = 128
        hidden_layer_size3 = 128

        '''Policy Mean'''
        self.policy_fc1 = nn.Linear(num_states		  , hidden_layer_size1)
        self.policy_fc2 = nn.Linear(hidden_layer_size1, hidden_layer_size2)
        self.policy_fc3 = nn.Linear(hidden_layer_size2, hidden_layer_size3)
        self.policy_fc4 = nn.Linear(hidden_layer_size3, num_actions)
        '''Policy Distributions'''
        self.log_std = nn.Parameter(torch.zeros(num_actions))

        '''Value'''
        self.value_fc1 = nn.Linear(num_states		 , hidden_layer_size1)
        self.value_fc2 = nn.Linear(hidden_layer_size1, hidden_layer_size2)
        self.value_fc3 = nn.Linear(hidden_layer_size2, hidden_layer_size3)
        self.value_fc4 = nn.Linear(hidden_layer_size3, 1)

        self.initParameters()

    def initParameters(self):
        '''Policy'''
        if self.policy_fc1.bias is not None:
            self.policy_fc1.bias.data.zero_()
        if self.policy_fc2.bias is not None:
            self.policy_fc2.bias.data.zero_()
        if self.policy_fc3.bias is not None:
            self.policy_fc3.bias.data.zero_()
        if self.policy_fc4.bias is not None:
            self.policy_fc4.bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.policy_fc1.weight)
        torch.nn.init.xavier_uniform_(self.policy_fc2.weight)
        torch.nn.init.xavier_uniform_(self.policy_fc3.weight)
        torch.nn.init.xavier_uniform_(self.policy_fc4.weight)
        '''Value'''
        if self.value_fc1.bias is not None:
            self.value_fc1.bias.data.zero_()

        if self.value_fc2.bias is not None:
            self.value_fc2.bias.data.zero_()

        if self.value_fc3.bias is not None:
            self.value_fc3.bias.data.zero_()

        if self.value_fc4.bias is not None:
            self.value_fc4.bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.value_fc1.weight)
        torch.nn.init.xavier_uniform_(self.value_fc2.weight)
        torch.nn.init.xavier_uniform_(self.value_fc3.weight)
        torch.nn.init.xavier_uniform_(self.value_fc4.weight)

    def forward(self, x):
        '''Policy'''
        p_mean = F.relu(self.policy_fc1(x))
        p_mean = F.relu(self.policy_fc2(p_mean))
        p_mean = F.relu(self.policy_fc3(p_mean))
        p_mean = self.policy_fc4(p_mean)

        p = MultiVariateNormal(p_mean, self.log_std.exp())
        '''Value'''
        v = F.relu(self.value_fc1(x))
        v = F.relu(self.value_fc2(v))
        v = F.relu(self.value_fc3(v))
        v = self.value_fc4(v)

        return p,v


Episode = namedtuple('Episode', ('s', 'a', 'r', 'value', 'logprob'))


class EpisodeBuffer(object):
    def __init__(self):
        self.data = []

    def push(self, *args):
        self.data.append(Episode(*args))

    def get_data(self):
        return self.data


Transition = namedtuple('Transition', ('s', 'a', 'logprob', 'TD', 'GAE'))


class ReplayBuffer(object):
    def __init__(self, buff_size = 10000):
        super(ReplayBuffer, self).__init__()
        self.buffer = deque(maxlen=buff_size)

    def sample(self, batch_size):
        index_buffer = [i for i in range(len(self.buffer))]
        indices = random.sample(index_buffer, batch_size)
        return indices, self.np_buffer[indices]

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def clear(self):
        self.buffer.clear()


def worker(rnn_len, proc_num, state_sender, result_sender, action_receiver, reset_receiver, motion_receiver):
    """

    :type rnn_len: int
    :type proc_num: int
    :type result_sender: Connection
    :type state_sender: Connection
    :type action_receiver: Connection
    :type reset_receiver: Connection
    :type motion_receiver: Connection
    :return:
    """

    # reset variable
    # 0 : go on (no reset)
    # 1 : soft reset ( reset world )
    # 2 : hard reset ( new world )

    env = SkateDartEnv()

    state = None
    while True:
        reset_flag = reset_receiver.recv()
        if reset_flag == 1:
            state = env.reset()
            # state = env.hard_reset()
        elif reset_flag == 2:
            state = env.hard_reset()

        state_sender.send(state)
        action = action_receiver.recv()
        state, reward, is_done, _ = env.step(action)
        result_sender.send((reward, is_done, proc_num))


class PPO(object):
    def __init__(self, env_name, num_slaves=1, eval_print=True, eval_log=True, visualize_only=False):
        np.random.seed(seed=int(time.time()))
        self.env_name = env_name
        self.env = SkateDartEnv()
        self.num_slaves = num_slaves
        self.num_state = self.env.observation_space.shape[0]
        self.num_action = self.env.action_space.shape[0]
        self.num_epochs = 10
        self.num_evaluation = 0
        self.num_training = 0

        self.gamma = 0.99
        self.lb = 0.95
        self.clip_ratio = 0.2

        # self.buffer_size = 2048
        # self.batch_size = 128
        # self.replay_buffer = ReplayBuffer(10000)
        self.buffer_size = 4096
        self.batch_size = 256
        self.replay_buffer = ReplayBuffer(100000)

        self.total_episodes = []

        self.model = Model(self.num_state, self.num_action).float()
        # self.optimizer = optim.Adam(self.model.parameters(), lr=7E-4)
        self.optimizer = optim.Adam(self.model.parameters(), lr=2E-4)
        # self.optimizer_old = optim.Adam(self.model.parameters(), lr=1E-4)
        self.w_entropy = 0.0

        self.sum_return = 0.
        self.num_episode = 0

        self.saved = False
        self.save_directory = self.env_name + '_' + 'model_'+time.strftime("%Y%m%d%H%M") + '/'
        if not self.saved and not os.path.exists(self.save_directory) and not visualize_only:
            os.makedirs(self.save_directory)
            self.saved = True

        self.log_file = None
        if not visualize_only:
            self.log_file = open(self.save_directory + 'log.txt', 'w')
        self.eval_print = eval_print
        self.eval_log = eval_log

        self.state_sender = []  # type: list[Connection]
        self.result_sender = []  # type: list[Connection]
        self.state_receiver = []  # type: list[Connection]
        self.result_receiver = []  # type: list[Connection]
        self.action_sender = []  # type: list[Connection]
        self.reset_sender = []  # type: list[Connection]
        self.motion_sender = []  # type: list[Connection]
        self.envs = []  # type: list[Process]

        self.init_envs()

    def init_envs(self):
        for slave_idx in range(self.num_slaves):
            s_s, s_r = Pipe()
            r_s, r_r = Pipe()
            a_s, a_r = Pipe()
            reset_s, reset_r = Pipe()
            motion_s, motion_r = Pipe()
            p = Process(target=worker, args=(-1, slave_idx, s_s, r_s, a_r, reset_r, motion_r))
            self.state_sender.append(s_s)
            self.result_sender.append(r_s)
            self.state_receiver.append(s_r)
            self.result_receiver.append(r_r)
            self.action_sender.append(a_s)
            self.reset_sender.append(reset_s)
            self.motion_sender.append(motion_s)
            self.envs.append(p)
            p.start()

    def reinit_env(self, slave_idx):
        print('reinit_env: ', slave_idx)
        if self.envs[slave_idx].is_alive():
            self.envs[slave_idx].terminate()
        s_s, s_r = Pipe()
        r_s, r_r = Pipe()
        a_s, a_r = Pipe()
        reset_s, reset_r = Pipe()
        motion_s, motion_r = Pipe()
        p = Process(target=worker, args=(-1, slave_idx, s_s, r_s, a_r, reset_r, motion_r))
        self.state_sender[slave_idx] = s_s
        self.result_sender[slave_idx] = r_s
        self.state_receiver[slave_idx] = s_r
        self.result_receiver[slave_idx] = r_r
        self.action_sender[slave_idx] = a_s
        self.reset_sender[slave_idx] = reset_s
        self.motion_sender[slave_idx] = motion_s
        self.envs[slave_idx] = p
        p.start()

    def envs_get_states(self, terminated):
        states = []
        for recv_idx in range(len(self.state_receiver)):
            if terminated[recv_idx]:
                states.append([0.] * self.num_state)
            else:
                states.append(self.state_receiver[recv_idx].recv())
        return states

    def envs_send_actions(self, actions, terminated):
        for i in range(len(self.action_sender)):
            if not terminated[i]:
                self.action_sender[i].send(actions[i])

    def envs_get_status(self, terminated):
        status = [None for _ in range(self.num_slaves)]  # type: list[tuple[float|None, bool]]

        alive_receivers = [self.result_receiver[recv_idx] for recv_idx, x in enumerate(terminated) if not x]

        for receiver in alive_receivers:
            if receiver.poll(20):
                recv_data = receiver.recv()
                status[recv_data[2]] = (recv_data[0], recv_data[1])

        for j in range(len(status)):
            if terminated[j]:
                status[j] = (0., True)
            elif status[j] is None:
                # assertion error, reinit in GenerateTransition
                status[j] = (None, True)

        # status = []
        # for recv_idx in range(len(self.result_receiver)):
        #     if terminated[recv_idx]:
        #         status.append((0., True))
        #     else:
        #         status.append(self.result_receiver[recv_idx].recv())
        return zip(*status)

    def envs_resets(self, reset_flag):
        for i in range(len(self.reset_sender)):
            self.reset_sender[i].send(reset_flag)

    def envs_reset(self, i, reset_flag):
        self.reset_sender[i].send(reset_flag)

    def SaveModel(self):
        torch.save(self.model.state_dict(), self.save_directory + str(self.num_evaluation) + '.pt')

    def LoadModel(self, model_path):
        self.model.load_state_dict(torch.load(model_path))

    def ComputeTDandGAE(self):
            self.replay_buffer.clear()
            self.sum_return = 0.
            for epi in self.total_episodes:
                data = epi.get_data()
                size = len(data)
                states, actions, rewards, values, logprobs = zip(*data)

                values = np.concatenate((values, np.zeros(1)), axis=0)
                advantages = np.zeros(size)
                ad_t = 0

                for i in reversed(range(len(data))):
                    self.sum_return += rewards[i]
                    delta = rewards[i] + values[i + 1] * self.gamma - values[i]
                    ad_t = delta + self.gamma * self.lb * ad_t
                    advantages[i] = ad_t

                TD = values[:size] + advantages
                for i in range(size):
                    self.replay_buffer.push(states[i], actions[i], logprobs[i], TD[i], advantages[i])
            self.num_episode = len(self.total_episodes)

    def GenerateTransitions(self):
        del self.total_episodes[:]
        episodes = [EpisodeBuffer() for _ in range(self.num_slaves)]

        self.envs_resets(1)
        # self.env.Resets(False)

        local_step = 0
        terminated = [False] * self.num_slaves
        # print('Generate Transtions...')

        percent = 0
        while True:
            # update states
            states = self.envs_get_states(terminated)
            # states = self.env.GetStates()

            a_dist, v = self.model(torch.tensor(states).float())
            actions = a_dist.sample().detach().numpy()
            logprobs = a_dist.log_prob(torch.tensor(actions).float()).detach().numpy().reshape(-1)
            values = v.detach().numpy().reshape(-1)

            self.envs_send_actions(actions, terminated)
            rewards, is_done = self.envs_get_status(terminated)
            # self.env.Steps(actions)
            # rewards, is_done = self.env.GetRewards(), self.env.IsTerminalStates()
            for j in range(self.num_slaves):
                if terminated[j]:
                    continue

                assertion_occur = rewards[j] is None
                nan_occur = np.any(np.isnan(states[j])) or np.any(np.isnan(actions[j])) or (rewards[j] is not None and np.isnan(rewards[j]))
                if not nan_occur and not assertion_occur:
                    episodes[j].push(states[j], actions[j], rewards[j], values[j], logprobs[j])
                    local_step += 1

                if assertion_occur:
                    self.reinit_env(j)

                # if episode is terminated
                if is_done[j] or nan_occur:
                    if not is_done[j] and nan_occur:
                        self.print('network parameter nan')

                    # push episodes
                    if len(episodes[j].data) > 0:
                        self.total_episodes.append(episodes[j])

                    # if data limit is exceeded, stop simulations
                    if local_step < self.buffer_size:
                        episodes[j] = EpisodeBuffer()
                        if nan_occur:
                            self.envs_reset(j, 2)
                        else:
                            self.envs_reset(j, 1)
                        # self.env.Reset(True, j)
                    else:
                        terminated[j] = True
                else:
                    self.envs_reset(j, 0)
                    pass

            if local_step >= self.buffer_size:
                if all(terminated):
                    break

    def OptimizeModel(self):
        # print('Optimize Model...')
        self.ComputeTDandGAE()
        all_transitions = np.array(self.replay_buffer.buffer)

        for _ in range(self.num_epochs):
            np.random.shuffle(all_transitions)
            for i in range(len(all_transitions) // self.batch_size):
                transitions = all_transitions[i * self.batch_size:(i + 1) * self.batch_size]
                batch = Transition(*zip(*transitions))

                stack_s = np.vstack(batch.s).astype(np.float32)
                stack_a = np.vstack(batch.a).astype(np.float32)
                stack_lp = np.vstack(batch.logprob).astype(np.float32)
                stack_td = np.vstack(batch.TD).astype(np.float32)
                stack_gae = np.vstack(batch.GAE).astype(np.float32)

                a_dist, v = self.model(torch.tensor(stack_s).float())
                '''Critic Loss'''
                loss_critic = ((v - torch.tensor(stack_td).float()).pow(2)).mean()

                '''Actor Loss'''
                ratio = torch.exp(a_dist.log_prob(torch.tensor(stack_a).float()) - torch.tensor(stack_lp).float())
                stack_gae = (stack_gae - stack_gae.mean()) / (stack_gae.std() + 1E-5)
                surrogate1 = ratio * torch.tensor(stack_gae).float()
                surrogate2 = torch.clamp(ratio, min=1.0 - self.clip_ratio, max=1.0 + self.clip_ratio) * torch.tensor(stack_gae).float()
                loss_actor = - torch.min(surrogate1, surrogate2).mean()

                '''Entropy Loss'''
                loss_entropy = - self.w_entropy * a_dist.entropy().mean()

                loss = loss_critic + loss_actor + loss_entropy
                # loss = loss_critic + loss_actor
                self.optimizer.zero_grad()
                loss.backward(retain_graph=True)
                for param in self.model.parameters():
                    param.grad.data.clamp_(-0.5, 0.5)
                self.optimizer.step()

    # print('Done!')
    def Train(self):
        self.GenerateTransitions()
        self.OptimizeModel()
        self.num_training += 1

    def Evaluate(self):
        self.num_evaluation += 1
        self.print('noise : {:.4f}'.format(self.model.log_std.exp().mean()))
        self.print('Avg return : {:.4f}'.format(self.sum_return / self.num_episode))
        return self.sum_return/self.num_episode, 0

    def print(self, s):
        if self.eval_print:
            print(s)
        if self.eval_log:
            self.log_file.write(s+"\n")


if __name__ == "__main__":
    import sys
    pydart.init()
    tic = time.time()
    ppo = None  # type: PPO
    if len(sys.argv) < 2:
        ppo = PPO('gliding', 4)
    else:
        ppo = PPO(sys.argv[1], int(sys.argv[2]))

    if len(sys.argv) > 3:
        print("load {}".format(sys.argv[3]))
        ppo.LoadModel(sys.argv[3])

    # parser = argparse.ArgumentParser()
    # parser.add_argument('-m','--model',help='actor model directory')
    # args =parser.parse_args()
    # if args.model is not None:
    #     print("load {}".format(args.model))
    #     ppo.LoadModel(args.model)
    rewards = []
    steps = []
    # print('num states: {}, num actions: {}'.format(ppo.env.GetNumState(),ppo.env.GetNumAction()))

    max_avg_steps = 0
    max_avg_reward = 0.

    for i in range(50000):
        ppo.Train()
        print('# {}'.format(i+1))
        ppo.log_file.write('# {}'.format(i+1) + "\n")
        reward, step = ppo.Evaluate()
        rewards.append(reward)
        steps.append(step)
        if max_avg_steps < step or max_avg_reward < reward:
            ppo.SaveModel()
            if max_avg_steps < step:
                max_avg_steps = step
            if max_avg_reward < reward:
                max_avg_reward = reward

        # Plot(np.asarray(rewards),'reward',1,False)
        print("Elapsed time : {:.2f}s".format(time.time() - tic))
        ppo.log_file.write("Elapsed time : {:.2f}s".format(time.time() - tic) + "\n")
        ppo.log_file.flush()
