# from rl.core import Agent
import numpy as np

import pydart2 as pydart
from rl.dart_env import YulDartEnv

from collections import namedtuple
from collections import deque
from itertools import count
import random
import time
import os

import torch
from torch import nn, optim
import torch.nn.functional as F
import torchvision.transforms as T

MultiVariateNormal = torch.distributions.Normal
temp = MultiVariateNormal.log_prob
MultiVariateNormal.log_prob = lambda self, val: temp(self, val).sum(-1, keepdim=True)

temp2 = MultiVariateNormal.entropy
MultiVariateNormal.entropy = lambda self: temp2(self).sum(-1)
MultiVariateNormal.mode = lambda self: self.mean


class Model(nn.Module):
    def __init__(self, num_states, num_actions):
        super(Model, self).__init__()

        hidden_layer_size1 = 64
        hidden_layer_size2 = 32

        '''Policy Mean'''
        self.policy_fc1 = nn.Linear(num_states		  , hidden_layer_size1)
        self.policy_fc2 = nn.Linear(hidden_layer_size1, hidden_layer_size2)
        self.policy_fc3 = nn.Linear(hidden_layer_size2, num_actions)
        '''Policy Distributions'''
        self.log_std = nn.Parameter(torch.zeros(num_actions))        # add little noise
        # self.log_std = nn.Parameter(0.1 * torch.ones(num_actions)) # add large noise

        '''Value'''
        self.value_fc1 = nn.Linear(num_states		  ,hidden_layer_size1)
        self.value_fc2 = nn.Linear(hidden_layer_size1 ,hidden_layer_size2)
        self.value_fc3 = nn.Linear(hidden_layer_size2 ,1)

        self.initParameters()

    def initParameters(self):
        '''Policy'''
        if self.policy_fc1.bias is not None:
            self.policy_fc1.bias.data.zero_()

        if self.policy_fc2.bias is not None:
            self.policy_fc2.bias.data.zero_()

        if self.policy_fc3.bias is not None:
            self.policy_fc3.bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.policy_fc1.weight)
        torch.nn.init.xavier_uniform_(self.policy_fc2.weight)
        torch.nn.init.xavier_uniform_(self.policy_fc3.weight)
        '''Value'''
        if self.value_fc1.bias is not None:
            self.value_fc1.bias.data.zero_()

        if self.value_fc2.bias is not None:
            self.value_fc2.bias.data.zero_()

        if self.value_fc3.bias is not None:
            self.value_fc3.bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.value_fc1.weight)
        torch.nn.init.xavier_uniform_(self.value_fc2.weight)
        torch.nn.init.xavier_uniform_(self.value_fc3.weight)

    def forward(self, x):
        '''Policy'''
        p_mean = F.relu(self.policy_fc1(x))
        p_mean = F.relu(self.policy_fc2(p_mean))
        p_mean = self.policy_fc3(p_mean)

        p = MultiVariateNormal(p_mean, self.log_std.exp())
        '''Value'''
        v = F.relu(self.value_fc1(x))
        v = F.relu(self.value_fc2(v))
        v = self.value_fc3(v)

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


class PPO(object):
    def __init__(self, env_name, num_slaves=1, eval_print=True, eval_log=True, visualize_only=False):
        np.random.seed(seed=int(time.time()))
        self.env_name = env_name

        self.env = YulDartEnv(env_name)

        self.num_slaves = num_slaves
        self.num_state = self.env.observation_space.shape[0]
        self.num_action = self.env.action_space.shape[0]
        self.num_epochs = 10
        self.num_evaluation = 0
        self.num_training = 0

        self.gamma = 0.99
        self.lb = 0.95
        self.clip_ratio = 0.2

        self.buffer_size = 2048
        self.batch_size = 128
        self.replay_buffer = ReplayBuffer(10000)

        self.total_episodes = []

        self.model = Model(self.num_state, self.num_action).float()
        self.optimizer = optim.Adam(self.model.parameters(), lr=7E-4)
        self.w_entropy = 0.0

        self.saved = False
        self.save_directory = self.env_name + '_' + 'model_'+time.strftime("%m%d%H%M") + '/'
        if not self.saved and not os.path.exists(self.save_directory) and not visualize_only:
            os.makedirs(self.save_directory)
            self.saved = True

        self.log_file = None
        if not visualize_only:
            self.log_file = open(self.save_directory + 'log.txt', 'w')
        self.eval_print = eval_print
        self.eval_log = eval_log

    def SaveModel(self):
        torch.save(self.model.state_dict(), self.save_directory + str(self.num_evaluation) + '.pt')

    def LoadModel(self, model_path):
        self.model.load_state_dict(torch.load(model_path))

    def ComputeTDandGAE(self):
        self.replay_buffer.clear()
        for epi in self.total_episodes:
            data = epi.get_data()
            size = len(data)
            states, actions, rewards, values, logprobs = zip(*data)

            values = np.concatenate((values, np.zeros(1)), axis=0)
            advantages = np.zeros(size)
            ad_t = 0

            for i in reversed(range(len(data))):
                delta = rewards[i] + values[i + 1] * self.gamma - values[i]
                ad_t = delta + self.gamma * self.lb * ad_t
                advantages[i] = ad_t

            TD = values[:size] + advantages
            for i in range(size):
                self.replay_buffer.push(states[i], actions[i], logprobs[i], TD[i], advantages[i])

    def GenerateTransitions(self):
        del self.total_episodes[:]
        states = [None] * self.num_slaves
        actions = [None] * self.num_slaves
        rewards = [None] * self.num_slaves
        states_next = [None] * self.num_slaves
        episodes = [None] * self.num_slaves
        for j in range(self.num_slaves):
            episodes[j] = EpisodeBuffer()

        self.env.Resets(True)

        states = self.env.GetStates()

        local_step = 0
        terminated = [False] * self.num_slaves
        # print('Generate Transtions...')

        percent = 0
        while True:
            # new_percent = local_step*10//self.buffer_size
            # if (new_percent == percent) is not True:
            # percent = new_percent
            # print('{}0%'.format(percent))
            a_dist, v = self.model(torch.tensor(states).float())
            actions = a_dist.sample().detach().numpy()
            logprobs = a_dist.log_prob(torch.tensor(actions).float()).detach().numpy().reshape(-1)
            values = v.detach().numpy().reshape(-1)

            self.env.Steps(actions)
            for j in range(self.num_slaves):
                if terminated[j]:
                    continue

                nan_occur = False
                if np.any(np.isnan(states[j])) or np.any(np.isnan(actions[j])):
                    nan_occur = True
                else:
                    rewards[j] = self.env.GetReward(j)
                    episodes[j].push(states[j], actions[j], rewards[j], values[j], logprobs[j])
                    local_step += 1

                # if episode is terminated
                if self.env.IsTerminalState(j) or (nan_occur is True):
                    # push episodes
                    self.total_episodes.append(episodes[j])

                    # if data limit is exceeded, stop simulations
                    if local_step < self.buffer_size:
                        episodes[j] = EpisodeBuffer()
                        self.env.Reset(True, j)
                    else:
                        terminated[j] = True
            if local_step >= self.buffer_size:
                all_terminated = True
                for j in range(self.num_slaves):
                    if terminated[j] is False:
                        all_terminated = False

                if all_terminated is True:
                    break

            # update states
            states = self.env.GetStates()

    # print('Done!')
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
        is_multi = 'multi' in self.env_name
        self.num_evaluation += 1
        total_reward = 0
        total_step = 0
        for motion_idx in range(1 if not is_multi else len(self.env.ref_motions)):
            if is_multi:
                self.env.specify_motion_num(motion_idx)
            self.env.Resets(True)
            self.env.Reset(False, 0)
            states = self.env.GetStates()
            for j in range(len(states)):
                if np.any(np.isnan(states[j])):
                    self.print("state warning!!!!!!!! start")

            for t in count():
                action_dist, _ = self.model(torch.tensor(states).float())
                actions = action_dist.loc.detach().numpy()
                for j in range(len(actions)):
                    if np.any(np.isnan(actions[j])):
                        self.print("action warning!!!!!!!!" + str(t))

                self.env.Steps(actions)

                for j in range(len(states)):
                    if np.any(np.isnan(states[j])):
                        self.print("state warning!!!!!!!!"+str(t))

                for j in range(self.num_slaves):
                    if self.env.IsTerminalState(j) is False:
                        total_step += 1
                        total_reward += self.env.GetReward(j)
                states = self.env.GetStates()
                if all(self.env.IsTerminalStates()):
                    break
            if is_multi:
                self.env.specify_motion_num()
        self.print('noise : {:.3f}'.format(self.model.log_std.exp().mean()))
        if total_step is not 0:
            self.print('Epi reward : {:.2f}, Step reward : {:.2f} Total step : {}'
                  .format(total_reward / self.num_slaves, total_reward / total_step, total_step))
        else:
            self.print('bad..')
        return total_reward / self.num_slaves, total_step / self.num_slaves

    def print(self, s):
        if self.eval_print:
            print(s)
        if self.eval_log:
            self.log_file.write(s+"\n")

import matplotlib
import matplotlib.pyplot as plt
plt.ion()

def Plot(y,title,num_fig=1,ylim=True):
    global glob_count_for_plt

    plt.figure(num_fig)
    plt.clf()
    plt.title(title)
    plt.plot(y)
    plt.show()
    if ylim:
        plt.ylim([0,1])
    plt.pause(0.001)

import argparse


if __name__ == "__main__":
    import sys
    pydart.init()
    tic = time.time()
    ppo = None  # type: PPO
    if len(sys.argv) < 2:
        ppo = PPO('spin', 1)
    else:
        ppo = PPO(sys.argv[1], 1)

    if len(sys.argv) > 2:
        print("load {}".format(sys.argv[2]))
        ppo.LoadModel(sys.argv[2])

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

    for i in range(50000):
        ppo.Train()
        print('# {}'.format(i+1))
        ppo.log_file.write('# {}'.format(i+1) + "\n")
        reward, step = ppo.Evaluate()
        rewards.append(reward)
        steps.append(step)
        if i % 10 == 0 or max_avg_steps < step:
            ppo.SaveModel()
            if max_avg_steps < step:
                max_avg_steps = step

        Plot(np.asarray(rewards),'reward',1,False)
        print("Elapsed time : {:.2f}s".format(time.time() - tic))
        ppo.log_file.write("Elapsed time : {:.2f}s".format(time.time() - tic) + "\n")
        ppo.log_file.flush()