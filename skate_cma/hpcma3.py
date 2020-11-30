import os
import cma
import pickle
import time
import numpy as np
from multiprocessing import Pool

from skate_cma.skate_env3 import SkateDartEnv
from skate_cma.PenaltyType import PenaltyType


def f(x):
    env = SkateDartEnv(x[0])
    x0 = x[2]
    q0 = x[3]
    dq0 = x[4]
    duration0, duration1 = x[5], x[6]
    option0, option1, weight = x[7], x[8], x[9]
    is_last_state = x[10]
    solution = np.split(x[1], duration0+duration1+1)
    """
    x[0] : env_name
    x[1] : tested solution
    x[2] : solution offset(original pose)
    
    """

    obj_0 = 0.
    obj_1 = 0.
    env.update_ref_states(solution[:duration0+1], x0[:duration0+1], q0, dq0, x[5])
    env.set_penalty(option0, weight)
    env.save_com_init()
    for bs_idx in range(3*duration0):
        env.step(bs_idx/3)
        obj_0 += env.penalty_instant()
    obj_0 /= 3*duration0
    obj_0 += env.penalty_final()
    q = np.asarray(env.skel.q)
    dq = np.asarray(env.skel.dq)

    if not is_last_state:
        env.update_ref_states(solution[duration0:], x0[duration0:], q, dq, x[6])
        env.set_penalty(option1, weight)
        env.save_com_init()
        for bs_idx in range(3*duration1):
            env.step(bs_idx/3)
            obj_1 += env.penalty_instant()
        obj_1 /= 3*duration1
        obj_1 += env.penalty_final()

    obj_reg = weight[PenaltyType.REGULARIZE] * np.sum(np.square(np.asarray(x[1]))) / (duration0+duration1)

    return obj_0 + obj_1 + obj_reg


class HpCma(object):
    def __init__(self, env_name, num_slaves=1, sigma=.5, max_time=10., start_state_num=0, start_state_sol_dir='init_solution', cma_timeout=1800):
        self.env_name = env_name
        self.env = SkateDartEnv(env_name)

        self.max_time = max_time

        with open(env_name + '.skkey', 'rb') as skkey_file:
            self.skkey_states = pickle.load(skkey_file)

        self.log_dir = self.env_name + '_' + 'model_'+time.strftime("%Y%m%d%H%M") + '/'
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.angles = []
        count = 0
        state = self.skkey_states[0]
        self.state_duration = []
        while count < int(self.max_time * 10.):
            state_count = 0
            for _ in range(int(state.dt*10.)):
                self.angles.append(state.angles[6:])
                state_count += 1
                count += 1
                if count == int(self.max_time * 10.):
                    break
            self.state_duration.append(state_count)
            state = state.get_next()
        self.angles.append(self.angles[-1])

        self.dof = self.env.skel.num_dofs() - 6

        self.num_slaves = num_slaves
        self.sigma = sigma

        self.solutions = []
        self.start_state_num = start_state_num
        self.start_state_sol_dir = start_state_sol_dir

        self.cma_timeout = cma_timeout

        self.init_q = None
        self.init_dq = np.zeros_like(self.skkey_states[0].angles.copy())

        print('Start CMA')
        print('Motion name: ', self.env_name)
        print('Max motion time: ', self.max_time)
        print('Parallel with process #: ', self.num_slaves)
        print('CMA initial sigma: ', self.sigma)
        print('CMA time out: ', self.cma_timeout)
        print('Start motion state #:', self.start_state_num)
        if self.start_state_num > 0:
            print('Load previous solution from ', self.start_state_sol_dir)

    def set_init_dq(self, dq):
        assert(len(self.init_dq) == len(dq))
        self.init_dq = np.asarray(dq)

    def set_init_q(self, q):
        assert(len(self.init_dq) == len(q))
        self.init_q = np.asarray(q)

    def save_solution(self, i, solution):
        filename = self.log_dir + 'xbest.skcma'
        with open(filename, 'a') as file:
            strs = list(map(str, solution))
            strs.insert(0, str(i))
            file.write(' '.join(strs))
            file.write('\n')

    def run(self):
        q = self.skkey_states[0].angles.copy() if self.init_q is None else self.init_q
        dq = self.init_dq
        x0t = np.zeros_like(q[6:])

        self.env.reset()
        if self.start_state_num > 0:
            file_path = self.start_state_sol_dir + 'xbest.skcma'
            with open(file_path, 'r') as file:
                lines = file.read().splitlines()
                state_list_in_file = list(map(int, [line.split()[0] for line in lines]))

                for i in range(self.start_state_num):
                    solutions = [x0t]
                    state_index_in_file = state_list_in_file.index(i)
                    x_state = np.asarray(list(map(float, lines[state_index_in_file].split()[1:])))
                    solutions.extend(np.split(x_state, self.state_duration[i]))

                    self.save_solution(i, x_state)

                    x0 = self.angles[sum(self.state_duration[:i]):sum(self.state_duration[:i+1])+1]
                    self.env.update_ref_states(solutions, x0, q, dq, self.state_duration[i])
                    for bs_idx in range(3*self.state_duration[i]):
                        self.env.step(bs_idx/3)

                    x0t = solutions[-1]
                    q = np.asarray(self.env.skel.q)
                    dq = np.asarray(self.env.skel.dq)

        for i in range(self.start_state_num, len(self.state_duration)-1):
            penalty_option0 = [None] * len(PenaltyType)
            penalty_option1 = [None] * len(PenaltyType)
            penalty_weight = [1.] * len(PenaltyType)
            penalty_weight[PenaltyType.TORQUE] = 0.
            self.objective(i, penalty_option0, penalty_option1, penalty_weight)

            x0 = self.angles[sum(self.state_duration[:i]):sum(self.state_duration[:i+2])+1]

            solution = self.run_one_window(i,
                                           x0t, x0,
                                           q, dq,
                                           self.state_duration[i], self.state_duration[i+1],
                                           penalty_option0, penalty_option1, penalty_weight
                                           )

            self.env.reset()
            extended_solution = np.split(np.hstack((np.asarray(x0t), np.asarray(solution))), self.state_duration[i]+self.state_duration[i+1]+1)
            self.env.update_ref_states(extended_solution[:self.state_duration[i]+1], x0[:self.state_duration[i]+1], q, dq, self.state_duration[i])
            for bs_idx in range(3*self.state_duration[i]):
                self.env.step(bs_idx/3)
            q = np.asarray(self.env.skel.q)
            dq = np.asarray(self.env.skel.dq)

            solutions = np.split(solution, self.state_duration[i]+self.state_duration[i+1])
            self.solutions.extend(solutions[:self.state_duration[i]])
            x0t = self.solutions[-1]

            self.save_solution(i, np.hstack(np.split(solution, self.state_duration[i]+self.state_duration[i+1])[:self.state_duration[i]]))

        if True:
            # last window
            i = len(self.state_duration)-1
            penalty_option0 = [None] * len(PenaltyType)
            penalty_option1 = [None] * len(PenaltyType)
            penalty_weight = [1.] * len(PenaltyType)
            self.objective(i, penalty_option0, penalty_option1, penalty_weight)

            x0 = self.angles[sum(self.state_duration[:i]):]

            solution = self.run_one_window(i,
                                           x0t, x0,
                                           q, dq,
                                           self.state_duration[i], 0,
                                           penalty_option0, penalty_option1, penalty_weight,
                                           True
                                           )

            solutions = np.split(solution, self.state_duration[i])
            self.solutions.extend(solutions[:self.state_duration[i]])
            self.save_solution(i, solution)

    def run_one_window(self, idx, x0t, x0, q0, dq0, duration0, duration1, option0, option1, weight, is_last_state=False):
        """

        :param idx:
        :param x0t: window terminal solution from previous idx optimization
        :param x0: solution offset
        :param q0: starting position
        :param dq0: starting velocity
        :param duration0:
        :param duration1:
        :param option0:
        :param option1:
        :param weight:
        :param is_last_state:
        :return:
        """
        es = cma.CMAEvolutionStrategy(((duration0+duration1) * self.dof) * [0.], self.sigma, {'tolstagnation': 1000, 'timeout': self.cma_timeout})
        es.logger.save_to(self.log_dir+str(idx)+'_', True)
        while not es.stop():
            solutions = es.ask()
            extended_solutions = [np.hstack((np.asarray(x0t), np.asarray(solution))) for solution in solutions]
            env_names = [self.env_name for _ in range(len(solutions))]
            x0s = [x0 for _ in range(len(solutions))]
            qs = [q0 for _ in range(len(solutions))]
            dqs = [dq0 for _ in range(len(solutions))]
            duration0s = [duration0 for _ in range(len(solutions))]
            duration1s = [duration1 for _ in range(len(solutions))]
            option0s = [option0 for _ in range(len(solutions))]
            option1s = [option1 for _ in range(len(solutions))]
            weights = [weight for _ in range(len(solutions))]
            is_last_states = [is_last_state for _ in range(len(solutions))]
            with Pool(self.num_slaves) as p:
                objs = p.map(f, list(zip(env_names, extended_solutions, x0s, qs, dqs, duration0s, duration1s, option0s, option1s, weights, is_last_states)))
            es.tell(solutions, objs)
            es.logger.add()
            es.disp()

        es.result_pretty()
        # cma.plot()

        return np.asarray(es.result[0])

    @staticmethod
    def objective(i, penalty_option0, penalty_option1, penalty_weight):
        raise NotImplementedError
