#!/usr/bin/env python3
import gym
import torch
import numpy as np
import multiprocessing as mp
import os
import pickle
import sys
import time
import logging
import cma
import argparse

from torchmodel import StandardFCNet

def _makedir(name):
    if not os.path.exists(name):
        os.makedirs(name)

def get_logger():
    _makedir('log')
    _makedir('data')
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s: %(message)s')
    logger = logging.getLogger('MAIN')
    logger.setLevel(logging.DEBUG)
    return logger

class Task:
    
    def __init__(self, envname, hidden_size, max_steps, target, pop_size):

        self.task = envname
        self.env_fn = lambda: gym.make(self.task)
        self.repetitions = 10
        self.test_repetitions = 10
        env = self.env_fn()
        self.action_dim = env.action_space.shape[0]
        self.state_dim = env.observation_space.shape[0]
        self.reward_to_fitness = lambda r: r

        self.max_steps = max_steps
        self.pop_size = pop_size

        self.num_workers = mp.cpu_count()

        self.action_clip = lambda a: np.clip(a, -1, 1)
        self.target = target

        self.hidden_size = hidden_size
        self.model_fn = lambda: StandardFCNet(self.state_dim, self.action_dim, self.hidden_size)
        model = self.model_fn()
        self.initial_weight = model.get_weight()
        self.weight_decay = 0.005
        self.action_noise_std = 0
        self.sigma = 1
        self.tag = 'CMA-%d' % (hidden_size)

class BaseModel:
    def get_weight(self):
        weight = []
        for param in self.parameters():
            weight.append(param.data.numpy().flatten())
        weight = np.concatenate(weight, 0)
        return weight

    def set_weight(self, solution):
        offset = 0
        for param in self.parameters():
            param_shape = param.data.numpy().shape
            param_size = np.prod(param_shape)
            src_param = solution[offset: offset + param_size]
            if len(param_shape) > 1:
                src_param = src_param.reshape(param_shape)
            param.data = torch.FloatTensor(src_param)
            offset += param_size
        assert offset == len(solution)

class Normalizer:
    def __init__(self, filter_mean=True):
        self.m = 0
        self.v = 0
        self.n = 0.
        self.filter_mean = filter_mean

    def state_dict(self):
        return {'m': self.m,
                'v': self.v,
                'n': self.n}

    def load_state_dict(self, saved):
        self.m = saved['m']
        self.v = saved['v']
        self.n = saved['n']

    def __call__(self, o):
        self.m = self.m * (self.n / (self.n + 1)) + o * 1 / (1 + self.n)
        self.v = self.v * (self.n / (self.n + 1)) + (o - self.m) ** 2 * 1 / (1 + self.n)
        self.std = (self.v + 1e-6) ** .5  # std
        self.n += 1
        if self.filter_mean:
            o_ = (o - self.m) / self.std
        else:
            o_ = o / self.std
        return o_

class StaticNormalizer:
    def __init__(self, o_size):
        self.offline_stats = SharedStats(o_size)
        self.online_stats = SharedStats(o_size)

    def __call__(self, o_):
        o = torch.FloatTensor([o_] if np.isscalar(o_) else o_)
        self.online_stats.feed(o)
        if self.offline_stats.n[0] == 0:
            return o_
        std = (self.offline_stats.v + 1e-6) ** .5
        o = (o - self.offline_stats.m) / std
        o = o.numpy()
        if np.isscalar(o_):
            o = np.asscalar(o)
        else:
            o = o.reshape(o_.shape)
        return o

class SharedStats:
    def __init__(self, o_size):
        self.m = torch.zeros(o_size)
        self.v = torch.zeros(o_size)
        self.n = torch.zeros(1)
        self.m.share_memory_()
        self.v.share_memory_()
        self.n.share_memory_()

    def feed(self, o):
        n = self.n[0]
        new_m = self.m * (n / (n + 1)) + o / (n + 1)
        self.v.copy_(self.v * (n / (n + 1)) + (o - self.m) * (o - new_m) / (n + 1))
        self.m.copy_(new_m)
        self.n.add_(1)

    def zero(self):
        self.m.zero_()
        self.v.zero_()
        self.n.zero_()

    def load(self, stats):
        self.m.copy_(stats.m)
        self.v.copy_(stats.v)
        self.n.copy_(stats.n)

    def merge(self, B):
        A = self
        n_A = self.n[0]
        n_B = B.n[0]
        n = n_A + n_B
        delta = B.m - A.m
        m = A.m + delta * n_B / n
        v = A.v * n_A + B.v * n_B + delta * delta * n_A * n_B / n
        v /= n
        self.m.copy_(m)
        self.v.copy_(v)
        self.n.add_(B.n)

    def state_dict(self):
        return {'m': self.m.numpy(),
                'v': self.v.numpy(),
                'n': self.n.numpy()}

    def load_state_dict(self, saved):
        self.m = torch.FloatTensor(saved['m'])
        self.v = torch.FloatTensor(saved['v'])
        self.n = torch.FloatTensor(saved['n'])

def fitness_shift(x):
    x = np.asarray(x).flatten()
    ranks = np.empty(len(x))
    ranks[x.argsort()] = np.arange(len(x))
    ranks /= (len(x) - 1)
    ranks -= .5
    return ranks

class Worker(mp.Process):
    def __init__(self, id, task_q, result_q, stop):
        mp.Process.__init__(self)
        self.id = id
        self.task_q = task_q
        self.result_q = result_q
        self.stop = stop

    def run(self):
        np.random.seed()
        while not self.stop.value:
            if self.task_q.empty():
                continue
            id, solution = self.task_q.get()
            fitness, steps = self.evalfun(solution)
            self.result_q.put([id, fitness, steps])

class Evaluator:

    def __init__(self, config, state_normalizer):
        self.model = config.model_fn()
        self.repetitions = config.repetitions
        self.env = config.env_fn()
        self.state_normalizer = state_normalizer
        self.config = config

    def eval(self, solution):
        self.model.set_weight(solution)
        rewards = []
        steps = []
        for i in range(self.repetitions):
            reward, step = self.single_run()
            rewards.append(reward)
            steps.append(step)
        return -np.mean(rewards), np.sum(steps)

    def single_run(self):
        state = self.env.reset()
        total_reward = 0
        steps = 0
        while True:
            state = self.state_normalizer(state)
            action = self.model(np.stack([state])).data.numpy().flatten()
            action += np.random.randn(len(action)) * self.config.action_noise_std
            action = self.config.action_clip(action)
            state, reward, done, info = self.env.step(action)
            steps += 1
            total_reward += reward
            if done:
                return total_reward, steps


class CMAWorker(Worker):

    def __init__(self, id, state_normalizer, task_q, result_q, stop, config):
        Worker.__init__(self, id, task_q, result_q, stop)
        self.evalfun = Evaluator(config, state_normalizer).eval

def train(config, logger):
    task_queue = mp.SimpleQueue()
    result_queue = mp.SimpleQueue()
    stop = mp.Value('i', False)
    stats = SharedStats(config.state_dim)
    normalizers = [StaticNormalizer(config.state_dim) for _ in range(config.num_workers)]
    for normalizer in normalizers:
        normalizer.offline_stats.load(stats)

    workers = [CMAWorker(id, normalizers[id], task_queue, result_queue, stop, config) for id in range(config.num_workers)]
    for w in workers: w.start()

    opt = cma.CMAOptions()
    opt['tolfun'] = -config.target
    opt['popsize'] = config.pop_size
    opt['verb_disp'] = 0
    opt['verb_log'] = 0
    opt['maxiter'] = sys.maxsize
    es = cma.CMAEvolutionStrategy(config.initial_weight, config.sigma, opt)

    total_steps = 0
    initial_time = time.time()
    training_rewards = []
    training_steps = []
    training_timestamps = []
    test_mean, test_std = test(config, config.initial_weight, stats)
    logger.info('total steps %8d, %+4.0f(%+4.0f)' % (total_steps, test_mean, test_std))
    training_rewards.append(test_mean)
    training_steps.append(0)
    training_timestamps.append(0)
    while True:
        solutions = es.ask()
        for id, solution in enumerate(solutions):
            task_queue.put((id, solution))
        while not task_queue.empty():
            continue
        result = []
        while len(result) < len(solutions):
            if result_queue.empty():
                continue
            result.append(result_queue.get())
        result = sorted(result, key=lambda x: x[0])
        total_steps += np.sum([r[2] for r in result])
        cost = [r[1] for r in result]
        best_solution = solutions[np.argmin(cost)]
        elapsed_time = time.time() - initial_time
        test_mean, test_std = test(config, best_solution, stats)
        best = -np.min(cost)
        logger.info('total steps = %8d    test = %+4.0f (%4.0f)    best = %+4.0f (%+4.0f)    elapased time = %4.0f sec' %
            (total_steps, test_mean, test_std, best, config.target, elapsed_time))
        training_rewards.append(test_mean)
        training_steps.append(total_steps)
        training_timestamps.append(elapsed_time)
        #with open('data/%s-best_solution_%s.bin' % (TAG, config.task), 'wb') as f: # XXX gets stuck
        #    pickle.dump(solutions[np.argmin(result)], f)
        if best > config.target:
            logger.info('Best score of %f exceeds target %f' % (best, config.target))
            break
        if config.max_steps and total_steps > config.max_steps:
            logger.info('Maximum number of steps exceeded')
            stop.value = True
            break

        cost = fitness_shift(cost)
        es.tell(solutions, cost)
        # es.disp()
        for normalizer in normalizers:
            stats.merge(normalizer.online_stats)
            normalizer.online_stats.zero()
        for normalizer in normalizers:
            normalizer.offline_stats.load(stats)

    stop.value = True
    for w in workers: w.join()
    return [training_rewards, training_steps, training_timestamps]

def test(config, solution, stats):
    normalizer = StaticNormalizer(config.state_dim)
    normalizer.offline_stats.load_state_dict(stats.state_dict())
    evaluator = Evaluator(config, normalizer)
    evaluator.model.set_weight(solution)
    rewards = []
    for i in range(config.test_repetitions):
        reward, _ = evaluator.single_run()
        rewards.append(reward)
    return np.mean(rewards), np.std(rewards) / config.repetitions

def multi_runs(task, logger, runs=1):
    if not os.path.exists('log'):
        os.makedirs('log')
    fh = logging.FileHandler('log/%s-%s.txt' % (task.tag, task.task))
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    stats = []
    for run in range(runs):
        logger.info('Run %3d/%3d' % (run+1, runs))
        stats.append(train(task, logger))
        with open('data/%s-stats-%s.bin' % (task.tag, task.task), 'wb') as f:
            pickle.dump(stats, f)

def main():

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', type=str, default='LunarLanderContinuous-v2')
    parser.add_argument('--nhid', help='# of hidden units', type=int, default=64)
    parser.add_argument('--rgoal', help='reward goal', type=float, default=250)
    parser.add_argument('--max-steps', help='maximum number of steps', type=int, default=int(2e7))
    parser.add_argument('--pop-size', help='population size', type=int, default=64)
    args = parser.parse_args()

    task = Task(args.env, args.nhid, args.max_steps, args.rgoal, args.pop_size)

    logger = get_logger()

    p = mp.Process(target=multi_runs, args=(task,logger))

    p.start()
    p.join()

if __name__ == '__main__':
    main()
