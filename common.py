import multiprocessing as mp
import numpy as np
import logging
import os
import gym
import torch

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

class BasicTask:

    def __init__(self, envname):

        self.task = envname
        self.env_fn = lambda: gym.make(self.task)
        self.repetitions = 10
        self.test_repetitions = 10
        env = self.env_fn()
        self.action_dim = env.action_space.shape[0]
        self.state_dim = env.observation_space.shape[0]
        self.reward_to_fitness = lambda r: r

class ContinuousLunarLanderTask(BasicTask):

    def __init__(self, target=250, max_steps=int(4e7), pop_size=64):

        BasicTask.__init__(self, 'LunarLanderContinuous-v2')

        self.max_steps = max_steps
        self.pop_size = pop_size

        self.num_workers = mp.cpu_count()

        self.action_clip = lambda a: np.clip(a, -1, 1)
        self.target = target

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
