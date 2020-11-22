#!/usr/bin/env python3
import numpy as np
import multiprocessing as mp
import os
import pickle
import sys
import time
import logging

from common import SharedStats, StaticNormalizer, fitness_shift, Worker, get_logger, ContinuousLunarLanderTask

from torchmodel import StandardFCNet

import cma

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

class ContinuousLunarLanderTaskCMA(ContinuousLunarLanderTask):
    
    def __init__(self, hidden_size=16, max_steps=int(1e7)):

        ContinuousLunarLanderTask.__init__(self, max_steps=max_steps)

        self.hidden_size = hidden_size
        self.model_fn = lambda: StandardFCNet(self.state_dim, self.action_dim, self.hidden_size)
        model = self.model_fn()
        self.initial_weight = model.get_weight()
        self.weight_decay = 0.005
        self.action_noise_std = 0
        self.sigma = 1
        self.tag = 'CMA-%d' % (hidden_size)


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

    hidden_size = 16
    max_steps = 2e7
    task = ContinuousLunarLanderTaskCMA(hidden_size, max_steps)

    logger = get_logger()

    p = mp.Process(target=multi_runs, args=(task,logger))

    p.start()
    p.join()

if __name__ == '__main__':
    main()
