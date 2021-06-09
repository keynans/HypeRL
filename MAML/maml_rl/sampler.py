import gym
import torch
import multiprocessing as mp
import numpy as np

from maml_rl.envs.subproc_vec_env import SubprocVecEnv
from maml_rl.episode import BatchEpisodes
from torch.nn.utils.convert_parameters import (vector_to_parameters,
                                               parameters_to_vector)

def make_env(env_name):
    def _make_env():
        return gym.make(env_name)
    return _make_env

class BatchSampler(object):
    def __init__(self, env_name, batch_size, num_workers=mp.cpu_count() - 1, seed=0):
        self.env_name = env_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self.queue = mp.Queue()
        if num_workers > 0:
            self.envs = SubprocVecEnv([make_env(env_name) for _ in range(num_workers)],
                queue=self.queue)
        else:
            self.envs = gym.make(env_name)
        self._env = gym.make(env_name)
        self._env.seed(seed)
        self.envs.seed(seed)

    def sample(self,task, policy, determenistic=False, params=None, gamma=0.95, device='cpu'):
        episodes = BatchEpisodes(batch_size=self.batch_size, gamma=gamma, device=device)
        for i in range(self.batch_size):
            self.queue.put(i)
        for _ in range(self.num_workers):
            self.queue.put(None)
        observations, batch_ids = self.envs.reset()
        dones = [False]
        if params is not None:
                old_params = parameters_to_vector(policy.parameters())
                policy.load_state_dict(params)

        z = None
        if policy.is_hyper:
            z = policy.hyper.get_z(task)
            
        while (not all(dones)) or (not self.queue.empty()):
            with torch.no_grad():
                observations_tensor = torch.from_numpy(observations).to(device=device)
                normal, actions_tensor = policy(observations_tensor,task, params=params, z=z)
                if not determenistic:
                    actions_tensor = normal.sample()

                actions = actions_tensor.cpu().numpy()
            new_observations, rewards, dones, new_batch_ids, _ = self.envs.step(actions)
            episodes.append(observations, actions, rewards, batch_ids)
            observations, batch_ids = new_observations, new_batch_ids

        if params is not None:
            vector_to_parameters(old_params, policy.parameters())

        return episodes

    def reset_task(self, task):
        if self.num_workers > 0:
            tasks = [task for _ in range(self.num_workers)]
            reset = self.envs.reset_task(tasks)
            return all(reset)
        else:
            tasks = task
            reset = self.envs.reset_task(tasks)

    def sample_tasks(self, unseen_tasks, num_tasks):
        tasks = self._env.unwrapped.sample_tasks(unseen_tasks, num_tasks)
        return tasks

    def get_task_tensor(self, task, scale):
        min_ , max_ , mean, var = scale
        if 'velocity' in task.keys():
            task_v = (task['velocity'] - min_) / (max_ - min_)
            task_tensor = torch.Tensor([task_v])
        elif 'direction' in task.keys():
            task_x = task['direction'] / max_
            task_tensor = torch.Tensor([task_x])
        elif 'goal' in task.keys():
            task_x, task_y = ((task['goal'] - min_) * 2 / (max_ - min_))-1
            task_tensor = torch.Tensor([task_x, task_y])
        else:
          #   task_x, task_y = ((task['position'] - min_) * 2 / (max_ - min_))-1
             task_x, task_y = task['position']
             theta = np.arctan2(task_y, task_x)
             sin = np.sin(theta)
             cos= np.cos(theta)
             task_tensor = torch.Tensor([cos, sin])
             # task_tensor = torch.Tensor([task_x, task_y])
        return task_tensor

    def sample_unseen_task(self, tasks=[], num_of_unseen=1):
        unseen_task = []
        values = []
        for i in range(num_of_unseen):
            unseen_task += self._env.unwrapped.sample_unseen_task(tasks)
            x = list(unseen_task[-1].values())[0]
            try:
                values += list(list(unseen_task[-1].values())[0])
            except:
                values += [list(unseen_task[-1].values())[0]]
        t = np.asarray(values)
        min_ = min(t)
        max_ = max(t)
        mean = np.mean(t)
        var = np.var(t)
        return unseen_task, (min_, max_, mean, var)