import numpy as np
import torch
import rlkit.torch.pytorch_util as ptu
from rand_param_envs.hopper_rand_params import HopperRandParamsEnv

from . import register_env


@register_env('hopper-rand-params')
class HopperRandParamsWrappedEnv(HopperRandParamsEnv):
    def __init__(self, n_tasks=2, n_train_tasks=2 ,n_eval_tasks=2, randomize_tasks=True):
        super(HopperRandParamsWrappedEnv, self).__init__()
        tasks = self.sample_tasks(n_train_tasks, 1337)
        test_tasks = self.sample_tasks(n_eval_tasks,1338)
        self.norm = self.get_norm(tasks)
        self.tasks = tasks + test_tasks
        self.reset_task(0)

    def sample_tasks(self, num_tasks, seed):
        #np.random.seed(seed)
        velocities = np.random.uniform(self.min, self.max, size=(num_tasks,))
        tasks = [{'velocity': velocity} for velocity in velocities]
        return tasks
    
    def sample_test_tasks(self, num_tasks, seed):
        #np.random.seed(seed)
        velocities = np.random.uniform(self.test_min, self.test_max, size=(num_tasks,))
        tasks = [{'velocity': velocity} for velocity in velocities]
        return tasks

    def get_all_task_idx(self):
        return range(len(self.tasks))

    def reset_task(self, idx):
        self._task = self.tasks[idx]
        self._goal = idx
        self.set_task(self._task)
        self.reset()

    def get_tensor(self, idx):
        task = self.tasks[idx]['goal']
        min_ , max_ , _, _ = self.norm
        task_x = task / max_
        return  torch.Tensor([task_x]).unsqueeze(0)

    def get_tasks_tensors(self, indices):
        all_tasks = self.get_tensor(indices[0])
        for idx in indices[1:]:
            all_tasks = torch.cat([all_tasks, self.get_tensor(idx)],dim=0)
        return all_tasks.to(ptu.device)

    def get_norm(self):
        values = []
        for task in self.tasks:
            values += [task['goal']]
        t = np.asarray(values)
        min_ = min(t)
        max_ = max(t)
        mean = np.mean(t)
        var = np.var(t)
        return (min_, max_, mean, var)