import numpy as np
import torch
import rlkit.torch.pytorch_util as ptu
from .half_cheetah import HalfCheetahEnv

class HalfCheetahVelEnv(HalfCheetahEnv):
    """Half-cheetah environment with target velocity, as described in [1]. The
    code is adapted from
    https://github.com/cbfinn/maml_rl/blob/9c8e2ebd741cb0c7b8bf2d040c4caeeb8e06cc95/rllab/envs/mujoco/half_cheetah_env_rand.py

    The half-cheetah follows the dynamics from MuJoCo [2], and receives at each
    time step a reward composed of a control cost and a penalty equal to the
    difference between its current velocity and the target velocity. The tasks
    are generated by sampling the target velocities from the uniform
    distribution on [0, 2].

    [1] Chelsea Finn, Pieter Abbeel, Sergey Levine, "Model-Agnostic
        Meta-Learning for Fast Adaptation of Deep Networks", 2017
        (https://arxiv.org/abs/1703.03400)
    [2] Emanuel Todorov, Tom Erez, Yuval Tassa, "MuJoCo: A physics engine for
        model-based control", 2012
        (https://homes.cs.washington.edu/~todorov/papers/TodorovIROS12.pdf)
    """
    def __init__(self, task={}, n_tasks=2,n_train_tasks=2 ,n_eval_tasks=2, randomize_tasks=True):
        self._task = task
        tasks = self.sample_tasks(n_train_tasks, 1337)
        test_tasks = self.sample_tasks(n_eval_tasks,1338)
        self.tasks = tasks + test_tasks
        self.norm = self.get_norm()
        self._goal_vel = self.tasks[0].get('velocity', 0.0)
        self._goal = self._goal_vel
        super(HalfCheetahVelEnv, self).__init__()

    def step(self, action):
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]

        forward_vel = (xposafter - xposbefore) / self.dt
        forward_reward = -1.0 * abs(forward_vel - self._goal)
        ctrl_cost = 0.5 * 1e-1 * np.sum(np.square(action))

        observation = self._get_obs()
        reward = forward_reward - ctrl_cost
        done = False
        infos = dict(reward_forward=forward_reward,
            reward_ctrl=-ctrl_cost, task=self._task)
        return (observation, reward, done, infos)

    def sample_tasks(self, num_tasks, seed):
        #np.random.seed(seed)
        velocities = np.random.uniform(0.0, 3.0, size=(num_tasks,))
        tasks = [{'velocity': velocity} for velocity in velocities]
        return tasks

    def get_tensor(self, idx):
        task = self.tasks[idx]['velocity']
        min_ , max_ , _, _ = self.norm
        task_v = (task - min_) / (max_ - min_)
        return  torch.Tensor([task_v]).unsqueeze(0)

    def get_tasks_tensors(self, indices):
        all_tasks = self.get_tensor(indices[0])
        for idx in indices[1:]:
            all_tasks = torch.cat([all_tasks, self.get_tensor(idx)],dim=0)
        return all_tasks.to(ptu.device)

    def get_norm(self):
        values = []
        for task in self.tasks:
            values += [task['velocity']]
        t = np.asarray(values)
        min_ = min(t)
        max_ = max(t)
        mean = np.mean(t)
        var = np.var(t)
        return (min_, max_, mean, var)
        
    def get_all_task_idx(self):
        return range(len(self.tasks))

    def reset_task(self, idx):
        self._task = self.tasks[idx]
        self._goal_vel = self._task['velocity']
        self._goal = self._goal_vel
        self.reset()