import numpy as np
import torch
import rlkit.torch.pytorch_util as ptu
from .ant_multitask_base import MultitaskAntEnv


# Copy task structure from https://github.com/jonasrothfuss/ProMP/blob/master/meta_policy_search/envs/mujoco_envs/ant_rand_goal.py
class AntVelEnv(MultitaskAntEnv):
    def __init__(self, task={}, n_tasks=2, n_train_tasks=2 ,n_eval_tasks=2, randomize_tasks=True, **kwargs):
        self.min = 0.
        self.max = 3.
        tasks = self.sample_tasks(n_train_tasks)
        test_tasks = self.sample_tasks(n_eval_tasks)
        self.tasks = tasks + test_tasks
        self._task = task
        self.norm = self.get_norm()
        self._goal_vel = self.tasks[0].get('goal', 0.0)
        self._goal = self._goal_vel
        super(AntVelEnv, self).__init__(task, n_tasks, **kwargs)

    def step(self, action):
        xposbefore = self.get_body_com("torso")[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.get_body_com("torso")[0]

        forward_vel = (xposafter - xposbefore) / self.dt
        forward_reward = -1.0 * np.abs(forward_vel - self._goal) + 1.0
        survive_reward = 0.05

        ctrl_cost = 0.5 * 1e-2 * np.sum(np.square(action))
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))

        observation = self._get_obs()
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        notdone = np.isfinite(state).all() \
            and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        infos = dict(reward_forward=forward_reward, reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost, reward_survive=survive_reward,
            task=self._task)
        return (observation, reward, done, infos)


    def sample_tasks(self, num_tasks):#, seed):
        #np.random.seed(seed)
        velocities = np.random.uniform(self.min, self.max, size=(num_tasks,))
        tasks = [{'goal': velocity} for velocity in velocities]
        return tasks

    def get_tensor(self, idx):
        task = self.tasks[idx]['goal']
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
            values += [task['goal']]
        t = np.asarray(values)
        min_ = min(t)
        max_ = max(t)
        mean = np.mean(t)
        var = np.var(t)
        return (min_, max_, mean, var)
