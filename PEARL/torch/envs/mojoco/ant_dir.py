import numpy as np
import torch
import rlkit.torch.pytorch_util as ptu
from .ant_multitask_base import MultitaskAntEnv

class AntDirEnv(MultitaskAntEnv):

    def __init__(self, task={}, n_tasks=2, n_train_tasks=2 ,n_eval_tasks=2, forward_backward=False, randomize_tasks=True, **kwargs):
        self.forward_backward = forward_backward
        directions = [-1, 1]
        tasks = [{'goal': direction} for direction in directions]
        test_tasks = [{'goal': direction} for direction in directions]
        self.tasks = tasks + test_tasks
        self._task = task
        self.norm = self.get_norm()
        self._goal_dir = task.get('goal', 1)
        self._goal = self._goal_dir
        super(AntDirEnv, self).__init__(task, n_tasks, **kwargs)

    def step(self, action):
        torso_xyz_before = np.array(self.get_body_com("torso"))

        direct = (np.cos(self._goal), np.sin(self._goal))

        self.do_simulation(action, self.frame_skip)
        torso_xyz_after = np.array(self.get_body_com("torso"))
        torso_velocity = torso_xyz_after - torso_xyz_before
        forward_reward = np.dot((torso_velocity[:2]/self.dt), direct)

        ctrl_cost = .5 * 1e-2 * np.square(action).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 0.05
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        notdone = np.isfinite(state).all() \
                  and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        ob = self._get_obs()
        return ob, reward, done, dict(
            reward_forward=forward_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward,
            torso_velocity=torso_velocity,
        )

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