import torch
import numpy as np
import time
from torch.nn.utils.convert_parameters import (vector_to_parameters,
                                               parameters_to_vector)
from torch.distributions.kl import kl_divergence
from maml_rl.metalearner import MetaLearner

from maml_rl.utils.torch_utils import (weighted_mean, detach_distribution,
                                       weighted_normalize)
from maml_rl.utils.optimization import conjugate_gradient
t0 = time.time()

class Multi_MetaLearner(MetaLearner):
    """Meta-learner

    The meta-learner is responsible for sampling the trajectories/episodes 
    (before and after the one-step adaptation), compute the task loss, compute 
    the updated parameters based on the inner-loss, and perform the meta-update.

    [1] Chelsea Finn, Pieter Abbeel, Sergey Levine, "Model-Agnostic 
        Meta-Learning for Fast Adaptation of Deep Networks", 2017 
        (https://arxiv.org/abs/1703.03400)
    [2] Richard Sutton, Andrew Barto, "Reinforcement learning: An introduction",
        2018 (http://incompleteideas.net/book/the-book-2nd.html)
    [3] John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan, 
        Pieter Abbeel, "High-Dimensional Continuous Control Using Generalized 
        Advantage Estimation", 2016 (https://arxiv.org/abs/1506.02438)
    [4] John Schulman, Sergey Levine, Philipp Moritz, Michael I. Jordan, 
        Pieter Abbeel, "Trust Region Policy Optimization", 2015
        (https://arxiv.org/abs/1502.05477)
    """
    def __init__(self, sampler, policy, baseline, gamma=0.95,
                 fast_lr=0.5, tau=1.0, device='cpu'):
        MetaLearner.__init__(self, sampler, policy, baseline, gamma,
                 fast_lr, tau, device)
        self.to(device)
        self.par2=[]
        self.par3=[]
        print("multi meta")

    def task_loss(self,task, episodes, old_pi, params=None):
        # Fit the baseline to the training episodes
        self.baseline.fit(episodes)
        values = self.baseline(episodes)
        advantages = episodes.gae(values,tau=self.tau)
        advantages = weighted_normalize(advantages, weights=episodes.mask)

        with torch.set_grad_enabled(old_pi is None):
 
            z = None
            if self.policy.is_hyper:
                z = self.policy.hyper.get_z(task)
            pi, _ = self.policy(episodes.observations, task, params=params, z=z)
            if old_pi is None:
                old_pi = detach_distribution(pi)
            log_ratio = (pi.log_prob(episodes.actions)
                        - old_pi.log_prob(episodes.actions))
            if log_ratio.dim() > 2:
                log_ratio = torch.sum(log_ratio, dim=2)
            ratio = torch.exp(log_ratio)

            loss = -weighted_mean(ratio * advantages,
                        weights=episodes.mask)
            
            mask = episodes.mask
            if episodes.actions.dim() > 2:
                mask = mask.unsqueeze(2)
            kl = weighted_mean(kl_divergence(pi, old_pi), weights=mask)

        return loss, detach_distribution(pi), kl

    def sample(self, tasks, scale, first_order=False, deterministic=False):
        """Sample trajectories
        for all the tasks `tasks`.
        """
        episodes = []
        for task in tasks:
            self.sampler.reset_task(task)
            task_tensor = self.sampler.get_task_tensor(task, scale).to(device=self.device)
            train_episodes = self.sampler.sample(task_tensor, self.policy, deterministic,
                gamma=self.gamma, device=self.device)
    
            episodes.append((task_tensor, train_episodes, train_episodes))
        return episodes

    def kl_divergence(self, episodes, old_pis=None):
        kls = []
        if old_pis is None:
            old_pis = [None] * len(episodes)

        for (task, train_episodes,_), old_pi in zip(episodes, old_pis):
            z = None
            if self.policy.is_hyper:
                z = self.policy.hyper.get_z(task)
            pi, _ = self.policy(train_episodes.observations, task, z=z)

            if old_pi is None:
                old_pi = detach_distribution(pi)

            mask = train_episodes.mask
            if train_episodes.actions.dim() > 2:
                mask = mask.unsqueeze(2)
            kl = weighted_mean(kl_divergence(pi, old_pi), weights=mask)
            kls.append(kl)

        return torch.mean(torch.stack(kls, dim=0))

    def hessian_vector_product(self, episodes, damping=1e-2):
        """Hessian-vector product, based on the Perlmutter method."""
        def _product(vector):
            kl = self.kl_divergence(episodes)
            grads = torch.autograd.grad(kl, self.policy.parameters(),
                create_graph=True)
            flat_grad_kl = parameters_to_vector(grads)

            grad_kl_v = torch.dot(flat_grad_kl, vector)
            grad2s = torch.autograd.grad(grad_kl_v, self.policy.parameters())
            flat_grad2_kl = parameters_to_vector(grad2s)

            return flat_grad2_kl + damping * vector
        return _product

    def get_loss(self, episodes, old_pis=None):
        loss = []; pis = []; kls=[]
        if old_pis is None:
            old_pis = [None] * len(episodes)
        for (task, train_episodes,_), old_pi in zip(episodes, old_pis):
            l,pi,kl = self.task_loss(task, train_episodes, old_pi)
            loss.append(l); pis.append(pi), kls.append(kl)

        return torch.mean(torch.stack(loss, dim=0)),torch.mean(torch.stack(kls, dim=0)),pis
   
    def step(self, episodes, total_tasks, scale, device, max_kl=1e-3, cg_iters=10, cg_damping=1e-2,
             ls_max_steps=10, ls_backtrack_ratio=0.5, beta=0.001, regulation=False):
        """Meta-optimization step (ie. update of the initial parameters), based 
        on Trust Region Policy Optimization (TRPO, [4]).
        """
        
        old_loss, _, old_pis = self.get_loss(episodes)

        # Save the old parameters
        old_params = parameters_to_vector(self.policy.parameters())
        old_policy = type(self.policy)(self.policy.task_dim, self.policy.state_dim, self.policy.action_dim, self.policy.num_hidden,self.policy.use_task).to(device)
        vector_to_parameters(old_params, old_policy.parameters())

        #compute gradient and do a step
        self.gradient_step(old_loss, old_params, old_pis, episodes, max_kl, cg_iters, cg_damping,
             ls_max_steps, ls_backtrack_ratio)
        
        #compute regulation with new parameters
        if regulation:
            reg = 0
            batch_tasks = [task.cpu().numpy() for task,_,_ in episodes]
            for task in total_tasks:
                task = self.sampler.get_task_tensor(task, scale).to(device=self.device)
                if (task.cpu().numpy() == batch_tasks).all(1).any():
                    continue
                par = self.policy.main_net_params(task)
                old_par = old_policy.main_net_params(task)
                reg += torch.norm(old_par - par, 2) ** 2
            reg_loss = (beta / len(episodes)) * reg

            # do a step with updated loss
            vector_to_parameters(old_params, self.policy.parameters())
            loss = old_loss + reg_loss
            self.gradient_step(loss, old_params, old_pis, episodes, max_kl, cg_iters, cg_damping,
                ls_max_steps, ls_backtrack_ratio)


    def gradient_step(self, old_loss, old_params, old_pis, episodes,max_kl=1e-3, cg_iters=10, cg_damping=1e-2,
             ls_max_steps=10, ls_backtrack_ratio=0.5):

        grads = torch.autograd.grad(old_loss, self.policy.parameters(),retain_graph=True)
        grads = parameters_to_vector(grads)

        # Compute the step direction with Conjugate Gradient
        hessian_vector_product = self.hessian_vector_product(episodes,
            damping=cg_damping)
        stepdir = conjugate_gradient(hessian_vector_product, grads,
            cg_iters=cg_iters)

        # Compute the Lagrange multiplier
        shs = 0.5 * torch.dot(stepdir, hessian_vector_product(stepdir))
        lagrange_multiplier = torch.sqrt(shs / max_kl)

        step = stepdir / lagrange_multiplier

        # Line search
        step_size = 1.0
        for _ in range(ls_max_steps):
            vector_to_parameters(old_params - step_size * step,
                                 self.policy.parameters())
            loss, kl, _ = self.get_loss(episodes, old_pis=old_pis)
            improve = old_loss - loss
            if (improve.item() > 0.0) and (kl.item() < max_kl):
                break
            step_size *= ls_backtrack_ratio
        else:
            vector_to_parameters(old_params, self.policy.parameters())


        # Runs new task and evaluate the reward before and after grdient
    def evaluate_task(self, unseen_task, scale, episode_num):    
        episode = super().sample(unseen_task, scale)
        
        def total_rewards(episodes_rewards, aggregation=torch.mean):
            rewards = torch.mean(torch.stack([aggregation(torch.sum(rewards, dim=0))
                for rewards in episodes_rewards], dim=0))
            return rewards.item()
        before_rewards = total_rewards([ep.rewards for __, ep, _ in episode])
        after_rewards = total_rewards([ep.rewards for __, _, ep, in episode])

        return before_rewards, after_rewards
    #    episode = self.sample(unseen_task, scale)
        
    #    def total_rewards(episodes_rewards, aggregation=torch.mean):
    #        rewards = torch.mean(torch.stack([aggregation(torch.sum(rewards, dim=0))
    #            for rewards in episodes_rewards], dim=0))
    #        return rewards.item()
    #    before_rewards = total_rewards([ep.rewards for __, ep, _ in episode])

    #    return before_rewards,before_rewards