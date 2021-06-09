import numpy as np

import torch
from torch import nn as nn
import torch.nn.functional as F

import rlkit.torch.pytorch_util as ptu


def _product_of_gaussians(mus, sigmas_squared):
    '''
    compute mu, sigma of product of gaussians
    '''
    sigmas_squared = torch.clamp(sigmas_squared, min=1e-7)
    sigma_squared = 1. / torch.sum(torch.reciprocal(sigmas_squared), dim=0)
    mu = sigma_squared * torch.sum(mus / sigmas_squared, dim=0)
    return mu, sigma_squared


def _mean_of_gaussians(mus, sigmas_squared):
    '''
    compute mu, sigma of mean of gaussians
    '''
    mu = torch.mean(mus, dim=0)
    sigma_squared = torch.mean(sigmas_squared, dim=0)
    return mu, sigma_squared


def _natural_to_canonical(n1, n2):
    ''' convert from natural to canonical gaussian parameters '''
    mu = -0.5 * n1 / n2
    sigma_squared = -0.5 * 1 / n2
    return mu, sigma_squared


def _canonical_to_natural(mu, sigma_squared):
    ''' convert from canonical to natural gaussian parameters '''
    n1 = mu / sigma_squared
    n2 = -0.5 * 1 / sigma_squared
    return n1, n2


class HyperPEARLAgent(nn.Module):

    def __init__(self,
                 latent_dim,
                 context_encoder,
                 policy,
                 use_hyper_net,
                 **kwargs
    ):
        super().__init__()
        self.latent_dim = latent_dim

        self.policy = policy
        self.use_hyper_net = use_hyper_net


        self.recurrent = kwargs['recurrent']
        self.use_ib = kwargs['use_information_bottleneck']
        self.sparse_rewards = kwargs['sparse_rewards']
        self.use_next_obs_in_context = kwargs['use_next_obs_in_context']


    def get_action(self, obs, deterministic=False):
        ''' sample action from the policy, conditioned on the task embedding '''
        z = self.z
        obs = ptu.from_numpy(obs[None])
        if self.use_hyper_net:
            return self.policy.get_action(obs, z, deterministic=deterministic)
        else:
            return self.policy.get_action(torch.cat([obs, z], dim=1), deterministic=deterministic)
        

    def set_num_steps_total(self, n):
        self.policy.set_num_steps_total(n)

    def forward(self, obs, context):
        ''' given context, get statistics under the current policy of a set of observations '''

        t, b, _ = obs.size()
        obs = obs.view(t * b, -1)
        task_z = [z.repeat(b, 1) for z in context]
        task_z = torch.cat(task_z, dim=0)
        
        if self.use_hyper_net:
            policy_outputs = self.policy(obs, task_z.detach(), reparameterize=True, return_log_prob=True)
        else:
            in_ = torch.cat([obs, task_z.detach()], dim=1)
            policy_outputs = self.policy(in_, reparameterize=True, return_log_prob=True)
  
        return policy_outputs, task_z



    @property
    def networks(self):
        return [self.policy]




