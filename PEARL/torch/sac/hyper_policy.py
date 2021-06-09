import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal
from rlkit.torch.distributions import TanhNormal
from rlkit.torch.core import np_ify
from rlkit.torch.sac.hyper_network import Hyper_Network
from collections import OrderedDict

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20

class Hyper_Policy(nn.Module):
    # Hyper net that create weights from the state for a net that estimates function Q(S, A)
    def __init__(self,
            hidden_sizes,
            obs_dim,
            latent_dim,
            action_dim,
            use_reverse,
            use_combine,
            num_hidden=3,
            input_size=0,
            **kwargs):
        super(Hyper_Policy, self).__init__()

        self.reverse = use_reverse
        output_dim = action_dim
        log_head = True
        
        if self.reverse:
            meta_v_dim = obs_dim
            base_v_dim = latent_dim
        else:
            meta_v_dim = latent_dim
            base_v_dim = obs_dim
        
        self.hyper = Hyper_Network(meta_v_dim, base_v_dim, output_dim, log_head)
    
    def forward(self,
            obs, 
            task,
            reparameterize=False,
            deterministic=False,
            return_log_prob=False):

        if self.reverse:
            meta_v_dim = obs
            base_v_dim = task
        else:
            meta_v_dim = task
            base_v_dim = obs

        mu, log_std = self.hyper(meta_v_dim, base_v_dim)
		
        # constrain log value in finite range to avoid NaN loss values
        log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)

        std = torch.exp(log_std) 

        log_prob = None
        expected_log_prob = None
        mean_action_log_prob = None
        pre_tanh_value = None
        if deterministic:
            action = torch.tanh(mu)
        else:
            tanh_normal = TanhNormal(mu, std)
            if return_log_prob:
                if reparameterize:
                    action, pre_tanh_value = tanh_normal.rsample(
                        return_pretanh_value=True
                    )
                else:
                    action, pre_tanh_value = tanh_normal.sample(
                        return_pretanh_value=True
                    )
                log_prob = tanh_normal.log_prob(
                    action,
                    pre_tanh_value=pre_tanh_value
                )
                log_prob = log_prob.sum(dim=1, keepdim=True)
            else:
                if reparameterize:
                    action = tanh_normal.rsample()
                else:
                    action = tanh_normal.sample()

        return (
            action, mu, log_std, log_prob, expected_log_prob, std,
            mean_action_log_prob, pre_tanh_value,
        )


    def get_action(self, obs, z, deterministic=False):
        actions = self.get_actions(obs, z, deterministic=deterministic)
        return actions[0, :], {}

    @torch.no_grad()
    def get_actions(self, obs, z, deterministic=False):
        outputs = self.forward(obs, z, deterministic=deterministic)[0]
        return np_ify(outputs)