import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from rlkit.torch.sac.hyper_network import Hyper_Network

class Hyper_QNetwork(nn.Module):
    # Hyper net that create weights from the state for a net that estimates function Q(S, A)
    def __init__(self,
            hidden_sizes,
            output_size,
            obs_dim,
            latent_dim,
            action_dim,
            use_reverse,
            use_combine,
            num_hidden=3,
            input_size=0,
            **kwargs):
        super(Hyper_QNetwork, self).__init__()
	
        self.combine = use_combine
        self.reverse = use_reverse
        self.value_func = True if action_dim == 0 else False
        log_head = False
        output_dim = 1

        meta_v_dim = obs_dim
        base_v_dim = action_dim

        if not self.value_func and self.combine:
            meta_v_dim += latent_dim
        else:
            base_v_dim += latent_dim

        # Q function net
        self.hyper = Hyper_Network(meta_v_dim, base_v_dim, output_dim, log_head)

    def forward(self, obs, actions, task_z):

        if not self.value_func:
            if self.combine:
                meta_v = torch.cat([obs, task_z],dim=1)
                base_v = actions
            else:
                meta_v = obs
                base_v = torch.cat([actions, task_z],dim=1)
        
        else:
                meta_v = obs
                base_v = task_z
        
        q = self.hyper(meta_v, base_v)
        
        return q
  

