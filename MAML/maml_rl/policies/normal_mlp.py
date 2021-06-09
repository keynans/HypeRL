import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from collections import OrderedDict
from maml_rl.policies.policy import Policy, weight_init

class NormalMLPPolicy(Policy):
    """Policy network based on a multi-layer perceptron (MLP), with a 
    `Normal` distribution output, with trainable standard deviation. This 
    policy network can be used on tasks with continuous action spaces (eg. 
    `HalfCheetahDir`). The code is adapted from 
    https://github.com/cbfinn/maml_rl/blob/9c8e2ebd741cb0c7b8bf2d040c4caeeb8e06cc95/sandbox/rocky/tf/policies/maml_minimal_gauss_mlp_policy.py
    """
    def __init__(self, task_dim, input_size, output_size, hidden_sizes=(),use_task=False,
                 nonlinearity=F.relu, init_std=1.0, min_std=1e-6):
        super(NormalMLPPolicy, self).__init__(
            input_size=input_size, output_size=output_size)

        self.task_dim = task_dim
        self.state_dim = input_size
        self.num_layers = len(hidden_sizes) + 1
        self.action_dim = output_size
        self.num_hidden = hidden_sizes
        self.nonlinearity = nonlinearity
        self.min_log_std = math.log(min_std)
        self.use_task = use_task
        self.is_hyper=False

        if use_task:
            input_size += task_dim

        layer_sizes = (input_size,) + hidden_sizes
        for i in range(1, self.num_layers):
            self.add_module('layer{0}'.format(i),
                nn.Linear(layer_sizes[i - 1], layer_sizes[i]))
        self.mu = nn.Linear(layer_sizes[-1], output_size)

        self.sigma = nn.Parameter(torch.Tensor(output_size))
        self.sigma.data.fill_(math.log(init_std))
        self.apply(weight_init)

    def forward(self, input, task=None, params=None, z=None):
#        if params is None:
        params = OrderedDict(self.named_parameters())

        output = input
        if self.use_task and task is not None:
            task = task.unsqueeze(0).repeat(input.shape[0],1)
            dim=1
            if len(input.shape) ==  3:
                task = task.unsqueeze(1).repeat(1,input.shape[1],1)
                dim=2

            output = torch.cat([input, task],dim=dim)
        
        for i in range(1, self.num_layers):
            output = F.linear(output,
                weight=params['layer{0}.weight'.format(i)],
                bias=params['layer{0}.bias'.format(i)])
            output = self.nonlinearity(output)
        mu = F.linear(output, weight=params['mu.weight'],
            bias=params['mu.bias'])
        scale = torch.exp(torch.clamp(params['sigma'], min=self.min_log_std))

        return Normal(loc=mu, scale=scale), mu
