from hyper_network import Hyper_QNetwork
import torch
import torch.nn as nn
import torch.nn.parameter as P

class Hyper_Critic_Random(nn.Module):
    # Hyper net that create weights from the state for a net that estimates function Q(S, A)
    def __init__(self, state_dim, action_dim, num_hidden=1):
        super(Hyper_Critic_Random, self).__init__()
        meta_v_dim = 5
        base_v_dim = state_dim + action_dim
        
        self.q1 = Hyper_QNetwork(meta_v_dim, base_v_dim)
        self.q2 = Hyper_QNetwork(meta_v_dim, base_v_dim)
        self.embedding = P.Parameter(torch.rand(meta_v_dim), requires_grad=True)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = self.q1(self.embedding.repeat(sa.shape[0],1), sa)
        q2 = self.q2(self.embedding.repeat(sa.shape[0],1), sa)
        return q1,q2

    def Q1(self, state, action, debug):
        sa = torch.cat([state, action], 1)
        q1 = self.q1(self.embedding.repeat(sa.shape[0],1), sa, debug)
        return q1
