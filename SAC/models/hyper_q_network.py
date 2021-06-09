from .hyper_network import Hyper_QNetwork
import torch.nn as nn

class Hyper_Critic(nn.Module):
    # Hyper net that create weights from the state for a net that estimates function Q(S, A)
    def __init__(self, state_dim, action_dim, num_hidden=1):
        super(Hyper_Critic, self).__init__()
        meta_v_dim = state_dim
        base_v_dim = action_dim
        self.q1 = Hyper_QNetwork(meta_v_dim, base_v_dim)
        self.q2 = Hyper_QNetwork(meta_v_dim, base_v_dim)

    def forward(self, state, action):
        q1 = self.q1(state, action)
        q2 = self.q2(state, action)
        return q1,q2

    def Q1(self, state, action):
        q1 = self.q1(state, action)
        return q1
