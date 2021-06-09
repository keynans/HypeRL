import torch
import torch.nn as nn
import torch.nn.functional as F

from .hyper_network import Meta_Embadding
from .hyper_network import ResBlock
from helper import init_weights

class QNetwork(nn.Module):
    def __init__(self,s_dim,a_dim,h_dim):
        super(QNetwork,self).__init__()

        self.linear1 = nn.Linear(s_dim+a_dim,h_dim)
        self.linear2 = nn.Linear(h_dim,h_dim)
        self.linear3 = nn.Linear(h_dim,1)

        self.apply(init_weights)

    def forward(self,s,a):
        x = torch.cat((s,a),dim=1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)

        return x

# Res35 
class Deep_Critic(nn.Module):
	def __init__(self,s_dim,a_dim,h_dim):
		super(Deep_Critic, self).__init__()
		h_size = 256
		n_blocks = 35

		self.q1 = nn.Sequential(nn.Linear(s_dim + a_dim, h_size),)
		for i in range(n_blocks):
			self.q1.add_module("block_{}".format(i), ResBlock(h_size, h_size))
		self.q1.add_module("relu_1", nn.ReLU())
		self.q1.add_module("fc",nn.Linear(h_size, 1))

	def forward(self, state, action):
		sa1 = torch.cat([state, action], 1)
		q1 = self.q1(sa1)
		return q1


class Critic(nn.Module):
    def __init__(self,s_dim,a_dim,h_dim):
        super(Critic,self).__init__()
        D2rl = False
        res35 = False
        feature_emb = False

        if D2rl:
            self.q1 = D2rl_Q(s_dim,a_dim,h_dim)
            self.q2 = D2rl_Q(s_dim,a_dim,h_dim)
        elif res35:
            self.q1 = Deep_Critic(s_dim,a_dim,h_dim)
            self.q2 = Deep_Critic(s_dim,a_dim,h_dim)
        elif feature_emb:
            self.q1 = Super_Critic(s_dim,a_dim,h_dim)
            self.q2 = Super_Critic(s_dim,a_dim,h_dim)
        else:
            self.q1 = QNetwork(s_dim,a_dim,h_dim)
            self.q2 = QNetwork(s_dim,a_dim,h_dim)


        self.apply(init_weights)

    def forward(self,s,a):
        q1 = self.q1(s,a)
        q2 = self.q2(s,a)
        return q1,q2

    def Q1(self, s, a):
        q1 = self.q1(s, a)
        return q1


class D2rl_Q(nn.Module):
	def __init__(self, state_dim, action_dim, h_size):
		super(D2rl_Q, self).__init__()

		in_dim = state_dim + action_dim + h_size
		self.l1_1 = nn.Linear(state_dim + action_dim, h_size)
		self.l1_2 = nn.Linear(in_dim, h_size)
		self.l1_3 = nn.Linear(in_dim, h_size)
		self.l1_4 = nn.Linear(in_dim, h_size)
		self.out1 = nn.Linear(h_size, 1)

	def forward(self, s,a):
		xu = torch.cat([s,a], dim=1)
		x1 = F.relu(self.l1_1(xu))
		x1 = torch.cat([x1, xu], dim=1)
		x1 = F.relu(self.l1_2(x1))
		x1 = torch.cat([x1, xu], dim=1)
		x1 = F.relu(self.l1_3(x1))
		x1 = torch.cat([x1, xu], dim=1)
		x1 = F.relu(self.l1_4(x1))
		x1 = self.out1(x1)
		return x1

# feature embedding
class Super_Critic(nn.Module):
	def __init__(self,s_dim,a_dim,h_dim):
		super(Super_Critic, self).__init__()
		print("super")
		h_size = 256
		# Q1 architecture
		self.q1_emb = nn.Sequential(
			Meta_Embadding(s_dim, 1024),
			nn.Linear(1024, 10)
		)
		self.q1 = nn.Sequential(
				nn.Linear(10 + a_dim, h_size),
				nn.ReLU(),
				nn.Linear(h_size, 1)
		)

		# Q2 architecture
		self.q2_emb = nn.Sequential(
			Meta_Embadding(s_dim, 1024),
			nn.Linear(1024, 10)
		)
		self.q2 = nn.Sequential(
				nn.Linear(10 + a_dim, h_size),
				nn.ReLU(),
				nn.Linear(h_size, 1)
		)

		self.apply(init_weights)

	def forward(self, state, action):
		z1 = self.q1_emb(state)
		z2 = self.q2_emb(state)
		sa1 = torch.cat([z1, action], 1)
		sa2 = torch.cat([z2, action], 1)
		q1 = self.q1(sa1)
		q2 = self.q2(sa2)
		return q1, q2

	def Q1(self, s, a):
		z1 = self.q1_emb(s)
		sa1 = torch.cat([z1, a], 1)
		q1 = self.q1(sa1)
		return q1