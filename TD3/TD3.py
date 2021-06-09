import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from hyper_q_network import Hyper_Critic
from hyper_q_network_reverse import Hyper_Critic_Reverse
from hyper_q_network_random import Hyper_Critic_Random
from hyper_network import Meta_Embadding
from hyper_network import ResBlock

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print (device)

# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477


class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action):
		super(Actor, self).__init__()
		self.l1 = nn.Linear(state_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, action_dim)
		
		self.max_action = max_action
		

	def forward(self, state):
		a = F.relu(self.l1(state))
		a = F.relu(self.l2(a))
		return self.max_action * torch.tanh(self.l3(a))

class Critic(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Critic, self).__init__()
		h_size = 256
		d2rl = False
		state_features = False
		res_35 = False

		if d2rl:
			self.q1 = D2rl_Q(state_dim, action_dim, h_size)
			self.q2 = D2rl_Q(state_dim, action_dim, h_size)

		elif res_35:
			self.q1 = Super_Critic(state_dim, action_dim, h_size)
			self.q2 = Super_Critic(state_dim, action_dim, h_size)

		elif state_features:
			self.q1 = Deep_Critic(state_dim, action_dim, h_size)
			self.q2 = Deep_Critic(state_dim, action_dim, h_size)

		else:
			self.q1 = nn.Sequential(
				nn.Linear(state_dim + action_dim, h_size),
				nn.ReLU(),
				nn.Linear(h_size, h_size),
				nn.ReLU(),
				nn.Linear(h_size, 1)
			)
			# Q2 architecture
			self.q2 = nn.Sequential(
				nn.Linear(state_dim + action_dim, h_size),
				nn.ReLU(),
				nn.Linear(h_size, h_size),
				nn.ReLU(),
				nn.Linear(h_size, 1)
			)

	def forward(self, state, action, logger=None):
		if logger is not None:
			logger['w1'][-1].append(self.l1.weight.detach().cpu().numpy())
			logger['w2'][-1].append(self.l2.weight.detach().cpu().numpy())
		sa = torch.cat([state, action], 1)
		q1 = self.q1(sa)
		q2 = self.q2(sa)
		return q1, q2


	def Q1(self, state, action,logger=None):
		sa = torch.cat([state, action], 1)
		q1 = self.q1(sa)
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

	def forward(self, xu):
		x1 = F.relu(self.l1_1(xu))
		x1 = torch.cat([x1, xu], dim=1)
		x1 = F.relu(self.l1_2(x1))
		x1 = torch.cat([x1, xu], dim=1)
		x1 = F.relu(self.l1_3(x1))
		x1 = torch.cat([x1, xu], dim=1)
		x1 = F.relu(self.l1_4(x1))
		x1 = self.out1(x1)
		return x1

# res35 critic
class Super_Critic(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Super_Critic, self).__init__()
		h_size = 1024
		n_blocks = 35
		
		# Q1 architecture
		self.q1_emb = nn.Sequential(nn.Linear(state_dim, h_size),
		ResBlock(h_size, h_size),
		ResBlock(h_size, h_size),
		ResBlock(h_size, h_size),
		ResBlock(h_size, h_size),
		nn.Linear(h_size, state_dim)
		)

		self.q1 = nn.Sequential(
				nn.Linear(state_dim + action_dim, 256),
				nn.ReLU(),
				nn.Linear(256, 1)
		)

		# Q2 architecture
		self.q2_emb = nn.Sequential(nn.Linear(state_dim, h_size),
		ResBlock(h_size, h_size),
		ResBlock(h_size, h_size),
		ResBlock(h_size, h_size),
		ResBlock(h_size, h_size),
		nn.Linear(h_size, state_dim)
		)

		self.q2 = nn.Sequential(
				nn.Linear(state_dim + action_dim, 256),
				nn.ReLU(),
				nn.Linear(256, 1)
		)


	def init_layers(self):

		for module in self.q1_emb.modules():
			if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear)):
				fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(module.weight)
				bound = 1. / (2. * math.sqrt(fan_in))
				torch.nn.init.uniform_(module.weight, -bound, bound)
		for module in self.q2_emb.modules():
			if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear)):
				fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(module.weight)
				bound = 1. / (2. * math.sqrt(fan_in))
				torch.nn.init.uniform_(module.weight, -bound, bound)
		
	def forward(self, state, action):
		z1 = self.q1_emb(state)
		z2 = self.q2_emb(state)
		sa1 = torch.cat([z1, action], 1)
		sa2 = torch.cat([z2, action], 1)
		q1 = self.q1(sa1)
		q2 = self.q2(sa2)
		return q1, q2


	def Q1(self, state, action, logger):
		z1 = self.q1_emb(state)
		sa1 = torch.cat([z1, action], 1)
		q1 = self.q1(sa1)
		return q1

# state_features critic
class Deep_Critic(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Deep_Critic, self).__init__()
		h_size = 256
		n_blocks = 70
		# Q1 architecture
		self.q1 = nn.Sequential(nn.Linear(state_dim + action_dim, h_size),)
		for i in range(n_blocks):
			self.q1.add_module("block_{}".format(i), ResBlock(h_size, h_size))
		self.q1.add_module("relu_1", nn.ReLU())
		self.q1.add_module("fc",nn.Linear(h_size, 1))

		# Q2 architecture
		self.q2 = nn.Sequential(nn.Linear(state_dim + action_dim, h_size),)
		for i in range(n_blocks):
			self.q2.add_module("block_{}".format(i), ResBlock(h_size, h_size))
		self.q2.add_module("relu_1", nn.ReLU())
		self.q2.add_module("fc",nn.Linear(h_size, 1))

	def forward(self, state, action):
		sa1 = torch.cat([state, action], 1)
		sa2 = torch.cat([state, action], 1)
		q1 = self.q1(sa1)
		q2 = self.q2(sa2)
		return q1, q2


	def Q1(self, state, action, logger):
		sa1 = torch.cat([state, action], 1)
		q1 = self.q1(sa1)
		return q1

class TD3(object):
	def __init__(
		self,
		state_dim,
		action_dim,
		max_action,
		discount=0.99,
		tau=0.005,
		policy_noise=0.2,
		noise_clip=0.5,
		policy_freq=2,
		hyper={},
		hyper_lr=5e-5,
		policy_lr=3e-4,
		logger=None
	):

		self.actor = Actor(state_dim, action_dim, max_action).to(device)
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=policy_lr)
	
		if hyper["no_hyper"]:
			critic = Critic
			lr =  3e-4

		else:
			if hyper["reverse_hyper"]:
				critic = Hyper_Critic_Reverse
			elif hyper["random_hyper"]:
				critic = Hyper_Critic_Random
			else:
				critic = Hyper_Critic
			lr = hyper_lr


		self.critic = critic(state_dim, action_dim).to(device)
		self.critic_target = critic(state_dim, action_dim).to(device)
		self.copy_params(self.critic_target,self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

		self.max_action = max_action
		self.discount = discount
		self.tau = tau
		self.policy_noise = policy_noise
		self.noise_clip = noise_clip
		self.policy_freq = policy_freq
		self.total_it = 0
		self.logger = logger

	def select_action(self, state):
		state = torch.FloatTensor(state.reshape(1, -1)).to(device)
		return self.actor(state).cpu().data.numpy().flatten()


	def train(self, replay_buffer, batch_size=100, debug=False, train_actor=True):

		self.total_it += 1

		# Sample replay buffer 
		state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

		with torch.no_grad():
			# Select action according to policy and add clipped noise
			noise =  (
				torch.randn_like(action) * self.policy_noise
			).clamp(-self.noise_clip, self.noise_clip)
			
			next_action = (
				self.actor_target(next_state) + noise
			).clamp(-self.max_action, self.max_action)

			# Compute the target Q value
			target_Q1, target_Q2 = self.critic_target(next_state, next_action)
			target_Q = torch.min(target_Q1, target_Q2)
			target_Q = reward + not_done * self.discount * target_Q

		# Get current Q estimates
		if debug and (self.total_it-1) % 10000 == 0:
			current_Q1, current_Q2 = self.critic(state, action, self.logger)
		else:
			current_Q1, current_Q2 = self.critic(state, action)
		self.loss1 = F.mse_loss(current_Q1, target_Q)
		self.loss2 = F.mse_loss(current_Q2, target_Q)
		self.logger['loss1'][-1].append(self.loss1.item())

		# Compute critic loss
		critic_loss = self.loss1 + self.loss2

		# Optimize the critic
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		
		self.logger['critic'][-1].append(self.compute_gradient(self.critic.q1.parameters()))
		self.critic_optimizer.step()

		# Delayed policy updates
		if self.total_it % self.policy_freq == 0:
			if train_actor:
		
			# Compute actor loss
			# update to min (Q) for hyper and Q1 for regular
				if debug:
					actor_loss = -self.critic.Q1(state, self.actor(state), self.logger).mean()
				else:
					actor_loss = -self.critic.Q1(state, self.actor(state), None).mean()
			
				self.logger['q1'][-1].append(actor_loss.item())
				# Optimize the actor 
				self.actor_optimizer.zero_grad()

				self.disable_gradients()
				actor_loss.backward() 
				self.logger['policy'][-1].append(self.compute_gradient(self.actor.parameters()))
				self.enable_gradients()
				self.actor_optimizer.step()

			# Update the frozen target models
			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
		
			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

		return self.logger

	def compute_gradient(self, parameters):
		if isinstance(parameters, torch.Tensor):
			parameters = [parameters]
		parameters = [p for p in parameters if p.grad is not None]
		if len(parameters) == 0:
			return torch.tensor(0.)
		device = parameters[0].grad.device
		total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2.0).to(device) for p in parameters]), 2.0)
		return total_norm.item()
	
	def disable_gradients(self):
		for p in self.critic.parameters():
				p.requires_grad = False
	
	
	def enable_gradients(self):
		for p in self.critic.parameters():
				p.requires_grad = True

	def save(self, filename):
		torch.save(self.critic.state_dict(), filename + "_critic")
		torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
		
		torch.save(self.actor.state_dict(), filename + "_actor")
		torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")


	def reduce_lr(self, net_optimizer, lr=1e-5):
		for param_group in net_optimizer.param_groups:
			if param_group['lr'] != lr:
				print ("### lr drop %s ###"%lr)
			param_group['lr'] = lr

	def load(self, filename):
		self.critic.load_state_dict(torch.load(filename + "_critic"))
		self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
		self.critic_target = copy.deepcopy(self.critic)

		self.actor.load_state_dict(torch.load(filename + "_actor"))
		self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
		self.actor_target = copy.deepcopy(self.actor)
		
	def copy_params(self, target, source):
		for target_param, param in zip(target.parameters(), source.parameters()):
			target_param.data.copy_(param.data)