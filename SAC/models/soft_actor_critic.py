import torch
import torch.optim as opt
import torch.nn.functional as F
import numpy as np

from .q_network import Critic, Super_Critic
from .hyper_q_network import Hyper_Critic
from .policy_network import PolicyNetwork
from helper import ReplayMemory, copy_params, soft_update

class SoftActorCritic(object):
    def __init__(self,observation_space,action_space, args, logger=None):
        self.s_dim = observation_space.shape[0]
        self.a_dim = action_space.shape[0]
        self.alpha = args.alpha
        self.entropy_tunning = args.entropy_tunning
        self.is_Hyper = not args.no_hyper
        self.logger = logger

        # create component networks
        if self.is_Hyper:
            network = Hyper_Critic
            qlr = args.hyper_lr
        else:
            network = Critic
            qlr = args.lr
            

        self.q_network = network(self.s_dim,self.a_dim,args.hidden_dim).to(args.device)
        self.target_q_network = network(self.s_dim,self.a_dim,args.hidden_dim).to(args.device)
        self.policy_network = PolicyNetwork(self.s_dim, self.a_dim, args.hidden_dim, action_space,args.min_log,
                                args.max_log,args.epsilon, args.device).to(args.device)


        # copy weights from q networks to target networks
        copy_params(self.target_q_network, self.q_network)
        
        # optimizers
        self.policy_network_opt = opt.Adam(self.policy_network.parameters(),args.lr)
        self.q_network_opt = opt.Adam(self.q_network.parameters(),qlr)

        
        self.device = args.device

        # automatic entropy tuning
        if args.entropy_tunning:
            self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(args.device)).item()
            self.log_alpha = torch.tensor([0.], requires_grad=True, device=args.device)
            self.alpha_optim = opt.Adam([self.log_alpha], lr=args.lr)
                
        self.replay_memory = ReplayMemory(int(args.replay_memory_size))

    def get_action(self, s):
        state = torch.FloatTensor(s).to(self.device).unsqueeze(0)
        action, _, _, _ = self.policy_network.sample_action(state)
        return action.detach().cpu().numpy()[0]


    def update_params(self, batch_size, gamma, tau, i):

        states, actions, rewards, next_states, ndones = self.replay_memory.sample(batch_size)
        
        # make sure all are torch tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        ndones = torch.FloatTensor(np.float32(ndones)).unsqueeze(1).to(self.device)

        # compute targets
        with torch.no_grad():
            next_action, next_log_pi,_, _ = self.policy_network.sample_action(next_states)
            next_target_q1, next_target_q2 = self.target_q_network(next_states,next_action)
            next_target_q = torch.min(next_target_q1,next_target_q2) - (self.alpha*next_log_pi).unsqueeze(1)
            next_q = rewards + gamma*ndones*next_target_q

        # compute losses
        q1, q2 = self.q_network(states,actions)

        q1_loss = F.mse_loss(q1,next_q)
        q2_loss = F.mse_loss(q2,next_q)

        q_loss = q1_loss + q2_loss 

        # gradient descent
        self.q_network_opt.zero_grad()
        q_loss.backward()
        self.q_network_opt.step()

        pi, log_pi, mean, log_std = self.policy_network.sample_action(states)

        q1_pi, q2_pi = self.q_network(states,pi)
        min_q_pi = torch.min(q1_pi,q2_pi)
        policy_loss = ((self.alpha * log_pi) - min_q_pi).mean()

        self.policy_network_opt.zero_grad()

        self.disable_critic_gradients()
        policy_loss.backward()
        self.enable_critic_gradients()
         
        self.policy_network_opt.step()

        # alpha loss
        if self.entropy_tunning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = self.log_alpha.exp()
        else:
            alpha_loss = torch.tensor(0.).to(self.device)

        # update target network params
        soft_update(self.target_q_network,self.q_network, tau)

        return self.logger

    def disable_critic_gradients(self):
        for p in self.q_network.parameters():
            p.requires_grad = False
		
    def enable_critic_gradients(self):
        for p in self.q_network.parameters():
            p.requires_grad = True

    def compute_gradient(self, parameters):
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        if len(parameters) == 0:
            return torch.tensor(0.)
        device = parameters[0].grad.device
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2.0).to(device) for p in parameters]), 2.0)
        return total_norm.item()