import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal

from collections import OrderedDict
from maml_rl.policies.policy import Policy, weight_init


class ResBlock(nn.Module):

    def __init__(self, layer, ltype="linear"):
        super(ResBlock, self).__init__()

   
        if ltype == "linear":
            self.fc = nn.Sequential(
                                nn.Linear(layer, layer, bias=True),
                                nn.ELU(),
                                nn.Linear(layer, layer, bias=True),
                               )
        elif ltype == "conv1":
            self.fc = nn.Sequential(
                                nn.Conv1d(layer, layer, kernel_size=3,padding=1),
                                nn.ELU(),
                                nn.Conv1d(layer, layer, kernel_size=3,padding=1),
                               )

    def forward(self, x):
        
        h = self.fc(x)
        return F.elu(x + h)


class Head(nn.Module):

    def __init__(self, latent_dim, output_dim_in, output_dim_out):
        super(Head, self).__init__()
        
        h_layer = 1024
        self.output_dim_in = output_dim_in
        self.output_dim_out = output_dim_out
                
        # Q function net
        self.W = nn.Sequential(
            nn.Linear(h_layer, output_dim_in * output_dim_out)
        )

        self.b = nn.Sequential(
            nn.Linear(h_layer, output_dim_out)
        )
        self.s = nn.Sequential(
            nn.Linear(h_layer, output_dim_out)
        )

        self.init_layers()

    def forward(self, x):

        w = self.W(x).view(-1, self.output_dim_out, self.output_dim_in)
        b = self.b(x).view(-1, self.output_dim_out, 1)
        s = 1. + self.s(x).view(-1, self.output_dim_out, 1)

        return w, b, s
   
    def init_layers(self):
        for b in self.b.modules():
            if isinstance(b, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                torch.nn.init.zeros_(b.weight)  

class Task_Embadding(nn.Module):

    def __init__(self, task_dim, z_dim, emb_dim):

        super(Task_Embadding, self).__init__()

        f_layer = 512
        self.z_dim = z_dim
        self.emb_dim = emb_dim

        self.hyper = nn.Sequential(
            nn.Linear(task_dim, self.emb_dim),
            nn.Tanh(),
			nn.Linear(self.emb_dim, f_layer, bias=True),
			nn.ELU(),
			ResBlock(f_layer),
			ResBlock(f_layer),
            ResBlock(f_layer),
            ResBlock(f_layer),
            ResBlock(f_layer),
			nn.Linear(f_layer, self.z_dim, bias=True),
            nn.ELU(),
		)

        self.init_layers()

    def forward(self, task):
		
        # f heads
        z = self.hyper(task)
        return z

    def init_layers(self):

        # init f with fanin
        for module in self.hyper.modules():
            if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(module.weight)
                bound = 1. / math.sqrt(fan_in)
                torch.nn.init.uniform_(module.weight, -bound, bound) 


class Reverse_Hyper_Policy(Policy):
    # Hyper net that create weights from the state for a net that estimates function Q(S, A)
    def __init__(self, task_dim, state_dim, action_dim, num_hidden,use_task=True, var_tasks=1.0 , init_std=1.0, min_std=1e-6):
        super(Reverse_Hyper_Policy, self).__init__(input_size=state_dim, output_size=action_dim)

        self.min_log_std = math.log(min_std)
        self.sigma = nn.Parameter(torch.Tensor(action_dim))
        self.sigma.data.fill_(math.log(init_std))

        self.use_task = True
        self.task_dim = task_dim
        self.state_dim = state_dim
        self.num_hidden = num_hidden
        self.action_dim = action_dim

        self.emb_dim = 64
        self.g_layer = 32
        self.z_dim = 1024
        
        self.hyper = Task_Embadding(state_dim, self.z_dim, self.emb_dim)

        # Q function net
        self.layer1 = Head(self.z_dim,self.emb_dim,self.g_layer)
        self.hidden = nn.ModuleList(Head(self.z_dim,self.g_layer,self.g_layer) for i in range(num_hidden))
        self.last_layer = Head(self.z_dim, self.g_layer, action_dim)

        self.state_emb = nn.Linear(task_dim, self.emb_dim)
    

    def forward(self, state, task, params=None):
        shape = state.shape
        task = task.unsqueeze(0).repeat(shape[0],1)
        if len(shape) ==  3:
            task = task.unsqueeze(1).repeat(1,shape[1],1)


        z = self.hyper(state)

		#state embedding
        emb = torch.tanh(self.state_emb(task).view(-1, self.emb_dim,1))
   
        # g first layer
        w ,b ,s = self.layer1(z)
        out = F.elu(torch.bmm(w, emb) * s + b)
   
        # g hidden layers
        for i, layer in enumerate(self.hidden):
            w, b, s = self.hidden[i](z)
            out = F.elu(torch.bmm(w, out) * s + b)

        # g final layer
        w, b, s = self.last_layer(z)
        mu = F.elu(torch.bmm(w, out) * s + b)
        mu = mu.squeeze().view(*shape[:-1], -1)

        scale = torch.exp(torch.clamp(self.sigma, min=self.min_log_std))
        return Normal(loc=mu, scale=scale)


    def main_net_params(self, task):
        z = self.hyper(task)

        # get parameters
        w ,b ,s = self.layer1(z)
        params = torch.cat((torch.flatten(w),torch.flatten(b),torch.flatten(s)))
        for i, layer in enumerate(self.hidden):
            w, b, s = self.hidden[i](z)
            params = torch.cat((params,torch.flatten(w),torch.flatten(b),torch.flatten(s)))
        w, b, s = self.last_layer(z)
        params = torch.cat((params,torch.flatten(w),torch.flatten(b),torch.flatten(s)))

        return params


        
