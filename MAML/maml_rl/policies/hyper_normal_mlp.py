import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal

from collections import OrderedDict
from maml_rl.policies.policy import Policy, weight_init

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ResBlock(nn.Module):

    def __init__(self, in_size, out_size):
        super(ResBlock, self).__init__()
        self.fc = nn.Sequential(
                            nn.ReLU(),
                            nn.Linear(in_size, out_size),
                            nn.ReLU(),
                            nn.Linear(out_size, out_size),
                            )

    def forward(self, x):
        h = self.fc(x)
        return x + h

class Head(nn.Module):

    def __init__(self, latent_dim, output_dim_in, output_dim_out, sttdev):
        super(Head, self).__init__()
        
        h_layer = latent_dim
        self.output_dim_in = output_dim_in
        self.output_dim_out = output_dim_out
     
        self.W1 = nn.Linear(h_layer, output_dim_in * output_dim_out)
        self.b1 = nn.Linear(h_layer, output_dim_out)
        self.s1 = nn.Linear(h_layer, output_dim_out)

        self.init_layers(sttdev)

    def forward(self, x):

        # weights, bias and scale for dynamic layer
        w = self.W1(x).view(self.output_dim_in, self.output_dim_out)
        b = self.b1(x)
        s = 1. + self.s1(x)
                    
        return w, b, s
   
    def init_layers(self,stddev):
        

        torch.nn.init.uniform_(self.W1.weight, -stddev, stddev)
        torch.nn.init.uniform_(self.b1.weight, -stddev, stddev)
        torch.nn.init.uniform_(self.s1.weight, -stddev, stddev)

        torch.nn.init.zeros_(self.W1.bias)
        torch.nn.init.zeros_(self.s1.bias)
        torch.nn.init.zeros_(self.b1.bias)
                        
class Meta_Embadding(nn.Module):

    def __init__(self, meta_dim, z_dim):

        super(Meta_Embadding, self).__init__()

        self.z_dim = z_dim
        self.hyper = nn.Sequential(

			nn.Linear(meta_dim, 256),
            ResBlock(256, 256),
            ResBlock(256, 256),
            
            nn.Linear(256, 512),
            ResBlock(512, 512),
            ResBlock(512, 512),
            
            nn.Linear(512, 1024),
            ResBlock(1024, 1024),
            ResBlock(1024, 1024),
            
		)

        self.init_layers()

    def forward(self, meta_v):
        z = self.hyper(meta_v).view(-1, self.z_dim)
        return z

    def init_layers(self):

        for module in self.hyper.modules():
            if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(module.weight)
                bound = 1. / (2. * math.sqrt(fan_in))
                torch.nn.init.uniform_(module.weight, -bound, bound)

class Hyper_Network(nn.Module):
    def __init__(self,meta_v_dim, base_v_dim, output_dim):
        super(Hyper_Network, self).__init__()
	    
        dynamic_layer = 256
        z_dim = 1024
        self.output_dim = output_dim
        self.hyper = Meta_Embadding(meta_v_dim, z_dim)

        # Q function net
        self.layer1 = Head(z_dim, base_v_dim, dynamic_layer, sttdev=0.05)
        self.last_layer = Head(z_dim, dynamic_layer, output_dim,  sttdev=0.008)

    def forward(self, meta_v, base_v, debug=None):
        
        # produce dynmaic weights
        z = self.hyper(meta_v)
        w1 ,b1 ,s1 = self.layer1(z)
        w2, b2, s2 = self.last_layer(z)
        
        # dynamic network pass
        out = F.relu(torch.matmul(base_v,w1) * s1 + b1)
        out = torch.matmul(out, w2) * s2 + b2 

        return out

    def get_z(self, meta_v):
        z = self.hyper(meta_v)
        return z

    def dynamic(self, z, base_v):
        w1 ,b1 ,s1 = self.layer1(z)
        w2, b2, s2 = self.last_layer(z)
        
        # dynamic network pass
        out = F.relu(torch.matmul(base_v,w1) * s1 + b1)
        out = torch.matmul(out, w2) * s2 + b2 

        return out


class Hyper_Policy(Policy):
    # Hyper net that create weights from the state for a net that estimates function Q(S, A)
    def __init__(self, task_dim, state_dim, action_dim, num_hidden, use_task=True, var_tasks=1.0 , init_std=1.0, min_std=1e-6):
        super(Hyper_Policy, self).__init__(input_size=state_dim, output_size=action_dim)
        self.task_dim = task_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_hidden = num_hidden  
        self.use_task = use_task         
        self.is_hyper=True 
        
        self.min_log_std = math.log(min_std)
        self.sigma = nn.Parameter(torch.Tensor(action_dim))
        self.sigma.data.fill_(math.log(init_std))

        self.hyper = Hyper_Network(task_dim, state_dim, action_dim)

    def forward(self, state, task, params=None, z=None):

        if z is None:
            z  = self.hyper.get_z(task) 
        
        mu = self.hyper.dynamic(z, state)
        scale = torch.exp(torch.clamp(self.sigma, min=self.min_log_std))

        return Normal(loc=mu, scale=scale) ,mu


        
