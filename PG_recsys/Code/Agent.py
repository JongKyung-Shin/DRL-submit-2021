import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

class Policy_Network(nn.Module):
    def __init__(self, obs_dim, n_actions, device):
        super(Policy_Network, self).__init__()

        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.device = device

        self.l1 = nn.Linear(obs_dim, 64)
        self.l2 = nn.Linear(64, 256)
        self.l3 = nn.Linear(256, n_actions)
        #self.l3 = nn.Linear(128, 4)
    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        x = F.softmax(x, dim=-1)

        return x

    def get_action_logprob(self, obs, next_action, device):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32)
        obs = obs.to(self.device)
        #print(obs)
        output = self.forward(obs)
        categorical = Categorical(output)
        #action = categorical.sample()
        #action = action.to(device)
        #action = torch.argmax(output)
        #next_action = next_action.to(device)
        #logprob = categorical.log_prob(next_action)
        
        action = torch.argmax(output)
        next_action = torch.Tensor([next_action]).to(device)
        logprob = categorical.log_prob(next_action)

        #logprob = categorical.log_prob(action)
        #print("action : {0}, next_action : {1}, logprob :{2}".format(action, next_action,logprob))
        #prob = output.squeeze()[action]

        return action.item(), logprob, output

def agent_train(logprobs, returns, optim, optim2):
    optim.zero_grad()
    # Cumulate gradients
    for ret, logprob in zip(returns, logprobs):
        j = -1 * logprob * ret * 0.001
        j.backward()
    optim.step()
    optim2.step()