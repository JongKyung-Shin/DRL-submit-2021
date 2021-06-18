import torch
import numpy as np 
class rec_env:
    def __init__(self):
        self.observation_space = torch.Tensor(np.zeros((1,1549)))
        self.action_space = torch.Tensor(np.zeros((1,8603)))
    def step(action):
        reward = 0
        next_state = False # 다음
        done = False
        return next_state, reward, done
    def reset():
        pass
    