from IPython.display import clear_output
import matplotlib.pyplot as plt
import numpy as np
import torch

def drop_duplicate(stream_data):
    for key in stream_data.keys():
        new_list = []
        good_list = stream_data[key]["purchase_list"]
        for good in good_list:
            if good not in new_list:
                new_list.append(good)
        stream_data[key]["purchase_list"] = new_list
    return stream_data

def plot(rewards,  y_title):
    clear_output(True)
    plt.figure(figsize=(20,5))
    # plt.title('')
    plt.ylabel(y_title)
    plt.xlabel('episode')
    plt.plot(rewards)
    plt.show()

    
class ReplayBuffer:
    """
    https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/ddpg/ddpg.py
    with some modifications
    """
    def __init__(self, obs_shape, act_shape, buffer_size):
        buffer_obs_shape = tuple([buffer_size]) + obs_shape
        buffer_act_shape = tuple([buffer_size]) + act_shape
        self.obs_buf = np.zeros(buffer_obs_shape, dtype=np.float32)
        self.act_buf = np.zeros(buffer_act_shape, dtype=np.float32)
        self.ret_buf = np.zeros(buffer_size, dtype=np.float32)
        self.adv_buf = np.zeros(buffer_size, dtype=np.float32)
        self.prob_old_buf = np.zeros(buffer_size, dtype=np.float32)
        self.pointer, self.size, self.buffer_size = 0, 0, buffer_size

    def store(self, obs, act, ret, adv, prob_old):
        self.obs_buf[self.pointer] = obs
        self.act_buf[self.pointer] = act
        self.ret_buf[self.pointer] = ret
        self.adv_buf[self.pointer] = adv
        self.prob_old_buf[self.pointer] = prob_old
        self.pointer = (self.pointer+1) % self.buffer_size
        self.size = min(self.size+1, self.buffer_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     act=self.act_buf[idxs],
                     ret=self.ret_buf[idxs],
                     adv=self.adv_buf[idxs],
                     prob_old=self.prob_old_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}

    def __len__(self):
        return self.size