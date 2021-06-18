## main 에서 모든 데이터 import 시킨 후 
# state_generator에서 state를 가져고
# Agent에서 action을 가져온다.
# RL_algorithm에서 RL 알고리즘을 가져옴
# main은 깔끔하게 데이터 import만 되어있도록 코드를 작성하기
# %% 
import numpy as np
import json
import pandas as pd
import os
import sys

from torch.distributions.utils import logits_to_probs
from Utils import plot
from state_generator import CNN
from state_generator import state_generation
from Utils import drop_duplicate
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from Agent import Policy_Network
from Agent import agent_train
from tqdm import tqdm
__file__ = "main.py"
dirname = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(dirname)
#필요한 데이터 모두 import
loc_data = pd.read_csv(dirname + "/Data/base_mapping_file.csv")
masking_map = np.load(dirname  + "/Data/mapping_array_by_aisle.npy")
key_list_df = pd.read_csv(dirname + "/Data/key_3_list.csv")
products_by_md = np.load(dirname+"/Data/products_by_md_H_소분류.npy")
price_df = pd.read_csv(dirname + "/Data/price_inf.csv")
with open(dirname + "/Data/json_file/purchase_202001_ver4.json", "r") as jsonfile:
    stream_data = json.load(jsonfile)
stream_data= drop_duplicate(stream_data)
key_list = key_list_df["key"].to_list()
all_product_list = []
small_inf_p_list = []
for key in tqdm(stream_data.keys()):
    for good_inf in stream_data[key]["purchase_list"]:
        good_code = good_inf["상품코드"]
        good_small = good_inf["소분류"]
        if good_code not in all_product_list:
            all_product_list.append(good_code)
        if good_small not in small_inf_p_list:
            small_inf_p_list.append(good_small)
'''
all_product_onehot_dic = {}
for product in tqdm(all_product_list):
    temp = [0]*len(all_product_list)
    idx = all_product_list.index(product)
    temp[idx] = 1
    all_product_onehot_dic[product] = temp
'''
# %%
#Experiment_Setting
GPU_NUM = 3
device = torch.device(f"cuda:{GPU_NUM}") if torch.cuda.is_available() else torch.device("cpu")
# Import environment

#shape
obs_shape = torch.Tensor(np.zeros((1,1549))).shape
act_shape = torch.Tensor(np.zeros((1,8603))).shape
obs_dim = obs_shape[1]
n_actions = act_shape[1]
#Policy model(REINFORCE)
policy_model = Policy_Network(obs_dim = obs_dim, n_actions = n_actions, device = device)
policy_model.to(device)
#Opimizer
optimizer = optim.Adam(policy_model.parameters())
cnn_model = CNN(device = device)
cnn_model.to(device)
optimizer2 = optim.Adam(cnn_model.parameters())
#Training Parameter
total_step = 1000 # iteration 몇번돌지
batch_size = 32
gamma = 0.99
#List for saving results
rewards = []
#List for saving return and logprobablity
logprobs = []
returns = []
loss_list = []
# %%
episode_reward = 0
episode_loss = 0
best_test_reward = -999999
key_reward = 0
key_loss = 0
for epoch in range(total_step):
    print("epoch : ",epoch)
    for key in key_list[1:2]:
        for t in range(0,len(stream_data[key]["purchase_list"])-1, 1):
            print("t : ", t)
            state_vector =  state_generation(t, key, stream_data ,cnn_model, loc_data, masking_map, products_by_md, small_inf_p_list, device)
            print("state_vector :",state_vector)
            next_action = stream_data[key]["purchase_list"][t+1]["상품코드"]
            action, logprob, prob = policy_model.get_action_logprob(state_vector, torch.LongTensor([all_product_list.index(next_action)]) ,device) # index가 나옴
            prob = prob.unsqueeze(dim = 0)
            #print("현재상품명 : ",stream_data[key]["purchase_list"][t]["상품명"])
            
            print("다음 상품코드 : ", next_action)
            print("예측 상품코드", all_product_list[action])
            print("다음상품명 : ",stream_data[key]["purchase_list"][t+1]["상품명"])
            print("현재 통로 위치: ",stream_data[key]["purchase_list"][t]["대분류"])
            
            #print(torch.sum(prob))
            #print(prob)
            criterion = nn.CrossEntropyLoss()
            loss = criterion(prob.to("cpu"), torch.LongTensor([all_product_list.index(next_action)]))
            #print(loss)            
            # if action == all_product_list.index(next_action):
            #     reward = 100*t
            # else:
            #     reward = 1           

            try:
                reward = price_df[price_df.상품코드 == int(all_product_list[action])]["단가"].to_list()[0]
            except:
                reward = 1
            #reward = 1
            print("reward : ", reward)
            logprobs.append(logprob)
            returns.append(reward)
            episode_reward += reward
            episode_loss += loss
            print("**********************")

    for i in range(len(returns)-2, -1,-1):
        returns[i] += returns[i+1]*gamma
    rewards.append(episode_reward)
    loss_list.append(episode_loss/len(returns))
    agent_train(logprobs, returns, optimizer, optimizer2)
    logprobs = []
    returns = []
    episode_reward = 0
    episode_loss = 0
    if epoch%100 == 0:
        plot(loss_list, "mean_loss")
    
