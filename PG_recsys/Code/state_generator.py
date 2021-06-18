#%%
import numpy as np
from numpy.core.fromnumeric import product
import pandas as pd
import json
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
#%%
def customer_information_function(key, stream_data,device):
    cus_info_list = []
    sex_list = {'M': 0, 'F': 1}
    c_type = {'사업자' : 0, '개인': 1}
    b_type = ['야간업소', '개인', '한식', '커피/베이커리', '분식', '기타', '일식', '중식', '유통업체', '급식', '양식']
    customer_inf_list = ["성별", "회원구분", "업종유형"]
    
    for c_inf in customer_inf_list:
        if c_inf == "성별":
            cus_info_list.append(sex_list[stream_data[key][c_inf]])
        elif c_inf == "회원구분":
            cus_info_list.append(c_type[stream_data[key][c_inf]])
        else:
            temp = [0]*len(b_type)
            idx = b_type.index(stream_data[key][c_inf])
            temp[idx] = 1
            cus_info_list = cus_info_list + temp
    return torch.Tensor(cus_info_list).to(device)

# %%
# map_loc_t : t 시점에서 고객의 위치(구매발생위치), 
# loc_data : 전체 상품 및 위치 데이터, 
# cumulative_map : 지금까지 지나온 매대를 모두 masking한 30*38짜리 array
# masking_map : 매장 도면 상으로 통로 및 매대구분된 30*38짜리 array
def state_mapping_function(map_loc_t, loc_data, cumulative_map, masking_map):
    current_map = np.zeros((30,38))
    current_map[int(map_loc_t[0])][int(map_loc_t[1])] = 100  # 현재위치에 가중치 더 높게
    aisle_t = loc_data[(loc_data["row"] == int(map_loc_t[0])) & (loc_data["column"] == int(map_loc_t[1]))]["통로구분"]
    aisle_id = aisle_t.unique()[0]
    all_location_aisle = loc_data[loc_data["통로구분"] == aisle_id][["row", "column"]].drop_duplicates()
    all_location_aisle = np.array(all_location_aisle)
    for row, col in all_location_aisle:
        cumulative_map[row][col] = 10  # 현재위치를 포함하는 통로 공간정보에 가중치 높게
    state_arr = np.dstack((cumulative_map, masking_map))
    state_arr = np.dstack((state_arr, current_map))
    return state_arr, cumulative_map
# %%
act_shape = tuple([1])
act_shape
# %%
class CNN(nn.Module):
    def __init__(self,device):
        super(CNN, self).__init__()
        # ImgIn shape=(1, 30, 38, 840)
        #    Conv     -> (1, 28, 36, 128)
        #    Pool     -> (1, 14, 18, 128)
        self.device = device
        self.layer1 = nn.Sequential(
            nn.Conv2d(2840, 128, kernel_size=4, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #nn.Upsample(scale_factor = 2, mode ="bilinear", align_corners=False),
        )
        # ImgIn shape=(1, 14, 18, 128)
        #    Conv     -> (1, 12, 16, 128)
        #    Pool     -> (1, 6, 8, 36)
        self.layer2 = nn.Sequential(
            nn.Conv2d(128, 32, kernel_size=4, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #nn.Upsample(scale_factor = 2, mode ="bilinear", align_corners=False),
        )
        
        self.layer3 = nn.Sequential(
            nn.Linear(1536, 256),
            nn.Linear(256, 100),
        )
    
    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        x = x.to(self.device)
        out = self.layer1(x)
        out = self.layer2(out)
        out = torch.squeeze(out)
        out = torch.flatten(out)
        #out = torch.unsqueeze(out, dim = 0)
        #out = self.layer3(out)
        #out = F.softmax(out/3, dim=1)
        #out = torch.squeeze(out)
       
        return out
        
def state_generation(i, key, stream_data, cnn_model, loc_data, masking_map, products_by_md_copy, small_inf_p_list,device):
    
    #가장 최근에 구매가 발생한 매대의 위치 추적
    map_loc_t_row = stream_data[key]["purchase_list"][i]["row"]
    map_loc_t_col = stream_data[key]["purchase_list"][i]["column"]
    map_loc_t = np.array([map_loc_t_row, map_loc_t_col])

    #t시점의 고객 위치 정보와 매장배치도가 합쳐진 vector 생성
    masking_map, _ = state_mapping_function(map_loc_t, loc_data, np.zeros((30,38)), masking_map)
    masking_map = np.transpose(masking_map, (2,0,1))
    masking_map = torch.from_numpy(np.array(masking_map, dtype= np.float32))
    #매대별 물품 정보 텐서 및 차원 변환
    map_loc_t_code = stream_data[key]["purchase_list"][i]["소분류"]
    products_by_md = products_by_md_copy.copy()
    products_by_md[int(map_loc_t_row)][int(map_loc_t_col)][small_inf_p_list.index(map_loc_t_code)] = 100
    products_by_md = np.transpose(products_by_md, (2,0,1))
    products_by_md = torch.from_numpy(np.array(products_by_md, dtype= np.float32))
    
    # element-wise 곱 (아다마르 곱)
    for i in range(masking_map.shape[0]):
        if i == 0:
            main = torch.mul(products_by_md, masking_map[0])
        else:
            temp = torch.mul(products_by_md, masking_map[0])
            main = torch.cat([main, temp])
    main = main.unsqueeze(dim= 0)
    # CNN을 통한 embedding vector
    cnn_embedding = cnn_model(main)
    # 고객 정보와 통합
    custom_inf = customer_information_function(key, stream_data, device)
    state_vector = torch.cat([cnn_embedding, custom_inf])
    #state_vector = cnn_embedding
    return state_vector

# %%

