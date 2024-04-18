import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# 忘却損失
# 過去モデルの出力の平均に近づけるようにする
class Forgetting_Loss(nn.Module):
    
    def __init__(self, ):
        super().__init__()
        pass
    
    
    def forward(self, z_list, z_list_past):
        
        # パッチ数
        num_patch = len(z_list)
        
        # 現在モデルの特徴量，特徴量の平均
        z_list = torch.stack(list(z_list), dim=0)
        z_avg = z_list.mean(dim=0)
        
        # 過去モデルの特徴量，特徴量の平均
        z_list_past = torch.stack(list(z_list_past), dim=0)
        z_avg_past = z_list_past.mean(dim=0)

        z_sim = 0
        for i in range(num_patch):
            z_sim += F.cosine_similarity(z_list[i], z_avg_past, dim=1).mean()
        
        z_sim = z_sim / num_patch
        z_sim_out = z_sim.clone().detach()
        
        return -z_sim, z_sim_out
            