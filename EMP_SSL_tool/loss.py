import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# パッチ毎の特徴量の平均を近づける
class Similarity_Loss(nn.Module):
    def __init__(self, ):
        super().__init__()
        pass

    def forward(self, z_list, z_avg):
        z_sim = 0
        num_patch = len(z_list)
        z_list = torch.stack(list(z_list), dim=0)
        #print("z_list.shape : ", z_list.shape)       # torch.Size([20, 100, 1024])
        z_avg = z_list.mean(dim=0)
        #print("z_avg.shape : ", z_avg.shape)         # torch.Size([100, 1024])
        
        #print(xzy)
        
        z_sim = 0
        for i in range(num_patch):
            z_sim += F.cosine_similarity(z_list[i], z_avg, dim=1).mean()
            
        z_sim = z_sim/num_patch
        z_sim_out = z_sim.clone().detach()
                
        return -z_sim, z_sim_out
    


class MultiCropContrastive_Loss_v2(nn.Module):
    def __init__(self, temp):
        
        super(MultiCropContrastive_Loss_v2, self).__init__()
        
        self.criterion = nn.CrossEntropyLoss()
        self.temp = temp
        
    
    def forward(self, features):
        
        num_patch = len(features)
        batch_size = features[0].shape[0]
        
        # 特徴量の平均を計算
        z_list = torch.stack(list(features), dim=0)
        z_avg = z_list.mean(dim=0)
        #print("z_avg.shape : ", z_avg.shape)                    # torch.Size([100, 1024])
        
        # 特徴量の平均をパッチ数分だけ連結 （ここをなくしたい）
        #z_avg = torch.cat([z_avg]*num_patch, dim=0)
        
        
        # 特徴量の連結
        features = torch.cat(features, dim=0)
        #print("features.shape : ", features.shape)              # torch.Size([2000, 1024])
        
        
        # ラベルの作成
        labels = torch.cat([torch.arange(len(z_avg))], dim=0)
        #print("labels.shape : ", labels.shape)                  # torch.Size([100])
        
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        #print("labels.shape : ", labels.shape)                  # torch.Size([100, 100])
        
        labels = torch.cat([labels]*num_patch, dim=0)
        labels = labels.cuda()
        #print("labels.shape : ", labels.shape)                  # torch.Size([2000, 100])
        
        """
        print("labels[0] : ", labels[0])
        print("labels[98] : ", labels[98])
        print("labels[99] : ", labels[99])
        print("labels[100] : ", labels[100])
        print("labels[101] : ", labels[101])
        print("labels[102] : ", labels[102])
        """
        
        # 特徴量の正規化
        features = F.normalize(features, dim=1)
        z_avg =F.normalize(z_avg, dim=1)
        
        
        # 類似度行列の計算
        similarity_matrix = torch.matmul(features, z_avg.T)
        #print("similarity_matrix.shape : ", similarity_matrix.shape)       # torch.Size([2000, 100])
        
        
        # 正例の計算
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
        #print("positives.shape : ", positives.shape)         # torch.Size([2000, 1])
        
        
        # 負例の計算
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)
        #print("negatives.shape : ", negatives.shape)         # torch.Size([2000, 99])
        
        
        # logitsの計算
        logits = torch.cat([positives, negatives], dim=1)
        #print("logits.shape : ", logits.shape)              # torch.Size([2000, 100])
        
        # labelsの再作成
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        
        logits = logits / self.temp
        
        loss = self.criterion(logits, labels)
        return loss
    
    
class DisSimilarity_Loss(nn.Module):
    
    def __init__(self, ):
        super().__init__()
        pass
    
    
    def forward(self, z_list, z_avg):
        
        z_dissim = 0
        num_patch = len(z_list)
        z_list = torch.stack(list(z_list), dim=0)
        #print("z_list.shape : ", z_list.shape)       # torch.Size([20, 100, 1024])
        z_avg = z_list.mean(dim=0)
        #print("z_avg.shape : ", z_avg.shape)         # torch.Size([100, 1024])
    
        z_dissim = 0
        count = 0
        for i in range(num_patch):
            for j, z in enumerate(z_list[i]):     # i番目のパッチの中の一つの特徴量zを取り出す
                for k, avg in enumerate(z_avg):   # 特徴量の平均avg
                    if j != k:
                        #print("z.shape : ", z.shape)
                        #print("avg.shape : ", avg.shape)
                        z_dissim += F.cosine_similarity(z, avg, dim=0) - 1.0
                        count += 1
                        #print("z_dissim : ", z_dissim)
                        #print(xyz)
        
        z_dissim = z_dissim / count
        
        return z_dissim

class DisSimilarity_Loss_Efficiency(nn.Module):
    
    def __init__(self, ):
        super().__init__()
        pass
    
    
    def forward(self, z_list, z_avg):
        
        num_patch = len(z_list)
        z_list = torch.stack(list(z_list), dim=0)
        #print("z_list.shape : ", z_list.shape)       # torch.Size([20, 100, 1024])
        z_avg = z_list.mean(dim=0)
        #print("z_avg.shape : ", z_avg.shape)         # torch.Size([100, 1024])
        
        # 各パッチについて、全ての特徴量と全平均との間のコサイン類似度を計算
        # 形状: [num_patch, num_features, num_features]
        cos_sim = F.cosine_similarity(z_list[:, :, None, :], z_avg[None, None, :, :], dim=3) - 1
        
        #print("cos_sim.shape : ", cos_sim.shape)
        
         # 自己比較を除外するためのマスクを作成
        mask = ~torch.eye(cos_sim.size(1), dtype=torch.bool).to(cos_sim.device)
    
        # マスクを適用して不類似度を計算
        cos_sim = cos_sim * mask[None, :, :]

        # 不類似度の合計とカウント
        z_dissim = cos_sim.sum()
        count = mask.sum() * num_patch

        # 平均不類似度を計算
        z_dissim = z_dissim / count
        
        return z_dissim
        
    
# 負例を使用したInfoNCE Lossの改良版
class InfoNCE_Loss_ver2(nn.Module):
    def __init__(self, temp):
        
        super(InfoNCE_Loss_ver2, self).__init__()
        
        self.criterion = nn.CrossEntropyLoss()
        self.temp = temp
        
    
    def forward(self, features):
        
        num_patch = len(features)
        batch_size = features[0].shape[0]
        
        # 特徴量の平均を計算
        z_list = torch.stack(list(features), dim=0)
        z_avg = z_list.mean(dim=0)
        #print("z_avg.shape : ", z_avg.shape)                    # torch.Size([100, 1024])
        
        # 特徴量の平均をパッチ数分だけ連結 （ここをなくしたい）
        #z_avg = torch.cat([z_avg]*num_patch, dim=0)
        
        
        # 特徴量の連結
        features = torch.cat(features, dim=0)
        #print("features.shape : ", features.shape)              # torch.Size([2000, 1024])
        
        
        # ラベルの作成
        labels = torch.cat([torch.arange(len(z_avg))], dim=0)
        #print("labels.shape : ", labels.shape)                  # torch.Size([100])
        
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        #print("labels.shape : ", labels.shape)                  # torch.Size([100, 100])
        
        labels = torch.cat([labels]*num_patch, dim=0)
        labels = labels.cuda()
        #print("labels.shape : ", labels.shape)                  # torch.Size([2000, 100])
        
        """
        print("labels[0] : ", labels[0])
        print("labels[98] : ", labels[98])
        print("labels[99] : ", labels[99])
        print("labels[100] : ", labels[100])
        print("labels[101] : ", labels[101])
        print("labels[102] : ", labels[102])
        """
        
        # 特徴量の正規化
        features = F.normalize(features, dim=1)
        z_avg =F.normalize(z_avg, dim=1)
        
        
        # 類似度行列の計算
        similarity_matrix = torch.matmul(features, z_avg.T)
        #print("similarity_matrix.shape : ", similarity_matrix.shape)       # torch.Size([2000, 100])
        
        
        # 正例の計算
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
        #print("positives.shape : ", positives.shape)         # torch.Size([2000, 1])
        
        
        # 負例の計算
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)
        #print("negatives.shape : ", negatives.shape)         # torch.Size([2000, 99])
        
        
        # logitsの計算
        logits = torch.cat([positives, negatives], dim=1)
        #print("logits.shape : ", logits.shape)              # torch.Size([2000, 100])
        
        # labelsの再作成
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        
        logits = logits / self.temp
        
        loss = self.criterion(logits, labels)
        return loss
        
        
    
# 負例を使用したInfoNCE Loss
class InfoNCE_Loss(nn.Module):
    def __init__(self, temp):
        super().__init__()
        
        self.criterion = nn.CrossEntropyLoss()
        self.temp = temp
        
        
    
    def forward(self, features):
        
        #print("len(features) : ", len(features))            # 10  (パッチ数が10の時)
        #print("features[0].shape : ", features[0].shape)    # torch.Size([100, 1024])
        num_patch = len(features)
        batch_size =  features[0].shape[0]
        
        
        
        # 特徴量の平均を計算
        #print("features[0][0] : ", features[0][0])
        #print("features[0][1] : ", features[0][1])
        z_list = torch.stack(list(features), dim=0)
        #print("z_list.shape : ", z_list.shape)       # z_list.shape :  torch.Size([10, 100, 1024])
        #print("z_list[0][0] : ", z_list[0][0])
        #print("z_list[0][1] : ", z_list[0][1])
        z_avg = z_list.mean(dim=0)
        #print("z_avg.shape : ", z_avg.shape)         # z_avg.shape :  torch.Size([100, 1024])
        
        z_avg = torch.cat([z_avg]*num_patch, dim=0)
        #print("z_avg.shape : ", z_avg.shape)         # torch.Size([1000, 1024])
        #print("z_avg[0] : ", z_avg[0])
        #print("z_avg[100] : ", z_avg[100])
        
        
        # 特徴量の連結
        #print("features[0][0] : ", features[0][0])
        #print("features[0][1] : ", features[0][1])
        features = torch.cat(features, dim=0)
        #print("featuers[0] : ", features[0])
        #print("features[1] : ", features[1])
        #print("features.shape : ", features.shape)          # torch.Size([1000, 1024])
        
        
        #labels = torch.cat([torch.arange(batch_size) for i in range(num_patch)], dim=0)
        #print("labels.shape : ", labels.shape)               # torch.Size([1000])
        labels = torch.cat([torch.arange(len(features))], dim=0)
        #print("labels.shape : ", labels.shape)               #torch.Size([1000])
        #print("labels : ", labels)
        
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        #print("labels.shape : ", labels.shape)               # torch.Size([1000, 1000])
        #print("labels : ", labels)
        
        
        labels = labels.cuda()
        #print("labels.shape : ", labels.shape)    # torch.Size([1000, 1000])
        #print("labels : ", labels)                # 対角成分が1の1000*1000の行列
        #print("labels.sum() : ", labels.sum())    # 
        
        features = F.normalize(features, dim=1)
        z_avg = F.normalize(z_avg, dim=1)
        
        
        # 類似度行列の計算
        similarity_matrix = torch.matmul(features, z_avg.T)
        #print("similarity_matrix.shape : ", similarity_matrix.shape)    # torch.Size([1000, 1000])
        
        
        # マスクの作成
        #mask = torch.eye(labels.shape[0], dtype=torch.bool).cuda()
        #print("mask : ", mask)
        #print("mask.shape : ", mask.shape)          # torch.Size([1000, 1000])
        
        # maskを使用して自己との類似度を測らないようにする（必要ない）
        #labels = labels[~mask].view(labels.shape[0], -1)
        #print("labels : ", labels)
        #print("labels.shape : ", labels.shape)      # ([1000, 999])
        
        # maskを使用して自己との類似度を測らないようにする（必要ない）
        #similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        #print("similarity_matrix : ", similarity_matrix)
        #print("similarity_matrix.shape : ", similarity_matrix.shape)   # torch.Size([1000, 999])
        
        
        # 正例の計算
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
        #print("positives.shape : ", positives.shape)                               # torch.Size([1000, 1])
        #print("positives[0:3] : ", positives[0:3])
        
        
        negatives = similarity_matrix[~labels.bool()].view(labels.shape[0], -1)
        #print("negatives.shape : ", negatives.shape)                               # torch.Size([1000, 999])
        #print("negatives : ", negatives)
        
        
        logits = torch.cat([positives, negatives], dim=1)
        #print("logits.shape : ", logits.shape)                                     # torch.Size([1000, 1000])
        #print("logits : ", logits)
        
        
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        #print("labels : ", labels)
        #print("labels.shape : ", labels.shape)  # torch.Size([1000])
        
        
        logits = logits / self.temp
        
        loss = self.criterion(logits, labels)
        return loss
    
    
    
# past/momentum encoderの出力にencoderの出力が近づくための損失
class Forgetting_Similarity_Loss(nn.Module):
    def __init__(self, ):
        super().__init__()
        pass
    
    def forward(self, z_list, z_avg, z_list_extra, z_avg_extra):
        
        z_sim = 0
        num_patch = len(z_list)
        
        # 現在のencoder
        z_list = torch.stack(list(z_list), dim=0)
        z_avg = z_list.mean(dim=0)

        # past/momentum encoder
        z_list_extra = torch.stack(list(z_list_extra), dim=0)
        z_avg_extra = z_list_extra.mean(dim=0)
        
        
        z_sim = 0
        for i in range(num_patch):
            z_sim += F.cosine_similarity(z_list[i], z_avg_extra, dim=1).mean()
        
        z_sim = z_sim/num_patch
        z_sim_out = z_sim.clone().detach()
        
        return -z_sim, z_sim_out
        
        
    

# fast-mocoのcombineを用いて，近づける特徴量の数を増やす．
class Similarity_Loss_v2(nn.Module):
    def __init__(self, ):
        super().__init__()
        pass
    
    
    def forward(self, z_list, z_avg):
        
        z_sim = 0
        
        #print("len(z_list) : ", len(z_list))        # 80
        z_list = torch.stack(list(z_list), dim=0)   
        z_avg = z_list.mean(dim=0)
        #print("z_list.shape : ", z_list.shape)      # torch.Size([80, 100, 1024])
        #print("z_avg.shape : ", z_avg.shape)        # torch.Size([100, 1024])
        
        
        #num_patch = len(z_list)
        #z_sim = 0
        #for i in range(num_patch):
        #    z_sim += F.cosine_similarity(z_list[i], z_avg, dim=1).mean()
        #z_sim = z_sim/num_patch
        #print("z_sim : ", z_sim)
        
        
        
        """ divide & combine の操作  """
        divide = 5
        partition = int(len(z_list) / divide)
        #print("partition : ", partition)        # 16
        z_list_divide1 = z_list[0:partition]
        z_list_divide2 = z_list[partition:partition*2]
        z_list_divide3 = z_list[partition*2:partition*3]
        z_list_divide4 = z_list[partition*3:partition*4]
        z_list_divide5 = z_list[partition*4:partition*5]
        
        # 形状確認
        #print("len(z_list_divide1) : ", len(z_list_divide1))    # 16
        #print("len(z_list_divide2) : ", len(z_list_divide2))    # 16
        #print("len(z_list_divide3) : ", len(z_list_divide3))    # 16
        #print("len(z_list_divide4) : ", len(z_list_divide4))    # 16
        #print("len(z_list_divide5) : ", len(z_list_divide5))    # 16
        
        #print("z_list_divide1[0].shape : ", z_list_divide1[0].shape)   # torch.Size([100, 1024])
        #print("z_list_divide2[0].shape : ", z_list_divide2[0].shape)   # torch.Size([100, 1024])
        #print("z_list_divide3[0].shape : ", z_list_divide3[0].shape)   # torch.Size([100, 1024])
        #print("z_list_divide4[0].shape : ", z_list_divide4[0].shape)   # torch.Size([100, 1024])
        #print("z_list_divide5[0].shape : ", z_list_divide5[0].shape)   # torch.Size([100, 1024])
        
        
        # クラス（データ）毎に特徴量の平均を計算し，近づけるデータ数の数を増やす
        tmp_list = []
        z_list_extra = []
        for i in range(len(z_list_divide1)):
            
            tmp = (z_list_divide1[i]+z_list_divide2[i]) / 2.0
            tmp_list.append(tmp)
            
            tmp = (z_list_divide1[i]+z_list_divide3[i]) / 2.0
            tmp_list.append(tmp)
            
            tmp = (z_list_divide1[i]+z_list_divide4[i]) / 2.0
            tmp_list.append(tmp)
            
            tmp = (z_list_divide1[i]+z_list_divide5[i]) / 2.0
            tmp_list.append(tmp)
            
            
            tmp = (z_list_divide2[i]+z_list_divide3[i]) / 2.0
            tmp_list.append(tmp)
            
            tmp = (z_list_divide2[i]+z_list_divide4[i]) / 2.0
            tmp_list.append(tmp)
            
            tmp = (z_list_divide2[i]+z_list_divide5[i]) / 2.0
            tmp_list.append(tmp)
            
            
            tmp = (z_list_divide3[i]+z_list_divide4[i]) / 2.0
            tmp_list.append(tmp)
            
            tmp = (z_list_divide3[i]+z_list_divide5[i]) / 2.0
            tmp_list.append(tmp)
            
            
            tmp = (z_list_divide4[i]+z_list_divide5[i]) / 2.0
            tmp_list.append(tmp)
            
            #print("len(tmp_list) : ", len(tmp_list))            # 10
            #print("tmp_list[0].shape : ", tmp_list[0].shape)    # torch.Size([100, 1024])
            
            
        
        z_list_extra = torch.stack(tmp_list)
        #print("z_list_extra.shape : ", z_list_extra.shape)      # torch.Size([160, 100, 1024])
        
        z_extra_avg = z_list_extra.mean(dim=0)
        #print("z_extra_avg.shape : ", z_extra_avg.shape)        # torch.Size([100, 1024])
        
        # 二つとも同じになることを確認
        #print("z_avg[0][0:10] : ", z_avg[0][0:10])
        #print("z_extra_avg[0][0:10] : ", z_extra_avg[0][0:10])
        
        z_list = torch.cat((z_list, z_list_extra), dim=0)
        #print("z_list.shape : ", z_list.shape)                  # torch.Size([240, 100, 1024])
        
        
        num_patch = z_list.shape[0]
        z_sim = 0
        for i in range(num_patch):
            z_sim += F.cosine_similarity(z_list[i], z_avg, dim=1).mean()
            
        z_sim = z_sim/num_patch
        z_sim_out = z_sim.clone().detach()
        
        #print("z_sim : ", z_sim)
        
        #print(xyz)
        
        
        return -z_sim, z_sim_out

    
class TotalCodingRate(nn.Module):
    
    def __init__(self, eps=0.01):
        super(TotalCodingRate, self).__init__()
        self.eps = eps
    
    def compute_discrimn_loss(self, W):
        
        p, m = W.shape
        #print("p : ", p)     # 次元数．　(1024)
        #print("m : ", m)     # 画像数． (100)
        
        I = torch.eye(p, device=W.device)
        scalar = p / (m * self.eps)
        logdet = torch.logdet(I + scalar * W.matmul(W.T))
        return logdet / 2.
    
    def forward(self, X):
        return - self.compute_discrimn_loss(X.T)
    

class TotalCodingRate4Contrastive(nn.Module):
    
    def __init__(self, eps=0.01, alpha=0.01):
        
        super(TotalCodingRate4Contrastive, self).__init__()
        self.eps = eps
        self.alpha = alpha
        
    def compute_discrimn_loss(self, W):
        
        p, m = W.shape
        
        I = torch.eye(p, device=W.device)
        scalar = p / (m * self.eps)
        logdet = torch.logdet(I + scalar * W.matmul(W.T))
        return logdet / 2.
    
    def forward(self, X):
        
        logdet = self.compute_discrimn_loss(X.T)
        #self.logdet = logdet
        loss = torch.exp(1. / (logdet * self.alpha))
        
        return loss, logdet
