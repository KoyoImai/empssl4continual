import torch
from torch.utils.data import RandomSampler, Sampler

import math
import random
import numpy as np


class IidSampler(Sampler):
    
    def __init__(self,
                 dataset,
                 batch_size,
                 trial=7):
        
        
        # データセットの総データ数
        self.num_samples = len(dataset)
        
        # バッチサイズ
        self.batch_size = batch_size
        
        # 確認したバッチ数
        self.num_batches_seen = 0
        
        
        # datasetのラベルをnumpy配列に変換
        if torch.is_tensor(dataset.dataset.targets):
            self.labels = dataset.dataset.targets.detach().cpu().numpy()
        else:
            self.labels = np.array(dataset.dataset.targets)
        
    
    def advance_batches_seen(self):
        
        self.num_batches_seen += 1
        return self.num_batches_seen
    
    
    def __iter__(self):
        
        idx = list(range(self.num_samples))
        random.shuffle(idx)
        
        #print("idx[:50] : ", idx[:50])
        #print("idx[-50:] : ", idx[-50:])
        
        return iter(idx)
        
    
    
    def __len__(self):
        return self.num_samples
        
        
        


class SeqSampler(Sampler):
    
    def __init__(self,
                 dataset,
                 batch_size,
                 blend_ratio=0,
                 n_concurrent_classes=1,
                 train_samples_per_cls=2500,
                 trial=7):
        

        
        
        # データセットの総データ数
        self.num_samples = len(dataset)
        #print("self.num_samples : ", self.num_samples)
        
        # バッチサイズ　（いらないかも）
        self.batch_size = batch_size
        
        # クラスの境界をどの程度ぼかすか
        self.blend_ratio = blend_ratio
        
        # 一度に学習するクラス数
        self.n_concurrent_classes = n_concurrent_classes
        
        # クラス事のサンプル数
        self.train_samples_per_cls = train_samples_per_cls
        
        
        # datasetのラベルをnumpy配列に変換
        if torch.is_tensor(dataset.dataset.targets):
            self.labels = dataset.dtaset.targets.detach().cpu().numpy()
        else:
            self.labels = np.array(dataset.dataset.targets)
            
        self.classes = list(set(self.labels))
        self.n_classes = len(self.classes)
        
        #print("self.classes : ", self.classes)
        # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        
        #print("self.n_classes : ", self.n_classes)
        # 20   
        
        
    def __iter__(self):
        
        cmin = []
        cmax = []
        for i in range(int(self.n_classes / self.n_concurrent_classes)):
            for _ in range(self.n_concurrent_classes):
                cmin.append(i * self.n_concurrent_classes)
                cmax.append((i + 1) * self.n_concurrent_classes)
        print("cmin : ", cmin)
        print("cmax : ", cmax)
        
        # フィルタの作成
        filter_fn = lambda y: np.logical_and(
            np.greater_equal(y, cmin[c]), np.less(y, cmax[c]))
        
        # class incremental入力の設定
        sample_idx = []
        for c in self.classes:
            filtered_train_ind = filter_fn(self.labels)
            #print("filtered_train_ind : ", filtered_train_ind)
            filtered_ind = np.arange(self.labels.shape[0])[filtered_train_ind]
            np.random.shuffle(filtered_ind)
            #print("filtered_ind : ", filtered_ind)
            
            cls_idx = self.classes.index(c)
            if len(self.train_samples_per_cls) == 1:
                sample_num = self.train_samples_per_cls[0]
            else:
                assert len(self.train_samples_per_cls) == len(self.classes)
                sample_num = self.train_samples_per_cls[cls_idx]
                
            
            print("type(sample_num) : ", type(sample_num))
            
            sample_idx.append(filtered_ind.tolist()[:sample_num])
            print('Class [{}, {}): {} samples'.format(cmin[cls_idx], cmax[cls_idx],
                                                      sample_num))

        
        
        # タスクの境界をぼかす
        if self.blend_ratio > 0.0:
            for c in range(len(self.classes)):
                if c > 0:
                    blendable_sample_num = int(min(len(sample_idx[c]), len(sample_idx[c-1])) * self.blend_ratio / 2)
                    blend_prob = np.arange(0.5, 0.05, -0.45 / blendable_sample_num)
                    assert blend_prob.size == blendable_sample_num, 'unmatched sample and probability count'
                    
                    for ind in range(blendable_sample_num):
                        if random.random() < blend_prob[ind]:
                            tmp = sample_idx[c-1][-ind-1]
                            sample_idx[c-1][-ind-1] = sample_idx[c][ind]
                            sample_idx[c][ind] = tmp
                
        
        
        final_idx = []
        for sample in sample_idx:
            final_idx += sample

        print("len(final_idx) : ", len(final_idx))
        
        return iter(final_idx)
        
        
    def __len__(self):
        if len(self.train_samples_per_cls) == 1:
            return self.n_classes * self.train_samples_per_cls[0]
        else:
            return sum(self.train_samples_per_cls)
    
    
    #def set_epoch(self, epoch: int, instance: int = 0) -> None:
        #self.sampler.set_epoch()

        
class CORe50SeqSampler(Sampler):
    
    def __init__(self,
                 dataset,
                 batch_size,
                 blend_ratio=0,
                 n_concurrent_classes=1,
                 train_samples_per_cls=None,
                 trial=7):
        
        # データセットの総データ数
        self.num_samples = len(dataset)
        #print("self.num_samples : ", self.num_samples)
        
        # バッチサイズ　（いらないかも）
        self.batch_size = batch_size
        
        # クラスの境界をどの程度ぼかすか
        self.blend_ratio = blend_ratio
        
        # 一度に学習するクラス数
        self.n_concurrent_classes = n_concurrent_classes
        
        # クラス事のサンプル数
        self.train_samples_per_cls = train_samples_per_cls
        
        
        # datasetのラベルをnumpy配列に変換
        if torch.is_tensor(dataset.labels):
            self.labels = dataset.labels.detach().cpu().numpy()
        else:
            self.labels = np.array(dataset.labels)
            
        self.classes = list(set(self.labels))
        self.n_classes = len(self.classes)
        
        print("self.classes : ", self.classes)
        # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        print(ghjkl)
        
        #print("self.n_classes : ", self.n_classes)
        # 20   
        
        
        
    def __iter__(self):
        
        cmin = []
        cmax = []
        for i in range(int(self.n_classes / self.n_concurrent_classes)):
            for _ in range(self.n_concurrent_classes):
                cmin.append(i * self.n_concurrent_classes)
                cmax.append((i + 1) * self.n_concurrent_classes)
        print("cmin : ", cmin)
        print("cmax : ", cmax)
        
        # フィルタの作成
        filter_fn = lambda y: np.logical_and(
            np.greater_equal(y, cmin[c]), np.less(y, cmax[c]))
        
        # class incremental入力の設定
        sample_idx = []
        for c in self.classes:
            filtered_train_ind = filter_fn(self.labels)
            #print("filtered_train_ind : ", filtered_train_ind)
            filtered_ind = np.arange(self.labels.shape[0])[filtered_train_ind]
            #np.random.shuffle(filtered_ind)
            #print("filtered_ind : ", filtered_ind)
            
            cls_idx = self.classes.index(c)
            if len(self.train_samples_per_cls) == 1:
                sample_num = self.train_samples_per_cls[0]
            else:
                assert len(self.train_samples_per_cls) == len(self.classes)
                sample_num = self.train_samples_per_cls[cls_idx]
                
            
            print("type(sample_num) : ", type(sample_num))
            
            sample_idx.append(filtered_ind.tolist()[:sample_num])
            print('Class [{}, {}): {} samples'.format(cmin[cls_idx], cmax[cls_idx],
                                                      sample_num))

        
        
        # タスクの境界をぼかす
        if self.blend_ratio > 0.0:
            for c in range(len(self.classes)):
                if c > 0:
                    blendable_sample_num = int(min(len(sample_idx[c]), len(sample_idx[c-1])) * self.blend_ratio / 2)
                    blend_prob = np.arange(0.5, 0.05, -0.45 / blendable_sample_num)
                    assert blend_prob.size == blendable_sample_num, 'unmatched sample and probability count'
                    
                    for ind in range(blendable_sample_num):
                        if random.random() < blend_prob[ind]:
                            tmp = sample_idx[c-1][-ind-1]
                            sample_idx[c-1][-ind-1] = sample_idx[c][ind]
                            sample_idx[c][ind] = tmp
                
        
        
        final_idx = []
        for sample in sample_idx:
            final_idx += sample

        print("len(final_idx) : ", len(final_idx))
        
        return iter(final_idx)
        
        
    def __len__(self):
        if len(self.train_samples_per_cls) == 1:
            return self.n_classes * self.train_samples_per_cls[0]
        else:
            return sum(self.train_samples_per_cls)

        
        
class ImageNet100SeqSampler(Sampler):
    
    def __init__(self,
                 dataset,
                 batch_size,
                 blend_ratio=0,
                 n_concurrent_classes=1,
                 train_samples_per_cls=None,
                 trial=7):
        
        
        # 確認したバッチ数
        self.num_batches_seen = 0
        
        # データセットの総データ数
        self.num_samples = len(dataset)
        #print("self.num_samples : ", self.num_samples)
        
        # バッチサイズ　（いらないかも）
        self.batch_size = batch_size
        
        # クラスの境界をどの程度ぼかすか
        self.blend_ratio = blend_ratio
        
        # 一度に学習するクラス数
        self.n_concurrent_classes = n_concurrent_classes
        
        # クラス事のサンプル数
        self.train_samples_per_cls = train_samples_per_cls
        
        
        # datasetのラベルをnumpy配列に変換
        if torch.is_tensor(dataset.labels):
            self.labels = dataset.task_labels.detach().cpu().numpy()
        else:
            self.labels = np.array(dataset.task_labels)
            
        self.classes = list(set(self.labels))
        self.n_classes = len(self.classes)
        
        #print("self.classes : ", self.classes)
        # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        
        #print("self.n_classes : ", self.n_classes)
        # 20   
        
        
        
    def __iter__(self):
        
        cmin = []
        cmax = []
        for i in range(int(self.n_classes / self.n_concurrent_classes)):
            for _ in range(self.n_concurrent_classes):
                cmin.append(i * self.n_concurrent_classes)
                cmax.append((i + 1) * self.n_concurrent_classes)
        print("cmin : ", cmin)
        print("cmax : ", cmax)
        
        # フィルタの作成
        filter_fn = lambda y: np.logical_and(
            np.greater_equal(y, cmin[c]), np.less(y, cmax[c]))
        
        # class incremental入力の設定
        sample_idx = []
        for c in self.classes:
            filtered_train_ind = filter_fn(self.labels)
            #print("filtered_train_ind : ", filtered_train_ind)
            filtered_ind = np.arange(self.labels.shape[0])[filtered_train_ind]
            np.random.shuffle(filtered_ind)
            #print("filtered_ind : ", filtered_ind)
            
            cls_idx = self.classes.index(c)
            if len(self.train_samples_per_cls) == 1:
                sample_num = self.train_samples_per_cls[0]
            else:
                assert len(self.train_samples_per_cls) == len(self.classes)
                sample_num = self.train_samples_per_cls[cls_idx]
                
            
            print("type(sample_num) : ", type(sample_num))
            
            sample_idx.append(filtered_ind.tolist()[:sample_num])
            print('Class [{}, {}): {} samples'.format(cmin[cls_idx], cmax[cls_idx],
                                                      sample_num))

        
        
        # タスクの境界をぼかす
        if self.blend_ratio > 0.0:
            for c in range(len(self.classes)):
                if c > 0:
                    blendable_sample_num = int(min(len(sample_idx[c]), len(sample_idx[c-1])) * self.blend_ratio / 2)
                    blend_prob = np.arange(0.5, 0.05, -0.45 / blendable_sample_num)
                    assert blend_prob.size == blendable_sample_num, 'unmatched sample and probability count'
                    
                    for ind in range(blendable_sample_num):
                        if random.random() < blend_prob[ind]:
                            tmp = sample_idx[c-1][-ind-1]
                            sample_idx[c-1][-ind-1] = sample_idx[c][ind]
                            sample_idx[c][ind] = tmp
                
        
        
        final_idx = []
        for sample in sample_idx:
            final_idx += sample

        print("len(final_idx) : ", len(final_idx))
        
        return iter(final_idx)
        
        
    def __len__(self):
        if len(self.train_samples_per_cls) == 1:
            return self.n_classes * self.train_samples_per_cls[0]
        else:
            return sum(self.train_samples_per_cls) 
    
    def advance_batches_seen(self):
        
        self.num_batches_seen += 1
        return self.num_batches_seen
        
        
"""        
class ImageNet100SeqSampler(Sampler):
    
    def __init__(self,
                 dataset,
                 trial):
        
        # データセットの総数
        self.num_samples = len(dataset)
        
        
        
    def __iter__(self):
        
        # サンプルのインデックス用リストを初期化
        sample_idx = []
        
        # 
        for idx in range(self.num_samples):
            sample_idx.append(idx)
        
        
        return iter(sample_idx)
    
    
    def __len__(self):
        
        return self.num_samples
        
        
"""