import torch
from torch.utils.data import RandomSampler, Sampler
from torch.nn import functional as F 

import math
import random
import numpy as np
from collections import deque
 

    
"""  通常バッファ用のBatchSampler  """
class SimpleBufferBatchSampler(Sampler):
    
    def __init__(self,
                 buffer_size: int,
                 repeat: int,
                 sampler:Sampler,
                 batch_size: int) -> None:
        
        self.sampler = sampler
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.repeat = repeat
        
        
        # バッファの初期化
        self.buffer = []
        self.db_head = 0
        self.num_batches_seen = 0
        self.num_batches_yielded = 0
        self.batch_history = 0
        self.init_from_ckpt = False
        
    
    
    """  確認済みのバッチ数を加算  """
    def advance_batches_seen(self):
        self.num_batches_seen += 1
        return self.num_batches_seen
    
    
    
    """  バッファからデータをサンプリング  """
    def sample_k(self, q, k):
        
        if len(q)<2*k:
            return random.sample(q, k=0)
        elif k <= len(q):
            return random.sample(q[:-k], k=k)
        else:
            return random.choice(q[:-k], k=k)
    
    
    
    """  バッファにデータを追加  """
    def add_to_buffer(self, n):
        
        if self.db_head >= len(self.all_indices):
            return True
        
        indices_to_add = self.all_indices[self.db_head:self.db_head + n]
        for idx in indices_to_add:
            self.buffer += [{
                'idx': idx,
                'feature': None,
                'lifespan': 0,
                'label': None,
                'seen': False,
                
            }]
            
        self.db_head += len(indices_to_add)
        
        
        # lifespanを加算
        for b in self.buffer:
            b['lifespan'] += 1
        
        return False, indices_to_add
    
    
    
    """  バッファ内データを削除  """
    def resize_buffer(self, n):
        
        # n : バッファサイズ
        
        n2rm = len(self.buffer) - n
        if n2rm <= 0:
            return
        
        
        
    
    
    
    
    """  __iter__()を定義  """
    def __iter__(self):
        
        self.all_indices = list(self.sampler)
        if not self.init_from_ckpt:
            self.db_head = 0
            self.num_batches_seen = 0
            self.num_batches_yielded = 0
            self.batch_history = deque(maxlen=128)
            
        
        for i in range(self.num_batches_yielded - self.num_batches_seen, 0, -1):
            
            yield self.batch_history[-i]
            
        
        assert self.buffer_size <= len(self.all_indices)
        
        
        
        while self.num_batches_yielded < len(self):
            
            # self.bufferにデータを追加
            done, indices_to_add = self.add_to_buffer(self.batch_size)
            
            # 入力バッチの確認
            self.indices_to_add = indices_to_add
            
            
            for j in range(self.repeat):
                
                """  入力バッチとメモリバッチを連結して学習  """
                
                # 入力バッチの作成
                batch_idx_incom = [idx for idx in self.indices_to_add]
                
                # メモリバッチの作成
                batch = self.sample_k(self.buffer, self.batch_size)
                batch_idx_mem = [b['idx'] for b in batch]
                
                # 入力バッチとメモリバッチの連結
                batch_idx = batch_idx_incom + batch_idx_mem
                
                
                self.num_batches_yielded += 1
                self.batch_history += [batch_idx]
                
                
                yield batch_idx
                
            
            # バッファ内データの削除
            self.resize_buffer(self.buffer_size)
                
        
        self.init_from_ckpt = False
        
    def __len__(self) -> int:
        return len(self.sampler) * self.repeat // self.batch_size
        
        
        
def tensorize_buffer(buffer):
    
    buffer_tensor = {}
    for k in buffer[0]:
        
        tens_list = [s[k] for s in buffer]
        if all(t is None for t in tens_list):
            continue
        if k == "fn":
            continue
        
        dummy = [t for t in tens_list if t is not None][0] * 0.
        tens_list = [t if t is not None else dummy for t in tens_list]
        
        try:
            if isinstance(tens_list[0], torch.Tensor):
                tens = torch.stack(tens_list)
            elif isinstance(tens_list[0], (int, bool, float)):
                tens = torch.tensor(tens_list)
            else:
                tens = torch.tensor(tens_list)
            buffer_tensor[k] = tens
        except Exception as e:
            print(e)
    
    return buffer_tensor
        
        


    
"""  MinRedバッファ用のBatchSampler  """
class MinRedBufferBatchSampler(Sampler):
    
    def __init__(self,
                 buffer_size: int,
                 repeat: int,
                 data_type: str,
                 sampler: Sampler,
                 batch_size: int) -> None:
        
        
        self.sampler = sampler
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.repeat = repeat
        
        self.gamma = 0.5
        
        # バッファ内データのカウント用
        self.label_dict = {}
        
        # バッファの初期化
        self.buffer = []
        self.db_head = 0
        self.num_batches_seen = 0
        self.num_batches_yielded = 0
        self.batch_history = 0
        self.init_from_ckpt = False
        self.delete_buffer = None
        
        if data_type=="cifar10":
            self.num_class = 10
            self.num_task = 10
        elif data_type=='cifar100':
            self.num_class = 20
            self.num_task = 20
        elif data_type=="split-cifar10":
            self.num_class = 5
            self.num_task = 5
        elif data_type=='imagenet100':
            self.num_class = 100
            self.num_task = 10
        elif data_type=='core50':
            self.num_class = 10
            self.num_task=10
        else:
            assert False
     
    
    
    """  バッファ内データのラベルをカウント  """
    def buffer_data_count(self):
        
        # 各ラベルのカウントを初期化
        self.label_dict = {f"D{i}": 0 for i in range(self.num_task)}
            
        # バッファ内の各データに対してカウント
        for b in self.buffer:
            try:
                #print("b['label'].item() : ", b['label'].item())
                label = f"D{b['label'].item()}"
                self.label_dict[label] += 1
            except:
                zxcvb = 1
    
    
    """  バッファ内データを返却  """
    def return_label_count(self):
        return self.label_dict
    
    
    
    
    """ バッファにデータを追加  """
    def add_to_buffer(self, n):
        if self.db_head >= len(self.all_indices):
            return True
        
        # バッファにインデックスを保存
        indices_to_add = self.all_indices[self.db_head:self.db_head + n]
        for idx in indices_to_add:
            self.buffer += [{
                'idx': idx,          # datasetのインデックス
                'feature': None,     # データの特徴量
                'lifespan': 0,       # バッファ滞在期間
                'label': None,       # データのラベル
                'seen': False,       # データを使用したかのフラグ
            }]
        self.db_head += len(indices_to_add)
        
        
        # lifespanを加算
        for b in self.buffer:
            b['lifespan'] += 1
        
        
        return False, indices_to_add
        
    
    
    """  バッファ内データのサンプリング  """
    def sample_k(self, q, k):
        
        # バッファ内データの数が,2*self.batch_size未満の場合,入力バッチのみで学習を行う
        if len(q) < 2*k:
            return random.sample(q, k=0)
        # バッファ内データがバッチサイズ以上なら，重複なしでランダムにバッチとして取り出す
        elif k <= len(q):
            return random.sample(q[:-k], k=k)
        else:
            return random.choices(q[:-k], k=k)
    
    
    def advance_batches_seen(self):
        
        self.num_batches_seen += 1
        return self.num_batches_seen
    
    
    """  バッファ内のデータを削除  """
    def resize_buffer(self, n):
        
        # n : バッファサイズ
        
        # 削除するデータ数の確認
        n2rm = len(self.buffer) - n
        if n2rm <= 0:
            return
        
        
        # 類似度が高いサンプルを一つずつ削除
        def max_coverage_reduction(x, n2rm):
            
            sim = (torch.einsum('ad,bd->ab', x, x) + 1) / 2
            sim.fill_diagonal_(-10.)
            
            idx2rm = []
            for i in range(n2rm):
                neigh_sim = sim.max(dim=1)[0]
                most_similar_idx = torch.argmax(neigh_sim)
                idx2rm += [most_similar_idx.item()]
                sim.index_fill_(0, most_similar_idx, -10.)
                sim.index_fill_(1, most_similar_idx, -10.)
            return idx2rm

        
        
        
        # self.bufferからインデックスとバッファの内容を取り出す
        buffer = [(b, i) for i, b in enumerate(self.buffer) if b['seen']]
        
        
        # 削除対象のデータ数が少ない場合
        if len(buffer) < 2 * n2rm:
            lifespans = [b['lifespan'] for b in self.buffer]
            idx2rm = torch.tensor(lifespans).argsort(
                descending=True)[:n2rm].tolist()
            
        else:
            # 類似度が最も高い5つを計算
            feats = torch.stack([b['feature'] for b, i in buffer], 0)
            
            # idx2rmはbufferのインデックスが含まれる?
            idx2rm = max_coverage_reduction(feats, n2rm)
            
            # bufferのインデックスからself.bufferのインデックスを獲得?
            idx2rm = [buffer[i][1] for i in idx2rm]
            
        idx2rm =set(idx2rm)
        
        self.buffer = [b for i, b in enumerate(self.buffer) if i not in idx2rm]
        
        self.delete_buffer = [b for i, b in enumerate(self.buffer) if i in idx2rm]
    
    
    
    
    """  バッファ内データの特徴量などを更新  """
    def update_sample_stats(self, sample_info):
        
        # 辞書     データセットのインデックス : self.bufferのインデックス
        db2buff = {b['idx']: i for i, b in enumerate(self.buffer)}
        
        #print("db2buf : ", db2buff)
        #print("len(db2buff) : ", len(db2buff))
        #print("len(self.buffer) : ", len(self.buffer))
        
        sample_index = sample_info['meta']['index'].detach().cpu()    # データセットのインデックス
        sample_label = sample_info['meta']['label']                   # データのラベル
        
        z = sample_info['feature'][:].detach()
        sample_features = F.normalize(z, p=2, dim=-1)
        
        def polyak_avg(val, avg, gamma):
            return (1 - gamma) * val + gamma * avg
        
        #print("sample_index : ", sample_index)
        #print("db2buff : ", db2buff)
        
        for i in range(len(sample_index)):
            db_idx = sample_index[i].item()         # データセットのインデックス
            if db_idx in db2buff:
                b = self.buffer[db2buff[db_idx]]    # バッファ
                
                if not b['seen']:
                    b['feature'] = sample_features[i]
                else:
                    b['feature'] = F.normalize(polyak_avg(
                        b['feature'], sample_features[i], self.gamma),
                                               p=2,
                                               dim=-1)
                    #b['feature'] = sample_features[i]
                b['label'] = sample_label[i]
                b['seen'] = True
            #print("b['label'] : ", b['label'])
                
        samples = [
            self.buffer[db2buff[idx]] for idx in sample_index.tolist() if idx in db2buff
        ]
        
        if not samples:
            return {}
        else:
            return tensorize_buffer(samples)

    
    
    
    """  __iter__()の定義  """ 
    def __iter__(self):
        
        self.all_indices = list(self.sampler)
        #print("self.all_indices : ", self.all_indices)
        
        if not self.init_from_ckpt:
            self.db_head = 0
            self.num_batches_seen = 0
            self.num_batches_yielded = 0
            self.batch_history = deque(maxlen=128)
        
        
        # モデルが見ていないバッチを再送信
        for i in range(self.num_batches_yielded - self.num_batches_seen, 0, -1):
            
            yield self.batch_history[-i]
        
        
        # バッファサイズが全データ数よりも少ないかを確認
        assert self.buffer_size <= len(self.all_indices)
        
        
        
        while self.num_batches_yielded < len(self):
            
            # self.bufferにデータを追加
            done, indices_to_add = self.add_to_buffer(self.batch_size)
            
            #print("indices_to_add : ", indices_to_add)
            
            # 入力バッチの確認
            self.indices_to_add = indices_to_add
            
            
            for j in range(self.repeat):

                """  入力バッチとメモリバッチを連結して学習  """

                # 入力バッチの作成
                batch_idx_incom = [idx for idx in self.indices_to_add]
                
                # メモリバッチの作成
                batch = self.sample_k(self.buffer, self.batch_size)
                batch_idx_mem = [b['idx'] for b in batch]
                
                # 入力バッチとメモリバッチの連結
                batch_idx = batch_idx_incom + batch_idx_mem
                
                
                self.num_batches_yielded += 1
                self.batch_history += [batch_idx]
                
                yield batch_idx
                
            
            # バッファ内のデータをカウント
            # num_updates=1のときは，この位置でデータをカウントする．
            #self.buffer_data_count()
            #print("self.label_dict : ", self.label_dict)
            
            # バッファ内データの削除
            self.resize_buffer(self.buffer_size)
            
            
            # バッファ内のデータをカウント
            # この位置だと，num_updates=1の時に挙動がおかしくなる
            self.buffer_data_count()
            #print("self.label_dict : ", self.label_dict)
            
            
            
            
            
            
        self.init_from_ckpt = False
    
    
    
    def fetch_buffer_data(self):
        
        #print("len(self.buffer) : ", len(self.buffer))
        num_data = len(self.buffer)
        #print("self.buffer['index'] : ", self.buffer['index']) 
        
        idx = [buffer['idx'] for buffer in self.buffer]
        #print("idx : ", idx)
        
        
        return idx
    
    
    def fetch_feature_label(self):
        
        num_data = len(self.buffer)
        
        #features = [buff['feature'] for buff in self.buffer]
        #label = [buff['label'] for buff in self.buffer]
        
        features = [buff['feature'] for buff in self.buffer if buff['feature'] is not None]
        labels = [buff['label'].item() for buff in self.buffer if buff['feature'] is not None]
        #print("len(features) : ", len(features))
        
        #print("labels : ", labels)
        
        return features, labels
        
    
    def fetch_delete_feature_label(self):
        
        num_data = len(self.buffer)
        features = [buff['feature'] for buff in self.delete_buffer if buff['feature'] is not None]
        labels = [buff['label'].item() for buff in self.delete_buffer if buff['feature'] is not None]
    
        return features, labels
    
    
    def fetch_stream_data(self):
        
        return self.indices_to_add
    
    
    
    
    
    def __len__(self) -> int:
        return len(self.sampler) * self.repeat // self.batch_size
    
    
    #def set_epoch(self, epoch: int) -> None:
    #    self.epoch = epoch
    #    self.sampler.set_epoch(epoch=epoch)



"""  MinRedバッファ用のBatchSampler  """
class MinRedBufferBatchSampler_confirm_buff(Sampler):
    
    def __init__(self,
                 buffer_size: int,
                 repeat: int,
                 data_type: str,
                 sampler: Sampler,
                 batch_size: int) -> None:
        
        
        self.sampler = sampler
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.repeat = repeat
        
        self.gamma = 0.5
        
        # バッファ内データのカウント用
        self.label_dict = {}
        
        # バッファの初期化
        self.buffer = []
        self.db_head = 0
        self.num_batches_seen = 0
        self.num_batches_yielded = 0
        self.batch_history = 0
        self.init_from_ckpt = False
        self.delete_buffer = None
        
        if data_type=="cifar10":
            self.num_class = 10
            self.num_task = 10
        elif data_type=='cifar100':
            self.num_class = 20
            self.num_task = 20
        elif data_type=="split-cifar10":
            self.num_class = 5
            self.num_task = 5
        elif data_type=='imagenet100':
            self.num_class = 100
            self.num_task = 10
        elif data_type=='core50':
            self.num_class = 10
            self.num_task=10
        else:
            assert False
        
        
        # バッファ内データの入れ替わりを確認するため
        self.now_task = None
        self.num_change_buffer_data = [None, None, None, None]
     
    
    
    """  現在学習中のタスクを設定  """
    def define_now_task(self, now_task):
        
        self.now_task = now_task
        #print("self.now_task : ", self.now_task)
        
        
    def return_num_change_buffer_data(self):
        
        return self.num_change_buffer_data
    
    """  バッファ内データのラベルをカウント  """
    def buffer_data_count(self):
        
        # 各ラベルのカウントを初期化
        self.label_dict = {f"D{i}": 0 for i in range(self.num_task)}
            
        # バッファ内の各データに対してカウント
        for b in self.buffer:
            try:
                #print("b['label'].item() : ", b['label'].item())
                label = f"D{b['label'].item()}"
                self.label_dict[label] += 1
            except:
                zxcvb = 1
    
    
    """  バッファ内データを返却  """
    def return_label_count(self):
        return self.label_dict
    
    
    
    
    """ バッファにデータを追加  """
    def add_to_buffer(self, n):
        if self.db_head >= len(self.all_indices):
            return True
        
        # バッファにインデックスを保存
        indices_to_add = self.all_indices[self.db_head:self.db_head + n]
        for idx in indices_to_add:
            self.buffer += [{
                'idx': idx,          # datasetのインデックス
                'feature': None,     # データの特徴量
                'lifespan': 0,       # バッファ滞在期間
                'label': None,       # データのラベル
                'seen': False,       # データを使用したかのフラグ
            }]
        self.db_head += len(indices_to_add)
        
        
        # lifespanを加算
        for b in self.buffer:
            b['lifespan'] += 1
        
        
        return False, indices_to_add
        
    
    
    """  バッファ内データのサンプリング  """
    def sample_k(self, q, k):
        
        # バッファ内データの数が,2*self.batch_size未満の場合,入力バッチのみで学習を行う
        if len(q) < 2*k:
            return random.sample(q, k=0)
        # バッファ内データがバッチサイズ以上なら，重複なしでランダムにバッチとして取り出す
        elif k <= len(q):
            return random.sample(q[:-k], k=k)
        else:
            return random.choices(q[:-k], k=k)
    
    
    def advance_batches_seen(self):
        
        self.num_batches_seen += 1
        return self.num_batches_seen
    
    
    """  バッファ内のデータを削除  """
    def resize_buffer(self, n):
        
        # n : バッファサイズ
        
        # 削除するデータ数の確認
        n2rm = len(self.buffer) - n
        if n2rm <= 0:
            return
        
        
        """
        def max_coverage_reduction(x, n2rm):
            
            sim = (torch.einsum('ad,bd->ab', x, x) + 1) / 2
            sim.fill_diagonal_(-10.)
            
            idx2rm = []
            for i in range(n2rm):
                neigh_sim = sim.max(dim=1)[0]
                most_similar_idx = torch.argmax(neigh_sim)
                idx2rm += [most_similar_idx.item()]
                sim.index_fill_(0, most_similar_idx, -10.)
                sim.index_fill_(1, most_similar_idx, -10.)
            return idx2rm

        """
        
        
        # 類似度が高いサンプルを一つずつ削除
        def max_coverage_reduction(x, n2rm):
            
            print("x.shape : ", x.shape)     # torch.Size([228, 1024])
            
            
            sim = (torch.einsum('ad,bd->ab', x, x) + 1) / 2
            sim.fill_diagonal_(-10.)
            
            #print("sim.shape : ", sim.shape)       # torch.Size([228, 228])
            
            idx2rm = []
            most_simil = []
            for i in range(n2rm):
                
                # torch.tensor.max()は，最大値とインデックスを返す
                neigh_sim_0 = sim.max(dim=1)[0]             # 最大値
                neigh_sim_1 = sim.max(dim=1)[1]             # インデックス
                #print("neigh_sim_0.shape : ", neigh_sim_0.shape)       # torch.Size([228])
                #print("neigh_sim_1.shape : ", neigh_sim_1.shape)
                #print("neigh_sim_0[10:15] : ", neigh_sim_0[10:15])
                #print("neigh_sim_1[10:15] : ", neigh_sim_1[10:15])
                #print(asdfg)

                # torch.argmax()は，全ての要素の中で最大値のインデックスを返す
                most_similar_idx_0 = torch.argmax(neigh_sim_0)           # バッファから削除するデータのインデックス
                most_similar_idx_1 = neigh_sim_1[most_similar_idx_0]     # バッファから削除するデータと最も類似度が高いインデックス
                #print("most_similar_idx_0 : ", most_similar_idx_0)
                #print("most_similar_idx_1 : ", most_similar_idx_1)
                #print("most_similar_idx_0.shape : ", most_similar_idx_0.shape)
                #print("most_similar_idx_1.shape : ", most_similar_idx_1.shape)
                
                #print()
                #print(asd)
                
                
                idx2rm += [most_similar_idx_0.item()]
                most_simil += [most_similar_idx_1.item()]
                #print("len(idx2rm) : ", len(idx2rm))           # 下と同じ
                #print("len(omst_simil) : ", len(most_simil))   # 上と同じ
                
                sim.index_fill_(0, most_similar_idx_0, -10.)
                sim.index_fill_(1, most_similar_idx_0, -10.)
            return idx2rm, most_simil

        
        
        
        # self.bufferからインデックスとバッファの内容を取り出す
        buffer = [(b, i) for i, b in enumerate(self.buffer) if b['seen']]
        
        
        # 削除対象のデータ数が少ない場合
        if len(buffer) < 2 * n2rm:
            lifespans = [b['lifespan'] for b in self.buffer]
            idx2rm = torch.tensor(lifespans).argsort(
                descending=True)[:n2rm].tolist()
            
        else:
            # 類似度が最も高い5つを計算
            feats = torch.stack([b['feature'] for b, i in buffer], 0)
            
            # idx2rmはbufferのインデックスが含まれる?
            idx2rm, most_simil = max_coverage_reduction(feats, n2rm)
            
            #print("most_simil : ", most_simil)
            
            # bufferのインデックスからself.bufferのインデックスを獲得?
            idx2rm = [buffer[i][1] for i in idx2rm]           # 削除するデータのインデックス
            most_simil = [buffer[i][1] for i in most_simil]   # 削除するデータと最も類似するデータ
            
            
            #print("---- aaaa ----")
            #print("len(idx2rm) : ", len(idx2rm))           # 下と同じ
            #print("len(omst_simil) : ", len(most_simil))   # 上と同じ
            #print()
        
        
        # バッファから削除するデータの類似度の組み合わせ
        del_label = []
        for i in idx2rm:
            del_label += [self.buffer[i]['label'].item()]
        self.del_label = del_label
        
        most_simil_label = []
        for i in most_simil:
            most_simil_label += [self.buffer[i]['label'].item()]
        self.most_simil_label = most_simil_label

        # バッファから削除するデータ_最も類似度が高いデータ = []
        new_new = 0
        new_old = 0
        old_new = 0
        old_old = 0
        for i in range(len(self.del_label)):
            if self.del_label[i] == self.now_task and self.most_simil_label[i] == self.now_task:
                new_new += 1
            elif self.del_label[i] == self.now_task and self.most_simil_label[i] != self.now_task:
                new_old += 1
            elif self.del_label[i] != self.now_task and self.most_simil_label[i] == self.now_task:
                old_new += 1
            elif self.del_label[i] != self.now_task and self.most_simil_label[i] != self.now_task:
                old_old += 1
        
        self.new_new = new_new
        self.new_old = new_old
        self.old_new = old_new
        self.old_old = old_old
        
        #print("self.new_new : ", self.new_new)
        #print("self.new_old : ", self.new_old)
        #print("self.old_new : ", self.old_new)
        #rint("self.old_old : ", self.old_old)
                
        self.num_change_buffer_data = [self.new_new, self.new_old, self.old_new, self.old_old]
        
        
        #if del_label[0] == 1:
        #    print("del_label : ", del_label)
        #    print("most_label : ", most_simil_label)
        
        
        idx2rm =set(idx2rm)
        
        self.buffer = [b for i, b in enumerate(self.buffer) if i not in idx2rm]
        
        self.delete_buffer = [b for i, b in enumerate(self.buffer) if i in idx2rm]
    
    
    
    
    """  バッファ内データの特徴量などを更新  """
    def update_sample_stats(self, sample_info):
        
        # 辞書     データセットのインデックス : self.bufferのインデックス
        db2buff = {b['idx']: i for i, b in enumerate(self.buffer)}
        
        #print("db2buf : ", db2buff)
        #print("len(db2buff) : ", len(db2buff))
        #print("len(self.buffer) : ", len(self.buffer))
        
        sample_index = sample_info['meta']['index'].detach().cpu()    # データセットのインデックス
        sample_label = sample_info['meta']['label']                   # データのラベル
        
        z = sample_info['feature'][:].detach()
        sample_features = F.normalize(z, p=2, dim=-1)
        
        def polyak_avg(val, avg, gamma):
            return (1 - gamma) * val + gamma * avg
        
        #print("sample_index : ", sample_index)
        #print("db2buff : ", db2buff)
        
        for i in range(len(sample_index)):
            db_idx = sample_index[i].item()         # データセットのインデックス
            if db_idx in db2buff:
                b = self.buffer[db2buff[db_idx]]    # バッファ
                
                if not b['seen']:
                    b['feature'] = sample_features[i]
                else:
                    b['feature'] = F.normalize(polyak_avg(
                        b['feature'], sample_features[i], self.gamma),
                                               p=2,
                                               dim=-1)
                    #b['feature'] = sample_features[i]
                b['label'] = sample_label[i]
                b['seen'] = True
            #print("b['label'] : ", b['label'])
                
        samples = [
            self.buffer[db2buff[idx]] for idx in sample_index.tolist() if idx in db2buff
        ]
        
        if not samples:
            return {}
        else:
            return tensorize_buffer(samples)

    
    
    
    """  __iter__()の定義  """ 
    def __iter__(self):
        
        self.all_indices = list(self.sampler)
        #print("self.all_indices : ", self.all_indices)
        
        if not self.init_from_ckpt:
            self.db_head = 0
            self.num_batches_seen = 0
            self.num_batches_yielded = 0
            self.batch_history = deque(maxlen=128)
        
        
        # モデルが見ていないバッチを再送信
        for i in range(self.num_batches_yielded - self.num_batches_seen, 0, -1):
            
            yield self.batch_history[-i]
        
        
        # バッファサイズが全データ数よりも少ないかを確認
        assert self.buffer_size <= len(self.all_indices)
        
        
        
        while self.num_batches_yielded < len(self):
            
            # self.bufferにデータを追加
            done, indices_to_add = self.add_to_buffer(self.batch_size)
            
            #print("indices_to_add : ", indices_to_add)
            
            # 入力バッチの確認
            self.indices_to_add = indices_to_add
            
            
            for j in range(self.repeat):

                """  入力バッチとメモリバッチを連結して学習  """

                # 入力バッチの作成
                batch_idx_incom = [idx for idx in self.indices_to_add]
                
                # メモリバッチの作成
                batch = self.sample_k(self.buffer, self.batch_size)
                batch_idx_mem = [b['idx'] for b in batch]
                
                # 入力バッチとメモリバッチの連結
                batch_idx = batch_idx_incom + batch_idx_mem
                
                
                self.num_batches_yielded += 1
                self.batch_history += [batch_idx]
                
                yield batch_idx
                
            
            # バッファ内のデータをカウント
            # num_updates=1のときは，この位置でデータをカウントする．
            #self.buffer_data_count()
            #print("self.label_dict : ", self.label_dict)
            
            # バッファ内データの削除
            self.resize_buffer(self.buffer_size)
            
            
            # バッファ内のデータをカウント
            # この位置だと，num_updates=1の時に挙動がおかしくなる
            self.buffer_data_count()
            #print("self.label_dict : ", self.label_dict)
            
            
            
            
            
            
        self.init_from_ckpt = False
    
    
    
    def fetch_buffer_data(self):
        
        #print("len(self.buffer) : ", len(self.buffer))
        num_data = len(self.buffer)
        #print("self.buffer['index'] : ", self.buffer['index']) 
        
        idx = [buffer['idx'] for buffer in self.buffer]
        #print("idx : ", idx)
        
        
        return idx
    
    
    def fetch_feature_label(self):
        
        num_data = len(self.buffer)
        
        #features = [buff['feature'] for buff in self.buffer]
        #label = [buff['label'] for buff in self.buffer]
        
        features = [buff['feature'] for buff in self.buffer if buff['feature'] is not None]
        labels = [buff['label'].item() for buff in self.buffer if buff['feature'] is not None]
        #print("len(features) : ", len(features))
        
        #print("labels : ", labels)
        
        return features, labels
        
    
    def fetch_delete_feature_label(self):
        
        num_data = len(self.buffer)
        features = [buff['feature'] for buff in self.delete_buffer if buff['feature'] is not None]
        labels = [buff['label'].item() for buff in self.delete_buffer if buff['feature'] is not None]
    
        return features, labels
    
    
    def fetch_stream_data(self):
        
        return self.indices_to_add
    
    
    
    
    
    def __len__(self) -> int:
        return len(self.sampler) * self.repeat // self.batch_size
    
    
    #def set_epoch(self, epoch: int) -> None:
    #    self.epoch = epoch
    #    self.sampler.set_epoch(epoch=epoch)
    
    
    
    
"""  Randomバッファ用のBatchSampler  """
class RandomBufferBatchSampler(Sampler):
    
    def __init__(self,
                 buffer_size: int,
                 repeat: int,
                 data_type: str,
                 sampler: Sampler,
                 batch_size: int) -> None:
        
        
        self.sampler = sampler
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.repeat = repeat
        
        self.gamma = 0.5
        
        # バッファ内データのカウント用
        self.label_dict = {}
        
        # バッファの初期化
        self.buffer = []
        self.db_head = 0
        self.num_batches_seen = 0
        self.num_batches_yielded = 0
        self.batch_history = 0
        self.init_from_ckpt = False
        
        if data_type=="cifar10":
            self.num_class = 10
            self.num_task = 10
        elif data_type=='cifar100':
            self.num_class = 20
            self.num_task = 20
        elif data_type=="split-cifar10":
            self.num_class = 5
            self.num_task = 5
        elif data_type=='imagenet100':
            self.num_class = 100
            self.num_task = 10
        elif data_type=='core50':
            self.num_class = 10
            self.num_task=10
        else:
            assert False
     
    
    
    """  バッファ内データのラベルをカウント  """
    def buffer_data_count(self):
        
        # 各ラベルのカウントを初期化
        self.label_dict = {f"D{i}": 0 for i in range(self.num_task)}
            
        # バッファ内の各データに対してカウント
        for b in self.buffer:
            try:
                #print("b['label'].item() : ", b['label'].item())
                label = f"D{b['label'].item()}"
                self.label_dict[label] += 1
            except:
                zxcvb = 1
    
    
    """  バッファ内データを返却  """
    def return_label_count(self):
        return self.label_dict
    
    
    
    
    """ バッファにデータを追加  """
    def add_to_buffer(self, n):
        if self.db_head >= len(self.all_indices):
            return True
        
        # バッファにインデックスを保存
        indices_to_add = self.all_indices[self.db_head:self.db_head + n]
        for idx in indices_to_add:
            self.buffer += [{
                'idx': idx,          # datasetのインデックス
                'feature': None,     # データの特徴量
                'lifespan': 0,       # バッファ滞在期間
                'label': None,       # データのラベル
                'seen': False,       # データを使用したかのフラグ
            }]
        self.db_head += len(indices_to_add)
        
        
        # lifespanを加算
        for b in self.buffer:
            b['lifespan'] += 1
        
        
        return False, indices_to_add
        
    
    
    """  バッファ内データのサンプリング  """
    def sample_k(self, q, k):
        
        # バッファ内データの数が,2*self.batch_size未満の場合,入力バッチのみで学習を行う
        if len(q) < 2*k:
            return random.sample(q, k=0)
        # バッファ内データがバッチサイズ以上なら，重複なしでランダムにバッチとして取り出す
        elif k <= len(q):
            return random.sample(q[:-k], k=k)
        else:
            return random.choices(q[:-k], k=k)
    
    
    def advance_batches_seen(self):
        
        self.num_batches_seen += 1
        return self.num_batches_seen
    
    
    """  バッファ内のデータを削除  """
    def resize_buffer(self, n):
        
        # n : バッファサイズ
        
        # 削除するデータ数の確認
        n2rm = len(self.buffer) - n
        if n2rm <= 0:
            return
        
        
        # 類似度が高いサンプルを一つずつ削除
        def max_coverage_reduction(x, n2rm):
            
            sim = (torch.einsum('ad,bd->ab', x, x) + 1) / 2
            sim.fill_diagonal_(-10.)
            
            idx2rm = []
            for i in range(n2rm):
                neigh_sim = sim.max(dim=1)[0]
                most_similar_idx = torch.argmax(neigh_sim)
                idx2rm += [most_similar_idx.item()]
                sim.index_fill_(0, most_similar_idx, -10.)
                sim.index_fill_(1, most_similar_idx, -10.)
            return idx2rm

        
        
        
        # self.bufferからインデックスとバッファの内容を取り出す
        buffer = [(b, i) for i, b in enumerate(self.buffer) if b['seen']]
        
        
        # 削除対象のデータ数が少ない場合
        if len(buffer) < 2 * n2rm:
            lifespans = [b['lifespan'] for b in self.buffer]
            idx2rm = torch.tensor(lifespans).argsort(
                descending=True)[:n2rm].tolist()
            
        else:
            # ランダムでデータを削除
            feats = torch.stack([b['feature'] for b, i in buffer], 0)
            #print("feats : ", feats)
            
            num_feats = feats.shape[0]
            print("num_feats : ", num_feats)
            
            # idx2rmはbufferのインデックスが含まれる?
            idx2rm = random.sample(range(num_feats), n2rm)
            
            # bufferのインデックスからself.bufferのインデックスを獲得?
            idx2rm = [buffer[i][1] for i in idx2rm]
        
        #print("idx2rm : ", idx2rm)
        #print(asdfg)
        idx2rm =set(idx2rm)
        
        self.buffer = [b for i, b in enumerate(self.buffer) if i not in idx2rm]
        #print("len(self.buffer) : ", len(self.buffer))
        
    
    
    
    
    """  バッファ内データの特徴量などを更新  """
    def update_sample_stats(self, sample_info):
        
        # 辞書     データセットのインデックス : self.bufferのインデックス
        db2buff = {b['idx']: i for i, b in enumerate(self.buffer)}
        
        
        #print("db2buf : ", db2buff)
        #print("len(db2buff) : ", len(db2buff))
        #print("len(self.buffer) : ", len(self.buffer))
        
        sample_index = sample_info['meta']['index'].detach().cpu()    # データセットのインデックス
        sample_label = sample_info['meta']['label']                   # データのラベル
        
        z = sample_info['feature'][:].detach()
        sample_features = F.normalize(z, p=2, dim=-1)
        
        def polyak_avg(val, avg, gamma):
            return (1 - gamma) * val + gamma * avg
        
        #print("sample_index : ", sample_index)
        #print("db2buff : ", db2buff)
        
        for i in range(len(sample_index)):
            db_idx = sample_index[i].item()         # データセットのインデックス
            if db_idx in db2buff:
                b = self.buffer[db2buff[db_idx]]    # バッファ
                
                if not b['seen']:
                    b['feature'] = sample_features[i]
                else:
                    b['feature'] = F.normalize(polyak_avg(
                        b['feature'], sample_features[i], self.gamma),
                                               p=2,
                                               dim=-1)
                
                b['label'] = sample_label[i]
                b['seen'] = True
            #print("b['label'] : ", b['label'])
                
        samples = [
            self.buffer[db2buff[idx]] for idx in sample_index.tolist() if idx in db2buff
        ]
        
        if not samples:
            return {}
        else:
            return tensorize_buffer(samples)

    
    
    
    """  __iter__()の定義  """ 
    def __iter__(self):
        
        self.all_indices = list(self.sampler)
        #print("self.all_indices : ", self.all_indices)
        
        if not self.init_from_ckpt:
            self.db_head = 0
            self.num_batches_seen = 0
            self.num_batches_yielded = 0
            self.batch_history = deque(maxlen=128)
        
        
        # モデルが見ていないバッチを再送信
        for i in range(self.num_batches_yielded - self.num_batches_seen, 0, -1):
            
            yield self.batch_history[-i]
        
        
        # バッファサイズが全データ数よりも少ないかを確認
        assert self.buffer_size <= len(self.all_indices)
        
        
        
        while self.num_batches_yielded < len(self):
            
            # self.bufferにデータを追加
            done, indices_to_add = self.add_to_buffer(self.batch_size)
            
            #print("indices_to_add : ", indices_to_add)
            
            # 入力バッチの確認
            self.indices_to_add = indices_to_add
            
            
            for j in range(self.repeat):

                """  入力バッチとメモリバッチを連結して学習  """

                # 入力バッチの作成
                batch_idx_incom = [idx for idx in self.indices_to_add]
                
                # メモリバッチの作成
                batch = self.sample_k(self.buffer, self.batch_size)
                batch_idx_mem = [b['idx'] for b in batch]
                
                # 入力バッチとメモリバッチの連結
                batch_idx = batch_idx_incom + batch_idx_mem
                
                
                self.num_batches_yielded += 1
                self.batch_history += [batch_idx]
                
                yield batch_idx
                
            
            # バッファ内のデータをカウント
            # num_updates=1のときは，この位置でデータをカウントする．
            #self.buffer_data_count()
            #print("self.label_dict : ", self.label_dict)
            
            # バッファ内データの削除
            self.resize_buffer(self.buffer_size)
            
            
            # バッファ内のデータをカウント
            # この位置だと，num_updates=1の時に挙動がおかしくなる
            self.buffer_data_count()
            #print("self.label_dict : ", self.label_dict)
            
            
            
            
            
            
        self.init_from_ckpt = False
        
    
    
    def __len__(self) -> int:
        return len(self.sampler) * self.repeat // self.batch_size
    
    
    
    
