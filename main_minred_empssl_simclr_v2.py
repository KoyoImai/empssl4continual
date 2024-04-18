"""
コードの説明：
    contrastive_lossをコサイン類似度ではなく，InfoNCE損失に変更する

関連コード：
    datasets.aug
    datasets.datasets_minred_empssl 
    datasets.sampler_minred_empssl
    datasets.batchsampler_minred_empssl
    EMP_SSL_tool.lars
    EMP_SSL_tool.loss

例1：
python main_minred_empssl.py --num_updates 10 --num_patch 80 --log_name ${log_name} --senario {iid/seq} --buffer_type minred --workers 1

例2:
export log_name="r18_empssl_imnet100_minred8B_lr001_patch10_seq_updates3_tcr01_sim200_1_17"
export FLAG="--senario seq --buffer_type minred --buffer_size 1024 --num_patch 10 --workers 1 --optim_lr 0.01 --epoch 1 --data_type imagenet100 --num_updates 3 --infonce_temp 0.07 --patch_sim 200 --empssl_tcr 0.1"
python main_minred_empssl_simclr_v2.py --log_name ${log_name} ${FLAG}

"""


import random
import logging
import os
import copy
import warnings
import torch
import torch.nn as nn
import argparse
import random
import numpy as np
import sys
import wandb
import csv
import torch.optim.lr_scheduler as lr_scheduler    # 学習率スケジューラ


# 自作 or 自作（改作）
from model.model_empssl import encoder                                    # modelの作成
#from datasets.aug import ContrastiveLearningViewGenerator                 # データ拡張
import datasets.datasets_minred_empssl                                    # データセットの作成
from datasets.sampler_minred_empssl import SeqSampler, IidSampler         # Samplerの作成
from datasets.sampler_minred_empssl import ImageNet100SeqSampler          # Samplerの作成
from datasets.sampler_minred_empssl import CORe50SeqSampler               # Samplerの作成
from datasets.batchsampler_minred_empssl import SimpleBufferBatchSampler  # BatchSamplerの作成(Simple)
from datasets.batchsampler_minred_empssl import MinRedBufferBatchSampler  # BatchSamplerの作成(MinRed)
from EMP_SSL_tool.lars import LARSWrapper                                 # 最適化手法
from EMP_SSL_tool.loss import InfoNCE_Loss_ver2, TotalCodingRate          # 損失関数
from EMP_SSL_tool.write_csv import write_csv                              # ~C~U~B~B~C~[~A~M__~AA




# Wandbの設定
#wandb.init(project="empssl_splitcifar10_minred256_lr001_patch20_seq_simclrv2_tcr01_sim50_1_22")
#wandb.init(project="practice4")




# コマンドライン引数の処理
def parse_option():
    
    parser = argparse.ArgumentParser('argument for training')
    
    
    """  その他  """
    parser.add_argument('--workers', type=int, default=10)
    
    
    """  記録周りの引数  """
    parser.add_argument('--log_root_dir', type=str, default='/home/kouyou/MinRed/continuous_ssl_problem/Logs')
    parser.add_argument('--log_name', type=str, default='practice')
    parser.add_argument('--log_ckpt_dir', type=str, default='/home/kouyou/MinRed/continuous_ssl_problem/Logs/checkpoints')
    parser.add_argument('--csv_dir', type=str, default="/home/kouyou/MinRed/continuous_ssl_problem/Logs/csv")
    
    
    
    """  データ周りの引数  """
    #parser.add_argument('--train_filelist', type=str, default='/home/kouyou/CIFAR100')
    parser.add_argument('--data_type', type=str, default='cifar100')
    parser.add_argument('--trial', type=int, default=7)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--blend_ratio', type=float, default=0)
    parser.add_argument('--n_concurrent_classes', type=int, default=1)
    parser.add_argument('--train_samples_per_cls', type=int, nargs='+', default=[2500])
    
    
    """  シナリオ周りの引数  """
    parser.add_argument('--senario', type=str, default='')
    
    
    
    
    """  バッファ周りの引数  """
    parser.add_argument('--buffer_type', type=str, default="minred")
    parser.add_argument('--buffer_size', type=int, default=256)
    parser.add_argument('--num_updates', type=int, default=10)
    
    
    """  最適化周りの引数  """
    parser.add_argument('--optim_lr', type=float, default=0.03)
    parser.add_argument('--optim_momentum', type=int, default=0.9)
    parser.add_argument('--optim_weight_decay', type=float, default=0.0001)
    parser.add_argument('--optim_lr_scheduler', action='store_true')
    
    
    
    """  学習周りの引数  """
    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--start_epoch', type=int, default=0)
    
    
    
    
    """  EMP-SSL周りの引数  """
    parser.add_argument('--num_patch', type=int, default=20)
    parser.add_argument('--empssl_eps', type=float, default=0.2)
    parser.add_argument('--patch_sim', type=int, default=50)
    parser.add_argument('--empssl_tcr', type=float, default=1.0)
    
    
    """  infoNCE周りの引数  """
    parser.add_argument('--infonce_temp', type=float, default=0.07)
    
    """  modelの保存に関する引数  """
    parser.add_argument('--save_prog', type=int, default=1)
    
    
    
    opt = parser.parse_args()
    return opt
    


def main():
    
    
    """ コマンドライン引数の処理  """
    args = parse_option()
    print("args : ", args)
    
    
    """  wandb  """
    wandb.init(project=f"{args.log_name}")
    
    
    """  ディレクトリの作成  """
    # checkpoint用のディレクトリ作成
    logdir = os.path.join(args.log_ckpt_dir, args.log_name)
    os.makedirs(logdir, exist_ok=True)
    
    # csv用のディレクトリ作成
    logdir = os.path.join(args.csv_dir, args.log_name)
    os.makedirs(logdir, exist_ok=True)
    csv_path = logdir
    
    

    """ シード値の固定  """
    random.seed(args.trial)
    np.random.seed(args.trial)
    torch.manual_seed(args.trial)
    
    
    """ GPUの設定  """
    device = torch.device("cuda")
    
    
    """ モデルの定義  """
    if args.data_type == 'cifar100' or args.data_type == 'cifar10' or args.data_type == 'split-cifar10':
        model = encoder(arch="resnet18-cifar")
    elif args.data_type == 'imagenet100' or args.data_type == "core50":
        model = encoder(arch="resnet18-imagenet")
    model.cuda()
    model = nn.DataParallel(model)
    
    
    print("model : ", model)
    
    
    
    """  データセットの定義  """
    
    # データ拡張の定義
    if args.data_type == 'cifar10' or args.data_type == 'cifar100' or args.data_type == "split-cifar10":
        from datasets.aug import ContrastiveLearningViewGenerator
        print("aug4cifar")
        augmentation = ContrastiveLearningViewGenerator(num_patch=args.num_patch, data_type=args.data_type)
    elif args.data_type == 'imagenet100':
        from datasets.aug4imnet import ContrastiveLearningViewGenerator
        print("aug4imanet")
        augmentation = ContrastiveLearningViewGenerator(num_patch=args.num_patch, data_type=args.data_type)
    elif args.data_type == 'core50':
        from datasets.aug4core50 import ContrastiveLearningViewGenerator
        augmentation = ContrastiveLearningViewGenerator(num_patch=args.num_patch, data_type=args.data_type)
        print("aug4core50")
        
    else:
        assert False

    
    # trainfnameを定義
    if args.data_type == 'cifar100':
        trainfname = "/home/kouyou/CIFAR100"
        train_samples_per_cls = args.train_samples_per_cls
    elif args.data_type == 'cifar10':
        trainfname = "/home/kouyou/CIFAR10"
        train_samples_per_cls = args.train_samples_per_cls
    elif args.data_type == 'split-cifar10':
        trainfname = "/home/kouyou/CIFAR10"
        train_samples_per_cls = [10000]
    elif args.data_type == 'imagenet100':
        trainfname = '/home/kouyou/ImageNet-kari/ImageNet100/train_filelist'      # filelist.txtまでのパス
        filelist = 'train_filelist.txt'                                           # filelist.txtの名前
        num_task = 10
        train_samples_per_cls = args.train_samples_per_cls
    elif args.data_type == "core50":
        trainfname = '/home/kouyou/MinRed/continuous_ssl_problem/datasets/empssl_lists/core50/train_filelist_random.txt'
        train_samples_per_cls = args.train_samples_per_cls
    else:
        assert False
    
    
    # datasetの定義
    if args.data_type == 'standard':
        print("Unimplemented")
        assert False
    elif args.data_type == 'sequential':
        print("Unimplemented")
        assert False
    elif args.data_type == 'class_sequential':
        print("Unimplemented")
        assert False
    elif args.data_type == 'cifar10':
        train_dataset = datasets.datasets_minred_empssl.Cifar10Dataset(
            trainfname,
            transforms=augmentation,
            train=True,
            download=True,
            trial=args.trial
        )
    elif args.data_type == 'cifar100':
        train_dataset = datasets.datasets_minred_empssl.Cifar100Dataset(
            trainfname,
            transforms=augmentation,
            train=True,
            download=True,
            trial=args.trial
        )
    elif args.data_type == 'split-cifar10':
        train_dataset = datasets.datasets_minred_empssl.SplitCifar10Dataset(
            trainfname,
            transforms=augmentation,
            train=True,
            download=True,
            trial=args.trial
        )
    elif args.data_type == 'imagenet100':
        train_dataset = datasets.datasets_minred_empssl.ImageNet100Dataset(
            trainfname,
            transforms=augmentation,
            trial=True,
            num_task=num_task,
            #train_samples_per_cls=args.train_samples_per_cls
            train_samples_per_cls=12000
        )
    elif args.data_type == "core50":
        train_dataset = datasets.datasets_minred_empssl.CORe50Dataset(
            trainfname,
            transforms=augmentation,
            #train=True,
            #num_task=num_task
        )
    else:
        assert False
    
    
    # Samplerの定義
    if args.senario == 'iid' and args.data_type != 'imagenet100' and args.data_type != 'core50':

        #train_sampler = IidSampler(train_dataset,
        #                            batch_size=args.batch_size,
        #                            trial=args.trial)
        train_sampler = None
    elif args.senario == 'seq' and args.data_type != 'imagenet100' and args.data_type != 'core50':
        train_sampler = SeqSampler(train_dataset, 
                                   batch_size=args.batch_size,
                                   blend_ratio=args.blend_ratio,
                                   n_concurrent_classes=args.n_concurrent_classes,
                                   train_samples_per_cls=train_samples_per_cls,
                                   trial=args.trial)
    elif args.senario == 'seq' and args.data_type == 'imagenet100':
        train_sampler = ImageNet100SeqSampler(train_dataset,
                                              batch_size=args.batch_size,
                                              blend_ratio=args.blend_ratio,
                                              n_concurrent_classes=args.n_concurrent_classes,
                                              train_samples_per_cls=train_samples_per_cls)
    elif args.senario == 'seq' and args.data_type == 'core50':
        train_sampler = CORe50SeqSampler(train_dataset,
                                         batch_size=args.batch_size,
                                         blend_ratio=args.blend_ratio,
                                         n_concurrent_classes=args.n_concurrent_classes,
                                         train_samples_per_cls=train_samples_per_cls)
    
    else:
        assert False
    
    
    # BatchSamplerの定義
    if args.buffer_type == 'none':
        batch_sampler = None
        
    elif args.buffer_type == 'simple':
        buff_siz = args.buffer_size
        batch_sampler = SimpleBufferBatchSampler(
            buffer_size=buff_siz,
            repeat=args.num_updates,
            sampler=train_sampler,
            batch_size=args.batch_size
        )
    elif args.buffer_type == 'random':
        buff_siz = args.buffer_size
        batch_sampler = MinRedBufferBatchSampler(
            buffer_size=buff_siz,
            repeat=args.num_updates,
            data_type=args.data_type,
            sampler=train_sampler,
            batch_size=args.batch_size
        )
    elif args.buffer_type == 'minred':
        buff_siz = args.buffer_size
        batch_sampler = MinRedBufferBatchSampler(
            buffer_size=buff_siz,
            repeat=args.num_updates,
            data_type=args.data_type,
            sampler=train_sampler,
            batch_size=args.batch_size
        )
    else:
        raise NotImplementedError
    

    
    # train_loaderの定義
    if batch_sampler is not None:
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_sampler=batch_sampler,
            num_workers=args.workers,
            #num_workers = 10,
            pin_memory=True,
            prefetch_factor=1
        )
        
    else:
        if train_sampler is None:
            train_loader = torch.utils.data.DataLoader(
                dataset=train_dataset,
                #sampler=train_sampler,
                shuffle=True,
                batch_size=args.batch_size,
                num_workers=args.workers,
                #num_workers=10,
                pin_memory=True
            )
        elif train_sampler is not None:
            train_loader = torch.utils.data.DataLoader(
                dataset=train_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.workers,
                pin_memory=True,
                sampler=train_sampler,
                prefetch_factor=1
            )
        
    
    print("len(train_loader) : ", len(train_loader))
    
    
    # data_loaderの確認
    # ラベルは順番に現れてることは確認(12/26)  
    """
    f = open("./label.txt", 'a')
    count = 0
    for data in train_loader:
        
        #print("len(data['input']) : ", len(data['input']))
        #print("data['input'][0][0] : ", data['input'][0][0])
        #print("data['input'][1][0] : ", data['input'][1][0])
        
        images = data['input']
        images = torch.cat(images, dim=0)
        #print("images.shape : ", images.shape)    # torch.Size([2000, 3, 32, 32])
        #print(hjk)
        #label = data['meta']['label'][:args.batch_size]
        label = data['meta']['fn'][:args.batch_size]
        for lbl in label:
            count += 1
            #f.write(str(lbl.item()))
            f.write(lbl)
            f.write("\n")
            if count % 500 == 0:
                print(count)
        
    print(ghjk)
    """
    
    
    """  Optimizerの定義  """
    
    # 最適化するパラメータ
    optim_params = model.parameters()
    
    # Optimizerの作成
    optimizer = torch.optim.SGD(optim_params,
                                args.optim_lr,
                                momentum=args.optim_momentum,
                                weight_decay=args.optim_weight_decay)
    optimizer = LARSWrapper(optimizer, eta=0.005, clip=True, exclude_bias_n_norm=True)
    
    
    """  学習率schedulerの定義  """
    if args.data_type == 'cifar100':
        num_converage = (50000 // args.batch_size) * args.epoch
    elif args.data_type == 'cifar10':
        num_converage = (50000 // args.batch_size) * args.epoch
    elif args.data_type == 'imagenet100':
        num_converage = (120000 // args.batch_size) * args.epoch
    elif args.data_type == 'split-cifar10':
        num_converage = (50000 // args.batch_size) * args.epoch
    elif args.data_type == 'core50':
        #print("len(train_dataset) : ", len(train_dataset))
        num_converage = (len(train_dataset) // args.batch_size) * args.epoch
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_converage, eta_min=0, last_epoch=-1)
    
    
    
    """  損失関数の定義  """
    contrastive_loss = InfoNCE_Loss_ver2(args.infonce_temp)
    criterion = TotalCodingRate(eps=args.empssl_eps)
    
    contrastive_loss = contrastive_loss.cuda()
    criterion = criterion.cuda()
    
    
    
    """  学習開始  """
    for epoch in range(args.start_epoch, args.epoch):
        print("Train Epoch {}".format(epoch))
        sys.stdout.flush()
        
        #batch_sampler.set_epoch(epoch=epoch)
        
        train(train_loader,
              model,
              criterion,
              contrastive_loss,
              optimizer,
              scheduler,
              epoch,
              csv_path,
              args)
    
    
    
def train(train_loader,
          model,
          criterion,
          contrastive_loss,
          optimizer,
          scheduler,
          epoch,
          csv_path,
          args):
    
    
    """  modelをtrainモードに変更  """
    model.train()
    
    
    """  train_loaderからデータを取り出して学習  """
    for data in train_loader:
        
        
        # 確認済みバッチ数をカウント
        if args.buffer_type != "none":
            batch_i = train_loader.batch_sampler.advance_batches_seen()
        else:
            batch_i = train_loader.dataset.advance_batches_seen()
        
        
        # エポックの進み具合をカウント
        effective_epoch = epoch + (batch_i / len(train_loader))
        
        # 学習率の調整
        
        
        """  学習用データの用意  """
        images = data['input']
        images = torch.cat(images, dim=0)
        #print("images.shape : ", images.shape)     # torch.Size([2000, 3, 32, 32])
        

        # 学習用データをGPUに配置
        if torch.cuda.is_available():
            images = images.cuda()
        
        
        
        #画像の並びが公式と同じか確認（12/30）
        """
        mean = torch.tensor([0.5, 0.5, 0.5]).view(-1, 1, 1).cuda()
        std = torch.tensor([0.5, 0.5, 0.5]).view(-1, 1, 1).cuda()

        
        
        from torchvision.transforms.functional import to_pil_image
        for i in range(images.shape[0]):
            save_image = images[i] * std + mean
            save_image = save_image.clip(0, 1)
            save_image = to_pil_image(save_image)

            filename = f"./image/save_data_1_{i:05d}.pdf"
            save_image.save(filename)

            if i >301:
              assert False
        """
        
        # modelの出力を獲得
        z_proj, feature1, feature2 = model(images)
        #print("feature1 : ", feature1)
        #print(fghjk)
        
        #print("z_proj.shape : ", z_proj.shape)
        #print("feature1.shape : ", feature1.shape)
        #print("feature2.shape : ", feature2.shape)

        z_list = z_proj.chunk(args.num_patch, dim=0)
        z_avg = chunk_avg(z_proj, args.num_patch)
        
        
        # 損失計算
        loss_contrast = contrastive_loss(z_list)
        loss_TCR = cal_TCR(z_proj, criterion, args.num_patch)
        
        #loss_TCR = 0.0
        loss = args.patch_sim * loss_contrast + args.empssl_tcr * loss_TCR 
        #loss = loss_contrast
        
        lr = optimizer.param_groups[0]['lr']
        
        print("batch_i : {}/{}, lr : {:.5f} loss_contrast : {:.4f}, loss_TCR : {:.4f}, loss : {:.4f}".format(batch_i, len(train_loader), lr, loss_contrast, loss_TCR, loss))
        
        
        
        # バッファ内データの特徴量などを更新
        with torch.no_grad():
            data['feature'] = z_avg.detach()
            if not args.buffer_type.startswith('none'):
                
                stats = train_loader.batch_sampler.update_sample_stats(data)
                
                # バッファ内のラベル数をカウント
                label_dict = train_loader.batch_sampler.return_label_count()
                
            else:
                label_dict = None
        
        
        # 最適化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        

        write_csv(csv_path, batch_i, loss, name="loss")                   # loss~A~[~A~M__~A
        write_csv(csv_path, batch_i, loss_contrast, name="loss_contrast") # loss_contrast~A~[~A~M__~A
        write_csv(csv_path, batch_i, loss_TCR, name="loss_TCR")           # loss_TCR~A~[~A~M__~A
        #write_csv(csv_path, batch_i, loss_dissim, name="loss_dissim")     # loss_dissim~A~[~A~M__~A
        
        # 学習率scheduler
        if args.optim_lr_scheduler:
            scheduler.step()
            
            
        
        
        # modelのパラメータを保存
        step = int(len(train_loader.batch_sampler) / 100)
        step = step * args.save_prog
        if batch_i % step == 0:
            
            path_empssl = os.path.join(args.log_ckpt_dir, args.log_name)
            model_path = os.path.join(path_empssl, f"empssl_model_{epoch}_{batch_i:05d}.pth")
            torch.save(model.state_dict(), model_path)
            #print(xyz)
        
        if batch_i == len(train_loader.batch_sampler):
            path_empssl = os.path.join(args.log_ckpt_dir, args.log_name)
            model_path = os.path.join(path_empssl, f"empssl_model_{epoch}q.pth")
            torch.save(model.state_dict(), model_path)
        
        
        # wandbで記録
        wandb.log({'batch_i': batch_i,
                   'loss': loss,
                   'loss_contrast': loss_contrast,
                   'loss_TCR': loss_TCR,
                   'label_dict': label_dict})
        
        
        
def chunk_avg(x, n_chunks=2, normalize=False):
    x_list = x.chunk(n_chunks, dim=0)
    x = torch.stack(x_list, dim=0)
    if not normalize:
        return x.mean(0)
    else:
        return F.normalize(x.mean(0), dim=1)
    
    
def cal_TCR(z, criterion, num_patches):
    
    z_list = z.chunk(num_patches, dim=0)
    #print("z_list[0].shape : ", z_list[0].shape)   # torch.Size([200, 1024])
    
    
    loss = 0
    for i in range(num_patches):
        loss += criterion(z_list[i])
    loss = loss / num_patches
    return loss


if __name__ == '__main__':
    main()
