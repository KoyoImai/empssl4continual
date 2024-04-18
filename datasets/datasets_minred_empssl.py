import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.utils.data as data
import numpy as np
import random
import os


def sparse2coarse(targets):
    """Convert Pytorch CIFAR100 sparse targets to coarse targets.'
    Code copied from https://github.com/ryanchankh/cifar100coarse/blob/master/sparse2coarse.py
    Usage:
        trainset = torchvision.datasets.CIFAR100(path)
        trainset.targets = sparse2coarse(trainset.targets)
    """
    coarse_labels = np.array([ 4,  1, 14,  8,  0,  6,  7,  7, 18,  3,
                               3, 14,  9, 18,  7, 11,  3,  9,  7, 11,
                               6, 11,  5, 10,  7,  6, 13, 15,  3, 15,
                               0, 11,  1, 10, 12, 14, 16,  9, 11,  5,
                               5, 19,  8,  8, 15, 13, 14, 17, 18, 10,
                               16, 4, 17,  4,  2,  0, 17,  4, 18, 17,
                               10, 3,  2, 12, 12, 16, 12,  1,  9, 19,
                               2, 10,  0,  1, 16, 12,  9, 13, 15, 13,
                              16, 19,  2,  4,  6, 19,  5,  5,  8, 19,
                              18,  1,  2, 15,  6,  0, 17,  8, 14, 13])
    return coarse_labels[targets]


def splitcifar10labels(targets):
    
    coarse_labels = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
    
    return coarse_labels[targets]


def buffer_augmentation(data_type):
    
    if data_type == 'cifar100':
        data_insize = 32
    elif data_type == 'cifar10':
        data_insize = 32
    elif data_type == 'imagenet100':
        data_insize = 224
    else:
        assert False
    
    # 可視化のみに使うデータ拡張
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])

    augmentation = transforms.Compose([
        transforms.Resize(size=(data_insize, data_insize)),
        transforms.ToTensor(),
        normalize,
    ])
    
    return augmentation


class Cifar100Dataset(data.Dataset):
    
    def __init__(self, root, transforms, train, download, trial):
        
        data.Dataset.__init__(self)
        
        # 初期化
        self.root = root
        self.train = train
        self.download = download
        
        # データセットのダウンロード
        self.dataset = torchvision.datasets.CIFAR100(root=self.root, download=self.download, train=self.train)
        
        # 変更前のラベルを獲得
        self.original_labels = self.dataset.targets        
        
        # ラベルを変更
        self.dataset.targets = sparse2coarse(self.dataset.targets)
        
        
        # 確認したバッチ数
        self.num_batches_seen = 0
        
        # データ拡張
        if not isinstance(transforms, list):
            transforms = [transforms]
        self.transforms = transforms
        
        
        augmentation = buffer_augmentation(data_type="cifar100")
        
        
        if not isinstance(augmentation, list):
            augmentation = [augmentation]
        self.augmentation = augmentation
        
        
        
    def __getitem__(self, index):
        
        image, label = self.dataset[index]
        
        meta = {}
        i = 0
        transform = None
        if self.transforms is not None:
            i = np.random.randint(len(self.transforms))
            transform = self.transforms[i]
            
        if transform is not None:
            im1 = transform(image)
            
        """  データ拡張前のオリジナルデータ  """
        i = 0
        augmentation = None
        if self.augmentation is not None:
            i = np.random.randint(len(self.augmentation))
            augmentation = self.augmentation[i]
        
        if augmentation is not None:
            orig_im = augmentation(image)
        
        meta['transind'] = i
        meta['index'] = index
        meta['label'] = label
        
        out = {
            'input': im1,
            'orig_input': orig_im,
            'meta': meta,
        }
        
        return out
        
    def __len__(self):
        return len(self.dataset)
        
    
    def get_buffer_data(self, index):
        
        image, _ = self.dataset[index]
        label = self.original_labels[index]
        
        i = 0
        transform = None
        if self.transforms is not None:
            i = np.random.randint(len(self.transforms))
            transform = self.transforms[i]
        
        if transform is not None:
            im = transform(image)
        
        
        
        ##  __getitem__()からの追加部分
        # ここでデータとラベルを用意する
        #print("len(im) : ", len(im))            # len(im) :  20
        #print("im[0].shape : ", im[0].shape)    # im[0].shape :  torch.Size([3, 32, 32])
        im = torch.stack(im, dim=0)
        
        #label = torch.from_numpy(label).clone()
        #print("label : ", label)
        #print("label.shape : ", label.shape)
        #print(ghjkl)
        
        label = torch.tensor(label)
        
        return im, label
    
    
    def get_original_data(self, index):
        
        image, _ = self.dataset[index]
        label = self.original_labels[index]
        
        i = 0
        augmentation = None
        if self.augmentation is not None:
            i = np.random.randint(len(self.augmentation))
            augmentation = self.augmentation[i]
        
        if augmentation is not None:
            im = augmentation(image)
        
        
        #im = torch.stack(im, dim=0)
        label = torch.tensor(label)
        
        return im, label
    

    
    def advance_batches_seen(self):
        
        self.num_batches_seen += 1
        return self.num_batches_seen
    
    
class Cifar100LinearDataset(data.Dataset):
    
    def __init__(self, root, transforms, train, download, trial):
        
        data.Dataset.__init__(self)
        
        # 初期化
        self.root = root
        self.train = train
        self.download = download
        
        # データセットのダウンロード
        self.dataset = torchvision.datasets.CIFAR100(root=self.root, download=self.download, train=self.train)
        
        # ラベルを変更
        self.dataset.targets = sparse2coarse(self.dataset.targets)
        
        
        # 確認したバッチ数
        self.num_batches_seen = 0
        
        if not isinstance(transforms, list):
            transforms = [transforms]
        self.transforms = transforms
        
    def __getitem__(self, index):
        
        image, label = self.dataset[index]
        
        meta = {}
        i = 0
        transform = None
        if self.transforms is not None:
            i = np.random.randint(len(self.transforms))
            transform = self.transforms[i]
            
        if transform is not None:
            im1 = transform(image)
        
        meta['transind'] = i
        meta['index'] = index
        meta['label'] = label
        
        out = {
            'input': im1,
            'meta': meta,
        }
        
        return out
        
    def __len__(self):
        return len(self.dataset)
    
    
    
    def advance_batches_seen(self):
        
        self.num_batches_seen += 1
        return self.num_batches_seen
        

class Cifar100EvalDataset(data.Dataset):
    
    def __init__(self, list_fname, transform=None, train=False, download=True):
        
        data.Dataset.__init__(self)
        
        self.root = list_fname
        self.train = train
        self.download = download
        
        self.dataset = torchvision.datasets.CIFAR100(root=self.root, download=self.download, train=self.train)
        
        self.transform = transform
        
        
    def __getitem__(self, index):
        
        image, target = self.dataset[index]
        
        if self.transform is not None:
            image = self.transform(image)
            
            
        #print("type(image) : ", type(image))
        #print("type(target) : ", type(target))
        #print("type(index) : ", type(index))
        
        
        out = {
            'input': image,
            'target': torch.tensor(target),
            'index': index,
        }
        
        return out
        
    def __len__(self):
        return len(self.dataset)
    
        
        

class Cifar100EvalEachTaskDataset(data.Dataset):
    
    def __init__(self, list_fname, included_classes, transform=None, train=False, download=True):
        
        data.Dataset.__init__(self)
        
        self.root = list_fname
        self.train = train
        self.download = download
        self.included_classes = included_classes
        
        
        # CIFAR100データセットのダウンロード
        self.dataset = torchvision.datasets.CIFAR100(root=self.root, download=self.download, train=self.train)
        
        
        # 指定されたクラスのみを含むようにフィルタリング
        self.filter_dataset()
        
        # フィルタリングされたデータセットの確認
        print("len(self.dataset.data) : ", len(self.dataset.data))
        
        
        self.transform = transform
    
    
    
    def filter_dataset(self):
        self.dataset.data = [x for x, y in zip(self.dataset.data, self.dataset.targets) if y in self.included_classes]
        self.dataset.targets = [y for y in self.dataset.targets if y in self.included_classes]

        
    def __getitem__(self, index):
        
        image, target = self.dataset[index]
        
        
        if self.transform is not None:
            image = self.transform(image)
            
        
        out = {
            'input': image,
            'target': torch.tensor(target),
            'index': index,
        }
        
        return out
        
    def __len__(self):
        return len(self.dataset)

    
class Cifar100tSNEDataset(data.Dataset):
    
    def __init__(self, list_fname, included_classes, class_size=100, transform=None, train=False, download=True):
        
        data.Dataset.__init__(self)
        
        self.root = list_fname
        self.train = train
        self.download = download
        self.included_classes = included_classes
        self.class_size = class_size
        
        
        # CIFAR100データセットのダウンロード
        self.dataset = torchvision.datasets.CIFAR100(root=self.root, download=self.download, train=self.train)
        
        
        # 指定されたクラスのみを含むようにフィルタリング
        self.filter_dataset()
        
        # self.datasetに含まれるデータをクラスごとにclass_sizeに統一
        self.filter_dataset_v2()
        
        # フィルタリングされたデータセットの確認
        print("len(self.dataset.data) : ", len(self.dataset.data))
        
        
        self.transform = transform
    
    
    
    def filter_dataset(self):
        self.dataset.data = [x for x, y in zip(self.dataset.data, self.dataset.targets) if y in self.included_classes]
        self.dataset.targets = [y for y in self.dataset.targets if y in self.included_classes]

    def filter_dataset_v2(self):
        
        class_data = {cls: [] for cls in self.included_classes}
        for x, y in zip(self.dataset.data, self.dataset.targets):
            if y in self.included_classes:
                class_data[y].append(x)
        
        
        # 各クラスからランダムでclass_size分だけ選択して取り出す
        print("len(class_data) : ", len(class_data))
        for cls in class_data:
            if len(class_data[cls]) > self.class_size:
                #print("len(class_data[cls]) : ", len(class_data[cls]))
                #print("type(class_data) : ", type(class_data))
                #print("type(class_data[cls]) : ", type(class_data[cls]))
                #print("class_data[cls][0].shape : ", class_data[cls][0].shape)
                #class_data[cls] = np.random.choice(class_data[cls], self.class_size, replace=False).tolist()
                indices = np.random.choice(len(class_data[cls]), self.class_size, replace=False)
                class_data[cls] = [class_data[cls][i] for i in indices]
                #print("len(class_data[cls]) : ", len(class_data[cls]))
        
        
        # 新しいデータセットを作成
        new_data = []
        new_targets = []
        for cls, data in class_data.items():
            new_data.extend(data)
            new_targets.extend([cls] * len(data))
        
        self.dataset.data = new_data
        self.dataset.targets = new_targets
        
        #print("len(self.dataset.data) : ", len(self.dataset.data))
        #print("len(self.dataset.targets) : ", len(self.dataset.data))
        

        
    def __getitem__(self, index):
        
        image, target = self.dataset[index]
        
        
        if self.transform is not None:
            image = self.transform(image)
            
        
        out = {
            'input': image,
            'target': torch.tensor(target),
            'index': index,
        }
        
        return out
        
    def __len__(self):
        return len(self.dataset)
    
    
class Cifar100UmapDataset(data.Dataset):
    
    def __init__(self, list_fname, included_classes, transform=None, train=False, download=True):
        
        data.Dataset.__init__(self)
        
        self.root = list_fname
        self.train = train
        self.download = download
        self.included_classes = included_classes
        
        
        # CIFAR100データセットのダウンロード
        self.dataset = torchvision.datasets.CIFAR100(root=self.root, download=self.download, train=self.train)
        
        
        # 指定されたクラスのみを含むようにフィルタリング
        self.filter_dataset()
        
        # フィルタリングされたデータセットの確認
        print("len(self.dataset.data) : ", len(self.dataset.data))
        
        
        self.transform = transform
    
    
    
    def filter_dataset(self):
        self.dataset.data = [x for x, y in zip(self.dataset.data, self.dataset.targets) if y in self.included_classes]
        self.dataset.targets = [y for y in self.dataset.targets if y in self.included_classes]

        
    def __getitem__(self, index):
        
        image, target = self.dataset[index]
        
        
        if self.transform is not None:
            image = self.transform(image)
            
        
        out = {
            'input': image,
            'target': torch.tensor(target),
            'index': index,
        }
        
        return out
        
    def __len__(self):
        return len(self.dataset)


class SplitCifar10Dataset(data.Dataset):
    
    def __init__(self, root, transforms, train, download, trial):
        
        data.Dataset.__init__(self)
        
        # 初期化
        self.root = root
        self.train = train
        self.download = download
        
        # データセットのダウンロード
        self.dataset = torchvision.datasets.CIFAR10(root=self.root, download=self.download, train=self.train)
        
        # 確認したバッチ数
        self.num_batches_seen = 0
        
        
        # Split-CIFAR10に合わせてラベルを変更
        self.dataset.targets = splitcifar10labels(self.dataset.targets)
        
        
        
        # データ拡張の設定
        if not isinstance(transforms, list):
            transforms = [transforms]
        self.transforms = transforms
        
        
    def __getitem__(self, index):
        
        image, label = self.dataset[index]
        
        meta = {}
        i = 0
        transform = None
        if self.transforms is not None:
            i = np.random.randint(len(self.transforms))
            transform = self.transforms[i]
        
        if transform is not None:
            im1 = transform(image)
            
        meta['transind'] = i
        meta['index'] = index
        meta['label'] = label
        
        out = {
            'input': im1,
            'meta': meta,
        }
        
        
        return out
        
        
    def __len__(self):
        return len(self.dataset)

    
class Cifar10Dataset(data.Dataset):
    
    def __init__(self, root, transforms, train, download, trial):
        
        data.Dataset.__init__(self)
        
        # 初期化
        self.root = root
        self.train = train
        self.download = download
        
        
        # データセットのダウンロード
        self.dataset = torchvision.datasets.CIFAR10(root=self.root, download=self.download, train=self.train)
        
        
        # 確認したバッチ数
        self.num_batches_seen = 0
        
        
        # データ拡張の設定
        if not isinstance(transforms, list):
            transforms = [transforms]
        self.transforms = transforms
        
        
        augmentation = buffer_augmentation(data_type="cifar10")
        
        
        if not isinstance(augmentation, list):
            augmentation = [augmentation]
        self.augmentation = augmentation
        
        
    def __getitem__(self, index):
        
        image, label = self.dataset[index]
        
        
        meta = {}
        i = 0
        transform = None
        if self.transforms is not None:
            i = np.random.randint(len(self.transforms))
            transform = self.transforms[i]
        
        if transform is not None:
            im1 = transform(image)
        
            
        meta['transind'] = i
        meta['index'] = index
        meta['label'] = label
        
        out = {
            'input': im1,
            'meta': meta,
        }
        
        
        return out
     
        
        
    def get_buffer_data(self, index):
        
        image, label = self.dataset[index]
        #label = self.original_labels[index]
        
        i = 0
        transform = None
        if self.transforms is not None:
            i = np.random.randint(len(self.transforms))
            transform = self.transforms[i]
        
        if transform is not None:
            im = transform(image)
        
        
        
        ##  __getitem__()からの追加部分
        # ここでデータとラベルを用意する
        #print("len(im) : ", len(im))            # len(im) :  20
        #print("im[0].shape : ", im[0].shape)    # im[0].shape :  torch.Size([3, 32, 32])
        im = torch.stack(im, dim=0)
        
        #label = torch.from_numpy(label).clone()
        #print("label : ", label)
        #print("label.shape : ", label.shape)
        #print(ghjkl)
        
        label = torch.tensor(label)
        
        return im, label
        
    
    
    def get_original_data(self, index):
        
        image, label = self.dataset[index]
        
        
        i = 0
        augmentation = None
        if self.augmentation is not None:
            i = np.random.randint(len(self.augmentation))
            augmentation = self.augmentation[i]
        
        if augmentation is not None:
            im = augmentation(image)
        
        
        #im = torch.stack(im, dim=0)
        label = torch.tensor(label)
        
        return im, label
    
    
    def __len__(self):
        return len(self.dataset)
    
    
    def advance_batches_seen(self):
        
        self.num_batches_seen += 1
        return self.num_batches_seen
    
    

class Cifar10EvalDataset(data.Dataset):
    
    def __init__(self, list_fname, transform=None, train=False, download=True):
        
        self.root = list_fname
        self.train = train
        self.download = download
        
        
        self.dataset = torchvision.datasets.CIFAR10(root=self.root, download=self.download, train=self.train)
        
        self.transform = transform
        
        
    def __getitem__(self, index):
        
        image, target = self.dataset[index]
        
        if self.transform is not None:
            image = self.transform(image)
            
        
        
        out = {
            'input': image,
            'target': torch.tensor(target),
            'index': index,
        }
        
        return out
    
    
    def __len__(self):
        return len(self.dataset)
     

class Cifar10tSNEDataset(data.Dataset):
    
    def __init__(self, list_fname, included_classes, class_size=100, transform=None, train=False, download=True):
        
        data.Dataset.__init__(self)
        
        self.root = list_fname
        self.train = train
        self.download = download
        self.included_classes = included_classes
        self.class_size = class_size
        
        
        # CIFAR10データセットのダウンロード
        self.dataset = torchvision.datasets.CIFAR10(root=self.root, download=self.download, train=self.train)
        
        
        # 指定されたクラスのみを含むようにフィルタリング
        #self.filter_dataset()
        
        # self.datasetに含まれるデータをクラスごとにclass_sizeに統一
        self.filter_dataset_v2()
        
        # フィルタリングされたデータセットの確認
        print("len(self.dataset.data) : ", len(self.dataset.data))
        
        
        self.transform = transform
    
    
    
    def filter_dataset(self):
        self.dataset.data = [x for x, y in zip(self.dataset.data, self.dataset.targets) if y in self.included_classes]
        self.dataset.targets = [y for y in self.dataset.targets if y in self.included_classes]

    def filter_dataset_v2(self):
        
        class_data = {cls: [] for cls in self.included_classes}
        for x, y in zip(self.dataset.data, self.dataset.targets):
            if y in self.included_classes:
                class_data[y].append(x)
        
        
        # 各クラスからランダムでclass_size分だけ選択して取り出す
        print("len(class_data) : ", len(class_data))
        for cls in class_data:
            if len(class_data[cls]) > self.class_size:
                #print("len(class_data[cls]) : ", len(class_data[cls]))
                #print("type(class_data) : ", type(class_data))
                #print("type(class_data[cls]) : ", type(class_data[cls]))
                #print("class_data[cls][0].shape : ", class_data[cls][0].shape)
                #class_data[cls] = np.random.choice(class_data[cls], self.class_size, replace=False).tolist()
                indices = np.random.choice(len(class_data[cls]), self.class_size, replace=False)
                class_data[cls] = [class_data[cls][i] for i in indices]
                #print("len(class_data[cls]) : ", len(class_data[cls]))
        
        
        # 新しいデータセットを作成
        new_data = []
        new_targets = []
        for cls, data in class_data.items():
            new_data.extend(data)
            new_targets.extend([cls] * len(data))
        
        self.dataset.data = new_data
        self.dataset.targets = new_targets
        
        #print("len(self.dataset.data) : ", len(self.dataset.data))
        #print("len(self.dataset.targets) : ", len(self.dataset.data))
        

        
    def __getitem__(self, index):
        
        image, target = self.dataset[index]
        
        
        if self.transform is not None:
            image = self.transform(image)
            
        
        out = {
            'input': image,
            'target': torch.tensor(target),
            'index': index,
        }
        
        return out
        
    def __len__(self):
        return len(self.dataset)
    
        
        
class SplitCifar10EvalEachTaskDataset(data.Dataset):
    
    def __init__(self, list_fname, included_classes, transform=None, train=False, download=True):
        
        data.Dataset.__init__(self)
        
        self.root = list_fname
        self.train = train
        self.download = download
        self.included_classes = included_classes
        
        
        # CIFAR100データセットのダウンロード
        self.dataset = torchvision.datasets.CIFAR10(root=self.root, download=self.download, train=self.train)
        
        
        # 指定されたクラスのみを含むようにフィルタリング
        self.filter_dataset()
        
        # フィルタリングされたデータセットの確認
        print("len(self.dataset.data) : ", len(self.dataset.data))
        
        
        self.transform = transform
    
    def filter_dataset(self):
        self.dataset.data = [x for x, y in zip(self.dataset.data, self.dataset.targets) if y in self.included_classes]
        self.dataset.targets = [y for y in self.dataset.targets if y in self.included_classes]
        
    
    def __getitem__(self, index):
        
        image, target = self.dataset[index]
        
        if self.transform is not None:
            image = self.transform(image)
            
        out = {
            'input': image,
            'target': torch.tensor(target),
            'index': index,
        }
        
        return out
    
    def __len__(self):
        return len(self.dataset)
    
    
        
"""  ここから先は，filelist.txtを読み込んでデータセットを作成  """



def set_task_label(task_id):
    
    if task_id == 0:
        task_label = ['n01978455', 'n02099849', 'n02090622', 'n03492542', 'n02138441', 'n03494278', 'n02859443', 'n04336792', 'n04136333', 'n01749939']
    elif task_id == 1:
        task_label = ['n13040303', 'n02869837', 'n04127249', 'n02119022', 'n02085620', 'n03379051', 'n01558993', 'n07836838', 'n02231487', 'n02104029']
    elif task_id == 2:
        task_label = ['n03891251', 'n02488291', 'n07715103', 'n02974003', 'n04485082', 'n02701002', 'n01735189', 'n02009229', 'n01983481', 'n03259280']
    elif task_id == 3:
        task_label = ['n02804414', 'n01729322', 'n03775546', 'n03903868', 'n03584829', 'n01855672', 'n02396427', 'n02100583', 'n02113799', 'n04111531']
    elif task_id == 4:
        task_label = ['n02086910', 'n03930630', 'n02106550', 'n02483362', 'n04429376', 'n03085013', 'n03017168', 'n02107142', 'n01980166', 'n04067472']
    elif task_id == 5:
        task_label = ['n04099969', 'n02089973', 'n02086240', 'n03594734', 'n03642806', 'n03785016', 'n01692333', 'n04418357', 'n03787032', 'n03837869']
    elif task_id == 6:
        task_label = ['n02087046', 'n03947888', 'n01820546', 'n02114855', 'n04238763', 'n03530642', 'n02108089', 'n07714571', 'n02788148', 'n02091831']
    elif task_id == 7:
        task_label = ['n04229816', 'n02116738', 'n03424325', 'n03032252', 'n02109047', 'n02105505', 'n02089867', 'n02123045', 'n02326432', 'n02113978']
    elif task_id == 8:
        task_label = ['n02018207', 'n02877765', 'n03062245', 'n04026417', 'n03637318', 'n07831146', 'n13037406', 'n04493381', 'n02093428', 'n04589890']
    elif task_id == 9:
        task_label = ['n03777754', 'n04592741', 'n07753275', 'n02172182', 'n01773797', 'n04517823', 'n03794056', 'n02259212', 'n04435653', 'n03764736']
    
    return task_label


def encode_filename(fn, max_len=200):
    assert len(
        fn
    ) < max_len, f"Filename is too long. Specified max length is {max_len}"
    fn = fn + '\n' + ' ' * (max_len - len(fn))
    fn = np.fromstring(fn, dtype=np.uint8)
    fn = torch.ByteTensor(fn)
    return fn


def decode_filename(fn):
    fn = fn.cpu().numpy().astype(np.uint8)
    fn = fn.tostring().decode('utf-8')
    fn = fn.split('\n')[0]
    return fn



# filelist.txtを読み込んでデータセットを作成
class ImageNet100Dataset(data.Dataset):
    
    def __init__(self,
                 train_filelist,
                 transforms=None,
                 fname_fmt='{:03d}.jpeg',
                 trial=7,
                 num_task=10,
                 train_samples_per_cls=12000):
        
        data.Dataset.__init__(self)
        
        self.num_batches_seen = 0

        # シード値の固定
        random.seed(trial)
        torch.manual_seed(trial)
        
        # filelist.txtが存在するかの確認
        assert (os.path.exists(train_filelist)
               ), '{} does not exist'.format(train_filelist)
        
        
        all_files = []
        task_samples = train_samples_per_cls
        for i in range(num_task):
            
            # 特定タスク用のfilelistからパスを読み込む
            with open(f"{train_filelist}/train_filelist{i}.txt", 'r') as f:
                task_files = f.read().splitlines()                     # 学習用データまでの全てのパス
                #print("type(task_files) : ", type(task_files))        # <class 'list'>
                #print("len(task_files) : ", len(task_files))          # 13000~12180
            
            # タスク毎のデータ数を揃える
            task_files = random.sample(task_files, task_samples)
            all_files += task_files
            print("len(all_files) : ", len(all_files))   
        
        
        
        # 各データに対応したラベルを獲得
        all_labels = [fn.split('/')[-2] for fn in all_files]
        
        # 重複なしでラベルを獲得（100種類）
        label_set = sorted(list(set(all_labels)))
        
        # ラベル毎に対応した数値を割り当て（ラベル自体は「n~」の形）
        label_set = {y: j for j, y in enumerate(label_set)}
        
        # ラベルを割り当てられた数値に変換
        all_labels = [label_set[lbl] for lbl in all_labels]
        #print("label_set : ", label_set)                    # 実際のラベル(0~100)とn~の組み合わせ
        #print("all_labels[0:50] : ", all_labels[0:50])
        #print("all_labels[12000:12050] : ", all_labels[12000:12050])
        
        
        # タスクラベルを割り当てる
        task_labels = []
        for i in range(num_task):
            for j in range(task_samples):
                task_labels.append(i)
                
        
        #print("task_labels : ", task_labels)
        #print("len(task_labels) : ", len(task_labels))   # len(task_labels) :  120000

        
        filenames = all_files
        labels = all_labels
        
        
        self.filenames = torch.stack([encode_filename(fn) for fn in filenames])
        self.labels = torch.tensor(labels)
        self.task_labels = torch.tensor(task_labels)
        
        if not isinstance(transforms, list):
            transforms = [transforms]
        self.transforms = transforms
    
    
    def __getitem__(self, index):
        
        
        MAX_TRIES = 50
        for i in range(MAX_TRIES):
            try:
                fname = decode_filename(self.filenames[index])
                im = datasets.folder.pil_loader(fname)
                break
            except Exception:
                if i == MAX_TRIES - 1:
                    raise ValueRrror(
                        f'Aborting. Failed to load {MAX_TRIES} times in a row. Check {fname}'
                    )
                print(f'Failed to load. {fname}')
                index = np.random.randint(len(self))
        
        meta = {}
        i = 0
        transform = None
        if self.transforms is not None:
            i = np.random.randint(len(self.transforms))
            transform = self.transforms[i]
        if transform is not None:
            im = transform(im)
            
        
        meta['transid'] = i
        meta['fn'] = fname
        meta['index'] = index
        #meta['label'] = self.labels[index].item()
        meta['label'] = self.task_labels[index]
        meta['data_label'] = self.labels[index]
        
        out = {
            'input': im,
            'meta': meta,
        }
        
        
        return out
        
    
    def __len__(self):
        
        return self.filenames.shape[0]


    def advance_batches_seen(self):
        self.num_batches_seen += 1
        return self.num_batches_seen


class StandardEvalDataset(data.Dataset):
    
    def __init__(self,
                 list_fname,
                 transform=None,
                 train=True,
                 trial=7,
                 num_task=10):
        
        data.Dataset.__init__(self)
        assert (
            os.path.exists(list_fname)), '{} does not exist'.format(list_fname)
        
        # シード値の固定
        random.seed(trial)
        torch.manual_seed(trial)
        
        
        #"""
        all_files = []
        if train:
            task_samples = 12000
        elif not train:
            task_samples = 500
        for i in range(num_task):
            
            if train:
                fname = f"{list_fname}/train_filelist{i}.txt"
            elif not train:
                fname = f'{list_fname}/val_filelist{i}.txt'
            
            # 特定タスク用のfilelistからパスを読み込む
            with open(fname, 'r') as f:
                task_files = f.read().splitlines()                     # 学習用データまでの全てのパス
                #print("type(task_files) : ", type(task_files))        # <class 'list'>
                print("len(task_files) : ", len(task_files))          # 13000~12180
            
            # タスク毎のデータ数を揃える
            #task_files = random.sample(task_files, task_samples)
            all_files += task_files
            #print("len(all_files) : ", len(all_files))   
        #"""
        """
        with open(list_fname, 'r') as f:
            all_files = f.read().splitlines()
        """
           
        all_labels = [fn.split('/')[-2] for fn in all_files]
        label_set = sorted(list(set(all_labels)))
        label_set = {y: i for i, y in enumerate(label_set)}
        all_labels = [label_set[lbl] for lbl in all_labels]
        
        filenames = all_files
        labels = all_labels
        
        self.filenames = torch.stack([encode_filename(fn) for fn in filenames])
        self.labels = torch.tensor(labels)
        
        #if not isinstance(transform, list):
        #    transform = [transform]
        self.transforms = transform
        
    
    def __getitem__(self, index):
        
        MAX_TRIES = 50
        for i in range(MAX_TRIES):
            try:
                fname = decode_filename(self.filenames[index])
                im = datasets.folder.pil_loader(fname)
                break
            
            except Exception:
                if i == MAX_TRIES - 1:
                    raise ValueError(
                        f'Aborting. Failed to load {MAX_TRIES} times in a row. Check {fname}'
                    )
                print(f'Failed to load. {fname}')
                index = np.random.randint(len(self))
        
        
        if self.transforms is not None:
            im = self.transforms(im)
        
        out = {
            'input': im,
            'target': self.labels[index],
            'index': index,
        }
        
        return out
        
    def __len__(self):
        return self.filenames.shape[0]
        
        
class ImageNet100EvalEachTaskDataset(data.Dataset):
    
    def __init__(self,
                 list_fname,
                 task_n=None,
                 transform=None,
                 train=True,
                 trial=7):
        
        data.Dataset.__init__(self)
        assert (
            os.path.exists(list_fname)), '{} does not exist'.format(list_fname)
        
        # シード値の固定
        random.seed(trial)
        torch.manual_seed(trial)
        
        all_files = []
        if train:
            fname = f"{list_fname}/train_filelist{task_n}.txt"
        elif not train:
            fname = f"{list_fname}/val_filelist{task_n}.txt"
        
        with open(fname, 'r') as f:
            task_files = f.read().splitlines()
            print("len(task_files) : ", len(task_files))
        
        all_files += task_files
        
        # ラベルの処理
        all_labels = [fn.split('/')[-2] for fn in all_files]
        label_seq = sorted(list(set(all_labels)))
        label_seq = {y: i for i, y in enumerate(label_seq)}
        all_labels = [label_seq[lbl] for lbl in all_labels]
        
        filenames = all_files
        labels = all_labels
        
        self.filenames = torch.stack([encode_filename(fn) for fn in filenames])
        self.labels = torch.tensor(labels)
        
        #if not isinstance(transform, list):
        #    transform = [transform]
        self.transforms = transform
        
        
    def __getitem__(self, index):
        
        
        MAX_TRIES = 50
        for i in range(MAX_TRIES):
            try:
                fname = decode_filename(self.filenames[index])
                im = datasets.folder.pil_loader(fname)
                break
                
            except Exception:
                if i == MAX_TRIES - 1:
                    raise ValueError(
                        f'Aborting. Failed to load {MAX_TRIES} times in a row. Check {fname}'
                    )
                print(f'Failed to load. {fname}')
                index = np.random.randint(len(self))
            
        if self.transforms is not None:
            im = self.transforms(im)
            
        out = {
            'input': im,
            'target': self.labels[index],
            'index': index,
        }
        
        return out
    

    def __len__(self):
        return self.filenames.shape[0]
                
                
        
        
        
        

# filelist.txtを読み込んでデータセットを作成        
class CORe50Dataset(data.Dataset):
    
    def __init__(self,
                 train_filelist,
                 transforms=None,
                 fname_fmt='{:03d}.jpeg',
                ):
        
        data.Dataset.__init__(self)
        
        # filelist.txtが存在するか確認
        assert (os.path.exists(train_filelist)
               ), '{} does not exist'.format(train_filelist)
        
        with open(train_filelist, 'r') as f:
            all_files = f.read().splitlines()
        
        # 各データに対応したラベルを獲得
        all_labels = [int(fn.split(" ")[-1]) for fn in all_files] # txtファイルに記載したラベル
        print(all_labels[100000:100005])
        
        all_files = [fn.split(" ")[0] for fn in all_files]
        
        filenames = all_files
        labels = all_labels
        
        self.filenames = torch.stack([encode_filename(fn) for fn in filenames])
        self.labels = torch.tensor(labels)
        #print("self.filenames.shape : ", self.filenames.shape)    # torch.Size([115000, 201])
        #print("self.labels.shape : ", self.labels.shape)          # torch.Size([115000])
        
        if not isinstance(transforms, list):
            transforms = [transforms]
        self.transforms = transforms
        
        # 確認したバッチ数
        self.num_batches_seen = 0
        
    def __getitem__(self, index):
        
        MAX_TRIES = 50
        #print("index : ", index)
        for i in range(MAX_TRIES):
            try:
                fname = decode_filename(self.filenames[index])
                im = datasets.folder.pil_loader(fname)
                break
            except Exception:
                if i == MAX_TRIES - 1:
                    raise ValueRrror(
                        f'Aborting. Failed to load {MAX_TRIES} times in a row. Check {fname}'
                    )
                print(f'Failed to load. {fname}')
                index = np.random.randint(len(self))
        
        meta = {}
        i = 0
        transform = None
        if self.transforms is not None:
            i = np.random.randint(len(self.transforms))
            transform = self.transforms[i]
        if transform is not None:
            im = transform(im)
        
        
        meta['transid'] = i
        meta['fn'] = fname
        meta['index'] = index
        meta['label'] = self.labels[index]
        
        out = {
            'input': im,
            'meta': meta,
        }
        
        return out
        
        
        
    def __len__(self):
        
        return self.filenames.shape[0]
    
    
    def advance_batches_seen(self):
        
        self.num_batches_seen += 1
        return self.num_batches_seen
    
    
# filelist.txtを読み込んでデータセットを作成        
class CORe50EvalDataset(data.Dataset):
    
    def __init__(self,
                 train_filelist,
                 transforms=None,
                 trial=7,
                ):
        
        data.Dataset.__init__(self)
        
        # filelist.txtが存在するか確認
        assert (os.path.exists(train_filelist)
               ), '{} does not exist'.format(train_filelist)
        
        with open(train_filelist, 'r') as f:
            all_files = f.read().splitlines()
        
        # 各データに対応したラベルを獲得
        all_labels = [int(fn.split(" ")[-1]) for fn in all_files] # txtファイルに記載したラベル
        #print(all_labels[100000:100005])
        
        all_files = [fn.split(" ")[0] for fn in all_files]
        
        filenames = all_files
        labels = all_labels
        
        self.filenames = torch.stack([encode_filename(fn) for fn in filenames])
        self.labels = torch.tensor(labels)
        #print("self.filenames.shape : ", self.filenames.shape)    # torch.Size([115000, 201])
        #print("self.labels.shape : ", self.labels.shape)          # torch.Size([115000])
        
        if not isinstance(transforms, list):
            transforms = [transforms]
        self.transforms = transforms
        
        # 確認したバッチ数
        self.num_batches_seen = 0
        
    def __getitem__(self, index):
        
        MAX_TRIES = 50
        #print("index : ", index)
        for i in range(MAX_TRIES):
            try:
                fname = decode_filename(self.filenames[index])
                im = datasets.folder.pil_loader(fname)
                break
            except Exception:
                if i == MAX_TRIES - 1:
                    raise ValueRrror(
                        f'Aborting. Failed to load {MAX_TRIES} times in a row. Check {fname}'
                    )
                print(f'Failed to load. {fname}')
                index = np.random.randint(len(self))
        
        i = 0
        transform = None
        if self.transforms is not None:
            i = np.random.randint(len(self.transforms))
            transform = self.transforms[i]
        if transform is not None:
            im = transform(im)
        
        out = {
            'input': im,
            'target': self.labels[index],
            'index': index,
        }
        
        return out
        
        
        
    def __len__(self):
        
        return self.filenames.shape[0]
    
    
    def advance_batches_seen(self):
        
        self.num_batches_seen += 1
        return self.num_batches_seen
