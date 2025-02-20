import torch
from torchvision import datasets
from torchvision import transforms
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS
import os
import numpy as np
import math


def get_loader(args):
    if args.dset == 'dvsg':
         train_set = DVS128Gesture(root=os.path.join(args.data_path, args.dset), train=True, data_type='frame', frames_number=args.T, split_by='number')
         test_set = DVS128Gesture(root=os.path.join(args.data_path, args.dset), train=False, data_type='frame', frames_number=args.T, split_by='number')

         train_data_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
         test_data_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=True
        )
         
         return train_data_loader, test_data_loader
    elif args.dset == 'cifardvs':
        origin_set = CIFAR10DVS(root=os.path.join(args.data_path, args.dset), data_type='frame', frames_number=args.T, split_by='number')

        train_set, test_set = split_to_train_test_set(0.9, origin_set, 10)
        
        train_data_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
        test_data_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=True
        )
         
        return train_data_loader, test_data_loader
    elif args.dset == 'cifar':
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

        train = datasets.CIFAR10(os.path.join(args.data_path, args.dset), train=True, download=True, transform=transform)
        test = datasets.CIFAR10(os.path.join(args.data_path, args.dset), train=False, download=True, transform=transform)
    elif args.dset == 'mnist':
        tr_transform = transforms.Compose([transforms.RandomCrop(args.img_size, padding=2), 
                                            transforms.ToTensor(), 
                                            transforms.Normalize([0.5], [0.5])])
        train = datasets.MNIST(os.path.join(args.data_path, args.dset), train=True, download=True, transform=tr_transform)

        te_transform = transforms.Compose([transforms.Resize([args.img_size, args.img_size]), 
                                           transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
        test = datasets.MNIST(os.path.join(args.data_path, args.dset), train=False, download=True, transform=te_transform)

    elif args.dset == 'fmnist':
        tr_transform = transforms.Compose([transforms.RandomCrop(args.img_size, padding=2),
                                            transforms.ToTensor(), transforms.Normalize([0], [1])])
        train = datasets.FashionMNIST(os.path.join(args.data_path, args.dset), train=True, download=True, transform=tr_transform)

        te_transform = transforms.Compose([transforms.Resize([args.img_size, args.img_size]), 
                                           transforms.ToTensor(),transforms.Normalize([0], [1])])
        test = datasets.FashionMNIST(os.path.join(args.data_path, args.dset), train=False, download=True, transform=te_transform)

    else:
        print("Unknown dataset")
        exit(0)

    train_loader = torch.utils.data.DataLoader(dataset=train,
                                                 batch_size=args.batch_size,
                                                 shuffle=True,
                                                 num_workers=args.num_workers,
                                                 drop_last=True)

    test_loader = torch.utils.data.DataLoader(dataset=test,
                                                batch_size=args.batch_size * 2,
                                                shuffle=False,
                                                num_workers=args.num_workers,
                                                drop_last=True)

    return train_loader, test_loader

def split_to_train_test_set(train_ratio: float, origin_dataset: torch.utils.data.Dataset, num_classes: int, random_split: bool = False):
    '''
    :param train_ratio: split the ratio of the origin dataset as the train set
    :type train_ratio: float
    :param origin_dataset: the origin dataset
    :type origin_dataset: torch.utils.data.Dataset
    :param num_classes: total classes number, e.g., ``10`` for the MNIST dataset
    :type num_classes: int
    :param random_split: If ``False``, the front ratio of samples in each classes will
            be included in train set, while the reset will be included in test set.
            If ``True``, this function will split samples in each classes randomly. The randomness is controlled by
            ``numpy.randon.seed``
    :type random_split: int
    :return: a tuple ``(train_set, test_set)``
    :rtype: tuple
    '''
    label_idx = []
    for i in range(num_classes):
        label_idx.append([])

    for i, item in enumerate(origin_dataset):
        y = item[1]
        if isinstance(y, np.ndarray) or isinstance(y, torch.Tensor):
            y = y.item()
        label_idx[y].append(i)
    train_idx = []
    test_idx = []
    if random_split:
        for i in range(num_classes):
            np.random.shuffle(label_idx[i])

    for i in range(num_classes):
        pos = math.ceil(label_idx[i].__len__() * train_ratio)
        train_idx.extend(label_idx[i][0: pos])
        test_idx.extend(label_idx[i][pos: label_idx[i].__len__()])

    return torch.utils.data.Subset(origin_dataset, train_idx), torch.utils.data.Subset(origin_dataset, test_idx)
