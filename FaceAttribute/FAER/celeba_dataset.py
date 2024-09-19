# !/usr/bin/env python
# -*-coding:utf-8 -*-
"""
# Time       ：2024/1/6 22:35
# Author     ：XuJ1E
# version    ：python 3.8
# File       : dataset.py
"""
import os
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, root, mode, transform=None):
        self.image_name = []
        self.labels = []
        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.Resize([224, 224]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        if mode == 'train':
            self.img_data = os.path.join(root, 'train')
            self.img_label = os.path.join(root, 'train_label.txt')
            with open(self.img_label, 'r') as f:
                data = f.readlines()
                for i in range(len(data)):
                    line = data[i].split()
                    self.image_name.append(line[0])
                    label = [int(x) for x in line[1:]]
                    self.labels.append(label)
        else:
            self.img_data = os.path.join(root, 'test')
            self.img_label = os.path.join(root, 'test_label.txt')
            with open(self.img_label, 'r') as f:
                data = f.readlines()
                for i in range(len(data)):
                    line = data[i].split()
                    self.image_name.append(line[0])
                    label = [int(x) for x in line[1:]]
                    self.labels.append(label)

    def __getitem__(self, item):
        img_path = os.path.join(self.img_data, self.image_name[item])
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        label = torch.tensor(self.labels[item]).float()
        label = torch.where(label > 0, torch.ones_like(label), torch.zeros_like(label))
        return img, label

    def __len__(self):
        return len(self.image_name)


if __name__ == '__main__':
    dataset = MyDataset(root=r'./dataset/CelebA/', mode='test')
    data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=12)
    # for image, target in data_loader:
    #     for i in range(40):
    #         print(target[:, i])
    print(dataset[1][0].shape)
    print(dataset[1][1])
