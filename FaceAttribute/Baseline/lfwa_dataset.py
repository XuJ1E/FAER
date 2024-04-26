# !/usr/bin/env python
# -*-coding:utf-8 -*-
"""
# Time       ：2024/3/18 20:31
# Author     ：XuJ1E
# version    ：python 3.8
# File       : lfwa_dataset.py
"""
import os
import mat73
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset


class LFWADataset(Dataset):
    def __init__(self, root, file_list='lfw_att_40.mat', mode='train', transform=None):
        self.root = root
        self.file_list = file_list
        self.transform = transform
        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.Resize([224, 224]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        if mode == 'train':
            self.idx = mat73.loadmat(os.path.join(self.root, "indices_train_test.mat"))['indices_img_train']

        else:
            self.idx = mat73.loadmat(os.path.join(self.root, "indices_train_test.mat"))['indices_img_test']

        df = mat73.loadmat(os.path.join(root, file_list))
        name_all = df['name']
        label_all = df['label']
        self.names = [name_all[int(i - 1)].replace("\\", "/") for i in self.idx]

        self.targets = [label_all[int(i - 1)] for i in self.idx]

    def __getitem__(self, index):
        sample = os.path.join(self.root, 'image_path/', self.names[index])
        sample = Image.open(sample).convert('RGB')
        sample = self.transform(sample)
        target = self.targets[index]
        target = torch.tensor(target).float()

        return sample, target

    def __len__(self):
        return len(self.names)


if __name__ == '__main__':
    dataset = LFWADataset(root='F:/ImageClassification/MM_FOR_ML/data/LFWA/', file_list='lfw_att_40.mat', mode='train')
    data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=12)
    print(dataset[1][0].shape)
    print(dataset[1][1])