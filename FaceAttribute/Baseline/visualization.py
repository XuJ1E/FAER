# !/usr/bin/env python
# -*-coding:utf-8 -*-
"""
# Time       ：2024/1/13 20:56
# Author     ：XuJ1E
# version    ：python 3.8
# File       : visualization.py
"""
import glob
import os
import numpy as np
import torch
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from torchvision import transforms
from models.convnext import convnext_base
from models.cam import GradCAM, show_cam_on_image
matplotlib.use('TkAgg')


if __name__ == '__main__':
    model = convnext_base(pretrained=False, num_classes=40, drop_path_rate=0.25)
    checkpoint = torch.load('./weight/model/BASELINE_model_best.pth', map_location='cpu')['state_dict']
    model.load_state_dict(checkpoint, strict=False)

    target_layer = [model.stages[-1]]
    transform = transforms.Compose([transforms.ToPILImage(),
                                    transforms.Resize(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    cnt = 1
    for p in glob.glob(r'F:\ImageClassification\MM_FOR_ML\data\CelebA\test\*.jpg'):
        cnt += 1
        img = Image.open(p).convert('RGB')
        img = img.resize((224, 224))
        img = np.array(img, dtype=np.uint8)

        img_tensor = transform(img)
        input_tensor = torch.unsqueeze(img_tensor, dim=0)
        cam = GradCAM(model=model, target_layers=target_layer, use_cuda=False)
        target_category = None
        grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)
        grayscale_cam = grayscale_cam[0, :]
        visualization = show_cam_on_image(img.astype(dtype=np.float32) / 255.,
                                          grayscale_cam,
                                          use_rgb=True)
        plt.imshow(visualization)
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.savefig(f'./result/data/test_cam_stage/{os.path.basename(p).split(".")[0]}.png', dpi=400)
        plt.close()
        print('number of processing: ', cnt)
