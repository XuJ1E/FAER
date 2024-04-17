# 基于上下文通道注意力机制的人脸属性估计与表情识别
#（Facial attribute estimation and expression recognition based on contextual channel attention mechanism）

![overview](./set/img/overall_architecture.svg)

A PyTorch implementation of the [FAER](https://kns.cnki.net/kcms2/article/abstract?v=v5HVlYuqh9qy9Jy50ovh3R_ohTNoNi1Tw2-GgzoZ7z8DdzkZ__gFP4MYpB-sBX-4B9uMnQMOMuFXbwyWaNdzjHNFlDiD6hReqGS5Upt4YNMx6bycOGrmzffSsQ4lXtT_3Nr8wZ-iNNs=&uniplatform=NZKPT&flag=copy).

  

## Training
We provide the training code for AffectNet and RAF-DB.  

For AffectNet、RAF-DB dataset, run:
```
Comming soon
```

For CelebA dataset, run:
```
CUDA_VISIBLE_DEVICES=0,1 python main_celeba.py 
```

## Models
Pre-trained models can be downloaded for evaluation as following:

|     task    	| accuracy 	| link 	|
|:-----------:	|:--------:	|:----:	|
| CelebA       	| 91.87%   	|Comming soon|
| AffectNet 	  | 66.66%    |Comming soon|  
| RAF-DB       	| 91.75%   	|Comming soon|


## Grad CAM++ Reproduction
![overview](./set/img/affectnet.jpg)

![overview](./set/img/celeba.svg)

## For more detail imformation, see:
```
![paper](https://github.com/XUJ1Er/FAER/blob/main/set/%E5%9F%BA%E4%BA%8E%E4%B8%8A%E4%B8%8B%E6%96%87%E9%80%9A%E9%81%93%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6%E7%9A%84%E4%BA%BA%E8%84%B8%E5%B1%9E%E6%80%A7%E4%BC%B0%E8%AE%A1%E4%B8%8E%E8%A1%A8%E6%83%85%E8%AF%86%E5%88%AB.pdf)
```
