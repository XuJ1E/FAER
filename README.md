# Facial attribute estimation and expression recognition based on contextual channel attention mechanism

![overview](./set/imgs/Copy (1) of overall architecture.svg)

A PyTorch implementation of the [DAN](http://kns.cnki.net/kcms/detail/51.1307.TP.20240408.1747.012.html.).

  

## Training
We provide the training code for AffectNet and RAF-DB.  

For AffectNet-8 dataset, run:
```
CUDA_VISIBLE_DEVICES=0 python affectnet.py --epochs 10 --num_class 8
```
For AffectNet-7 dataset, run:
```
CUDA_VISIBLE_DEVICES=0 python affectnet.py --epochs 10 --num_class 7
```

For RAF-DB dataset, run:
```
CUDA_VISIBLE_DEVICES=0 python rafdb.py
```

## Models
Pre-trained models can be downloaded for evaluation as following:

|     task    	| accuracy 	| link 	|
|:-----------:	|:--------:	|:----:	|
| CelebA 	| 91.87    	|Comming soon|
| AffectNet 	| 66.66     |Comming soon|  
|    RAF-DB   	| 91.75    	|Comming soon|


## Grad CAM++ Reproduction
![overview](./set/imgs/grad-cam affectnet1.emf)
![overview](./set/imgs/grad-cam celeba1.emf)

