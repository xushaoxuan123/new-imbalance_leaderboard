
## requirements
```shell
torch 2.1
sklearn
tensorboard
tqdm

```

# 方法
## AGM
epoch = 60

alpha = 1.0
```js
best_acc: 0.5916
path: /data/users/jiahao_li/ks/ckpt/best_model_AGM_of_sgd_epoch60_batch64_lr0.001_alpha1.0.pth
```

## MMCosine
### 方法简介
head 拆分成a v两部分，使用l2 norm后的权重进行计算

loss为a v两部分的loss和

### 实验结果
epoch = 100

alpha/scaling = 10.0

no contrastive loss
```js
best_acc:  0.6664
path : /data/users/jiahao_li/ks/ckpt/best_model_MMCosine_of_sgd_epoch100_batch64_lr0.001_alpha10.0.pth
```

## AMCo
###实验结果
epoch=100



## PMR

## CML


# todo
给AGM调参
观察MMCosine等方法的实现
重写PMR等方法框架
实现transformer架构