# inception

本章节主要介绍inception网络结构V1-V4。

该系列结构探索了分类模型精度和速度上的提升思路，主要由以下组成：
- inception v1，Going Deeper with Convolutions
- inception v2，Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift
- inception v3，Rethinking the Inception Architecture for Computer Vision
- inception v4，Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning

---
## inception v1

参考相应论文Going Deeper with Convolutions，主要提出如下图所示架构。

![](http://chenguanfuqq.gitee.io/tuquan2/img_2018_5/inception_v1_arch.png)

该网络结构将1x1，3x3，5x5的conv和3x3的pooling，stack在一起，一方面增加了网络的width，另一方面增加了网络对尺度的适应性。主要特点是提高了网络内部计算资源的利用率。利用该结构构造的GoogLeNet网络结构（Goog代表google，LeNet为了致敬LeNet5网络）获得了ILSVRC 2014图像分类和目标检测的第一名，其中具体的网络结构如下图所示。


### 参考资料
- [Going Deeper with Convolutions](https://arxiv.org/abs/1409.4842) inception架构最初始版本，主要创新是增加了3x3卷积、5x5卷积、max池化和1x1卷积以及相应的降维版本结构来降低参数数量，同时增加feature map的多样性提升精度。
- [inception.py](https://github.com/pytorch/vision/blob/master/torchvision/models/inception.py) pytorch中实现了inception v3网络结构。
- [googlenet.py](https://github.com/kuangliu/pytorch-cifar/blob/master/models/googlenet.py) 在cifar中实现了GoogLeNet网络结构。
- [inception_v1_googlenet.py](https://github.com/vadimkantorov/metriclearningbench/blob/master/inception_v1_googlenet.py) 纯粹的pytorch实现的inception v1 GoogLeNet网络结构，同时有预训练的imagenet权重。

---
## inception v2

### 参考资料

---
## inception v3

### 参考资料

---
## inception v4

### 参考资料
