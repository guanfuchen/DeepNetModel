# inception

本章节主要介绍inception网络结构V1-V4，该结构和ResNet网络结构一样，在一定程度上带来了新的网络结构设计思路，ResNet通过残差学习增加了网络的深度，Inception通过多个不同kernel的滤波器级联增加了网络的宽度。

引用自[理解ResNet、Inception与Xception](https://bacterous.github.io/2017/12/18/%E7%90%86%E8%A7%A3ResNet%E3%80%81Inception%E4%B8%8EXception/)。
> 第一个见解与对层的操作有关。在传统的卷积网络中，每一层都会从之前的层提取信息，以便将输入数据转换成更有用的表征。但是，不同类型的层会提取不同种类的信息。5×5 卷积核的输出中的信息就和 3×3 卷积核的输出不同，又不同于最大池化核的输出……在任意给定层，我们怎么知道什么样的变换能提供最「有用」的信息呢？
> 见解 1：为什么不让模型选择？
> Inception 模块会并行计算同一输入映射上的多个不同变换，并将它们的结果都连接到单一一个输出。换句话说，对于每一个层，Inception 都会执行 5×5 卷积变换、3×3 卷积变换和最大池化。然后该模型的下一层会决定是否以及怎样使用各个信息。
> 这种模型架构的信息密度更大了，这就带来了一个突出的问题：计算成本大大增加。不仅大型（比如 5×5）卷积过滤器的固有计算成本高，并排堆叠多个不同的过滤器更会极大增加每一层的特征映射的数量。而这种计算成本增长就成为了我们模型的致命瓶颈。



该系列结构探索了分类模型精度和速度上的提升思路，主要由以下组成：
- inception v1，Going Deeper with Convolutions
- inception v2，~~Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift~~（出现引用错误，inception v2和v3均引用自Rethinking the Inception Architcture这篇论文）Rethinking the Inception Architecture for Computer Vision
- inception v3，Rethinking the Inception Architecture for Computer Vision
- inception v4，Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning
- 除此之外增加Xception的相关总结，Xception: Deep Learning with Depthwise Separable Convolutions

---
## inception v1

参考相应论文Going Deeper with Convolutions，主要提出如下图所示架构。

![](http://chenguanfuqq.gitee.io/tuquan2/img_2018_5/inception_v1_arch.png)

上层的卷积块构造的inception结构如下所示：

![](http://chenguanfuqq.gitee.io/tuquan2/img_2018_5/inception_kernel_arch.png)

该网络结构将1x1，3x3，5x5的conv和3x3的pooling，stack在一起，一方面增加了网络的width，另一方面增加了网络对尺度的适应性。主要特点是提高了网络内部计算资源的利用率。利用该结构构造的GoogLeNet网络结构（Goog代表google，LeNet为了致敬LeNet5网络）获得了ILSVRC 2014图像分类和目标检测的第一名，其中具体的网络结构如下图所示。

### GoogLeNet网络架构

![](http://chenguanfuqq.gitee.io/tuquan2/img_2018_5/cnn-googlenet.png)


### 参考资料
- [Going Deeper with Convolutions](https://arxiv.org/abs/1409.4842) inception架构最初始版本，主要创新是增加了3x3卷积、5x5卷积、max池化和1x1卷积以及相应的降维版本结构来降低参数数量，同时增加feature map的多样性提升精度。
- [inception.py](https://github.com/pytorch/vision/blob/master/torchvision/models/inception.py) pytorch中实现了inception v3网络结构。
- [googlenet.py](https://github.com/kuangliu/pytorch-cifar/blob/master/models/googlenet.py) 在cifar中实现了GoogLeNet网络结构。
- [inception_v1_googlenet.py](https://github.com/vadimkantorov/metriclearningbench/blob/master/inception_v1_googlenet.py) 纯粹的pytorch实现的inception v1 GoogLeNet网络结构，同时有预训练的imagenet权重。
- [Going Deeper with Convolutions 阅读笔记](https://asdf0982.github.io/2017/06/26/GoogleNet/)
- [caffe bvlc_googlenet](https://github.com/BVLC/caffe/blob/master/models/bvlc_googlenet/train_val.prototxt)
- [GoogLeNet示例](https://nndl.github.io/v/cnn-googlenet.html)
- [One by One [ 1 x 1 ] Convolution - counter-intuitively useful](https://iamaaditya.github.io/2016/03/one-by-one-convolution/) 其中介绍了GoogLeNet中常用的1x1卷积操作，也可以查看中文[为什么GoogleNet中的Inception Module使用1*1 convolutions?](https://ziyubiti.github.io/2016/11/13/googlenet-inception/)。

---
## inception v2

~~inception v2主要提出了由于在训练期间每层输入的分布发生变换，使得训练深度神经网络非常复杂，往往需要较低的学习率和仔细的参数初始化来减慢训练速度。这种输入变换的现象在文中被称为Internal Covariate Shift，可以理解为内部样本点偏移，本文通过Batch Norm归一化的方法来解决这个问题，解决了网络训练速度较慢的现象，同时略微提升了网络的精度。~~（引用错误）

网络架构如下表所示：

| 类型 | size/stride | 输入大小 |
| --- | --- | --- |
| conv | 3x3/2 | 299x299x3 |
| conv | 3x3/1 | 149x149x32 |
| conv padded | 3x3/1 | 147x147x32 |
| pool | 3x3/2 | 147x147x64 |
| conv | 3x3/1 | 73x73x64 |
| conv | 3x3/2 | 71x71x80 |
| conv | 3x3/1 | 35x35x192 |
| 3xInception | 图5 | 35x35x288 |
| 5xInception | 图6 | 17x17x768 |
| 2xInception | 图7 | 8x8x1280 |
| pool | 8x8 | 8x8x2048 |
| linear | logits | 1x1x2048 |
| softmax | classifier | 1x1x1000 |

inception v2图5结构
![inception_v2_fig5](http://chenguanfuqq.gitee.io/tuquan2/img_2018_5/inception_v2_fig5.png)

inception v2图6结构
![inception_v2_fig6](http://chenguanfuqq.gitee.io/tuquan2/img_2018_5/inception_v2_fig6.png)

inception v2图7结构
![inception_v2_fig7](http://chenguanfuqq.gitee.io/tuquan2/img_2018_5/inception_v2_fig7.png)


### 参考资料

---
## inception v3

### 参考资料

---
## inception v4

### 参考资料

---
## Xception

### 参考资料

---
## 参考资料
- [理解ResNet、Inception与Xception](https://bacterous.github.io/2017/12/18/%E7%90%86%E8%A7%A3ResNet%E3%80%81Inception%E4%B8%8EXception/) 该博客从上层网络模型内在提出的哲学来讲，非常有意思。