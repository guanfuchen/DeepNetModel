# inception

本章节主要介绍inception网络结构V1-V4。

---
## inception v1

参考相应论文Going Deeper with Convolutions，主要提出如下图所示架构。

![](http://chenguanfuqq.gitee.io/tuquan2/img_2018_5/inception_v1_arch.png)

该网络结构将1x1，3x3，5x5的conv和3x3的pooling，stack在一起，一方面增加了网络的width，另一方面增加了网络对尺度的适应性。主要特点是提高了网络内部计算资源的利用率。利用该结构构造的GoogLeNet网络结构（Goog代表google，LeNet为了致敬LeNet5网络）获得了ILSVRC 2014图像分类和目标检测的第一名，其中具体的网络结构如下图所示。




---
## 参考资料
- [Going Deeper with Convolutions](https://arxiv.org/abs/1409.4842) inception架构最初始版本，主要创新是增加了3x3卷积、5x5卷积、max池化和1x1卷积以及相应的降维版本结构来降低参数数量，同时增加feature map的多样性提升精度。
