# ResNet

深度残差网络。

---
## ResidualBlock
残差块通过学习残差$F(x)$来表示$H(x)$，其中$F(x)=H(x)-x$。

![](http://chenguanfuqq.gitee.io/tuquan2/img_2018_5/resnet_residualblock.png)


---
## BasicBlock
浅层的残差块，主要应用在ResNet18和ResNet34中，其中的卷积层一般紧接着BN缓解梯度消失。

![](http://chenguanfuqq.gitee.io/tuquan2/img_2018_5/resnet_basicblock.png)

---
## Bottleneck
深层的残差块，主要应用在ResNet50、ResNet101和ResNet152中。

![](http://chenguanfuqq.gitee.io/tuquan2/img_2018_5/resnet_bottleneck.png)


---
## ResNet34网络架构

ResNet34网络架构由$(3+4+6+3)*2+2=34$个卷积层构成，其中stride为2时，需要对$x$进行下采样然后加入到$F(x)$中（图中虚线部分）。

![](http://chenguanfuqq.gitee.io/tuquan2/img_2018_5/resnet34_arch.png)

---
## 其他网络架构

其他网络架构细节如下图所示，包括ResNet18、ResNet34、ResNet50、ResNet101和ResNet152等。

![](http://chenguanfuqq.gitee.io/tuquan2/img_2018_5/resnet_arch_list.png)

---
## 参考资料

- Deep Residual Learning for Image Recognition深度残差网络论文。
- Identity Mappings in Deep Residual Networks另一篇相关深度残差网络论文，更多数学上的证明。
- [resnet.py](https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py) pytorch中resnet实现。
- [ResNet解读](https://satisfie.github.io/2016/09/15/ResNet%E8%A7%A3%E8%AF%BB/) 介绍了ResNet，极大地总结还原了论文中的核心思想。
- [ResNet-50-deploy.prototxt](https://github.com/KaimingHe/deep-residual-networks/blob/master/prototxt/ResNet-50-deploy.prototxt) Kaiming实现的resnet，和pytorch实现的有差别，主要在stride为2的Conv次序上。
- [pytorch-resnet](https://github.com/ruotianluo/pytorch-resnet) 将caffe中的模型转换为pytorch。
