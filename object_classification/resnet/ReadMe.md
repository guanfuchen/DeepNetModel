# ResNet

深度残差网络。

引用自[理解ResNet、Inception与Xception](https://bacterous.github.io/2017/12/18/%E7%90%86%E8%A7%A3ResNet%E3%80%81Inception%E4%B8%8EXception/#%E6%9C%AA%E6%9D%A5%E5%8F%91%E5%B1%95)
> ResNet 诞生于一个美丽而简单的观察：为什么非常深度的网络在增加更多层时会表现得更差？
> 直觉上推测，更深度的网络不会比更浅度的同类型网络表现更差吧，至少在训练时间上是这样（当不存在过拟合的风险时）。让我们进行一个思想实验，假设我们已经构建了一个 n 层网络，并且实现了一定准确度。那么一个 n+1 层网络至少也应该能够实现同样的准确度——只要简单复制前面 n 层，再在最后一层增加一层恒等映射就可以了。类似地，n+2、n+3 和 n+4 层的网络都可以继续增加恒等映射，然后实现同样的准确度。但是在实际情况下，这些更深度的网络基本上都会表现得更差。
> ResNet 的作者将这些问题归结成了一个单一的假设：直接映射是难以学习的。而且他们提出了一种修正方法：不再学习从 x 到 H(x) 的基本映射关系，而是学习这两者之间的差异，也就是「残差（residual）」。然后，为了计算 H(x)，我们只需要将这个残差加到输入上即可。
> 假设残差为 F(x)=H(x)-x，那么现在我们的网络不会直接学习 H(x) 了，而是学习 F(x)+x。
> ResNet 的每一个「模块（block）」都由一系列层和一个「捷径（shortcut）」连接组成，这个「捷径」将该模块的输入和输出连接到了一起。然后在元素层面上执行「加法（add）」运算，如果输入和输出的大小不同，那就可以使用零填充或投射（通过 1×1 卷积）来得到匹配的大小。


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

## ResNet101网络架构

ResNet101网络架构由$(3+4+23+3)*3+2=101$个卷积层构成，其中stride为2时，需要对$x$进行下采样然后加入到$F(x)$中（图中虚线部分）。

![](http://chenguanfuqq.gitee.io/tuquan2/img_2018_5/resnet101_arch.png)

---
## 其他网络架构

其他网络架构细节如下图所示，包括ResNet18、ResNet34、ResNet50、ResNet101和ResNet152等。

![](http://chenguanfuqq.gitee.io/tuquan2/img_2018_5/resnet_arch_list.png)

---
# Identity Mappings in Deep Residual Networks

Deep residual networks（ResNets）由许多“Residual Units”组成。每一个单元表达形式如下所示：

$$y_l=h(x_l)+F(x_l,W_l)$$
$$x_{l+1}=f(y_l)$$

其中$x_l$和$x_{l+1}$是第$l$个单元的输入和输出，$F$是残差函数。ResNets网络中$h(x_l)=x_l$是identity mapping，$f$是$ReLU$函数。

本文推倒发现如果$f(y_l)$和$h(x_l)$是identity mapping，信号能够从一个单元直接传播到另一个单元。当接近满足以上条件时，实验显示训练更加容易。

另外和ResNets不同的是，本文使用了预激活而不是后激活，提升了网络的精度。

如果$f$和$h$是identity mapping，那么上述公式变为如下：
$$x_{l+1}=x_l+F(x_l,W_l)$$
递推得到：
$$x_{l+2}=x_{l+1}+F(x_{l+1},W_{l+1})=x_l+F(x_l,W_l+F(x_{l+1},W_{l+1})$$
$$x_{L}=x_l+\sum_{i=l}^{L-1}{F(x_i,W_i)}$$
上述公式具有以下属性：
- 任何深度单元$L$的特征$x_L$都能表示为任何浅层单元$l$的特征$x_l$的加上残差函数；
- 特征$x_{L}=x_0+\sum_{i=0}^{L-1}{F(x_i,W_i)}$是所有先前残差网络地输出之和，这和plain的矩阵向量乘积不同；

反向传播误差如下：
**反向传播误差**

前向传播和反向传播显示了信号能够直接从一个单元直接传播到另一个单元。


---
## 参考资料

- Deep Residual Learning for Image Recognition深度残差网络论文。
- Identity Mappings in Deep Residual Networks另一篇相关深度残差网络论文，更多数学上的证明。
- [resnet.py](https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py) pytorch中resnet实现。
- [ResNet解读](https://satisfie.github.io/2016/09/15/ResNet%E8%A7%A3%E8%AF%BB/) 介绍了ResNet，极大地总结还原了论文中的核心思想。
- [ResNet-50-deploy.prototxt](https://github.com/KaimingHe/deep-residual-networks/blob/master/prototxt/ResNet-50-deploy.prototxt) Kaiming实现的resnet，和pytorch实现的有差别，主要在stride为2的Conv次序上。
- [pytorch-resnet](https://github.com/ruotianluo/pytorch-resnet) 将caffe中的模型转换为pytorch。
