# group convolution

群卷积的应用能大大降低计算量，被广泛应用在轻量级网络设计中。该结构最开始出现在AlexNet中，最初的设计是为了能够高效利用GPU将模型训练。

![](https://chenguanfuqq.gitee.io/tuquan2/img_2018_5/alexnetarchitecture.svg)

正常卷积操作

![](https://chenguanfuqq.gitee.io/tuquan2/img_2018_5/convlayer.svg)

group卷积操作，filters被分成了两个group。每一个group都只有原来一半的feature map。

![](https://chenguanfuqq.gitee.io/tuquan2/img_2018_5/filtergroups2.svg)

depthwise卷积

![](https://chenguanfuqq.gitee.io/tuquan2/img_2018_5/depthwise_convolution.png)

depthwise separable卷积

![](https://chenguanfuqq.gitee.io/tuquan2/img_2018_5/depthwise_separable_convolution.png)

---
## 参考资料
- [深度学习之群卷积（Group Convolution）](https://blog.csdn.net/hhy_csdn/article/details/80030468)
- [微软亚研院王井天IGC的演讲](https://edu.csdn.net/course/play/8320)，演讲主要针对其工作的IGC系列，值得
一看。
- [A Tutorial on Filter Groups (Grouped Convolution)](https://blog.yani.io/filter-group-tutorial/)，解释非常直观系统。
- pytorch使用文档可参考[Conv2d](https://pytorch.org/docs/master/nn.html#conv2d)。