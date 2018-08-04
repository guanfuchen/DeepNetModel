# R-FCN

## R-FCN-ResNet101

这里以resnet101为基础网络的R-FCN网络为例，其中resnet 101中conv4以及先前的网络层都保持不变（stride=16），去除最后的avg pool层和全连接层，同时conv5以后进行了如下变化：
- conv5的第一个卷积快stride=2修改为stride=1；
- conv5 stage中的所有卷积网络层（resnet building block中的中间网络层）使用空洞卷积，将dilation转换为2，对先前stride=1修改带来的感受野的降低进行弥补；
- 最后一层增加一层1x1，1024的卷积网络，变换原先的2048维度的卷积feature map，降低计算代价；

```
# conv5的第一个卷积快stride=2修改为stride=1
res5a_branch1 = mx.symbol.Convolution(name='res5a_branch1', data=conv_feat, num_filter=2048, pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
res5a_branch2a = mx.symbol.Convolution(name='res5a_branch2a', data=conv_feat, num_filter=512, pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
# conv5 stage中的所有卷积网络层（resnet building block中的中间网络层）使用空洞卷积，将dilation转换为2，对先前stride=1修改带来的感受野的降低进行弥补
res5a_branch2b = mx.symbol.Convolution(name='res5a_branch2b', data=res5a_branch2a_relu, num_filter=512, pad=(2, 2), kernel=(3, 3), stride=(1, 1), dilate=(2, 2), no_bias=True, cudnn_off=True)
res5b_branch2b = mx.symbol.Convolution(name='res5b_branch2b', data=res5b_branch2a_relu, num_filter=512, pad=(2, 2), kernel=(3, 3), stride=(1, 1), dilate=(2, 2), no_bias=True, cudnn_off=True)
res5c_branch2b = mx.symbol.Convolution(name='res5c_branch2b', data=res5c_branch2a_relu, num_filter=512, pad=(2, 2), kernel=(3, 3), stride=(1, 1), dilate=(2, 2), no_bias=True, cudnn_off=True)
```

具体的基础网络架构如下图所示

![](http://chenguanfuqq.gitee.io/tuquan2/img_2018_5/rfcn_resnet101_caffe_arch.png)

---
## 参考资料
- [R-FCN论文翻译——中文版](http://noahsnail.com/2018/01/22/2018-01-22-R-FCN%E8%AE%BA%E6%96%87%E7%BF%BB%E8%AF%91%E2%80%94%E2%80%94%E4%B8%AD%E6%96%87%E7%89%88/)
- [Understanding Region-based Fully Convolutional Networks (R-FCN) for object detection](https://medium.com/@jonathan_hui/understanding-region-based-fully-convolutional-networks-r-fcn-for-object-detection-828316f07c99) 这篇博客非常直观，详细参考。