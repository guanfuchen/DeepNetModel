# DeepNetModel

深度网络模型从LeNet5、AlexNet、VGGNet和ResNet等等不断改进，每一个模型独特的设计思路都值得好好记录下来，本仓库主要为了整理零散的网络资料，力求图示和代码精简地介绍每一个深度网络模型。


---
## 网络结构目录

- face_detection
    - ...
- object_classification
    - resnet
    - inception
    - network_in_network
    - mobilenet
    - shufflenet
    - alexnet
    - ...
- object_detection
    - R-FCN

---
### ResNet

深度残差网络使得百层网络的训练成为可能，其他deep learning模型中大量采用了该架构。

### Network in Network

caffe model zoo中提供了ImageNet预训练模型文件[Netowork in Network ILSVRC](https://drive.google.com/drive/folders/0B0IedYUunOQINEFtUi1QNWVhVVU)和CIFAR10预训练模型文件[Network in Network CIFAR10 Model](https://gist.github.com/mavenlin/e56253735ef32c3c296d)。

### Inception v1,v2,v3,v4

增加Inception v1，v2，v3和v4论文思路整理，具体查看[inception理解](./object_classification/inception/ReadMe.md)。

### 轻量级网络

#### MobileNet v1,v2

增加轻量级网络MobileNet v1和v2知识整理，具体查看[mobilenet理解](./object_classification/mobilenet/ReadMe.md)。

#### ShuffleNet

增加轻量级网络ShuffleNet知识整理，具体查看[shufflenet理解](./object_classification/shufflenet/ReadMe.md)。

### AlexNet

增加AlexNet知识整理，具体查看[alexnet理解](./object_classification/alexnet/ReadMe.md)。

### VGGNet

增加VGGNet知识整理，具体查看[vggnet理解](./object_classification/vgg/ReadMe.md)。


### R-FCN

增加R-FCN知识整理，具体查看[rfcn理解](./object_detection/rfcn/ReadMe.md)。



---
## 参考资料

- Deep Residual Learning for Image Recognition深度残差网络论文。
- [ResNet, AlexNet, VGGNet, Inception: Understanding various architectures of Convolutional Networks](http://cv-tricks.com/cnn/understand-resnet-alexnet-vgg-inception/) 科普了AlexNet等网络结构。
- [CNN卷积神经网络架构综述](https://chenzomi12.github.io/2016/12/13/CNN-Architectures/) CNN相关的网络架构综述博客，介绍了AlexNet、ZFNet、GoogLeNet、VGGNet和ResNet等常用深度神经网络，同时可以参考综述类文章An Analysis of Deep Neural Network Models for Practical Applications。
- [Caffe神经网络结构汇总](http://noahsnail.com/2017/06/01/2017-6-1-Caffe%E7%BD%91%E7%BB%9C%E7%BB%93%E6%9E%84%E6%80%BB%E7%BB%93/) 介绍了caffe中常用的分类网络的模型结构。
