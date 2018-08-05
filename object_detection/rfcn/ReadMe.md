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

## PSROIPooling

该网络是R-FCN网络架构思想的核心，以下从源码实现来解读该实现细节。

```cpp
// caffe中的网络流图都是从bottom流向top，和prototxt中对应的tag相同
template <typename Dtype>
void PSROIPoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {
  // 获取bottom_data数据和bottom_rois数据，将data数据对应rois进行位置敏感的roi池化操作
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_rois = bottom[1]->cpu_data();
  // 输出数据的layers，这里为top_data
  Dtype* top_data = top[0]->mutable_cpu_data();
  # 对应的池化通道mapping
  int* mapping_channel_ptr = mapping_channel_.mutable_cpu_data();
  int count = top[0]->count();
  // 最后输出的网络层数目同时进行相应的初始化0/-1
  caffe_set(count, Dtype(0), top_data);
  caffe_set(count, -1, mapping_channel_ptr);
  // 核心PSROIPooling
  PSROIPoolingForward(bottom[1]->num(), bottom_data, spatial_scale_,
    channels_, height_, width_, pooled_height_,
    pooled_width_, bottom_rois, output_dim_, group_size_,
    top_data, mapping_channel_ptr);
}
```

```cpp
template <typename Dtype>
static void PSROIPoolingForward(const int num, const Dtype* bottom_data, const Dtype spatial_scale, const int channels, const int height, const int width, const int pooled_height, const int pooled_width, const Dtype* bottom_rois, const int output_dim, const int group_size, Dtype* top_data, int* mapping_channel) {
    // roi数量num，对每一个roi进行ps roi池化操作
    for (int n = 0; n < num; ++n) {
        // roi是间隔5的（x，y，w，h，score），所以roi_add为对应的处理的roi index
        int roi_add = n*5;
        // [start, end) interval for spatial sampling
        // roi_batch_ind为对应的roi index
        int roi_batch_ind = bottom_rois[roi_add];
        // roi_start_w和roi_start_h需要下采样到当前的feature map的宽度和高度，这里以stride=16为例，那么spatial_scale=1/16.0
        // roi的存放顺序为x，y，w，h，score，其中x为roi左上角的坐标，w和h为对应的roi的宽度和高度，这里适当+1为了适当增加roi区域
        Dtype roi_start_w = static_cast<Dtype>(round(bottom_rois[roi_add + 1])) * spatial_scale;
        Dtype roi_start_h = static_cast<Dtype>(round(bottom_rois[roi_add + 2])) * spatial_scale;
        Dtype roi_end_w = static_cast<Dtype>(round(bottom_rois[roi_add + 3]) + 1.) * spatial_scale;
        Dtype roi_end_h = static_cast<Dtype>(round(bottom_rois[roi_add + 4]) + 1.) * spatial_scale;

        // Force too small ROIs to be 1x1
        // 将过小的ROIs设置为1x1
        Dtype roi_width = max<Dtype>(roi_end_w - roi_start_w, 0.1);
        Dtype roi_height = max<Dtype>(roi_end_h - roi_start_h, 0.1);

        // Compute w and h at bottom
        // 计算底层的w和h，也就是最后需要池化为pooled_height和pooled_width大小的区域，那么将roi分割为这些大小后，各自对应的bin的大小为bin_size_h和bin_size_w
        Dtype bin_size_h = roi_height / static_cast<Dtype>(pooled_height);
        Dtype bin_size_w = roi_width / static_cast<Dtype>(pooled_width);

        // ctop为输出的ps roi池化维度，ctop应该是增加的池化大小
        for (int ctop = 0; ctop < output_dim; ++ctop) {
            // 对应的每一个pooled_height和pooled_width中的grid进行对应的池化
            for (int ph = 0; ph < pooled_height; ++ph) {
                for (int pw = 0; pw < pooled_width; ++pw) {
                    int index = n*output_dim*pooled_height*pooled_width + ctop*pooled_height*pooled_width + ph*pooled_width + pw;
                    // The output is in order (n, ctop, ph, pw)
                    // 输出顺序是（n, ctop, ph, pw）也就是（#roi, output_dim, pooled_height, pooled_width）
                    // bin内的start和end
                    int hstart = floor(static_cast<Dtype>(ph) * bin_size_h + roi_start_h);
                    int wstart = floor(static_cast<Dtype>(pw)* bin_size_w + roi_start_w);
                    int hend = ceil(static_cast<Dtype>(ph + 1) * bin_size_h + roi_start_h);
                    int wend = ceil(static_cast<Dtype>(pw + 1) * bin_size_w + roi_start_w);
                    // Add roi offsets and clip to input boundaries
                    // 边界检测，需要在[0，height/width]之间
                    hstart = min(max(hstart, 0), height);
                    hend = min(max(hend, 0), height);
                    wstart = min(max(wstart, 0), width);
                    wend = min(max(wend, 0), width);
                    // 查看该bin是否是空的
                    bool is_empty = (hend <= hstart) || (wend <= wstart);
                    // 一组的宽度和高度
                    int gw = pw;
                    int gh = ph;
                    int c = (ctop*group_size + gh)*group_size + gw;

                    // 池化层最后输出的sum
                    Dtype out_sum = 0;
                    // 遍历每一个bin，从h到w进行遍历
                    for (int h = hstart; h < hend; ++h) {
                        for (int w = wstart; w < wend; ++w) {
                            // 从当前的index对应到真实数据bottom_data中的index
                            int bottom_index = h*width + w;
                            out_sum += bottom_data[(roi_batch_ind * channels + c) * height * width + bottom_index];
                        }
                    }
                    Dtype bin_area = (hend - hstart)*(wend - wstart);
                    if (is_empty){
                        top_data[index] = 0;
                    }
                    else{
                        // 将池化输出通过平局获取
                        top_data[index] = out_sum/bin_area;
                    }
                    // 对应的mapping_channel为c
                    mapping_channel[index] = c;
                }
            }
        }
    }
}

```

---
## 参考资料
- [R-FCN论文翻译——中文版](http://noahsnail.com/2018/01/22/2018-01-22-R-FCN%E8%AE%BA%E6%96%87%E7%BF%BB%E8%AF%91%E2%80%94%E2%80%94%E4%B8%AD%E6%96%87%E7%89%88/)
- [Understanding Region-based Fully Convolutional Networks (R-FCN) for object detection](https://medium.com/@jonathan_hui/understanding-region-based-fully-convolutional-networks-r-fcn-for-object-detection-828316f07c99) 这篇博客非常直观，详细参考。
- [pytorch_RFCN](https://github.com/PureDiors/pytorch_RFCN) pytorch实现的R-FCN网络。
- [psroi_pooling_layer.cu](https://github.com/daijifeng001/caffe-rfcn/blob/4bcfcd104bb0b9f0862e127c71bd845ddf036f14/src/caffe/layers/psroi_pooling_layer.cu) 作者caffe实现的PSROIPooling的源码，对应的CPU第三方实现[psroi_pooling_layer.cpp](https://github.com/soeaver/py-RFCN-priv/blob/master/caffe-priv/src/caffe/layers/psroi_pooling_layer.cpp)。
- [RFCN train_val.prototxt](https://github.com/daijifeng001/R-FCN/blob/master/models/rfcn_prototxts/ResNet-101L_res3a/train_val.prototxt)和[RFCN test.prototxt](https://github.com/daijifeng001/R-FCN/blob/master/models/rfcn_prototxts/ResNet-101L_res3a/test.prototxt)