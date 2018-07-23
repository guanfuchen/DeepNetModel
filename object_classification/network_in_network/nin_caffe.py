# -*- coding: utf-8 -*-
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

caffe_root = os.path.expanduser('~/GitHub/GitWeb/caffe/')
model_deploy = os.path.expanduser('~/Data/nin/deploy.prototxt')
model_caffemodel = os.path.expanduser('~/Data/nin/nin_imagenet.caffemodel')
sys.path.insert(0, caffe_root + 'python')
import caffe

if __name__ == '__main__':
    net = caffe.Net(model_deploy, model_caffemodel, caffe.TEST)

    mu = np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
    mu = mu.mean(1).mean(1)
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_mean('data', mu)
    transformer.set_raw_scale('data', 255)
    transformer.set_channel_swap('data', (2, 1, 0))
    net.blobs['data'].reshape(1, 3, 224, 224)

    image = caffe.io.load_image('../../data/cat.jpg')
    transformed_image = transformer.preprocess('data', image)
    # plt.imshow(image)
    # plt.show()

    net.blobs['data'].data[...] = transformed_image
    output = net.forward()
    output_prob = output['pool4'][0, :, 0, 0]
    print('output_prob.shape:', output_prob.shape)
    print('predicted class is:', output_prob.argmax())

    labels_file = '../../data/synset_words.txt'
    labels = np.loadtxt(labels_file, str, delimiter='\t')
    print('output label:', labels[output_prob.argmax()])

    top_inds = output_prob.argsort()[::-1][:5]  # reverse sort and take five largest items
    print('top_inds.shape:', top_inds.shape)
    print('top_inds:', top_inds)
    # print('probabilities:', output_prob[top_inds])
    print('labels:', labels[top_inds])
