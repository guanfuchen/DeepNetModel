# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.autograd import Variable
from torch.nn import Parameter
from torch import Tensor
import torch.nn.functional as F
import math


class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        # 4 means lstm gate number, including ingate, forgetget, cellgate, outgate
        self.x2h = nn.Linear(in_features=input_size, out_features=4 * hidden_size, bias=bias)
        self.h2h = nn.Linear(in_features=hidden_size, out_features=4 * hidden_size, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            # print(weight.size())
            weight.data.uniform_(-stdv, stdv)

    def forward(self, x, hidden):
        hx, cx = hidden
        x = x.view(-1, x.size(1))
        gates = self.x2h(x) + self.h2h(hx)
        gates = gates.squeeze()
        # 将gates按照通道数dim=1也就是4*hidden_size切分为四个
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = F.sigmoid(ingate)
        forgetgate = F.sigmoid(forgetgate)
        cellgate = F.tanh(cellgate)
        outgate = F.sigmoid(outgate)

        cy = F.mul(cx, forgetgate) + F.mul(ingate, cellgate)
        hy = F.mul(outgate, F.tanh(cy))
        return (hy, cy)

    def init_hidden(self, batch_size):
        return (Variable(torch.randn(batch_size, self.hidden_size)), Variable(torch.randn(batch_size, self.hidden_size)))


class CLSTMCell(nn.Module):
    """
    Convolution LSTMCell
    """
    def __init__(self, input_shape, input_channels, filter_size, num_features):
        """
        :param input_shape: input shape: H, W
        :param input_channels: input channels
        :param filter_size: filter size for convolution with state-to-state and input-to-state
        :param num_features: convolution output channles
        """
        super(CLSTMCell, self).__init__()
        self.shape = input_shape
        self.input_channels = input_channels
        self.filter_size = filter_size
        self.num_features = num_features
        self.padding = (filter_size - 1)/2 # padding for getting same output shape with input shape H and W
        self.conv = nn.Conv2d(in_channels=(self.input_channels + self.num_features), out_channels=4*self.num_features, kernel_size=self.filter_size, stride=1, padding=self.padding)

    def forward(self, input, hidden_state):
        hx, cx = hidden_state
        combined = torch.cat((input, hx), 1)
        gates = self.conv(combined)
        ingate, forgetgate, cellgate, outgate = torch.split(gates, self.num_features, dim=1) # 在通道1也就是C上将原先的4*num_features且各位num_features，分别为ingate, forgetgate, cellgate和outgate这四个门控中
        ingate = F.sigmoid(ingate) # sigmoid输出为0-1，表示某些输入通过，某些输入不通过
        forgetgate = F.sigmoid(forgetgate) # 同ingate，只不过针对的目标是原先的cell值
        cellgate = F.tanh(cellgate) # 输出门用tanh，也可以考虑使用relu等激活函数
        outgate = F.sigmoid(outgate) # 同ingate，只不过针对的目标是最后的update值

        cy = (forgetgate*cx) + (ingate*cellgate)
        hy = outgate * F.tanh(cy)
        return hy, cy

    def init_hidden(self, batch_size):
        return (Variable(torch.zeros(batch_size, self.num_features, self.shape[0], self.shape[1])), Variable(torch.zeros(batch_size, self.num_features, self.shape[0], self.shape[1])))

if __name__ == '__main__':
    # --------------LSTMCell输入为1维时序特征----------------
    batch_size = 3
    sample_seqlen = 6
    input_size = 10
    hidden_size = 10
    rnn = LSTMCell(input_size, hidden_size)
    input = torch.randn(sample_seqlen, batch_size, input_size)
    # hx = torch.randn(batch_size, hidden_size)
    # cx = torch.randn(batch_size, hidden_size)
    hx = None
    cx = None
    if hx is None and cx is None:
        hx , cx = rnn.init_hidden(batch_size)
    output = []
    for i in range(sample_seqlen):
        hx, cx = rnn(input[i], (hx, cx))
        output.append(hx)
    # 其中output包含了所有的hidden参数，如果仅仅输出最后一层，或者最多和输入图像相同时序长度的隐藏状态
    print(len(output))
    # --------------LSTMCell输入为1维时序特征----------------

    # --------------CLSTMCell输入为2维时序图像----------------
    batch_size = 3
    sample_seqlen = 6
    input_shape = (64, 64) # H, W
    input_channels = 1
    filter_size = 5
    num_features = 128
    rnn = CLSTMCell(input_shape, input_channels, filter_size, num_features)
    input = torch.randn(sample_seqlen, batch_size, input_channels, input_shape[0], input_shape[1])
    # hx = torch.randn(batch_size, hidden_size)
    # cx = torch.randn(batch_size, hidden_size)
    hx = None
    cx = None
    if hx is None and cx is None:
        hx , cx = rnn.init_hidden(batch_size)
    output = []
    for i in range(sample_seqlen):
        hx, cx = rnn(input[i], (hx, cx))
        output.append(hx)
    # 其中output包含了所有的hidden参数，如果仅仅输出最后一层，或者最多和输入图像相同时序长度的隐藏状态
    print(len(output))
    # --------------CLSTMCell输入为2维时序图像----------------
