# -*- coding: utf-8 -*-
import time

import torch
import visdom
from torch.nn import Conv2d
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import math
from torch.utils.data import Dataset
from torch.autograd import Variable
import numpy as np
import os

# use_cuda = True
# use_visdom = True
use_cuda = False
use_visdom = False

class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        """
        :param input_size: input_Hxinput_W
        :param hidden_size:
        :param bias:
        """
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

    def init_hidden(self, batch_size, use_cuda=False):
        if use_cuda:
            return (Variable(torch.zeros(batch_size, self.hidden_size)).cuda(), Variable(torch.zeros(batch_size, self.hidden_size)).cuda())
        else:
            return (Variable(torch.zeros(batch_size, self.hidden_size)), Variable(torch.zeros(batch_size, self.hidden_size)))


class CLSTMCell(nn.Module):
    """
    Convolution LSTMCell
    """
    def __init__(self, input_shape, input_channels, filter_size, num_features, bias=True, dilation=1):
        """
        :param input_shape: input shape: H, W
        :param input_channels: input channels
        :param filter_size: filter size for convolution with state-to-state and input-to-state
        :param num_features: convolution output channles
        """
        super(CLSTMCell, self).__init__()
        self.input_shape = input_shape
        self.input_channels = input_channels
        self.filter_size = filter_size
        self.num_features = num_features
        self.dilation = dilation
        # if self.dilation==1:
        #     self.padding = (filter_size - 1)/2 # padding for getting same output shape with input shape H and W
        # else:
        #     self.padding = self.dilation
        self.padding = (self.filter_size-1)*self.dilation/2
        self.bias = bias
        self.conv = nn.Conv2d(in_channels=(self.input_channels + self.num_features), out_channels=4*self.num_features, kernel_size=self.filter_size, stride=1, padding=self.padding, bias=self.bias, dilation=self.dilation)

    def forward(self, input, hidden_state):
        # cell means no time relation, input = Bx(input_channels)xHxW, hx = Bx(num_features)xHxW
        hx, cx = hidden_state
        combined = torch.cat((input, hx), 1) # Bx(input_channels+num_features)xHxW
        gates = self.conv(combined) # Bx(4*num_features)xHxW
        # print('gates.shape:', gates.shape)
        # ingate, forgetgate, cellgate, outgate all shape is Bx(num_features)xHxW
        ingate, forgetgate, cellgate, outgate = torch.split(gates, self.num_features, dim=1) # 在通道1也就是C上将原先的4*num_features且各位num_features，分别为ingate, forgetgate, cellgate和outgate这四个门控中
        ingate = F.sigmoid(ingate) # sigmoid输出为0-1，表示某些输入通过，某些输入不通过
        forgetgate = F.sigmoid(forgetgate) # 同ingate，只不过针对的目标是原先的cell值
        cellgate = F.tanh(cellgate) # 输出门用tanh，也可以考虑使用relu等激活函数
        outgate = F.sigmoid(outgate) # 同ingate，只不过针对的目标是最后的update值

        # forgetgate for the cx, ingate for cellgate, outgate for cy tanh
        cy = (forgetgate*cx) + (ingate*cellgate)
        hy = outgate * F.tanh(cy)
        return hy, cy

    def init_hidden(self, batch_size, use_cuda=False):
        if use_cuda:
            return (Variable(torch.zeros(batch_size, self.num_features, self.input_shape[0], self.input_shape[1])).cuda(), Variable(torch.zeros(batch_size, self.num_features, self.input_shape[0], self.input_shape[1])).cuda())
        else:
            return (Variable(torch.zeros(batch_size, self.num_features, self.input_shape[0], self.input_shape[1])), Variable(torch.zeros(batch_size, self.num_features, self.input_shape[0], self.input_shape[1])))

class MCLSTMCell(nn.Module):
    """
    Multiple Convolution LSTMCell like encoder
    """
    def __init__(self, input_shape, input_channels, i2s_filter_size, s2s_filter_size_list, num_features_list, num_layers, return_sequences=False, dilation=1):
        super(MCLSTMCell, self).__init__()
        self.input_shape = input_shape
        self.input_channels = input_channels
        self.i2s_filter_size = i2s_filter_size # [5]
        self.s2s_filter_size_list = s2s_filter_size_list # [5 5]
        self.num_features_list = num_features_list # [64 64]
        self.num_layers = num_layers
        self.return_sequences = return_sequences
        self.s2o_filter_size = 1 # output conv
        self.dilation = dilation

        cell_list = []
        cell_list.append(CLSTMCell(self.input_shape, self.input_channels, self.s2s_filter_size_list[0], self.num_features_list[0], dilation=self.dilation))

        self.return_sequences_conv = Conv2d(in_channels=sum(self.num_features_list), out_channels=input_channels, kernel_size=self.s2o_filter_size, padding=(self.s2o_filter_size - 1)/2)

        for idcell in xrange(1, self.num_layers):
            cell_list.append(CLSTMCell(self.input_shape, self.num_features_list[idcell-1], self.s2s_filter_size_list[idcell], self.num_features_list[idcell]))

        self.cell_list = nn.ModuleList(cell_list)


    def forward(self, input, hidden_state):
        current_input = input # input shape: TxBxCxHxW
        next_hidden = []
        seqlen = current_input.size(0)

        hidden_concat = None
        # 对每一层每一次时间循环
        for idlayer in xrange(self.num_layers):
            outputs = []
            # outputs_seq = []

            hidden_h, hidden_c = hidden_state[idlayer]
            current_layer = self.cell_list[idlayer]


            for t in xrange(seqlen):
                current_input_t = current_input[t, :, :, :, :] # 获取当前输入
                hidden_h, hidden_c = current_layer(current_input_t, (hidden_h, hidden_c)) # 第一层hidden_h和hidden_c使用0初始化变量，输出后作为下一次时间循环
                # 最后一层输出
                # if self.return_sequences and idlayer==self.num_layers-1:
                #     outputs_seq.append(self.return_sequences_conv(hidden_h))
                # print('hidden_h.size():', hidden_h.size())
                outputs.append(hidden_h) # 作为下一层的输入

            # print('hidden_h.size():', hidden_h.size())
            # print('hidden_c.size():', hidden_c.size())
            next_hidden.append((hidden_h, hidden_c)) # 仅仅将最后的hidden_h和hidden_c输出，共有num_layers个输出
            # current_input = torch.cat(outputs, 0).view(seqlen, *outputs[0].size()) # input shape: TxBx(num_features)xHxW
            current_input = torch.cat(outputs, 0).view(seqlen, *outputs[idlayer].size()) # input shape: TxBx(num_features)xHxW
            if hidden_concat is None:
                hidden_concat = Variable(current_input)
            else:
                hidden_concat = torch.cat((Variable(current_input), hidden_concat), 2)


        # print('hidden_concat.size():', hidden_concat.size())
        if self.return_sequences:
            # output sequences channel == input sequences channle
            # current_input = torch.cat(outputs_seq, 0).view(seqlen, *outputs_seq[0].size()) # input shape: TxBx(input_shape)xHxW
            # current_input = current_input.transpose(0, 1)
            out_all = []
            for t in xrange(seqlen):
                hidden_concat_t = hidden_concat[t, ...]
                out_t = self.return_sequences_conv(hidden_concat_t)
                out_all.append(out_t)
            out_all = torch.stack(out_all)
            # print('out_all.size():', out_all.size())
            current_input = out_all

        # next_hidden list len num_layers, every list item is (hidden_h, hidden_c)
        # current_input 为最后一层的hidden_h shape is TxBx(num_features)xHxW, if return_sequences then use Conv for every T features
        return next_hidden, current_input # last current_input is the output of the MCLSTMCell unit

    def init_hidden(self, batch_size, use_cuda=False):
        init_states=[]
        for i in xrange(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, use_cuda))
        return init_states

class MLSTMCell(nn.Module):
    """
    Multiple LSTMCell like encoder
    """
    def __init__(self, input_shape, input_channels, num_features, num_layers, return_sequences=False):
        super(MLSTMCell, self).__init__()
        self.input_shape = input_shape
        self.input_channels = input_channels

        self.num_features = num_features
        self.num_layers = num_layers
        self.return_sequences = return_sequences

        cell_list = []
        cell_list.append(LSTMCell(input_size=self.input_shape[0]*self.input_shape[1]*self.input_channels, hidden_size=self.num_features))

        self.return_sequences_linear = nn.Linear(in_features=num_features*num_layers, out_features=self.input_shape[0]*self.input_shape[1]*self.input_channels)

        for idcell in xrange(1, self.num_layers):
            cell_list.append(LSTMCell(input_size=self.num_features, hidden_size=self.num_features))

        self.cell_list = nn.ModuleList(cell_list)


    def forward(self, input, hidden_state):
        seqlen = input.size(0)
        batch_size = input.size(1)
        # print('input.size:', input.size())
        current_input = input.view(seqlen, batch_size, -1) # input shape: TxBxCxHxW
        # print('current_input.size:', current_input.size())
        next_hidden = []
        hidden_concat = None
        for idlayer in xrange(self.num_layers):
            hidden_h, hidden_c = hidden_state[idlayer]
            outputs = []
            # outputs_seq = []
            for t in xrange(seqlen):
                current_layer = self.cell_list[idlayer]
                current_input_t = current_input[t, :, :]
                hidden_h, hidden_c = current_layer(current_input_t, (hidden_h, hidden_c))
                # if self.return_sequences:
                #     outputs_seq.append(self.return_sequences_linear(hidden_h))
                outputs.append(hidden_h)

            next_hidden.append((hidden_h, hidden_c))
            current_input = torch.cat(outputs, 0).view(seqlen, *outputs[0].size()) # input shape: TxBx(num_features)xHxW

            if hidden_concat is None:
                hidden_concat = Variable(current_input)
            else:
                hidden_concat = torch.cat((Variable(current_input), hidden_concat), 2)

        # print('hidden_concat.size():', hidden_concat.size())
        if self.return_sequences:
            # output sequences channel == input sequences channle
            # current_input = torch.cat(outputs_seq, 0).view(seqlen, *outputs_seq[0].size()) # input shape: TxBx(input_shape)xHxW
            # current_input = current_input.transpose(0, 1)
            # current_input = current_input.view(seqlen, batch_size, self.input_channels, self.input_shape[0], self.input_shape[1])
            out_all = []
            for t in xrange(seqlen):
                hidden_concat_t = hidden_concat[t, ...]
                out_t = self.return_sequences_linear(hidden_concat_t)
                # print('out_t.shape:', out_t.shape)
                out_all.append(out_t)
            out_all = torch.stack(out_all)
            out_all = out_all.view(seqlen, batch_size, self.input_channels, self.input_shape[0], self.input_shape[1])
            current_input = out_all

        return next_hidden, current_input # last current_input is the output of the MCLSTMCell unit

    def init_hidden(self, batch_size, use_cuda=False):
        init_states=[]
        for i in xrange(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, use_cuda))
        return init_states


class MovingMNISTLoader(Dataset):
    def __init__(self, root, split="train", split_ratio=0.7):
        self.root = root
        self.split = split
        self.data = np.load(self.root).transpose(1, 0, 2, 3) # from TxSxHxW to SxTxHxW
        all_len = len(self.data[:, 0, 0, 0])
        split_index = int(all_len*split_ratio)
        if self.split=='train':
            self.data = self.data[:split_index]
        elif self.split=='val':
            self.data = self.data[split_index:]
        print('self.data.shape:', self.data.shape)

    def __len__(self):
        return len(self.data[:, 0, 0, 0])

    def __getitem__(self, index):
        img_np = self.data[index, ...]
        img_np = img_np/255.0
        img_tensor = torch.from_numpy(np.expand_dims(img_np, axis = 1)).float()
        return img_tensor

def crossentropyloss(pred, target):
    loss = -torch.sum(torch.log(pred)*target + torch.log(1-pred)*(1-target))
    return loss

class ResCLSTMCell(nn.Module):
    """
    Multiple Convolution LSTMCell Prediction Net
    """
    def __init__(self, input_shape, input_channels, filter_size, num_features):
        super(ResCLSTMCell, self).__init__()
        self.clstmcell_1 = CLSTMCell(input_shape, input_channels, filter_size, num_features)
        self.clstmcell_2 = CLSTMCell(input_shape, num_features, filter_size, num_features)

    def forward(self, input, hidden_state):
        out = F.relu(input)
        out_clstmcell_1_hc = self.clstmcell_1(out, hidden_state[0])
        out_clstmcell_1 = F.relu(out_clstmcell_1_hc[0])
        out_clstmcell_2_hc = self.clstmcell_2(out_clstmcell_1, hidden_state[1])
        out_clstmcell_2 = F.relu(out_clstmcell_2_hc[0])
        out = out_clstmcell_1+out_clstmcell_2
        # (hy, cy)

        return [out_clstmcell_1_hc, out_clstmcell_2_hc], out # tbchw

    def init_hidden(self, batch_size, use_cuda=False):
        return [
            self.clstmcell_1.init_hidden(batch_size, use_cuda),
            self.clstmcell_2.init_hidden(batch_size, use_cuda),
        ]


class MResCLSTMCell(nn.Module):
    """
    Multiple Convolution LSTMCell like encoder
    """
    def __init__(self, input_shape, input_channels, i2s_filter_size, s2s_filter_size_list, num_features_list, num_layers, return_sequences=False, dilation=1):
        super(MResCLSTMCell, self).__init__()
        self.input_shape = input_shape
        self.input_channels = input_channels
        self.i2s_filter_size = i2s_filter_size # [5]
        self.s2s_filter_size_list = s2s_filter_size_list # [5 5]
        self.num_features_list = num_features_list # [64 64]
        self.num_layers = num_layers
        self.return_sequences = return_sequences
        self.s2o_filter_size = 1 # output conv
        self.dilation = dilation

        cell_list = []
        cell_list.append(ResCLSTMCell(self.input_shape, self.input_channels, self.s2s_filter_size_list[0], self.num_features_list[0]))

        # self.return_sequences_conv = Conv2d(in_channels=sum(self.num_features_list), out_channels=input_channels, kernel_size=self.s2o_filter_size, padding=(self.s2o_filter_size - 1)/2)

        for idcell in xrange(1, self.num_layers):
            cell_list.append(ResCLSTMCell(self.input_shape, self.num_features_list[idcell-1], self.s2s_filter_size_list[idcell], self.num_features_list[idcell]))

        self.cell_list = nn.ModuleList(cell_list)


    def forward(self, input, hidden_state):
        current_input = input # input shape: TxBxCxHxW
        next_hidden = []
        seqlen = current_input.size(0)

        hidden_concat = None
        # 对每一层每一次时间循环
        for idlayer in xrange(self.num_layers):
            outputs = []
            # outputs_seq = []

            hidden_h, hidden_c = hidden_state[idlayer]
            hidden_hc = hidden_state[idlayer]
            current_layer = self.cell_list[idlayer]


            for t in xrange(seqlen):
                current_input_t = current_input[t, :, :, :, :] # 获取当前输入
                hidden_hc, res_hidden_out = current_layer(current_input_t, (hidden_hc)) # 第一层hidden_h和hidden_c使用0初始化变量，输出后作为下一次时间循环
                # print('res_hidden_out.shape:', res_hidden_out.shape)
                # 最后一层输出
                # if self.return_sequences and idlayer==self.num_layers-1:
                #     outputs_seq.append(self.return_sequences_conv(hidden_h))
                # print('hidden_h.size():', hidden_h.size())
                outputs.append(res_hidden_out) # 作为下一层的输入

            # print('hidden_h.size():', hidden_h.size())
            # print('hidden_c.size():', hidden_c.size())
            next_hidden.append(hidden_hc) # 仅仅将最后的hidden_h和hidden_c输出，共有num_layers个输出
            # current_input = torch.cat(outputs, 0).view(seqlen, *outputs[0].size()) # input shape: TxBx(num_features)xHxW
            current_input = torch.cat(outputs, 0).view(seqlen, *outputs[idlayer].size()) # input shape: TxBx(num_features)xHxW
            if hidden_concat is None:
                hidden_concat = Variable(current_input)
            else:
                hidden_concat = torch.cat((Variable(current_input), hidden_concat), 2)


        # # print('hidden_concat.size():', hidden_concat.size())
        # if self.return_sequences:
        #     # output sequences channel == input sequences channle
        #     # current_input = torch.cat(outputs_seq, 0).view(seqlen, *outputs_seq[0].size()) # input shape: TxBx(input_shape)xHxW
        #     # current_input = current_input.transpose(0, 1)
        #     out_all = []
        #     for t in xrange(seqlen):
        #         hidden_concat_t = hidden_concat[t, ...]
        #         out_t = self.return_sequences_conv(hidden_concat_t)
        #         out_all.append(out_t)
        #     out_all = torch.stack(out_all)
        #     # print('out_all.size():', out_all.size())
        #     current_input = out_all

        # next_hidden list len num_layers, every list item is (hidden_h, hidden_c)
        # current_input 为最后一层的hidden_h shape is TxBx(num_features)xHxW, if return_sequences then use Conv for every T features
        return next_hidden, current_input # last current_input is the output of the MCLSTMCell unit

    def init_hidden(self, batch_size, use_cuda=False):
        init_states=[]
        for i in xrange(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, use_cuda))
        return init_states


def torch_var_tbchw2bcthw(x):
    # TxBxCxHxW ----BxCxTxHxW
    return x.transpose(0, 1).transpose(1, 2).contiguous()

def torch_var_bcthw2tbchw(x):
    # BxCxTxHxW ---- TxBxCxHxW
    return x.transpose(0, 2).transpose(1, 2).contiguous()

# class ResCLSTMPredNet(nn.Module):
#     """
#     Multiple Convolution LSTMCell Prediction Net
#     """
#     def __init__(self, input_shape, input_channels):
#         super(ResCLSTMPredNet, self).__init__()
#         self.conv3d_1 = nn.Conv3d(in_channels=input_channels, out_channels=64, kernel_size=3, stride=1, padding=1)
#
#         self.resclstm_1 = ResCLSTM(input_shape, 64, 64)
#         self.resclstm_2 = ResCLSTM(input_shape, 64, 128)
#         self.resclstm_3 = ResCLSTM(input_shape, 128, 1)
#
#     def forward(self, input, init_states):
#         input_bcthw = torch_var_tbchw2bcthw(input)
#         out_bcthw = self.conv3d_1(input_bcthw)
#         out = torch_var_bcthw2tbchw(out_bcthw)
#         out = self.resclstm_1(out, init_states[0])
#         out = self.resclstm_2(out, init_states[1])
#         out = self.resclstm_3(out, init_states[2])
#         return out
#
#     def init_hidden(self, batch_size, use_cuda=False):
#         # return [self.resclstm_1.init_hidden(batch_size, use_cuda)]
#         return [self.resclstm_1.init_hidden(batch_size, use_cuda), self.resclstm_2.init_hidden(batch_size, use_cuda), self.resclstm_3.init_hidden(batch_size, use_cuda)]
#
# class CLSTMPredNet(nn.Module):
#     """
#     Multiple Convolution LSTMCell Prediction Net
#     """
#     def __init__(self, input_shape, input_channels):
#         super(CLSTMPredNet, self).__init__()
#         self.num_layers = 3
#         self.input_shape = input_shape
#         self.input_channels = input_channels
#         self.num_features_list = [128, 64, 64]
#         self.s2o_filter_size = 1
#         self.s2s_filter_size_list = [5, 5, 5]
#         self.mclstmcell_1 = MCLSTMCell(input_shape=input_shape, input_channels=input_channels, i2s_filter_size=5, s2s_filter_size_list=self.s2s_filter_size_list, num_features_list=self.num_features_list, num_layers=self.num_layers, return_sequences=False)
#
#         cell_list = []
#         cell_list.append(CLSTMCell(self.input_shape, self.input_channels, self.s2s_filter_size_list[0], self.num_features_list[0]))
#
#         self.return_sequences_conv = Conv2d(in_channels=sum(self.num_features_list), out_channels=input_channels, kernel_size=self.s2o_filter_size, padding=(self.s2o_filter_size - 1)/2)
#
#         for idcell in xrange(1, self.num_layers):
#             cell_list.append(CLSTMCell(self.input_shape, self.num_features_list[idcell-1], self.s2s_filter_size_list[idcell], self.num_features_list[idcell]))
#
#         self.cell_list = nn.ModuleList(cell_list)
#
#
#     def forward(self, input, init_states):
#         seqlen = input.size(0)
#         batch_size = input.size(1)
#         print('input.shape:', input.shape)
#         out = self.mclstmcell_1(input, init_states[0])
#
#         # change all hidden before concat to output_0
#         hidden_h_concat = None
#         for idlayer in range(self.num_layers):
#             # print('out[0][{}][0].shape:'.format(idlayer), out[0][idlayer][0].shape) # hidden_h
#             # print('out[0][{}][1].shape:'.format(idlayer), out[0][idlayer][1].shape) # hidden_c
#             hidden_h_idlayer = out[0][idlayer][0]
#             if hidden_h_concat is None:
#                 hidden_h_concat = hidden_h_idlayer
#             else:
#                 hidden_h_concat = torch.cat((hidden_h_concat, hidden_h_idlayer), 1)
#
#         print('hidden_h_concat.shape:', hidden_h_concat.shape)
#         out_0 = self.return_sequences_conv(hidden_h_concat)
#         print('out_0.shape:', out_0.shape)
#         current_input = out_0
#         for t in xrange(seqlen):
#             states = out[0]
#             outputs = []
#
#             for idlayer in xrange(self.num_layers):
#                 hidden_states = states[idlayer]
#                 hidden_states = self.cell_list[idlayer](current_input, hidden_states)
#
#                 outputs.append(hidden_states)
#
#                 states[idlayer] = hidden_states
#
#                 current_input = hidden_states
#
#         return out
#
#     def init_hidden(self, batch_size, use_cuda=False):
#         # return [self.resclstm_1.init_hidden(batch_size, use_cuda)]
#         return [self.mclstmcell_1.init_hidden(batch_size, use_cuda)]


if __name__ == '__main__':
    pass
    # # --------------LSTMCell输入为1维时序特征----------------
    # batch_size = 3
    # sample_seqlen = 6
    # input_size = 10
    # hidden_size = 10
    # rnn = LSTMCell(input_size, hidden_size)
    # input = torch.randn(sample_seqlen, batch_size, input_size)
    # # hx = torch.randn(batch_size, hidden_size)
    # # cx = torch.randn(batch_size, hidden_size)
    # hx = None
    # cx = None
    # if hx is None and cx is None:
    #     hx , cx = rnn.init_hidden(batch_size)
    # output = []
    # for i in range(sample_seqlen):
    #     hx, cx = rnn(input[i], (hx, cx))
    #     output.append(hx)
    # # 其中output包含了所有的hidden参数，如果仅仅输出最后一层，或者最多和输入图像相同时序长度的隐藏状态
    # print(len(output))
    # # --------------LSTMCell输入为1维时序特征----------------
    #
    # # --------------CLSTMCell输入为2维时序图像----------------
    # batch_size = 3
    # sample_seqlen = 6
    # input_shape = (64, 64) # H, W
    # input_channels = 1
    # filter_size = 5
    # num_features = 128
    # rnn = CLSTMCell(input_shape, input_channels, filter_size, num_features)
    # input = torch.randn(sample_seqlen, batch_size, input_channels, input_shape[0], input_shape[1])
    # # hx = torch.randn(batch_size, hidden_size)
    # # cx = torch.randn(batch_size, hidden_size)
    # hx = None
    # cx = None
    # if hx is None and cx is None:
    #     hx , cx = rnn.init_hidden(batch_size)
    # output = []
    # for i in range(sample_seqlen):
    #     hx, cx = rnn(input[i], (hx, cx))
    #     output.append(hx)
    # # 其中output包含了所有的hidden参数，如果仅仅输出最后一层，或者最多和输入图像相同时序长度的隐藏状态
    # print(len(output))
    # # --------------CLSTMCell输入为2维时序图像----------------
    #
    #
    # # --------------MCLSTMCell输入为2维时序图像----------------
    # batch_size = 3
    # sample_seqlen = 6
    # input_shape = (64, 64) # H, W
    # input_channels = 1
    # filter_size = 5
    # num_features = 128
    # num_layers = 2
    # rnn = MCLSTMCell(input_shape, input_channels, filter_size, num_features, num_layers)
    # input = torch.randn(sample_seqlen, batch_size, input_channels, input_shape[0], input_shape[1])
    # # hx = torch.randn(batch_size, hidden_size)
    # # cx = torch.randn(batch_size, hidden_size)
    # init_states = None
    # if init_states is None:
    #     init_states = rnn.init_hidden(batch_size)
    # output = rnn(input, init_states)
    # for i in xrange(num_layers):
    #     hidden_h, hidden_c = output[0][i]
    #     print('hidden_h.size():', hidden_h.size())
    #     print('hidden_c.size():', hidden_c.size())
    # # --------------MCLSTMCell输入为2维时序图像----------------



    # # --------------MLSTMCell输入为2维时序图像----------------
    # batch_size = 3
    # sample_seqlen = 6
    # input_shape = (64, 64) # H, W
    # input_channels = 1
    # filter_size = 5
    # num_layers = 2
    # num_features = 128
    # return_sequences = True
    # rnn = MLSTMCell(input_shape, input_channels, num_features, num_layers, return_sequences)
    # input = torch.randn(sample_seqlen, batch_size, input_channels, input_shape[0], input_shape[1])
    # # hx = torch.randn(batch_size, hidden_size)
    # # cx = torch.randn(batch_size, hidden_size)
    # init_states = None
    # if init_states is None:
    #     init_states = rnn.init_hidden(batch_size)
    # output = rnn(input, init_states)
    # last_hidden = output[1]
    # if return_sequences:
    #     pass
    #     last_hidden = last_hidden.view(sample_seqlen, batch_size, input_channels, input_shape[0], input_shape[1])
    #
    # print('last_hidden.size():', last_hidden.size())
    # # for i in xrange(num_layers):
    # #     hidden_h, hidden_c = output[0][i]
    #     # print('hidden_h.size():', hidden_h.size())
    #     # print('hidden_c.size():', hidden_c.size())
    # # --------------MLSTMCell输入为2维时序图像----------------


    # # --------------MCLSTMCell输入为2维时序图像----------------
    # batch_size = 3
    # sample_seqlen = 6
    # input_shape = (64, 64) # H, W
    # input_channels = 1
    # filter_size = 5
    # num_features = 128
    # num_layers = 2
    # output_channels = 64
    # rnn = ResCLSTM(input_shape, input_channels, output_channels)
    # rnn.init_hidden(batch_size)
    # input = torch.randn(sample_seqlen, batch_size, input_channels, input_shape[0], input_shape[1]) # TxBxCxHxW
    # # hx = torch.randn(batch_size, hidden_size)
    # # cx = torch.randn(batch_size, hidden_size)
    # output = rnn(input)
    # # for i in xrange(num_layers):
    # #     hidden_h, hidden_c = output[0][i]
    # #     print('hidden_h.size():', hidden_h.size())
    # #     print('hidden_c.size():', hidden_c.size())
    # # --------------MCLSTMCell输入为2维时序图像----------------

