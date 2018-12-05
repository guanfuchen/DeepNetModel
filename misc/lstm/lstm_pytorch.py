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

    def init_hidden(self, batch_size):
        return (Variable(torch.zeros(batch_size, self.hidden_size)), Variable(torch.zeros(batch_size, self.hidden_size)))


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
        self.input_shape = input_shape
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
        if use_cuda:
            return (Variable(torch.zeros(batch_size, self.num_features, self.input_shape[0], self.input_shape[1])).cuda(), Variable(torch.zeros(batch_size, self.num_features, self.input_shape[0], self.input_shape[1])).cuda())
        else:
            return (Variable(torch.zeros(batch_size, self.num_features, self.input_shape[0], self.input_shape[1])), Variable(torch.zeros(batch_size, self.num_features, self.input_shape[0], self.input_shape[1])))

class MCLSTMCell(nn.Module):
    """
    Multiple Convolution LSTMCell like encoder
    """
    def __init__(self, input_shape, input_channels, filter_size, num_features, num_layers, return_sequences=False):
        super(MCLSTMCell, self).__init__()
        self.input_shape = input_shape
        self.input_channels = input_channels
        self.filter_size = filter_size
        self.num_features = num_features
        self.num_layers = num_layers
        self.return_sequences = return_sequences

        cell_list = []
        cell_list.append(CLSTMCell(self.input_shape, self.input_channels, self.filter_size, self.num_features))

        self.return_sequences_conv = Conv2d(in_channels=self.num_features, out_channels=input_channels, kernel_size=self.filter_size, padding=(self.filter_size - 1)/2)

        for idcell in xrange(1, self.num_layers):
            cell_list.append(CLSTMCell(self.input_shape, self.num_features, self.filter_size, self.num_features))

        self.cell_list = nn.ModuleList(cell_list)


    def forward(self, input, hidden_state):
        current_input = input # input shape: TxBxCxHxW
        next_hidden = []
        seqlen = current_input.size(0)
        for idlayer in xrange(self.num_layers):
            hidden_h, hidden_c = hidden_state[idlayer]
            outputs = []
            outputs_seq = []
            for t in xrange(seqlen):
                current_layer = self.cell_list[idlayer]
                current_input_t = current_input[t, :, :, :, :]
                hidden_h, hidden_c = current_layer(current_input_t, (hidden_h, hidden_c))
                if self.return_sequences:
                    outputs_seq.append(self.return_sequences_conv(hidden_h))
                outputs.append(hidden_h)

            next_hidden.append((hidden_h, hidden_c))
            current_input = torch.cat(outputs, 0).view(seqlen, *outputs[0].size()) # input shape: TxBx(num_features)xHxW

        if self.return_sequences:
            # output sequences channel == input sequences channle
            current_input = torch.cat(outputs_seq, 0).view(seqlen, *outputs_seq[0].size()) # input shape: TxBx(input_shape)xHxW
            # current_input = current_input.transpose(0, 1)

        return next_hidden, current_input # last current_input is the output of the MCLSTMCell unit

    def init_hidden(self, batch_size):
        init_states=[]
        for i in xrange(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size))
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

        self.return_sequences_linear = nn.Linear(in_features=self.num_features, out_features=self.input_shape[0]*self.input_shape[1]*self.input_channels)

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
        for idlayer in xrange(self.num_layers):
            hidden_h, hidden_c = hidden_state[idlayer]
            outputs = []
            outputs_seq = []
            for t in xrange(seqlen):
                current_layer = self.cell_list[idlayer]
                current_input_t = current_input[t, :, :]
                hidden_h, hidden_c = current_layer(current_input_t, (hidden_h, hidden_c))
                if self.return_sequences:
                    outputs_seq.append(self.return_sequences_linear(hidden_h))
                outputs.append(hidden_h)

            next_hidden.append((hidden_h, hidden_c))
            current_input = torch.cat(outputs, 0).view(seqlen, *outputs[0].size()) # input shape: TxBx(num_features)xHxW

        if self.return_sequences:
            # output sequences channel == input sequences channle
            current_input = torch.cat(outputs_seq, 0).view(seqlen, *outputs_seq[0].size()) # input shape: TxBx(input_shape)xHxW
            # current_input = current_input.transpose(0, 1)
            current_input = current_input.view(seqlen, batch_size, self.input_channels, self.input_shape[0], self.input_shape[1])

        return next_hidden, current_input # last current_input is the output of the MCLSTMCell unit

    def init_hidden(self, batch_size):
        init_states=[]
        for i in xrange(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size))
        return init_states


class MovingMNISTLoader(Dataset):
    def __init__(self, root, split="train", split_ratio=0.9):
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

class MCLSTMPredNet(nn.Module):
    """
    Multiple Convolution LSTMCell Prediction Net
    """
    def __init__(self, input_shape, input_channels, filter_size, num_features, num_layers):
        super(MCLSTMPredNet, self).__init__()
        mclstmcell = MCLSTMCell(input_shape, input_channels, filter_size, num_features, num_layers)

    def forward(self, input):
        pass

if __name__ == '__main__':
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

    # --------------2层CLSTMCell实验相关----------------
    batch_size = 1
    input_shape = (64, 64) # H, W
    input_channels = 1
    filter_size = 3
    num_features = 16
    num_layers = 2
    model = MCLSTMCell(input_shape, input_channels, filter_size, num_features, num_layers, return_sequences=True)
    if use_cuda:
        model.cuda()

    local_path = os.path.expanduser('~/Data/mnist_test_seq.npy')
    train_dst = MovingMNISTLoader(local_path, split='train')
    train_loader = torch.utils.data.DataLoader(train_dst, batch_size=batch_size, shuffle=True)

    optimizer = optim.RMSprop(model.parameters(), lr=0.0002)

    init_states = None
    if init_states is None:
        init_states = model.init_hidden(batch_size)

    if use_visdom:
        vis = visdom.Visdom()
        vis.close()

    init_time = str(int(time.time()))
    loss_iteration_save_file = '/tmp/loss_iteration_{}.txt'.format(init_time)
    loss_iteration_save_fp = open(loss_iteration_save_file, 'wb')

    data_count = int(train_dst.__len__() * 1.0 / batch_size)
    for epoch in range(1, 1000, 1):
        loss_epoch = 0
        for i, train_data in enumerate(train_loader):
            imgs = train_data[:, 0:10, ...]
            labels = train_data[:, 10:20, ...]
            if use_cuda:
                imgs = imgs.cuda()
                labels = labels.cuda()
            imgs_transpose = imgs.transpose(0, 1)
            # print('imgs_transpose.shape:', imgs_transpose.shape)
            # print('labels.shape:', labels.shape)
            outputs = model(imgs_transpose, init_states)
            # hidden_h, hidden_c = outputs[0][num_layers-1]
            # print('hidden_h.shape:', hidden_h.shape)
            # print('hidden_c.shape:', hidden_c.shape)
            last_hidden = outputs[1]
            # print('last_hidden.shape:', last_hidden.shape)

            optimizer.zero_grad()

            loss = 0
            for seq in range(10):
                predframe = torch.sigmoid(last_hidden[seq].view(batch_size, -1))
                labelframe = labels[:, seq, ...].view(batch_size, -1)
                # print('predframe.shape:', predframe.shape)
                # print('labelframe.shape:', labelframe.shape)
                loss += crossentropyloss(predframe, labelframe)

            loss.backward()
            optimizer.step()

            loss_np = loss.cpu().data.numpy() * 1.0 / batch_size
            print "loss:", loss_np
            loss_epoch += loss_np
            loss_iteration_save_fp.write(str(loss_np))

            if use_visdom:
                win = 'loss_iteration'
                loss_np_expand = np.expand_dims(loss_np, axis=0)
                win_res = vis.line(X=np.ones(1) * (i + data_count * (epoch - 1) + 1), Y=loss_np_expand, win=win, update='append')
                if win_res != win:
                    vis.line(X=np.ones(1) * (i + data_count * (epoch - 1) + 1), Y=loss_np_expand, win=win, opts=dict(title=win, xlabel='iteration', ylabel='loss'))
            # break

        loss_avg_epoch = loss_epoch / (data_count * 1.0)
        if use_visdom:
            win = 'loss_epoch'
            loss_avg_epoch_expand = np.expand_dims(loss_avg_epoch, axis=0)
            win_res = vis.line(X=np.ones(1)*epoch, Y=loss_avg_epoch_expand, win=win, update='append')
            if win_res != win:
                vis.line(X=np.ones(1)*epoch, Y=loss_avg_epoch_expand, win=win, opts=dict(title=win, xlabel='epoch', ylabel='loss'))

    loss_iteration_save_fp.close()
    # --------------2层CLSTMCell实验相关----------------


    # # --------------2层LSTMCell实验相关----------------
    # batch_size = 20
    # input_shape = (64, 64) # H, W
    # input_channels = 1
    # num_layers = 2
    # num_features = 2048
    # model = MLSTMCell(input_shape=input_shape, input_channels=input_channels, num_features=num_features, num_layers=num_layers, return_sequences=True)
    # if use_cuda:
    #     model.cuda()
    #
    # local_path = os.path.expanduser('~/Data/mnist_test_seq.npy')
    # train_dst = MovingMNISTLoader(local_path, split='train')
    # train_loader = torch.utils.data.DataLoader(train_dst, batch_size=batch_size, shuffle=True)
    # val_dst = MovingMNISTLoader(local_path, split='val')
    # val_loader = torch.utils.data.DataLoader(val_dst, batch_size=batch_size, shuffle=True)
    #
    # optimizer = optim.RMSprop(model.parameters(), lr=0.0002)
    #
    # init_states = None
    # if init_states is None:
    #     init_states = model.init_hidden(batch_size)
    #
    #
    # val_init_states = None
    # if val_init_states is None:
    #     val_init_states = model.init_hidden(batch_size)
    #
    # if use_visdom:
    #     vis = visdom.Visdom()
    #     vis.close()
    #
    # init_time = str(int(time.time()))
    # loss_iteration_save_file = '/tmp/loss_iteration_{}.txt'.format(init_time)
    # loss_iteration_save_fp = open(loss_iteration_save_file, 'wb')
    #
    # data_count = int(train_dst.__len__() * 1.0 / batch_size)
    # val_data_count = int(val_dst.__len__() * 1.0 / batch_size)
    # for epoch in range(1, 1000, 1):
    #     loss_epoch = 0
    #     for i, train_data in enumerate(train_loader):
    #         model.train()
    #
    #         imgs = train_data[:, 0:10, ...]
    #         labels = train_data[:, 10:20, ...]
    #         if use_cuda:
    #             imgs = imgs.cuda()
    #             labels = labels.cuda()
    #         imgs_transpose = imgs.transpose(0, 1)
    #         # print('imgs_transpose.shape:', imgs_transpose.shape)
    #         # print('labels.shape:', labels.shape)
    #         outputs = model(imgs_transpose, init_states)
    #         # hidden_h, hidden_c = outputs[0][num_layers-1]
    #         # print('hidden_h.shape:', hidden_h.shape)
    #         # print('hidden_c.shape:', hidden_c.shape)
    #         last_hidden = outputs[1]
    #         # print('last_hidden.shape:', last_hidden.shape)
    #
    #         optimizer.zero_grad()
    #
    #         loss = 0
    #         for seq in range(10):
    #             predframe = torch.sigmoid(last_hidden[seq].view(batch_size, -1))
    #             labelframe = labels[:, seq, ...].view(batch_size, -1)
    #             # print('predframe.shape:', predframe.shape)
    #             # print('labelframe.shape:', labelframe.shape)
    #             loss += crossentropyloss(predframe, labelframe)
    #
    #         loss.backward()
    #         optimizer.step()
    #
    #         loss_np = loss.cpu().data.numpy() * 1.0 / batch_size
    #         # print "loss:", loss_np
    #         loss_epoch += loss_np
    #         loss_iteration_save_fp.write(str(loss_np))
    #
    #         if use_visdom:
    #             win = 'loss_iteration'
    #             loss_np_expand = np.expand_dims(loss_np, axis=0)
    #             win_res = vis.line(X=np.ones(1) * (i + data_count * (epoch - 1) + 1), Y=loss_np_expand, win=win, update='append')
    #             if win_res != win:
    #                 vis.line(X=np.ones(1) * (i + data_count * (epoch - 1) + 1), Y=loss_np_expand, win=win, opts=dict(title=win, xlabel='iteration', ylabel='loss'))
    #
    #
    #         val_loss_epoch = 0
    #         for val_i, val_data in enumerate(val_loader):
    #             model.eval()
    #
    #             val_imgs = val_data[:, 0:10, ...]
    #             val_labels = val_data[:, 10:20, ...]
    #             if use_cuda:
    #                 val_imgs = val_imgs.cuda()
    #                 val_labels = val_labels.cuda()
    #             val_imgs_transpose = val_imgs.transpose(0, 1)
    #             val_outputs = model(val_imgs_transpose, val_init_states)
    #             val_last_hidden = val_outputs[1]
    #
    #             val_loss = 0
    #             for val_seq in range(10):
    #                 val_predframe = torch.sigmoid(val_last_hidden[val_seq].view(batch_size, -1))
    #                 val_labelframe = val_labels[:, val_seq, ...].view(batch_size, -1)
    #                 # print('predframe.shape:', predframe.shape)
    #                 # print('labelframe.shape:', labelframe.shape)
    #                 val_loss += crossentropyloss(val_predframe, val_labelframe)
    #
    #
    #             val_loss_np = val_loss.cpu().data.numpy() * 1.0 / batch_size
    #             # print "val_loss_np:", val_loss_np
    #             val_loss_epoch += val_loss_np
    #         val_loss_avg_epoch = val_loss_epoch / (val_data_count * 1.0)
    #         print "val_loss_avg_epoch:", val_loss_avg_epoch
    #         # break
    #
    #     loss_avg_epoch = loss_epoch / (data_count * 1.0)
    #     if use_visdom:
    #         win = 'loss_epoch'
    #         loss_avg_epoch_expand = np.expand_dims(loss_avg_epoch, axis=0)
    #         win_res = vis.line(X=np.ones(1)*epoch, Y=loss_avg_epoch_expand, win=win, update='append')
    #         if win_res != win:
    #             vis.line(X=np.ones(1)*epoch, Y=loss_avg_epoch_expand, win=win, opts=dict(title=win, xlabel='epoch', ylabel='loss'))
    #
    # loss_iteration_save_fp.close()
    # # --------------2层LSTMCell实验相关----------------
