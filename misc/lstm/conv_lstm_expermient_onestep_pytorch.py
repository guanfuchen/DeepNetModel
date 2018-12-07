# -*- coding: utf-8 -*-
import argparse

import numpy as np
import time
import torch
import visdom
import os

from torch import optim, nn
from torch.nn import Conv2d
from torch.autograd import Variable
import torch.nn.functional as F

from lstm_pytorch import MLSTMCell, MovingMNISTLoader, crossentropyloss, MCLSTMCell, CLSTMCell


class OneStepMCLSTMCell(nn.Module):
    """
    Multiple Convolution LSTMCell like encoder
    """
    def __init__(self, input_shape, input_channels):
        super(OneStepMCLSTMCell, self).__init__()
        self.input_shape = input_shape
        self.input_channels = input_channels
        self.s2o_filter_size = 1

        self.mclstm_1_num_layers = 3
        self.mclstm_1_num_features_list = [128, 64, 64]
        self.mclstm_1_s2s_filter_size_list = [5, 5, 5]
        self.mclstm_1 = MCLSTMCell(self.input_shape, self.input_channels, i2s_filter_size=5, s2s_filter_size_list=self.mclstm_1_s2s_filter_size_list, num_features_list=self.mclstm_1_num_features_list, num_layers=self.mclstm_1_num_layers, return_sequences=False, dilation=1)
        # append all hidden_states in previous layers
        self.return_sequences_conv = Conv2d(in_channels=sum(self.mclstm_1_num_features_list), out_channels=input_channels, kernel_size=self.s2o_filter_size, padding=(self.s2o_filter_size - 1)/2)


    def forward(self, input, hidden_state):
        """
        hidden_state for every lstm init
        :param input: shape TxBxCxHxW
        :param hidden_state: list for MCLSTMCell
        :return:
        """
        # seqlen = input.size(0)
        # batch_size = input.size(1)
        out_mclstm_1 = self.mclstm_1(input, hidden_state[0]) # mclstm out [[[hidden_h, hidden_c], ... ,[hidden_h, hidden_c]], last_out_channel_hidden2input]

        out_mclstm_1_out = out_mclstm_1[0] # TxBx(num_features)xHxW
        out_concat = None
        for idlayer in xrange(self.mclstm_1_num_layers):
            hidden_h_mclstm_1_idlayer_hidden_h = out_mclstm_1_out[idlayer][0] # Bx(num_features_id)xHxW
            # print('hidden_h_mclstm_1_idlayer_hidden_h.shape:', hidden_h_mclstm_1_idlayer_hidden_h.shape)
            if out_concat is None:
                out_concat = hidden_h_mclstm_1_idlayer_hidden_h
            else:
                out_concat = torch.cat((out_concat, hidden_h_mclstm_1_idlayer_hidden_h), 1)
            # print('out_concat.shape:', out_concat.shape)

        # print('out_concat.shape:', out_concat.shape)
        out = self.return_sequences_conv(out_concat)
        # print('out.shape:', out.shape)
        # for t in xrange(seqlen):
        #     out_mclstm_1_out_t = out_mclstm_1_out[t, ...]
        #     out_mclstm_1_out_t = self.return_sequences_conv
        return out

    def init_hidden(self, batch_size, use_cuda=False):
        return [self.mclstm_1.init_hidden(batch_size, use_cuda)]


class OneStepMCLSTMCell_2(nn.Module):
    """
    Multiple Convolution LSTMCell like encoder
    """
    def __init__(self, input_shape, input_channels):
        super(OneStepMCLSTMCell_2, self).__init__()
        self.input_shape = input_shape
        self.input_channels = input_channels
        self.s2o_filter_size = 1

        self.mclstm_1_num_layers = 3
        self.mclstm_1_num_features_list = [128, 64, 64]
        self.mclstm_1_s2s_filter_size_list = [5, 5, 5]
        self.mclstm_1 = MCLSTMCell(self.input_shape, self.input_channels, i2s_filter_size=5, s2s_filter_size_list=self.mclstm_1_s2s_filter_size_list, num_features_list=self.mclstm_1_num_features_list, num_layers=self.mclstm_1_num_layers, return_sequences=True)
        # append all hidden_states in previous layers



    def forward(self, input, hidden_state):
        """
        hidden_state for every lstm init
        :param input: shape TxBxCxHxW
        :param hidden_state: list for MCLSTMCell
        :return:
        """
        # seqlen = input.size(0)
        # batch_size = input.size(1)
        out_mclstm_1 = self.mclstm_1(input, hidden_state[0]) # mclstm out [[[hidden_h, hidden_c], ... ,[hidden_h, hidden_c]], last_out_channel_hidden2input]
        out = out_mclstm_1[1]
        return out

    def init_hidden(self, batch_size, use_cuda=False):
        return [self.mclstm_1.init_hidden(batch_size, use_cuda)]

def downsample(inplanes, out_channels, kernel_size, stride, padding, bias):
    layers = []
    layers += [nn.Conv2d(inplanes, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)]
    layers += [nn.BatchNorm2d(out_channels)]
    layers += [nn.LeakyReLU(negative_slope=0.2)]
    return nn.Sequential(*layers)

class OneStepEncoderDecoer(nn.Module):
    """
    Multiple Convolution LSTMCell like encoder
    """
    def __init__(self, input_shape, input_channels):
        super(OneStepEncoderDecoer, self).__init__()
        self.input_shape = input_shape
        self.input_channels = input_channels

        self.out_filter_size = 1

        self.num_features_list = [16, 16, 16]
        self.filter_size_list = [3, 3, 3]

        self.clstmcell_1_1 = CLSTMCell(input_shape, input_channels, filter_size=self.filter_size_list[0], num_features=self.num_features_list[0])
        self.clstmcell_1_1_hc = None
        self.clstmcell_1_2 = CLSTMCell(input_shape, self.num_features_list[0], filter_size=self.filter_size_list[0], num_features=self.num_features_list[0])
        self.clstmcell_1_2_hc = None
        self.clstmcell_1_3 = CLSTMCell(input_shape, self.num_features_list[0], filter_size=self.filter_size_list[0], num_features=self.num_features_list[0])
        self.clstmcell_1_3_hc = None

        self.downsample_1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        input_shape = (input_shape[0]//2, input_shape[1]//2)

        self.clstmcell_2_1 = CLSTMCell(input_shape, self.num_features_list[0], filter_size=self.filter_size_list[1], num_features=self.num_features_list[1])
        self.clstmcell_2_1_hc = None
        self.clstmcell_2_2 = CLSTMCell(input_shape, self.num_features_list[1], filter_size=self.filter_size_list[1], num_features=self.num_features_list[1])
        self.clstmcell_2_2_hc = None
        self.clstmcell_2_3 = CLSTMCell(input_shape, self.num_features_list[1], filter_size=self.filter_size_list[1], num_features=self.num_features_list[1])
        self.clstmcell_2_3_hc = None

        self.downsample_2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        input_shape = (input_shape[0]//2, input_shape[1]//2)

        self.clstmcell_3_1 = CLSTMCell(input_shape, self.num_features_list[1], filter_size=self.filter_size_list[2], num_features=self.num_features_list[2])
        self.clstmcell_3_1_hc = None
        self.clstmcell_3_2 = CLSTMCell(input_shape, self.num_features_list[2], filter_size=self.filter_size_list[2], num_features=self.num_features_list[2])
        self.clstmcell_3_2_hc = None
        self.clstmcell_3_3 = CLSTMCell(input_shape, self.num_features_list[2], filter_size=self.filter_size_list[2], num_features=self.num_features_list[2])
        self.clstmcell_3_3_hc = None

        self.out_conv = Conv2d(in_channels=self.num_features_list[2], out_channels=input_channels, kernel_size=self.out_filter_size, padding=(self.out_filter_size - 1)/2)
        # self.out_conv = Conv2d(in_channels=10*self.num_features_list[2], out_channels=input_channels, kernel_size=self.out_filter_size, padding=(self.out_filter_size - 1)/2)

        self.upsample_1 = nn.UpsamplingBilinear2d(scale_factor=4)


    def forward(self, input, hidden_state):
        """
        hidden_state for every lstm init
        :param input: shape TxBxCxHxW
        :param hidden_state: list for MCLSTMCell
        :return:
        """
        seqlen = input.size(0)
        # batch_size = input.size(1)
        self.clstmcell_1_1_hc = hidden_state[0]
        self.clstmcell_1_2_hc = hidden_state[1]
        self.clstmcell_1_3_hc = hidden_state[2]

        self.clstmcell_2_1_hc = hidden_state[3]
        self.clstmcell_2_2_hc = hidden_state[4]
        self.clstmcell_2_3_hc = hidden_state[5]

        self.clstmcell_3_1_hc = hidden_state[6]
        self.clstmcell_3_2_hc = hidden_state[7]
        self.clstmcell_3_3_hc = hidden_state[8]

        out_concat = None

        for t in range(seqlen):
            input_t = input[t, ...]
            self.clstmcell_1_1_hc = self.clstmcell_1_1(input_t, self.clstmcell_1_1_hc)
            self.clstmcell_1_2_hc = self.clstmcell_1_2(self.clstmcell_1_1_hc[0], self.clstmcell_1_2_hc)
            self.clstmcell_1_3_hc = self.clstmcell_1_3(self.clstmcell_1_2_hc[0], self.clstmcell_1_3_hc)

            clstmcell_1_3_h = self.clstmcell_1_3_hc[0]
            clstmcell_1_3_h = self.downsample_1(clstmcell_1_3_h)

            # print('clstmcell_1_3_h.shape:', clstmcell_1_3_h.shape)

            self.clstmcell_2_1_hc = self.clstmcell_2_1(clstmcell_1_3_h, self.clstmcell_2_1_hc)
            self.clstmcell_2_2_hc = self.clstmcell_2_2(self.clstmcell_2_1_hc[0], self.clstmcell_2_2_hc)
            self.clstmcell_2_3_hc = self.clstmcell_2_3(self.clstmcell_2_2_hc[0], self.clstmcell_2_3_hc)

            clstmcell_2_3_h = self.clstmcell_2_3_hc[0]
            clstmcell_2_3_h = self.downsample_2(clstmcell_2_3_h)

            # print('clstmcell_2_3_h.shape:', clstmcell_2_3_h.shape)

            self.clstmcell_3_1_hc = self.clstmcell_3_1(clstmcell_2_3_h, self.clstmcell_3_1_hc)
            self.clstmcell_3_2_hc = self.clstmcell_3_2(self.clstmcell_3_1_hc[0], self.clstmcell_3_2_hc)
            self.clstmcell_3_3_hc = self.clstmcell_3_3(self.clstmcell_3_2_hc[0], self.clstmcell_3_3_hc)

            out = self.clstmcell_3_3_hc[0]
            # print('out.shape:', out.shape)
            # if out_concat is None:
            #     out_concat = out
            # else:
            #     out_concat = torch.cat((out_concat, out), 1)
            # print('out_concat.shape:', out_concat.shape)

        # print('out_concat.shape:', out_concat.shape)
        # out = self.out_conv(out_concat)
        out = self.out_conv(out)
        out = self.upsample_1(out)
        return out

    def init_hidden(self, batch_size, use_cuda=False):
        return [
            self.clstmcell_1_1.init_hidden(batch_size, use_cuda),
            self.clstmcell_1_2.init_hidden(batch_size, use_cuda),
            self.clstmcell_1_2.init_hidden(batch_size, use_cuda),

            self.clstmcell_2_1.init_hidden(batch_size, use_cuda),
            self.clstmcell_2_2.init_hidden(batch_size, use_cuda),
            self.clstmcell_2_3.init_hidden(batch_size, use_cuda),

            self.clstmcell_3_1.init_hidden(batch_size, use_cuda),
            self.clstmcell_3_2.init_hidden(batch_size, use_cuda),
            self.clstmcell_3_3.init_hidden(batch_size, use_cuda),
        ]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='training parameter setting')
    parser.add_argument('--use_cuda', type=bool, default=False, help='use cuda [ False ]')
    parser.add_argument('--use_visdom', type=bool, default=True, help='use visdom [ False ]')
    parser.add_argument('--save_epoch', type=int, default=-1, help='save model after epoch [ 1 ]')
    args = parser.parse_args()

    use_cuda = args.use_cuda
    use_visdom = args.use_visdom


    # --------------2层CLSTMCell实验相关----------------
    batch_size = 1
    input_shape = (64, 64) # H, W
    input_channels = 1
    # i2s_filter_size = 5
    # # s2s_filter_size_list = [5]
    # # num_features_list = [256]
    # s2s_filter_size_list = [5, 5, 5]
    # num_features_list = [128, 64, 64]
    # pred_len = 1
    # # s2s_filter_size_list = [3, 3, 3, 3, 3, 3]
    # # num_features_list = [64, 64, 128, 128, 64, 64]
    # # s2s_filter_size_list = [3, 3, 3, 3, 3]
    # # num_features_list = [32, 32, 64, 64, 128]
    # num_layers = len(num_features_list)
    # assert len(s2s_filter_size_list)==num_layers
    # assert len(num_features_list)==num_layers

    # model = MCLSTMCell(input_shape, input_channels, i2s_filter_size, s2s_filter_size_list, num_features_list, num_layers, return_sequences=True)
    model = OneStepMCLSTMCell(input_shape, input_channels)
    # model = OneStepMCLSTMCell_2(input_shape, input_channels)
    # model = OneStepEncoderDecoer(input_shape, input_channels)
    if use_cuda:
        model.cuda()

    local_path = os.path.expanduser('~/Data/mnist_test_seq.npy')
    train_dst = MovingMNISTLoader(local_path, split='train')
    train_loader = torch.utils.data.DataLoader(train_dst, batch_size=batch_size, shuffle=True)
    val_dst = MovingMNISTLoader(local_path, split='val')
    val_loader = torch.utils.data.DataLoader(val_dst, batch_size=batch_size, shuffle=True)

    optimizer = optim.RMSprop(model.parameters(), lr=0.0002, weight_decay=0.9)

    # init_states = None
    # if init_states is None:
    #     init_states = model.init_hidden(batch_size, use_cuda)

    if use_visdom:
        vis = visdom.Visdom()
        vis.close()

    init_time = str(int(time.time()))
    loss_iteration_save_file = '/tmp/loss_iteration_onestep_{}.txt'.format(init_time)
    loss_iteration_save_fp = open(loss_iteration_save_file, 'wb')

    data_count = int(train_dst.__len__() * 1.0 / batch_size)
    val_data_count = int(val_dst.__len__() * 1.0 / batch_size)
    for epoch in range(1, 1000, 1):
        loss_epoch = 0
        for i, train_data in enumerate(train_loader):
            model.train()

            imgs = train_data[:, 0:10, ...]
            labels = train_data[:, 10:20, ...]
            if use_cuda:
                imgs = imgs.cuda()
                labels = labels.cuda()
            imgs_transpose = imgs.transpose(0, 1)
            # print('imgs_transpose.shape:', imgs_transpose.shape)
            # print('labels.shape:', labels.shape)
            init_states = model.init_hidden(batch_size, use_cuda)
            outputs = model(imgs_transpose, init_states)
            # hidden_h, hidden_c = outputs[0][num_layers-1]
            # print('hidden_h.shape:', hidden_h.shape)
            # print('hidden_c.shape:', hidden_c.shape)
            last_hidden = outputs
            print('last_hidden.shape:', last_hidden.shape)

            optimizer.zero_grad()

            # ----------仅仅预测下一帧-------------
            predframe = torch.sigmoid(last_hidden.view(batch_size, -1))
            labelframe = labels[:, 0, ...].view(batch_size, -1)
            loss = crossentropyloss(predframe, labelframe)
            # ----------仅仅预测下一帧-------------

            loss.backward()
            optimizer.step()

            loss_np = loss.cpu().data.numpy() * 1.0 / batch_size
            print "loss:", loss_np
            loss_epoch += loss_np
            loss_iteration_save_fp.write(str(loss_np)+'\n')

            if use_visdom and i%10==0:
                win = 'Pred'
                pred_img = torch.sigmoid(last_hidden).cpu().data.numpy()[-1, 0, ...]
                pred_img[pred_img>0.5] = 1
                pred_img[pred_img<0.5] = 0
                vis.image(pred_img, win=win, opts=dict(title='Pred'))

                win = 'Pred_grey'
                pred_img = torch.sigmoid(last_hidden).cpu().data.numpy()[-1, 0, ...]
                vis.image(pred_img, win=win, opts=dict(title='Pred_grey'))

                win = 'GT'
                gt_img = labels.cpu().data.numpy()[0, 0, ...]
                vis.image(gt_img, win=win, opts=dict(title='GT'))

            if use_visdom:
                win = 'loss_iteration'
                loss_np_expand = np.expand_dims(loss_np, axis=0)
                win_res = vis.line(X=np.ones(1) * (i + data_count * (epoch - 1) + 1), Y=loss_np_expand, win=win, update='append')
                if win_res != win:
                    vis.line(X=np.ones(1) * (i + data_count * (epoch - 1) + 1), Y=loss_np_expand, win=win, opts=dict(title=win, xlabel='iteration', ylabel='loss'))

            # val_interval = 5000
            # if i%val_interval==0 and i!=0:
            #     val_loss_epoch = 0
            #     for val_i, val_data in enumerate(val_loader):
            #         model.eval()
            #
            #         val_imgs = val_data[:, 0:10, ...]
            #         val_labels = val_data[:, 10:20, ...]
            #         if use_cuda:
            #             val_imgs = val_imgs.cuda()
            #             val_labels = val_labels.cuda()
            #         val_imgs_transpose = val_imgs.transpose(0, 1)
            #         val_init_states = model.init_hidden(batch_size, use_cuda)
            #         val_outputs = model(val_imgs_transpose, val_init_states)
            #
            #         val_last_hidden = val_outputs
            #
            #         # ----------仅仅预测下一帧-------------
            #         val_predframe = torch.sigmoid(val_last_hidden[-1].view(batch_size, -1))
            #         val_labelframe = val_labels[:, 0, ...].view(batch_size, -1)
            #         val_loss = crossentropyloss(val_predframe, val_labelframe)
            #         # ----------仅仅预测下一帧-------------
            #
            #         val_loss_np = val_loss.cpu().data.numpy() * 1.0 / batch_size
            #         # print "val_loss_np:", val_loss_np
            #         val_loss_epoch += val_loss_np
            #     val_loss_avg_epoch = val_loss_epoch / (val_data_count * 1.0)
            #     print "val_loss_avg_epoch:", val_loss_avg_epoch

        loss_avg_epoch = loss_epoch / (data_count * 1.0)
        if use_visdom:
            win = 'loss_epoch'
            loss_avg_epoch_expand = np.expand_dims(loss_avg_epoch, axis=0)
            win_res = vis.line(X=np.ones(1)*epoch, Y=loss_avg_epoch_expand, win=win, update='append')
            if win_res != win:
                vis.line(X=np.ones(1)*epoch, Y=loss_avg_epoch_expand, win=win, opts=dict(title=win, xlabel='epoch', ylabel='loss'))

        if args.save_epoch > 0 and epoch%args.save_epoch==0 and epoch != 0:
            torch.save(model.state_dict(), 'experiment_{}_epoch_{}.pt'.format('conlvstmonestep', epoch))

    loss_iteration_save_fp.close()
    # --------------2层CLSTMCell实验相关----------------
