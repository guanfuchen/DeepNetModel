# -*- coding: utf-8 -*-
import argparse

import numpy as np
import time
import torch
import visdom
import os

from torch import optim, nn
from torch.nn import Conv2d

from lstm_pytorch import MLSTMCell, MovingMNISTLoader, crossentropyloss, MCLSTMCell, MResCLSTMCell


class OneStepMResCLSTMCell(nn.Module):
    """
    Multiple Convolution LSTMCell like encoder
    """
    def __init__(self, input_shape, input_channels):
        super(OneStepMResCLSTMCell, self).__init__()
        self.input_shape = input_shape
        self.input_channels = input_channels
        self.s2o_filter_size = 1

        self.mclstm_1_num_layers = 2
        self.mclstm_1_num_features_list = [128, 64]
        self.mclstm_1_s2s_filter_size_list = [5, 5]
        self.mclstm_1 = MResCLSTMCell(self.input_shape, self.input_channels, i2s_filter_size=5, s2s_filter_size_list=self.mclstm_1_s2s_filter_size_list, num_features_list=self.mclstm_1_num_features_list, num_layers=self.mclstm_1_num_layers, return_sequences=False, dilation=1)
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
        # print('len(out_mclstm_1_out):', len(out_mclstm_1_out))
        out_concat = None
        for idlayer in xrange(self.mclstm_1_num_layers):
            hidden_h_mclstm_1_idlayer_hidden_h = out_mclstm_1_out[idlayer][0][1] # Bx(num_features_id)xHxW
            # print('len(hidden_h_mclstm_1_idlayer_hidden_h):', len(hidden_h_mclstm_1_idlayer_hidden_h))
            # print('hidden_h_mclstm_1_idlayer_hidden_h.shape:', hidden_h_mclstm_1_idlayer_hidden_h.shape)
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
    # # s2s_filter_size_list = [5, 5]
    # # num_features_list = [64, 64]
    # s2s_filter_size_list = [3, 3]
    # num_features_list = [16, 16]
    # num_layers = len(s2s_filter_size_list)
    # assert len(s2s_filter_size_list)==num_layers
    # assert len(num_features_list)==num_layers

    # model = MResCLSTMCell(input_shape, input_channels, i2s_filter_size, s2s_filter_size_list, num_features_list, num_layers)
    model = OneStepMResCLSTMCell(input_shape, input_channels)
    # model = ResCLSTM(input_shape, input_channels, 1)
    if use_cuda:
        model.cuda()

    local_path = os.path.expanduser('~/Data/mnist_test_seq.npy')
    train_dst = MovingMNISTLoader(local_path, split='train')
    train_loader = torch.utils.data.DataLoader(train_dst, batch_size=batch_size, shuffle=True)
    val_dst = MovingMNISTLoader(local_path, split='val')
    val_loader = torch.utils.data.DataLoader(val_dst, batch_size=batch_size, shuffle=True)
    optimizer = optim.RMSprop(model.parameters(), lr=0.0002, weight_decay=0.9)
    # optimizer = optim.Adam(model.parameters(), lr=0.0002, weight_decay=0.9)

    init_states = None
    if init_states is None:
        init_states = model.init_hidden(batch_size, use_cuda)

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
            # print('last_hidden.shape:', last_hidden.shape)

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
            torch.save(model.state_dict(), 'experiment_{}_epoch_{}.pt'.format('resconlvstmonestep', epoch))

    loss_iteration_save_fp.close()
    # --------------2层CLSTMCell实验相关----------------

