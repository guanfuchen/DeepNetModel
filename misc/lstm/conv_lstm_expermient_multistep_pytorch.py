# -*- coding: utf-8 -*-
import argparse

import numpy as np
import time
import torch
import visdom
import os

from torch import optim, nn
from torch.nn import Conv2d
import torch.nn.functional as F

from lstm_pytorch import MLSTMCell, MovingMNISTLoader, crossentropyloss, MCLSTMCell, \
    torch_var_tbchw2bcthw, torch_var_bcthw2tbchw


class MultiStepMCLSTMCell(nn.Module):
    """
    Multiple Convolution LSTMCell like encoder
    """
    def __init__(self, input_shape, input_channels):
        super(MultiStepMCLSTMCell, self).__init__()
        self.input_shape = input_shape
        self.input_channels = input_channels
        self.s2o_filter_size = 1


        self.conv1 = nn.Conv3d(in_channels=self.input_channels, out_channels=40, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.mclstm_1_num_layers = 3
        self.mclstm_1_num_features_list = [40, 40, 40]
        self.mclstm_1_s2s_filter_size_list = [3, 3, 3]
        self.mclstm_1 = MCLSTMCell(input_shape=self.input_shape, input_channels=40, i2s_filter_size=5, s2s_filter_size_list=self.mclstm_1_s2s_filter_size_list, num_features_list=self.mclstm_1_num_features_list, num_layers=self.mclstm_1_num_layers, return_sequences=True)
        self.bn1 = nn.BatchNorm3d(40)


        self.conv2 = nn.Conv3d(in_channels=40, out_channels=40, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.mclstm_2_num_layers = 3
        self.mclstm_2_num_features_list = [40, 40, 40]
        self.mclstm_2_s2s_filter_size_list = [3, 3, 3]
        self.mclstm_2 = MCLSTMCell(input_shape=self.input_shape, input_channels=40, i2s_filter_size=5, s2s_filter_size_list=self.mclstm_2_s2s_filter_size_list, num_features_list=self.mclstm_2_num_features_list, num_layers=self.mclstm_2_num_layers, return_sequences=True)
        self.bn2 = nn.BatchNorm3d(40)

        self.conv_inputshape = nn.Conv3d(in_channels=40, out_channels=self.input_channels, kernel_size=(1, 1, 1), stride=1, padding=(0, 0, 0))


    def forward(self, input, hidden_state):
        """
        hidden_state for every lstm init
        :param input: shape TxBxCxHxW
        :param hidden_state: list for MCLSTMCell
        :return:
        """
        # seqlen = input.size(0)
        # batch_size = input.size(1)
        out_bcthw = torch_var_tbchw2bcthw(input)
        out_bcthw = self.conv1(out_bcthw)
        out = torch_var_bcthw2tbchw(out_bcthw)
        out = self.mclstm_1(out, hidden_state[0]) # mclstm out [[[hidden_h, hidden_c], ... ,[hidden_h, hidden_c]], last_out_channel_hidden2input]
        out = out[1] # TxBx(input_channels)xHxW
        out_bcthw = torch_var_tbchw2bcthw(out)
        out_bcthw = F.relu(self.bn1(out_bcthw))

        out_bcthw = self.conv2(out_bcthw)
        out = torch_var_bcthw2tbchw(out_bcthw)
        out = self.mclstm_2(out, hidden_state[1]) # mclstm out [[[hidden_h, hidden_c], ... ,[hidden_h, hidden_c]], last_out_channel_hidden2input]
        out = out[1] # TxBx(input_channels)xHxW
        out_bcthw = torch_var_tbchw2bcthw(out)
        out_bcthw = F.relu(self.bn2(out_bcthw))

        out_bcthw = self.conv_inputshape(out_bcthw)
        out = torch_var_bcthw2tbchw(out_bcthw)
        # out = F.relu(out+input)
        return out

    def init_hidden(self, batch_size, use_cuda=False):
        return [self.mclstm_1.init_hidden(batch_size, use_cuda), self.mclstm_2.init_hidden(batch_size, use_cuda)]


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
    # model = CLSTMPredNet(input_shape, input_channels)
    model = MultiStepMCLSTMCell(input_shape, input_channels)
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
    loss_iteration_save_file = '/tmp/loss_iteration_multistep_{}.txt'.format(init_time)
    loss_iteration_save_fp = open(loss_iteration_save_file, 'wb')

    data_count = int(train_dst.__len__() * 1.0 / batch_size)
    val_data_count = int(val_dst.__len__() * 1.0 / batch_size)
    for epoch in range(1, 1000, 1):
        loss_epoch = 0
        for i, train_data in enumerate(train_loader):
            model.train()
            imgs = train_data[:, 0:10, ...] # using train_loader must return batch fist
            labels = train_data[:, 10:20, ...]
            if use_cuda:
                imgs = imgs.cuda()
                labels = labels.cuda()
            imgs_transpose = imgs.transpose(0, 1) # change BxTxCxHxW to TxBxCxHxW
            # print('imgs_transpose.shape:', imgs_transpose.shape)
            # print('labels.shape:', labels.shape)
            init_states = model.init_hidden(batch_size, use_cuda)
            outputs = model(imgs_transpose, init_states) # MCLSTM outputs list index 0 num_layers*[hidden_h, hidden_c] index 1 hidden_last_seq or using Conv2D change the hidden_last_seq to the same channel
            # hidden_h, hidden_c = outputs[0][num_layers-1]
            # print('hidden_h.shape:', hidden_h.shape)
            # print('hidden_c.shape:', hidden_c.shape)
            last_hidden = outputs
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
            loss_iteration_save_fp.write(str(loss_np)+'\n')

            # if use_visdom and i%2==0:
            #     for seq in range(10):
            #         win = 'Pred_{}'.format(seq)
            #         pred_img =last_hidden.cpu().data.numpy()[seq, 0, ...]
            #         vis.image(pred_img, win=win, opts=dict(title='Pred_{}'.format(seq), caption='Prediction Frame'))
            #         win = 'GT_{}'.format(seq)
            #         gt_img = labels.cpu().data.numpy()[0, seq, ...][0, ...]
            #         vis.image(gt_img, win=win, opts=dict(title='GT_{}'.format(seq), caption='Ground Truth'))

            if use_visdom and i%10==0:
                win = 'Pred'
                pred_img = torch.sigmoid(last_hidden).cpu().data.numpy()[:, 0, ...]
                pred_img[pred_img>0.5] = 1
                pred_img[pred_img<0.5] = 0
                vis.images(pred_img, win=win, opts=dict(title='Pred', caption='Prediction Frame', nrow=10))
                win = 'Pred_grey'
                pred_img = torch.sigmoid(last_hidden).cpu().data.numpy()[:, 0, ...]
                vis.images(pred_img, win=win, opts=dict(title='Pred_grey', caption='Prediction Frame Grey', nrow=10))
                win = 'GT'
                gt_img = labels.cpu().data.numpy()[0, ...]
                vis.images(gt_img, win=win, opts=dict(title='GT', caption='Ground Truth', nrow=10))

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

