# -*- coding: utf-8 -*-

import argparse

import numpy as np
import time
import torch
from torch import nn
import visdom
import os
import torch.nn.functional as F

from lstm_pytorch import MLSTMCell, MovingMNISTLoader, crossentropyloss

'''
Basic convolution block consisting of convolution layer, batch norm and leaky relu
'''
class BasicConv(nn.Module):
    '''
    Arguments (basic parameters of a convolution layer):
        in_channels : input planes
        out_channels : output planes
        kernel_size : size of filter to be applied
        stride : striding to be used
        padding : padding to be used
    '''
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        super(BasicConv, self).__init__()

        self.basic_block = nn.Sequential(
                            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                        stride=stride, padding=padding),
                            nn.BatchNorm2d(out_channels),
                            nn.LeakyReLU(0.2, inplace=True),
                            )
    '''
    Computes a forward pass through the basic conv block

    Arguments:
        inp : input feature

    Return:
        out : output of the forward pass applied to input feature
    '''
    def forward(self, inp):
        out = self.basic_block(inp)
        return out

'''
Basic transpose convolution block consisting of transpose convolution layer, batch norm and leaky relu
'''
class BasicConvTranspose(nn.Module):
    '''
    Arguments (basic parameters of a transpose convolution layer):
        in_channels : input planes
        out_channels : output planes
        kernel_size : size of filter to be applied
        stride : striding to be used
        padding : padding to be used
    '''
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        super(BasicConvTranspose, self).__init__()

        self.basic_block = nn.Sequential(
                            nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                        stride=stride, padding=padding),
                            nn.BatchNorm2d(out_channels),
                            nn.LeakyReLU(0.2, inplace=True),
                            )

    '''
    Computes a forward pass through the basic transpose conv block

    Arguments:
        inp : input feature

    Return:
        out : output of the forward pass applied to input feature
    '''
    def forward(self, inp):
        out = self.basic_block(inp)
        return out

class DCGANEncoder(nn.Module):
    '''
    Arguments:
        in_channels : no. of channels in the input (can be either grayscale(1) or color image(3))
        out_channels : no. of channels in the feature vector (can be either content or pose features)
        normalize : whether to normalize the feature vector
    '''
    def __init__(self, in_channels, out_channels, normalize=False):
        super(DCGANEncoder, self).__init__()

        self.normalize = normalize

        self.block1 = BasicConv(in_channels, 64, 4, 2, 1)

        self.block2 = BasicConv(64, 128, 4, 2, 1)

        self.block3 = BasicConv(128, 256, 4, 2, 1)

        self.block4 = BasicConv(256, 512, 4, 2, 1)

        self.block5 = BasicConv(512, 512, 4, 2, 1)

        self.block6 = nn.Sequential(
                        nn.Conv2d(512, out_channels, 1, 1, 0),
                        nn.BatchNorm2d(out_channels),
                        nn.Tanh(),
                        )

    '''
    Computes a forward pass through the specified architecture

    Arguments:
        inp : input (generally image in grayscale or color form)

    Returns:
        out6 : output of the forward pass applied to the input image
        [out1, out2, out3, out4, out5] : outputs at each stage which may be used
               if skip connection functionality is included in the architecture

    Note : The encoder is coded so as to return skip connections at each stage everytime.
           These skip connections can be used in the decoder stages if required.
    '''
    def forward(self, inp):
        out1 = self.block1(inp)
        out2 = self.block2(out1)
        out3 = self.block3(out2)
        out4 = self.block4(out3)
        out5 = self.block5(out4)
        # print('out5.shape:', out5.shape)
        out6 = self.block6(out5)

        if self.normalize:
            out6 = F.normalize(out6, p=2)

        return out6, [out1, out2, out3, out4, out5]


'''
Decoder based on DC-GAN architecture
'''


class DCGANDecoder(nn.Module):
    '''
    Arguments:
        in_channels : no. of channels in the input feature vector
                      (generally concatenation of content and pose features)
        out_channels : no. of channels in the output (generally the original
                       image dimension - 1 for grayscale or 3 for color)
        use_skip : whether to use the skip connection functionality
    '''

    def __init__(self, in_channels, out_channels):
        super(DCGANDecoder, self).__init__()

        # if the skip connections are used, then the input at each stage is the
        # concatenation of current feature and feature vector from the encoder
        # hence double the channels, so mul_factor (multiplication factor) is
        # used to incorporate this effect
        self.mul_factor = 1

        self.block1 = BasicConvTranspose(in_channels, 512, 1, 1, 0)

        self.block2 = BasicConvTranspose(512 * self.mul_factor, 512, 4, 2, 1)

        self.block3 = BasicConvTranspose(512 * self.mul_factor, 256, 4, 2, 1)

        self.block4 = BasicConvTranspose(256 * self.mul_factor, 128, 4, 2, 1)

        self.block5 = BasicConvTranspose(128 * self.mul_factor, 64, 4, 2, 1)

        self.block6 = nn.Sequential(
            nn.ConvTranspose2d(64 * self.mul_factor, out_channels, 4, 2, 1),
            nn.Sigmoid(),
        )

    '''
    Computes a forward pass through the specified decoder architecture

    Arguments:
        content : content feature vector
        skip : skip connections (used only if requried)
        pose : pose feature vector

    Returns:
        out6 : result of the forward pass (generally the same dimension as the original image)
    '''

    def forward(self, content):

        # print('content.shape:', content.shape)
        out1 = self.block1(content)
        # print('out1.shape:', out1.shape)

        # if skip connections are to be used, then the input at each stage
        # is the concatenation of current feature vector and the skip
        # connection feature vector from the encoder
        inp2 = out1
        out2 = self.block2(inp2)

        inp3 = out2
        out3 = self.block3(inp3)

        inp4 = out3
        out4 = self.block4(inp4)

        inp5 = out4
        out5 = self.block5(inp5)

        inp6 = out5
        out6 = self.block6(inp6)

        return out6

class DCGANLSTMOneStep(nn.Module):
    """
    Multiple Convolution LSTMCell like encoder
    """
    def __init__(self, input_shape, input_channels):
        super(DCGANLSTMOneStep, self).__init__()
        self.input_shape = input_shape
        self.input_channels = input_channels
        self.encoder = DCGANEncoder(input_channels, 128)
        self.lstm = nn.LSTM(512, 512, 2)
        self.decoder = DCGANDecoder(128, input_channels)

    def forward(self, input):
        """
        :param input: shape TxBxCxHxW
        :return:
        """
        seq_len = input.shape[0]
        batch_size = input.shape[1]
        out = []
        x_encoder_out = []
        for t in range(seq_len):
            # print('input[{}, ...].shape:'.format(t), input[t, ...].shape)
            x_encoder, _ = self.encoder(input[t, ...])
            x_encoder_out.append(x_encoder)

        x_encoder_out_channel, x_encoder_out_h, x_encoder_out_w = x_encoder_out[0].shape[1:]
        x_encoder_out_features = x_encoder_out_channel*x_encoder_out_h*x_encoder_out_w
        x_encoder_out = torch.cat(x_encoder_out, 0).view(seq_len, batch_size, x_encoder_out_features)
        # print('x_encoder_out.shape:', x_encoder_out.shape)
        x_encoder_out, _ = self.lstm(x_encoder_out)
        # print('x_encoder_out.shape:', x_encoder_out.shape)
        x_encoder_out = x_encoder_out.view(seq_len, batch_size, x_encoder_out_channel, x_encoder_out_h, x_encoder_out_w )

        for t in range(seq_len):
            decoder = self.decoder(x_encoder_out[t])
            out.append(decoder)
        out = torch.cat(out, 0).view(seq_len, *out[0].size())
        # print('out.shape:', out.shape)
        return out[-1, ...]


if __name__ == '__main__':
    # --------------2层LSTMCell实验相关----------------
    parser = argparse.ArgumentParser(description='training parameter setting')
    parser.add_argument('--use_cuda', type=bool, default=False, help='use cuda [ False ]')
    parser.add_argument('--use_visdom', type=bool, default=True, help='use visdom [ False ]')
    parser.add_argument('--save_epoch', type=int, default=-1, help='save model after epoch [ 1 ]')
    args = parser.parse_args()

    use_cuda = args.use_cuda
    use_visdom = args.use_visdom
    batch_size = 2
    input_shape = (64, 64) # H, W
    input_channels = 1
    num_layers = 2
    num_features = 2048
    # model = MLSTMCell(input_shape=input_shape, input_channels=input_channels, num_features=num_features, num_layers=num_layers, return_sequences=True)
    model = DCGANLSTMOneStep(input_shape=input_shape, input_channels=input_channels)
    if use_cuda:
        model.cuda()

    local_path = os.path.expanduser('~/Data/mnist_test_seq.npy')
    train_dst = MovingMNISTLoader(local_path, split='train')
    train_loader = torch.utils.data.DataLoader(train_dst, batch_size=batch_size, shuffle=True)
    val_dst = MovingMNISTLoader(local_path, split='val')
    val_loader = torch.utils.data.DataLoader(val_dst, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.0002)

    if use_visdom:
        vis = visdom.Visdom()
        vis.close()

    init_time = str(int(time.time()))
    loss_iteration_save_file = '/tmp/loss_iteration_{}.txt'.format(init_time)
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
            outputs = model(imgs_transpose)
            # hidden_h, hidden_c = outputs[0][num_layers-1]
            # print('hidden_h.shape:', hidden_h.shape)
            # print('hidden_c.shape:', hidden_c.shape)
            # print('last_hidden.shape:', last_hidden.shape)

            optimizer.zero_grad()

            # loss = 0
            # for seq in range(10):
            #     predframe = torch.sigmoid(last_hidden[seq].view(batch_size, -1))
            #     labelframe = labels[:, seq, ...].view(batch_size, -1)
            #     # print('predframe.shape:', predframe.shape)
            #     # print('labelframe.shape:', labelframe.shape)
            #     loss += crossentropyloss(predframe, labelframe)

            predframe = torch.sigmoid(outputs.view(batch_size, -1))
            labelframe = labels[:, 0, ...].view(batch_size, -1)
            # print('predframe.shape:', predframe.shape)
            # print('labelframe.shape:', labelframe.shape)
            loss = crossentropyloss(predframe, labelframe)

            loss.backward()
            optimizer.step()

            loss_np = loss.cpu().data.numpy() * 1.0 / batch_size
            # print "loss:", loss_np
            loss_epoch += loss_np
            loss_iteration_save_fp.write(str(loss_np)+'\n')

            if use_visdom and i%2==0:
                win = 'Pred'
                pred_img = torch.sigmoid(outputs).cpu().data.numpy()[-1, 0, ...]
                pred_img[pred_img>0.5] = 1
                pred_img[pred_img<0.5] = 0
                vis.image(pred_img, win=win, opts=dict(title='Pred'))

                win = 'Pred_grey'
                pred_img = torch.sigmoid(outputs).cpu().data.numpy()[-1, 0, ...]
                vis.image(pred_img, win=win, opts=dict(title='Pred_grey'))

                win = 'GT'
                gt_img = labels.cpu().data.numpy()[0, 0, ...]
                vis.image(gt_img, win=win, opts=dict(title='GT'))

                win = 'Input'
                input_img = imgs.cpu().data.numpy()[0, ...]
                vis.images(input_img, win=win, opts=dict(title='Input', caption='Input'))

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
            #         val_outputs = model(val_imgs_transpose, val_init_states)
            #         val_last_hidden = val_outputs[1]
            #
            #         val_loss = 0
            #         for val_seq in range(10):
            #             val_predframe = torch.sigmoid(val_last_hidden[val_seq].view(batch_size, -1))
            #             val_labelframe = val_labels[:, val_seq, ...].view(batch_size, -1)
            #             # print('predframe.shape:', predframe.shape)
            #             # print('labelframe.shape:', labelframe.shape)
            #             val_loss += crossentropyloss(val_predframe, val_labelframe)
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
            torch.save(model.state_dict(), 'experiment_{}_epoch_{}.pt'.format('fclstmonestep', epoch))

    loss_iteration_save_fp.close()
    # --------------2层LSTMCell实验相关----------------
