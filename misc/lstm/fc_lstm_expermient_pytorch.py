# -*- coding: utf-8 -*-
import numpy as np
import time
import torch
import visdom
import os

from lstm_pytorch import MLSTMCell, MovingMNISTLoader, crossentropyloss

if __name__ == '__main__':
    # --------------2层LSTMCell实验相关----------------
    use_cuda = False
    use_visdom = True
    batch_size = 2
    input_shape = (64, 64) # H, W
    input_channels = 1
    num_layers = 2
    num_features = 2048
    model = MLSTMCell(input_shape=input_shape, input_channels=input_channels, num_features=num_features, num_layers=num_layers, return_sequences=True)
    if use_cuda:
        model.cuda()

    local_path = os.path.expanduser('~/Data/mnist_test_seq.npy')
    train_dst = MovingMNISTLoader(local_path, split='train')
    train_loader = torch.utils.data.DataLoader(train_dst, batch_size=batch_size, shuffle=True)
    val_dst = MovingMNISTLoader(local_path, split='val')
    val_loader = torch.utils.data.DataLoader(val_dst, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.0002)

    init_states = None
    if init_states is None:
        init_states = model.init_hidden(batch_size)


    val_init_states = None
    if val_init_states is None:
        val_init_states = model.init_hidden(batch_size)

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
            # print "loss:", loss_np
            loss_epoch += loss_np
            loss_iteration_save_fp.write(str(loss_np)+'\n')

            if use_visdom:
                win = 'loss_iteration'
                loss_np_expand = np.expand_dims(loss_np, axis=0)
                win_res = vis.line(X=np.ones(1) * (i + data_count * (epoch - 1) + 1), Y=loss_np_expand, win=win, update='append')
                if win_res != win:
                    vis.line(X=np.ones(1) * (i + data_count * (epoch - 1) + 1), Y=loss_np_expand, win=win, opts=dict(title=win, xlabel='iteration', ylabel='loss'))

            val_interval = 5000
            if i%val_interval==0 and i!=0:
                val_loss_epoch = 0
                for val_i, val_data in enumerate(val_loader):
                    model.eval()

                    val_imgs = val_data[:, 0:10, ...]
                    val_labels = val_data[:, 10:20, ...]
                    if use_cuda:
                        val_imgs = val_imgs.cuda()
                        val_labels = val_labels.cuda()
                    val_imgs_transpose = val_imgs.transpose(0, 1)
                    val_outputs = model(val_imgs_transpose, val_init_states)
                    val_last_hidden = val_outputs[1]

                    val_loss = 0
                    for val_seq in range(10):
                        val_predframe = torch.sigmoid(val_last_hidden[val_seq].view(batch_size, -1))
                        val_labelframe = val_labels[:, val_seq, ...].view(batch_size, -1)
                        # print('predframe.shape:', predframe.shape)
                        # print('labelframe.shape:', labelframe.shape)
                        val_loss += crossentropyloss(val_predframe, val_labelframe)

                    val_loss_np = val_loss.cpu().data.numpy() * 1.0 / batch_size
                    # print "val_loss_np:", val_loss_np
                    val_loss_epoch += val_loss_np
                val_loss_avg_epoch = val_loss_epoch / (val_data_count * 1.0)
                print "val_loss_avg_epoch:", val_loss_avg_epoch
                # break


        loss_avg_epoch = loss_epoch / (data_count * 1.0)
        if use_visdom:
            win = 'loss_epoch'
            loss_avg_epoch_expand = np.expand_dims(loss_avg_epoch, axis=0)
            win_res = vis.line(X=np.ones(1)*epoch, Y=loss_avg_epoch_expand, win=win, update='append')
            if win_res != win:
                vis.line(X=np.ones(1)*epoch, Y=loss_avg_epoch_expand, win=win, opts=dict(title=win, xlabel='epoch', ylabel='loss'))

    loss_iteration_save_fp.close()
    # --------------2层LSTMCell实验相关----------------
