# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt

def read_file_avg(loss_fname):
    loss_fp = open(loss_fname, 'rb')
    loss_fp_items = loss_fp.readlines()
    loss_list = []
    for loss_fp_item_id, loss_fp_item in enumerate(loss_fp_items):
        # print('loss_fp_item:', loss_fp_item)
        loss = float(loss_fp_item.strip())
        loss_list.append(loss)
        if loss_fp_item_id==2000:
            break
    loss_np = np.array(loss_list)
    # print('loss_list:', loss_list)
    batch_size = 5

    loss_acc = 0
    loss_avg_list = []
    for loss_np_id, loss_np_item in enumerate(loss_np):
        # print('loss_np_id:', loss_np_id)
        # print('loss_np_item:', loss_np_item)
        loss_acc += loss_np_item
        if (loss_np_id+1)%batch_size==0:
            loss_avg = loss_acc*1.0/batch_size
            loss_avg_list.append(loss_avg)
            # print('loss_avg:', loss_avg)
            # print('loss_acc:', loss_acc)
            loss_acc = 0
    return loss_avg_list

def save_list_to_file(loss_avg_list, fname):
    loss_fp = open(fname, 'wb')
    for loss_avg_list_item in loss_avg_list:
        loss_fp.write(str(loss_avg_list_item)+'\n')
    loss_fp.close()

if __name__ == '__main__':
    loss_fnames = []
    loss_fname = os.path.expanduser('~/GitHub/Quick/master_thesis/杂/预测实验/conv_lstm_loss_iteration_onestep_12_6_1.txt')
    loss_fnames.append(loss_fname)
    # loss_fname = os.path.expanduser('~/GitHub/Quick/master_thesis/杂/预测实验/fc_lstm_loss_iteration_onestep_12_6_1.txt')
    # loss_fnames.append(loss_fname)
    loss_fname = os.path.expanduser('~/GitHub/Quick/master_thesis/杂/预测实验/fc_lstm_lowchannel_loss_iteration_onestep_12_6_1.txt')
    loss_fnames.append(loss_fname)
    loss_fname = os.path.expanduser('~/GitHub/Quick/master_thesis/杂/预测实验/fc_lstm_channel_2048_tmp_loss_iteration_onestep_12_6_1.txt')
    loss_fnames.append(loss_fname)
    loss_fname = os.path.expanduser('~/GitHub/Quick/master_thesis/杂/预测实验/resconv_lstm_loss_iteration_onestep_12_6_1.txt')
    loss_fnames.append(loss_fname)
    for loss_fname in loss_fnames:
        loss_avg_list = read_file_avg(loss_fname)
        loss_avg_np = np.array(loss_avg_list)
        loss_avg_x = np.array(range(len(loss_avg_list)))
        plt.plot(loss_avg_x, loss_avg_np)
        loss_fname_nofix = loss_fname[:loss_fname.rfind('.')]
        save_list_to_file(loss_avg_list, fname='{}_smooth.txt'.format(loss_fname_nofix))

    plt.show()
