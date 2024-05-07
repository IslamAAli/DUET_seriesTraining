#%%
import os
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader

print('cuda: %s' % torch.cuda.is_available())

from src.dataset import EuRoC
from src.networks import IMUNetGyro, IMUNetAcc
from src.losses import IMULoss, IMULossGyro, IMULossAcc
from src.metrics import metric_aoe_test, metric_aoe_training, metric_ave_test, metric_ave_training, metric_aye_test, metric_rte_improvement_test, metric_rte_test
from src.training import train_model
from src.test import test_model, test_gyro_model, test_acc_model
import random

torch.manual_seed(1)
random.seed(1)
np.random.seed(1)

ENABLE_GYRO_NET = False
ENABLE_ACC_NET  = True

T = 84
training_samples = 16000 // (T * 2) * (T * 2)
train_seqs = [
    'MH_01_easy',
    'MH_03_medium',
    'MH_05_difficult',
    'V1_02_medium',
    'V2_01_easy',
    'V2_03_difficult'
]
test_seqs = [
    'MH_02_easy',
    'MH_04_difficult',
    'V1_03_difficult',
    'V2_02_medium'
]

# data split (different days)
# test_seqs = [
    # 'V1_02_medium',
    # 'V1_03_difficult',
    # 'V2_01_easy',
    # 'V2_02_medium',
    # 'V2_03_difficult'
# ]
# train_seqs = [
#     'MH_01_easy',
#     'MH_02_easy',
#     'MH_03_medium',
#     'MH_04_difficult',
#     'MH_05_difficult'
# ]

#%% Loading Data
print('Loading data...')
base_path = os.path.dirname(os.path.abspath(__file__))
euroc = EuRoC(base_path+'/original_datasets/euroc', base_path+'/data/euroc', train_seqs, test_seqs, training_samples, T)

#%% Model Parameters
in_channel = 6
out_channel = 3
layer_channels = [16, 32, 64, 128]
batch_size = 128
kernel_size = 5
dropout = 0.1
lr = 0.01
num_epochs = 1800
ckpt_path_gyro = base_path+'/models/euroc_gyro.pt'
ckpt_path_acc = base_path+'/models/euroc_acc.pt'

# data
train_iter = DataLoader(euroc, batch_size=batch_size, shuffle=False)
val_data = euroc.val

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Gyroscope network 
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
if ENABLE_GYRO_NET:
    print("## Gyro NET ##")
    # ===========================
    # train - Gyro
    # ===========================
    # Gyro Net
    gyro_net = IMUNetGyro(in_channel, layer_channels, out_channel, kernel_size, dropout, euroc.mean, euroc.std).cuda()
    loss_func_gyro = IMULossGyro(T=T).cuda()
    optimiser_gyro = torch.optim.Adam([
        {'params': gyro_net.parameters(), 'lr':lr, 'weight_decay': 0.1, },
        {'params': loss_func_gyro.parameters(), 'weight_decay': 0}
    ])
    metrics_dict_training_gyro = {'AOE': lambda x, y: metric_aoe_training(x, y)}
    metrics_for_early_stopping_gyro = ['AOE']

    # perform training
    running_time = train_model(gyro_net, optimiser_gyro, loss_func_gyro, metrics_dict_training_gyro, metrics_for_early_stopping_gyro, train_iter, val_data, epochs=num_epochs, patience=-1, ckpt_path=ckpt_path_gyro)
    print(f'training time: {running_time} s/epoch (time of gyro loss: {np.mean(loss_func_gyro.times_gyro)})')
    print('='.ljust(20, '='))

    # ===========================
    # test - Gyro
    # ===========================
    metric_dict_test_gyro = {'Absolute Orientation Error (AOE)': lambda x, y: metric_aoe_test(x, y),
                            'Absolute Yaw Error (AYE)': lambda x, y: metric_aye_test(x, y)}

    net = IMUNetGyro(in_channel, layer_channels, out_channel, kernel_size, dropout, euroc.mean, euroc.std).cuda()
    net.load_state_dict(torch.load(ckpt_path_gyro))
    net.eval()
    # running_time = test_model(net, metric_dict_test, euroc.test_seqs)
    running_time = test_gyro_model(net, metric_dict_test_gyro, euroc.test_seqs)
    print(f'running time: {running_time}')
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Aceelerometer Network
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
if ENABLE_ACC_NET:
    print("## Acc NET ##")
    # # ===========================
    # # train - Acc
    # # ===========================
    # # Acc Net
    # acc_net = IMUNetAcc(in_channel, layer_channels, out_channel, kernel_size, dropout, euroc.mean, euroc.std).cuda()
    # loss_func_acc = IMULossAcc(T=T).cuda()
    # optimiser_acc = torch.optim.Adam([
    #     {'params': acc_net.parameters(), 'lr':lr, 'weight_decay': 0.1, },
    #     {'params': loss_func_acc.parameters(), 'weight_decay': 0}
    # ])

    # metrics_dict_training_acc = {'AOE': lambda x, y: metric_aoe_training(x, y),
    #                             'AVE': lambda x, y: metric_ave_training(x, y)}
    # metrics_for_early_stopping_acc = ['AOE', 'AVE']

    # # perform training
    # running_time = train_model(acc_net, optimiser_acc, loss_func_acc, metrics_dict_training_acc, metrics_for_early_stopping_acc, train_iter, val_data, epochs=num_epochs, patience=-1, ckpt_path=ckpt_path_acc)
    # print(f'training time: {running_time} s/epoch (time of accel loss: {np.mean(loss_func_acc.times_accel)})')
    # print('='.ljust(20, '='))

    # ===========================
    # test - Acc
    # ===========================
    metric_dict_test_acc = {'Absolute Orientation Error (AOE)': lambda x, y: metric_aoe_test(x, y),
                        'Absolute Yaw Error (AYE)': lambda x, y: metric_aye_test(x, y),
                        'Absolute Velocity Error (AVE)': lambda x, y: metric_ave_test(x, y),
                        'Relative Translation Error (RTE)': lambda x, y: metric_rte_test(x, y, 5, 50), # rte: 5s 50 reps
                        'Improvement of Relative Translation Error (RAW, DUET, Improvement)': lambda x, y: metric_rte_improvement_test(x, y, 5, 50)}

    net = IMUNetAcc(in_channel, layer_channels, out_channel, kernel_size, dropout, euroc.mean, euroc.std).cuda()
    net.load_state_dict(torch.load(ckpt_path_acc))
    net.eval()
    # running_time = test_model(net, metric_dict_test, euroc.test_seqs)
    running_time = test_acc_model(net, metric_dict_test_acc, euroc.test_seqs)
    print(f'running time: {running_time}')
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++