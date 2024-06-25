#%%
import os
import torch
import numpy as np
import shutil
from torch import nn
from torch.utils.data import DataLoader
import pandas as pd

print('cuda: %s' % torch.cuda.is_available())

from src.dataset import EuRoC
from src.networks import IMUNetGyro, IMUNetAcc
from src.losses import IMULoss, IMULossGyro, IMULossAcc
from src.metrics import metric_aoe_test, metric_aoe_training, metric_ave_test, metric_ave_training, metric_aye_test, metric_rte_improvement_test, metric_rte_test
from src.training import train_model
from src.test import test_model, test_gyro_model, test_acc_model, get_predictions
import random
from src.export_data import export_data

torch.manual_seed(1)
random.seed(1)
np.random.seed(1)

ENABLE_GYRO_NET = True
ENABLE_ACC_NET  = True
ENABLE_FIXED_DATA_PROPAGATION = True

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
    'MH_01_easy',
    'MH_03_medium',
    'MH_05_difficult',
    'V1_02_medium',
    'V2_01_easy',
    'V2_03_difficult',
    'MH_02_easy',
    'MH_04_difficult',
    'V1_03_difficult',
    'V2_02_medium'
]

#%% Loading Data
print('Loading data...')
base_path = os.path.dirname(os.path.abspath(__file__))

corrected_imu_base_path = base_path+'/data/euroc_corrected'

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

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Gyroscope network 
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
if ENABLE_GYRO_NET:
    print("## Gyro NET ##")
    # ===========================
    # Data reading and preparation - Gyro
    # ===========================
    euroc = EuRoC(base_path+'/original_datasets/euroc', base_path+'/data/euroc', train_seqs, test_seqs, training_samples, T)
    train_iter = DataLoader(euroc, batch_size=batch_size, shuffle=False)
    val_data = euroc.val

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

    get_predictions(net, euroc.test_seqs, True, corrected_imu_base_path)   # predictions[0] => correction in gyro, predictions[2] => corrected gyro readings

    running_time = test_gyro_model(net, metric_dict_test_gyro, euroc.test_seqs)
    print(f'running time: {running_time}')
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Replace raw gyro data with fixed gyro data in the files in the corrected folder
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
if ENABLE_FIXED_DATA_PROPAGATION:
    # ===================================================================
    # copy gt data to folder as is
    # =================================================================== 
    source_folder =  base_path+"/data/euroc"

    # Destination folder path
    destination_folder = base_path+"/data/euroc_fixed_gyro_raw_acc"

    # List all files in the source folder
    files = os.listdir(source_folder)

    # Iterate through the files
    for file in files:
        # Check if the file ends with the postfix "_gt.csv"
        if file.endswith("_gt.csv"):
            # Form the full source path
            source_path = os.path.join(source_folder, file)
            
            # Form the full destination path
            destination_path = os.path.join(destination_folder, file)
            
            # Copy the file from source to destination
            shutil.copyfile(source_path, destination_path)
            print(f"[*INFO*] File '{file}' copied successfully.")

    print("[*INFO*] All ground turth files copied.")
    # ===================================================================

    # ===================================================================
    # replace the gyro columns in the imu file with corrected data
    # ===================================================================
    base_folder = base_path+"/data/euroc"
    corr_folder = base_path+"/data/euroc_corrected"
    dst_folder  = base_path+"/data/euroc_fixed_gyro_raw_acc"
    for seq in test_seqs:
        # file names 
        base_filename = seq+"_imu.csv"
        corr_filename = "gyro_"+seq+".csv"
        # Form full paths
        corr_file_path = os.path.join(corr_folder, corr_filename)
        base_file_path = os.path.join(base_folder, base_filename)
        
        # Read the first file (3 columns)
        corr_df = pd.read_csv(corr_file_path, header=None)
        
        # Read the second file (6 columns)
        base_df = pd.read_csv(base_file_path, header=None)
        base_df = base_df.drop(base_df.index[-1])
        
        # Replace the first three columns in the second file
        base_df.iloc[:, :3] = corr_df.iloc[:, :3]
        
        # Write the modified second file back to disk
        dst_filename = seq+"_imu.csv"
        dst_file_path = os.path.join(dst_folder, dst_filename)
        base_df.to_csv(dst_file_path, index=False)
        
        print(f"[*INFO*] Columns replaced in sequence: {seq}")

    print("[*INFO*] Corrected gyro inserted.")
    # ===================================================================

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Aceelerometer Network
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
if ENABLE_ACC_NET:
    print("## Acc NET ##")
    # ===========================
    # Data reading and preparation - Gyro
    # ===========================
    euroc = EuRoC(base_path+'/original_datasets/euroc', base_path+'/data/euroc_fixed_gyro_raw_acc', train_seqs, test_seqs, training_samples, T)
    train_iter = DataLoader(euroc, batch_size=batch_size, shuffle=False)
    val_data = euroc.val

    # ===========================
    # train - Acc
    # ===========================
    # Acc Net
    acc_net = IMUNetAcc(in_channel, layer_channels, out_channel, kernel_size, dropout, euroc.mean, euroc.std).cuda()
    loss_func_acc = IMULossAcc(T=T).cuda()
    optimiser_acc = torch.optim.Adam([
        {'params': acc_net.parameters(), 'lr':lr, 'weight_decay': 0.1, },
        {'params': loss_func_acc.parameters(), 'weight_decay': 0}
    ])

    metrics_dict_training_acc = {'AOE': lambda x, y: metric_aoe_training(x, y),
                                'AVE': lambda x, y: metric_ave_training(x, y)}
    metrics_for_early_stopping_acc = ['AOE', 'AVE']

    # perform training
    running_time = train_model(acc_net, optimiser_acc, loss_func_acc, metrics_dict_training_acc, metrics_for_early_stopping_acc, train_iter, val_data, epochs=num_epochs, patience=-1, ckpt_path=ckpt_path_acc)
    print(f'training time: {running_time} s/epoch (time of accel loss: {np.mean(loss_func_acc.times_accel)})')
    print('='.ljust(20, '='))

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

    get_predictions(net, euroc.test_seqs, False, corrected_imu_base_path)   # predictions[1] => correction in acc, predictions[3] => corrected acc readings

    running_time = test_acc_model(net, metric_dict_test_acc, euroc.test_seqs)
    print(f'running time: {running_time}')
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
