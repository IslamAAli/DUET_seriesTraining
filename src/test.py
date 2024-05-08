import torch
import numpy as np
import time
from src.export_data import export_data

def est_from_net(net, X):
    net.eval()
    with torch.no_grad():
        corr_gyro, corr_acc, hat_gyro, hat_acc = net(X.view(1, -1, 6))
        return corr_gyro[0], corr_acc[0], hat_gyro[0], hat_acc[0]

def est_gyro_from_net(net, X):
    net.eval()
    with torch.no_grad():
        corr_gyro, _, hat_gyro, _ = net(X.view(1, -1, 6))
        return corr_gyro[0], 0, hat_gyro[0], 0
    
def est_acc_from_net(net, X):
    net.eval()
    with torch.no_grad():
        _, corr_acc, hat_gyro, hat_acc = net(X.view(1, -1, 6))
        return 0, corr_acc[0], hat_gyro[0], hat_acc[0]
    
def get_predictions(net, test_data, net_type, save_path):
    times = []
    for name, data in test_data.items():
        print(f'predicting {name}...')
        imu = data['imu']
        # corr_gyro, corr_acc, hat_gyro, hat_acc 

        torch.cuda.synchronize()
        start_epoch = time.time()

        if net_type:
            pred = est_gyro_from_net(net, imu)
            export_data(pred[2], save_path, "gyro", name)
        else:
            pred = est_acc_from_net(net, imu)
            export_data(pred[3], save_path, "acc", name)

        torch.cuda.synchronize()
        end_epoch = time.time()
        elapsed = end_epoch - start_epoch
        times.append(elapsed / imu.shape[0])

def test_model(net, metrics_dict, test_data, saved_path=None, ckpt_path=None):
    # data: imu, ps, qs, vs, bgs, bas, rots, angular_velocities, accelerations 
    times = []
    for name, data in test_data.items():
        print(f'test {name}...')
        imu = data['imu']
        # corr_gyro, corr_acc, hat_gyro, hat_acc 

        torch.cuda.synchronize()
        start_epoch = time.time()
        pred = est_from_net(net, imu)
        torch.cuda.synchronize()
        end_epoch = time.time()
        elapsed = end_epoch - start_epoch
        times.append(elapsed / imu.shape[0])

        if saved_path is not None and ckpt_path is not None:
            with open(f"{saved_path}_{name}.csv", "a") as f:
                f.write(f'{ckpt_path},')
                for metric_name, metric_fn in metrics_dict.items():
                    metric = metric_fn(pred, data)
                    try:
                        len(metric)
                        for m in metric:
                            f.write(f'{m},')
                    except:
                        try:
                            f.write(f'{metric.cpu().item()},')
                        except:
                            f.write(f'{metric},')
                f.write('\n')
                f.close()
        else:
            for metric_name, metric_fn in metrics_dict.items():
                print(f'{metric_name}: {metric_fn(pred, data)}')

    with open("test_time.csv", "a") as f:
        f.write(f"{ckpt_path},{np.mean(times)}\n")
        f.close()

    return np.mean(times)

def test_gyro_model(net, metrics_dict, test_data, saved_path=None, ckpt_path=None):
    # data: imu, ps, qs, vs, bgs, bas, rots, angular_velocities, accelerations 
    times = []
    for name, data in test_data.items():
        print(f'\n => test {name}...')
        imu = data['imu']
        # corr_gyro, corr_acc, hat_gyro, hat_acc 

        torch.cuda.synchronize()
        start_epoch = time.time()
        pred = est_gyro_from_net(net, imu)
        torch.cuda.synchronize()
        end_epoch = time.time()
        elapsed = end_epoch - start_epoch
        times.append(elapsed / imu.shape[0])

        if saved_path is not None and ckpt_path is not None:
            with open(f"{saved_path}_{name}.csv", "a") as f:
                f.write(f'{ckpt_path},')
                for metric_name, metric_fn in metrics_dict.items():
                    metric = metric_fn(pred, data)
                    try:
                        len(metric)
                        for m in metric:
                            f.write(f'{m},')
                    except:
                        try:
                            f.write(f'{metric.cpu().item()},')
                        except:
                            f.write(f'{metric},')
                f.write('\n')
                f.close()
        else:
            for metric_name, metric_fn in metrics_dict.items():
                print(f'{metric_name}: {metric_fn(pred, data)}')

    with open("test_time.csv", "a") as f:
        f.write(f"{ckpt_path},{np.mean(times)}\n")
        f.close()

    return np.mean(times)

def test_acc_model(net, metrics_dict, test_data, saved_path=None, ckpt_path=None):
    # data: imu, ps, qs, vs, bgs, bas, rots, angular_velocities, accelerations 
    times = []
    for name, data in test_data.items():
        print(f'\n => test {name}...')
        imu = data['imu']
        # corr_gyro, corr_acc, hat_gyro, hat_acc 

        torch.cuda.synchronize()
        start_epoch = time.time()
        pred = est_acc_from_net(net, imu)
        torch.cuda.synchronize()
        end_epoch = time.time()
        elapsed = end_epoch - start_epoch
        times.append(elapsed / imu.shape[0])

        if saved_path is not None and ckpt_path is not None:
            with open(f"{saved_path}_{name}.csv", "a") as f:
                f.write(f'{ckpt_path},')
                for metric_name, metric_fn in metrics_dict.items():
                    metric = metric_fn(pred, data)
                    try:
                        len(metric)
                        for m in metric:
                            f.write(f'{m},')
                    except:
                        try:
                            f.write(f'{metric.cpu().item()},')
                        except:
                            f.write(f'{metric},')
                f.write('\n')
                f.close()
        else:
            for metric_name, metric_fn in metrics_dict.items():
                print(f'{metric_name}: {metric_fn(pred, data)}')

    with open("test_time.csv", "a") as f:
        f.write(f"{ckpt_path},{np.mean(times)}\n")
        f.close()

    return np.mean(times)

def test_model_raw(metrics_dict, test_data):
    # data: imu, ps, qs, vs, bgs, bas, rots, angular_velocities, accelerations 
    for name, data in test_data.items():
        print(f'test {name}...')
        imu = data['imu']
        # corr_gyro, corr_acc, hat_gyro, hat_acc 
        for metric_name, metric_fn in metrics_dict.items():
            print(f'{metric_name}: {metric_fn([None, None, imu[:, :3], imu[:, 3:]], data)}')
