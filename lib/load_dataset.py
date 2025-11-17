import os
import numpy as np
import pandas as pd

def load_st_dataset(dataset):
    #output B, N, D
    if dataset == 'PEMSD3':
        data_path = os.path.join('./data/PeMS03/PEMS03.npz')
        data = np.load(data_path)['data'][:, :, 0]  #onley the first dimension, traffic flow data

    elif dataset == 'PEMSD4':
        data_path = os.path.join('./data/PeMS04/PEMS04.npz')
        data = np.load(data_path)['data'][:, :, 0]  #onley the first dimension, traffic flow data

    elif dataset == 'PEMSD7':
        data_path = os.path.join('./data/PeMS07/PEMS07.npz')
        data = np.load(data_path)['data'][:, :, 0]  #onley the first dimension, traffic flow data

    elif dataset == 'PEMSD8':
        data_path = os.path.join('./data/PeMS08/PEMS08.npz')
        data = np.load(data_path)['data'][:, :, 0]  #onley the first dimension, traffic flow data

    elif dataset == 'xian_taxi':
        data_path = os.path.join('./data/xian_taxi/xian_taxi_400.npz')
        data = np.load(data_path)['data'][:, :, 0]  # onley the first dimension, traffic flow data

    elif dataset == 'PEMSD7(L)':
        data_path = os.path.join('./data/PEMS07(L)/PEMS07L.npz')
        data = np.load(data_path)['data'][:, :, 0]  #onley the first dimension, traffic flow data

    elif dataset == 'PEMSD7(M)':
        data_path = os.path.join('./data/PEMS07(M)/V_228.csv')
        data = np.array(pd.read_csv(data_path,header=None))  #onley the first dimension, traffic flow data
    elif dataset == 'METR-LA':
        data_path = os.path.join('./data/METR-LA/metr-la.h5')
        data = pd.read_hdf(data_path)
    elif dataset == 'PEMS-BAY':
        data_path = os.path.join('./data/PEMS-BAY/pems-bay.h5')
        data = pd.read_hdf(data_path)
    elif dataset == 'BJ':
        data_path = os.path.join('./data/BJ/BJ500.csv')
        data = np.array(pd.read_csv(data_path, header=0, index_col=0))
    else:
        raise ValueError
    if len(data.shape) == 2:
        data = np.expand_dims(data, axis=-1)
    print('Load %s Dataset shaped: ' % dataset, data.shape, data.max(), data.min(), data.mean(), np.median(data))
    return data


def load_st_speed(dataset):
    # output B, N, D
    if dataset == 'PEMSD3':
        # print("PEMSD3的shape",dataset.shape)
        data_path = os.path.join('./data/PeMS03/PEMS03.npz')
        speed = np.load(data_path)['data'][:, :, 2]
    elif dataset == 'PEMSD4':
        data_path = os.path.join('./data/PeMS04/PEMS04.npz')
        speed = np.load(data_path)['data'][:, :, 2]
    elif dataset == 'PEMSD7':
        data_path = os.path.join('./data/PeMS07/PEMS07.npz')
        speed = np.load(data_path)['data'][:, :, 2]
    elif dataset == 'PEMSD8':
        data_path = os.path.join('./data/PeMS08/PEMS08.npz')
        speed = np.load(data_path)['data'][:, :, 2]
    elif dataset == 'PEMSD7(L)':
        data_path = os.path.join('./data/PEMS07(L)/PEMS07L.npz')
        speed = np.load(data_path)['data'][:, :, 2]
    elif dataset == 'PEMSD7(M)':
        data_path = os.path.join('./data/PEMS07(M)/V_228.csv')
        speed = np.load(data_path)['data'][:, :, 2]
    elif dataset == 'xian_taxi':
        data_path = os.path.join('./data/xian_taxi/xian_taxi_400.npz')
        speed = np.load(data_path)['data'][:, :, 1]

    else:
        raise ValueError
    if len(speed.shape) == 2:
        speed = np.expand_dims(speed, axis=-1)

    print('Load %s Speed shaped: ' % dataset, speed.shape, speed.max(), speed.min(), speed.mean(), np.median(speed))

    return speed
def load_st_occupy(dataset):
    # output B, N, D
    if dataset == 'PEMSD3':
        data_path = os.path.join('./data/PeMS03/PEMS03.npz')
        occupy = np.load(data_path)['data'][:, :, 1]
    elif dataset == 'PEMSD4':
        data_path = os.path.join('./data/PeMS04/PEMS04.npz')
        occupy = np.load(data_path)['data'][:, :, 1]
    elif dataset == 'PEMSD7':
        data_path = os.path.join('./data/PeMS07/PEMS07.npz')
        occupy = np.load(data_path)['data'][:, :, 1]
    elif dataset == 'PEMSD8':
        data_path = os.path.join('./data/PeMS08/PEMS08.npz')
        occupy = np.load(data_path)['data'][:, :, 1]
    elif dataset == 'PEMSD7(L)':
        data_path = os.path.join('./data/PEMS07(L)/PEMS07L.npz')
        occupy = np.load(data_path)['data'][:, :, 1]
    elif dataset == 'PEMSD7(M)':
        data_path = os.path.join('./data/PEMS07(M)/V_228.csv')
        occupy = np.load(data_path)['data'][:, :, 1]
    else:
        raise ValueError
    if len(occupy.shape) == 2:
        occupy = np.expand_dims(occupy, axis=-1)

    print('Load %s occupy shaped: ' % dataset, occupy.shape, occupy.max(), occupy.min(), occupy.mean(), np.median(occupy))

    return occupy
#
# data_path = os.path.join('../data/PeMS07/PEMS07.npz')
# data = np.load(data_path)['data'][:, :, 0]  # onley the first dimension, traffic flow data
