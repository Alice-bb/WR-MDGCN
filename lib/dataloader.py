import torch
torch.cuda.empty_cache()
import numpy as np
import torch.utils.data
from lib.add_window import Add_Window_Horizon
from lib.load_dataset import load_st_dataset, load_st_occupy
from lib.load_dataset import load_st_speed
from lib.normalization import NScaler, MinMax01Scaler, MinMax11Scaler, StandardScaler, ColumnMinMaxScaler
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering


#数据归一化函数
def normalize_dataset(data, normalizer, column_wise=False):
    if normalizer == 'max01':
        if column_wise:
            minimum = data.min(axis=0, keepdims=True)
            maximum = data.max(axis=0, keepdims=True)
        else:
            minimum = data.min()
            maximum = data.max()
        scaler = MinMax01Scaler(minimum, maximum)
        data = scaler.transform(data)
        print('Normalize the dataset by MinMax01 Normalization')
    elif normalizer == 'max11':
        if column_wise:
            minimum = data.min(axis=0, keepdims=True)
            maximum = data.max(axis=0, keepdims=True)
        else:
            minimum = data.min()
            maximum = data.max()
        scaler = MinMax11Scaler(minimum, maximum)
        data = scaler.transform(data)
        print('Normalize the dataset by MinMax11 Normalization')
    elif normalizer == 'std':
        if column_wise:
            mean = data.mean(axis=0, keepdims=True)
            std = data.std(axis=0, keepdims=True)
        else:
            mean = data.mean()
            std = data.std()
        scaler = StandardScaler(mean, std)
        # data = scaler.transform(data)
        print('Normalize the dataset by Standard Normalization')
    elif normalizer == 'None':
        scaler = NScaler()
        data = scaler.transform(data)
        print('Does not normalize the dataset')
    elif normalizer == 'cmax':
        #column min max, to be depressed
        #note: axis must be the spatial dimension, please check !
        scaler = ColumnMinMaxScaler(data.min(axis=0), data.max(axis=0))
        data = scaler.transform(data)
        print('Normalize the dataset by Column Min-Max Normalization')
    else:
        raise ValueError
    return scaler

def split_data_by_days(data, val_days, test_days, interval=30):
    '''
    :param data: [B, *]
    :param val_days:
    :param test_days:
    :param interval: interval (15, 30, 60) minutes
    :return:
    用于按照天数划分数据集
    '''
    T = int((24*60)/interval)
    x=-T * int(test_days)
    test_data = data[-T*int(test_days):]
    val_data = data[-T*int(test_days + val_days): -T*int(test_days)]
    train_data = data[:-T*int(test_days + val_days)]
    return train_data, val_data, test_data

def split_data_by_ratio(data, val_ratio, test_ratio):
    data_len = data.shape[0]
    test_data = data[-int(data_len*test_ratio):]
    val_data = data[-int(data_len*(test_ratio+val_ratio)):-int(data_len*test_ratio)]
    train_data = data[:-int(data_len*(test_ratio+val_ratio))]
    return train_data, val_data, test_data

def data_loader(X, Y, batch_size, shuffle=True, drop_last=True):
    cuda = True if torch.cuda.is_available() else False
    TensorFloat = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    X, Y = TensorFloat(X), TensorFloat(Y)
    data = torch.utils.data.TensorDataset(X, Y)
    dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size,
                                             shuffle=shuffle, drop_last=drop_last)
    return dataloader

#准备数据集并返回数据加载器（dataloader）的函数
def get_dataloader(args, normalizer = 'std', tod=False, dow=False, weather=False, single=True):
    #load raw st dataset
    data = load_st_dataset(args.dataset)        # B, N, D

    # print("data", data[:, 48, 0])
    print("Loaded data shape:", data.shape)
    #normalize st data
    # scaler = normalize_dataset(data, normalizer, args.column_wise)

    L, N, F = data.shape

    feature_list = [data] #这行代码将 data 放入一个列表 feature_list 中，列表中现在只有一个元素，即原始数据。这种做法通常是为了方便后续对特征进行操作，比如添加更多特征或进行批处理。


    # numerical time_in_day
    time_ind    = [i%args.steps_per_day / args.steps_per_day for i in range(data.shape[0])]
    time_ind    = np.array(time_ind)
    time_in_day = np.tile(time_ind, [1, N, 1]).transpose((2, 1, 0))
    # print("time_in_day shape:", time_in_day.shape)
    #print("time_in_day的时间片设置:",time_in_day[:,:,0])
    feature_list.append(time_in_day)

    # numerical day_in_week
    day_in_week = [(i // args.steps_per_day)%args.days_per_week for i in range(data.shape[0])]
    day_in_week = np.array(day_in_week)
    day_in_week = np.tile(day_in_week, [1, N, 1]).transpose((2, 1, 0))
    # print("day_in_week shape:", day_in_week.shape)
    #print("day_in_week的时间片设置:", day_in_week[:, :, 0])
    feature_list.append(day_in_week)

    speed = load_st_speed(args.dataset)
    speed = np.array(speed)
    feature_list.append(speed)  # 扩展维度以匹配其他特征

    occupy = load_st_occupy(args.dataset)
    occupy = np.array(occupy)
    feature_list.append(occupy)

    steps_per_week = args.steps_per_day * 7
    past_week_avg_feature = np.zeros((L, N, F))
    past_day_avg_feature = np.zeros((L, N, F))
    recent_avg_feature = np.zeros((L, N, F))

    for i in range(data.shape[0]):
        # 1. 计算过去所有周相同时间片的流量平均值
        weekly_avgs = []
        max_weeks = min(2, int(i / steps_per_week))  # 确保最多取 22 天
        for week in range(1, max_weeks + 1):
        #for week in range(1, int(i / steps_per_week) + 1):
            weekly_avgs.append(data[i - week * steps_per_week, :, 0])
        if weekly_avgs:
            past_week_avg_feature[i, :, 0] = sum(weekly_avgs) / len(weekly_avgs)
        else:
            past_week_avg_feature[i, :, 0] = 0  # 如果没有周数据则填充零

        # 2. 计算过去所有天相同时间片的流量平均值（最多十四天）
        daily_avgs = []
        max_days = min(23, int(i / args.steps_per_day))  # 确保最多取 23 天
        for day in range(1, max_days + 1):
            daily_avgs.append(data[i - day * args.steps_per_day, :, 0])
        if daily_avgs:
            past_day_avg_feature[i, :, 0] = sum(daily_avgs) / len(daily_avgs)
        else:
            past_day_avg_feature[i, :, 0] = 0  # 如果没有天数据则填充零

        # 3. 计算前三个时间片的流量平均值
        recent_avgs = []
        for j in range(1, min(2, i + 1)):
            recent_avgs.append(data[i - j, :, 0])
        if recent_avgs:
            recent_avg_feature[i, :, 0] = sum(recent_avgs) / len(recent_avgs)
        else:
            recent_avg_feature[i, :, 0] = 0  # 如果没有最近时间片数据则填充零

    weekly_avg_feature = np.array(past_week_avg_feature)
    feature_list.append(weekly_avg_feature)
    daily_avg_feature = np.array(past_day_avg_feature)
    feature_list.append(daily_avg_feature)
    recent_avg_feature = np.array(recent_avg_feature)
    feature_list.append(recent_avg_feature)

    # data = np.concatenate(feature_list, axis=-1)
    x, y = Add_Window_Horizon(data, args.lag, args.horizon, single)
    print("Windowed data shapes - x:", x.shape, "y:", y.shape)
    #print("原始数据输入窗口的时间片设置:", x[10, :, 0, 0])
    #print("原始数据预测窗口的时间片设置:", y[10, :, 0, 0])
    x_day, y_day = Add_Window_Horizon(time_in_day, args.lag, args.horizon, single)
    #print("Windowed data shapes - x_day:", x_day.shape, "y_day:",y_day.shape)
    x_week, y_week = Add_Window_Horizon(day_in_week, args.lag, args.horizon, single)
    #print("Windowed data shapes - x_week:", x_week.shape, "y_week:", y_week.shape)
    # x_cluster, y_cluster = Add_Window_Horizon(cluster_labels, args.lag, args.horizon, single)
    x_speed, y_speed =  Add_Window_Horizon(speed, args.lag, args.horizon, single)
    x_occupy, y_occupy =  Add_Window_Horizon(occupy, args.lag, args.horizon, single)
    x_weekly, y_weekly =  Add_Window_Horizon(weekly_avg_feature, args.lag, args.horizon, single)
    x_daily, y_daily =  Add_Window_Horizon(daily_avg_feature, args.lag, args.horizon, single)
    x_recent, y_recent =  Add_Window_Horizon(recent_avg_feature, args.lag, args.horizon, single)
    x, y = np.concatenate([x, x_day, x_week, x_speed, x_occupy, x_weekly,x_daily,x_recent], axis=-1), np.concatenate([y, y_day, y_week], axis=-1) #B,T,N,3

    if args.test_ratio > 1:
        x_train, x_val, x_test = split_data_by_days(x, args.val_ratio, args.test_ratio)
        y_train, y_val, y_test = split_data_by_days(y, args.val_ratio, args.test_ratio)
        # day_train, day_val, day_test = split_data_by_days(time_in_day, args.val_ratio, args.test_ratio)
        # week_train, week_val, week_test = split_data_by_days(day_in_week, args.val_ratio, args.test_ratio)
    else:
        x_train, x_val, x_test = split_data_by_ratio(x, args.val_ratio, args.test_ratio)
        y_train, y_val, y_test = split_data_by_ratio(y, args.val_ratio, args.test_ratio)

        # week_train, week_val, week_test = split_data_by_ratio(day_in_week, args.val_ratio, args.test_ratio)
    # #normalize st data
    scaler = normalize_dataset(x_train[..., :args.input_dim], normalizer, args.column_wise)
    scaler1 = normalize_dataset(x_train[..., 3], normalizer, args.column_wise)
    scaler2 = normalize_dataset(x_train[..., 4], normalizer, args.column_wise)
    scaler3 = normalize_dataset(x_train[..., 5], normalizer, args.column_wise)
    scaler4 = normalize_dataset(x_train[..., 6], normalizer, args.column_wise)
    scaler5 = normalize_dataset(x_train[..., 7], normalizer, args.column_wise)

    x_train[..., :args.input_dim] = scaler.transform(x_train[..., :args.input_dim])
    x_val[..., :args.input_dim] = scaler.transform(x_val[..., :args.input_dim])
    x_test[..., :args.input_dim] = scaler.transform(x_test[..., :args.input_dim])
    x_train[..., 3] = scaler1.transform(x_train[..., 3])
    x_val[..., 3] = scaler1.transform(x_val[..., 3])
    x_test[..., 3] = scaler1.transform(x_test[..., 3])
    x_train[..., 4] = scaler2.transform(x_train[..., 4])
    x_val[..., 4] = scaler2.transform(x_val[..., 4])
    x_test[..., 4] = scaler2.transform(x_test[..., 4])
    x_train[..., 5] = scaler3.transform(x_train[..., 5])
    x_val[..., 5] = scaler3.transform(x_val[..., 5])
    x_test[..., 5] = scaler3.transform(x_test[..., 5])
    x_train[..., 6] = scaler4.transform(x_train[..., 6])
    x_val[..., 6] = scaler4.transform(x_val[..., 6])
    x_test[..., 6] = scaler4.transform(x_test[..., 6])
    x_train[..., 7] = scaler5.transform(x_train[..., 7])
    x_val[..., 7] = scaler5.transform(x_val[..., 7])
    x_test[..., 7] = scaler5.transform(x_test[..., 7])

    # #add time window
    # x_tra, y_tra = Add_Window_Horizon(data_train, args.lag, args.horizon, single)
    # x_val, y_val = Add_Window_Horizon(data_val, args.lag, args.horizon, single)
    # x_test, y_test = Add_Window_Horizon(data_test, args.lag, args.horizon, single)
    print('Train: ', x_train.shape, y_train.shape, x_train.max(), x_train.min(), x_train.mean(), np.median(x_train),
          y_train.max(), y_train.min(), y_train.mean())
    print('Val: ', x_val.shape, y_val.shape, x_val.max(), x_val.min(), x_val.mean(), np.median(x_val),
          y_val.max(), y_val.min(), y_val.mean())
    print('Test: ', x_test.shape, y_test.shape, x_test.max(), x_test.min(), x_test.mean(), np.median(x_test),
          y_test.max(), y_test.min(), y_test.mean())

    ##############get dataloader######################
    train_dataloader = data_loader(x_train, y_train, args.batch_size, shuffle=True, drop_last=True)
    if len(x_val[..., 0]) == 0:
        val_dataloader = None
    else:
        val_dataloader = data_loader(x_val, y_val, args.batch_size, shuffle=False, drop_last=True)
    test_dataloader = data_loader(x_test, y_test, args.batch_size, shuffle=False, drop_last=False)
    return train_dataloader, val_dataloader, test_dataloader, scaler

def get_adjacency_matrix2(distance_df_filename, num_of_vertices,
                         type_='connectivity', id_filename=None):
    '''
    Parameters
    ----------
    distance_df_filename: str, path of the csv file contains edges information

    num_of_vertices: int, the number of vertices

    type_: str, {connectivity, distance}

    Returns
    ----------
    A: np.ndarray, adjacency matrix

    '''
    import csv

    A = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                 dtype=np.float32)

    # Fills cells in the matrix with distances.
    with open(distance_df_filename, 'r') as f:
        f.readline()
        reader = csv.reader(f)
        for row in reader:
            if len(row) != 3:
                continue
            i, j, distance = int(row[0]), int(row[1]), float(row[2])
            if type_ == 'connectivity':
                A[i, j] = 1
                A[j, i] = 1
            elif type_ == 'distance':
                A[i, j] = 1 / distance
                A[j, i] = 1 / distance
            else:
                raise ValueError("type_ error, must be "
                                 "connectivity or distance!")
    return A


if __name__ == '__main__':
    import argparse
    #MetrLA 207; BikeNYC 128; SIGIR_solar 137; SIGIR_electric 321
    DATASET = 'SIGIR_electric'
    if DATASET == 'MetrLA':
        NODE_NUM = 207
    elif DATASET == 'BikeNYC':
        NODE_NUM = 128
    elif DATASET == 'SIGIR_solar':
        NODE_NUM = 137
    elif DATASET == 'SIGIR_electric':
        NODE_NUM = 321
    parser = argparse.ArgumentParser(description='PyTorch dataloader')
    parser.add_argument('--dataset', default=DATASET, type=str)
    parser.add_argument('--num_nodes', default=NODE_NUM, type=int)
    parser.add_argument('--val_ratio', default=0.1, type=float)
    parser.add_argument('--test_ratio', default=0.2, type=float)
    parser.add_argument('--lag', default=12, type=int)
    parser.add_argument('--horizon', default=12, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    args = parser.parse_args()
    train_dataloader, val_dataloader, test_dataloader, scaler = get_dataloader(args, normalizer = 'std', tod=False, dow=False, weather=False, single=True)
