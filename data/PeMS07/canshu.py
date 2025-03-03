
import numpy as np
import torch
import pywt

class StandardScaler:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        if type(data) == torch.Tensor and type(self.mean) == np.ndarray:
            self.std = torch.from_numpy(self.std).to(data.device).type(data.dtype)
            self.mean = torch.from_numpy(self.mean).to(data.device).type(data.dtype)
        return (data * self.std) + self.mean

def normalize_dataset(data, normalizer, column_wise=False):
    if normalizer == 'std':
        if column_wise:
            mean = data.mean(axis=0, keepdims=True)
            std = data.std(axis=0, keepdims=True)
        else:
            mean = data.mean()
            std = data.std()
        scaler = StandardScaler(mean, std)
        print('Normalize the dataset by Standard Normalization')
    else:
        raise ValueError
    return scaler

def get_flow(file_name):
    flow_data = np.load(file_name)
    flow_data = flow_data['data']
    return flow_data

def separate_signals(x, wavelet='coif1', level=2, threshold=None, dynamic=True):
    period_signal = torch.zeros_like(x)
    anomaly_signal = torch.zeros_like(x)
    anomaly_continued = torch.zeros_like(x[0, :])
    low_freq_thresholds = []
    high_freq_thresholds = []
    count = 0
    if dynamic:
        diff = x[1:, :] - x[:-1, :]
        diff = torch.abs(diff)
        # print("diff",diff[:, 5])
        std_diff = torch.std(diff)
        mean_diff = torch.mean(diff)
        print("000000",mean_diff)
        print("1111",std_diff)
        threshold = mean_diff + 1.5 * std_diff
        print("22222222",threshold)

    for i in range(1, x.shape[0]):
        coeffs = pywt.wavedec(x[i, :].cpu().numpy(), wavelet, level=level)
        # print(f"Coeffs shapes for index {i}: {[c.shape for c in coeffs]}")
        period_coeffs = [coeffs[0]] + [np.zeros_like(c) for c in coeffs[1:]]
        anomaly_coeffs = [np.zeros_like(coeffs[0])] + coeffs[1:]

        period_reconstructed = torch.tensor(pywt.waverec(period_coeffs, wavelet)).to(x.device)
        anomaly_reconstructed = torch.tensor(pywt.waverec(anomaly_coeffs, wavelet)).to(x.device)


        if period_reconstructed.shape[0] != x[i, :].shape[0]:
            # 只取前307个元素，或使用其他处理方式
            period_reconstructed = period_reconstructed[:x[i, :].shape[0]]

        # anomaly_reconstructed = torch.tensor(pywt.waverec(anomaly_coeffs, wavelet)).to(x.device)
        diff = x[i, :] - x[i - 1, :]
        anomaly_mask = torch.abs(diff) > threshold
        # anomaly_continued = anomaly_continued * (torch.abs(diff) < (0.2 * threshold))
        # anomaly_mask = anomaly_mask | (anomaly_continued > 0)

        period_reconstructed_strength = torch.abs(period_reconstructed)
        # print("period_reconstructed_strength",period_reconstructed_strength)
        low_freq_threshold = torch.mean(period_reconstructed_strength) * 0.8
        low_freq_thresholds.append(low_freq_threshold.item())  # 存储为 Python 标量
        anomaly_reconstructed_strength = torch.abs(anomaly_reconstructed)  # 获取突变信号的强度
        high_freq_threshold = torch.mean(anomaly_reconstructed_strength)  # 异常信号强度阈值
        high_freq_thresholds.append(high_freq_threshold.item())  # 存储为 Python 标量

        # print("anomaly_mask shape:", anomaly_mask.shape)
        # print("period_reconstructed_strength shape:", period_reconstructed_strength.shape)
        count+=1

        # print("anomaly_mask.sum()",anomaly_mask.sum())
        if (period_reconstructed_strength > low_freq_threshold).any() and (~anomaly_mask).sum() > (anomaly_mask.numel() * 0.7):
            anomaly_mask = anomaly_mask & (period_reconstructed_strength > low_freq_threshold)

        # print("000000",anomaly_mask)
        period_signal[i, :] = x[i, :] * (~anomaly_mask)
        anomaly_signal[i, :] = x[i, :] * anomaly_mask

        anomaly_continued = anomaly_mask
    average_low_freq_threshold = torch.mean(torch.tensor(low_freq_thresholds))
    average_high_freq_threshold = torch.mean(torch.tensor(high_freq_thresholds))
    std_high_freq_threshold = torch.std(torch.tensor(high_freq_thresholds))

    print("average_low_freq_threshold", average_low_freq_threshold)
    print("average_high_freq_threshold", average_high_freq_threshold)
    print("std_high_freq_threshold", std_high_freq_threshold)
    print("count",count)
    return period_signal, anomaly_signal

if __name__ == "__main__":
    traffic_data = get_flow("PEMS07.npz")

    print(traffic_data.shape)

    scaler = normalize_dataset(traffic_data[..., 0], 'std')
    traffic_data= scaler.transform(traffic_data[..., 0])
    print('Train: ', traffic_data.shape, traffic_data.max(), traffic_data.min(), traffic_data.mean(), np.median(traffic_data))
    traffic_data = torch.tensor(traffic_data)
    # 分离信号
    period_signal, anomaly_signal = separate_signals(traffic_data)
    # # 使用示例
    # # 选择第五个节点（index 4）第十天的数据
    # node_index = 10  # 第五个节点
    # day_index = 9  # 第十天，索引从0开始
    #
    # # 提取第十天的数据，假设每天有288个时间步
    # traffic_data = traffic_data[(day_index * 288) :(day_index * 288+200), :]  # shape (288, N, D)
    # traffic_data = torch.tensor(traffic_data)
    # # 分离信号
    # period_signal, anomaly_signal = separate_signals(traffic_data)
    # print("p",period_signal.shape)
    #
    #
    # # print(period_signal[:,5])
    # # print(anomaly_signal[:,5])
    #
    #
    # # # 可视化 # plt.figure(figsize=(15, 6))
    # #     # plt.plot(traffic_data[:, 55, 0].detach().numpy(), label='Original Signal', color='blue', linewidth=4, alpha=0.8)
    # import matplotlib.pyplot as plt
    #
    #
    # plt.figure(figsize=(15, 6))
    # plt.plot(traffic_data[:, node_index].detach().numpy(), label='Original Signal', color='blue', linewidth=4, alpha=0.8)
    # plt.plot(period_signal[:, node_index].detach().numpy(), label='Period Signal', color='black', linewidth=1.3,  alpha=0.6)
    # plt.plot(anomaly_signal[:, node_index].detach().numpy(), label='Anomaly Signal', color='red', linewidth=1.3, linestyle='--', alpha=0.8)
    # plt.title('Separated Period and Anomaly Signals for Node 5 on Day 10')
    # # 设置图例位置
    # plt.legend(loc='upper right')
    #
    # # # 调整Y轴范围以增强异常信号的视觉效果
    # # plt.ylim([0, 350])
    #
    # plt.xlabel('Time Step')
    # plt.ylabel('Flow')
    # plt.legend()
    # # plt.grid()
    # plt.show()
