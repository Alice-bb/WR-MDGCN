
from model.DDGCRNCell import DDGCRNCell

from model.GWCCell import GWCCell
import numpy as np

import torch
import torch.nn as nn
torch.cuda.empty_cache()
'''
小波分解在信号处理中确实是一种强大的工具，因为它可以将信号分解为不同的频率分量，从而分离出周期性信号（低频分量）和突变信号（高频分量）。然而，直接依赖小波分解来区分周期性信号和突变信号也有一定的局限性。这就是为什么在你的代码中先使用差分法，然后用小波分解来补充的原因。
'''
import pywt

#'db4', 'haar', 'coif1', 'sym4'
def separate_signals(x, wavelet='sym3', level=4):
    # 初始化
    period_signal = torch.zeros_like(x)
    anomaly_signal = torch.zeros_like(x)

    for i in range(1, x.shape[1]):
        # 将每个时间步的数据取出，送入小波变换
        coeffs = pywt.wavedec(x[:, i, :, 0].cpu().numpy(), wavelet, level=level, mode='per')

        # 保留低频部分的系数作为周期信号
        period_coeffs = [coeffs[0]] + [np.zeros_like(c) for c in coeffs[1:]]  # 高频为0，保留低频（周期性）
        # 保留高频部分的系数作为异常信号
        anomaly_coeffs = [np.zeros_like(coeffs[0])] + coeffs[1:]  # 低频为0，保留高频（突变）

        # 小波逆变换得到周期信号和突变信号
        period_reconstructed = torch.tensor(pywt.waverec(period_coeffs, wavelet, mode='per')).to(x.device)
        anomaly_reconstructed = torch.tensor(pywt.waverec(anomaly_coeffs, wavelet, mode='per')).to(x.device)
        if period_reconstructed.shape[1] != x[:, i, :, :].shape[1]:
        # 只取前307个元素，或使用其他处理方式
            period_reconstructed = period_reconstructed[:, :x[:, i, :,:].shape[1]]
        if anomaly_reconstructed.shape[1] != x[:, i, :, :].shape[1]:
        # 只取前307个元素，或使用其他处理方式
            anomaly_reconstructed = anomaly_reconstructed[:, :x[:, i, :,:].shape[1]]
        # 将周期信号和突变信号分配到结果中

        period_signal[:, i, :, 0] = period_reconstructed
        period_signal[:, i, :, 1] = x[:, i, :, 1]
        period_signal[:, i, :, 2] = x[:, i, :, 2]
        period_signal[:, i, :, 3] = x[:, i, :, 3]
        period_signal[:, i, :, 4] = x[:, i, :, 4]
        period_signal[:, i, :, 5] = x[:, i, :, 5]
        period_signal[:, i, :, 6] = x[:, i, :, 6]
        period_signal[:, i, :, 7] = x[:, i, :, 7]

        anomaly_signal[:, i, :, 0] = anomaly_reconstructed
        anomaly_signal[:, i, :, 1] = x[:, i, :, 1]
        anomaly_signal[:, i, :, 2] = x[:, i, :, 2]
        anomaly_signal[:, i, :, 3] = x[:, i, :, 3]
        anomaly_signal[:, i, :, 4] = x[:, i, :, 4]
        # anomaly_signal[:, i, :, 5] = x[:, i, :, 5]
        # anomaly_signal[:, i, :, 6] = x[:, i, :, 6]
        # anomaly_signal[:, i, :, 7] = x[:, i, :, 7]
    return period_signal, anomaly_signal



class GWCRM(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, diffusion_steps, embed_dim, num_layers=1):
        super(GWCRM, self).__init__()
        assert num_layers >= 1, 'At least one DCRNN layer in the Encoder.'
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers
        self.GWCRM_cells = nn.ModuleList()
        self.GWCRM_cells.append(GWCCell(node_num, dim_in, dim_out, diffusion_steps, embed_dim))
        for _ in range(1, num_layers):
            self.GWCRM_cells.append(GWCCell(node_num, dim_out, dim_out, diffusion_steps, embed_dim))

    def forward(self, x, init_state, node_embeddings,speed,occupy):
        #shape of x: (B, T, N, D)
        #shape of init_state: (num_layers, B, N, hidden_dim)
        assert x.shape[2] == self.node_num and x.shape[3] == self.input_dim
        seq_length = x.shape[1]     #x=[batch,steps,nodes,input_dim]
        current_inputs = x
        output_hidden = []
        for i in range(self.num_layers):

            state = init_state[i]   #state=[batch,steps,nodes,input_dim]
            inner_states = []
            for t in range(seq_length):   #如果有两层GRU，则第二层的GGRU的输入是前一层的隐藏状态
                state = self.GWCRM_cells[i](current_inputs[:, t, :, :], state, [node_embeddings[0][:, t, :, :], node_embeddings[1]],speed[:, t, :, :],occupy[:,t,:,:])#state=[batch,steps,nodes,input_dim]
                # state = self.dcrnn_cells[i](current_inputs[:, t, :, :], state,[node_embeddings[0], node_embeddings[1]])
                inner_states.append(state)   #一个list，里面是每一步的GRU的hidden状态
            output_hidden.append(state)  #每层最后一个GRU单元的hidden状态
            current_inputs = torch.stack(inner_states, dim=1)
            #拼接成完整的上一层GRU的hidden状态，作为下一层GRRU的输入[batch,steps,nodes,hiddensize]
        return current_inputs, output_hidden

    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.GWCRM_cells[i].init_hidden_state(batch_size))
        return torch.stack(init_states, dim=0)      #(num_layers, B, N, hidden_dim)

class DGCRM(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim, num_layers=1):
        super(DGCRM, self).__init__()
        assert num_layers >= 1, 'At least one DCRNN layer in the Encoder.'
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers
        self.embed_dim = embed_dim

        # 自适应时间步权重生成模块
        self.time_weight_generator = nn.Sequential(
            nn.Linear(2 * dim_in, 40),
            nn.ReLU(),
            nn.Linear(40, 2),
            nn.ReLU(),
            nn.Linear(2, 1),
            nn.Sigmoid()  # 输出的权重在 [0, 1] 之间
        )


        self.a = nn.Parameter(torch.FloatTensor([0.2]))  # Global memory weight
        self.b = nn.Parameter(torch.FloatTensor([0.2]))  # Update weight
        self.DGCRM_cells = nn.ModuleList()
        self.DGCRM_cells.append(DDGCRNCell(node_num, dim_in, dim_out, cheb_k, embed_dim))
        for _ in range(1, num_layers):
            self.DGCRM_cells.append(DDGCRNCell(node_num, dim_out, dim_out, cheb_k, embed_dim))

    def forward(self, x, init_state, node_embeddings, time, day, speed, occupy, weekly, daily, recent):
        assert x.shape[2] == self.node_num and x.shape[3] == self.input_dim
        seq_length = x.shape[1]
        batch_size = x.shape[0]
        # 初始化全局记忆
        global_memory = torch.zeros(batch_size, self.node_num, self.DGCRM_cells[0].hidden_dim).to(x.device)

        ## 000000000000 weekly      weekly
        importance_scores0 = torch.abs(x - weekly)  # Shape: [B, T, N, 1]
        importance_scores0 = importance_scores0.mean(dim=2, keepdim=True)  # Reduce over node dimension [B, T, 1, 1]
        # Normalize importance scores to get attention weights
        importance_weights0 = importance_scores0 / (
                    importance_scores0.sum(dim=1, keepdim=True) + 1e-6)  # Shape: [B, T, 1, 1]
        # weighted_input0 = torch.matmul(x,importance_weights0)  # Shape: [B, T, N, 1]
        weighted_input0 = x * importance_weights0  # Shape: [B, T, N, 1]


        # 1111111111111   daily  daily
        importance_scores1 = torch.abs(x - daily)  # Shape: [B, T, N, 1]
        importance_scores1 = importance_scores1.mean(dim=2, keepdim=True)  # Reduce over node dimension [B, T, 1, 1]
        # Normalize importance scores to get attention weights
        importance_weights1 = importance_scores1 / (
                importance_scores1.sum(dim=1, keepdim=True) + 1e-6)  # Shape: [B, T, 1, 1]
        weighted_input1 = x * importance_weights1  # Shape: [B, T, N, 1]


        # 2222222222222recent    recent   recent
        importance_scores2 = torch.abs(x - recent)  # Shape: [B, T, N, 1]
        importance_scores2 = importance_scores2.mean(dim=2, keepdim=True)  # Reduce over node dimension [B, T, 1, 1]
        importance_weights2 = importance_scores2 / (
                importance_scores2.sum(dim=1, keepdim=True) + 1e-6)  # Shape: [B, T, 1, 1]
        # Apply importance weights to the input
        weighted_input2 = x * importance_weights2  # Shape: [B, T, N, 1]

        current_inputs = x + weighted_input0 + weighted_input1 + weighted_input2
        # current_inputs = x

        output_hidden = []
        for i in range(self.num_layers):
            state = init_state[i]
            inner_states = []


            for t in range(seq_length):
                state = self.DGCRM_cells[i](current_inputs[:, t, :, :], state,
                                            [node_embeddings[0][:, t, :, :], node_embeddings[1]],
                                            speed[:, t, :, :], occupy[:, t, :, :],time[:, t, :, :],day[:, t, :, :])
                # 生成时间步自适应权重
                time_input = torch.cat([time[:, t, :, :], day[:, t, :, :]], dim=-1)  # 结合时间信息
                time_weight = self.time_weight_generator(time_input)  # [B, N, 1]：每个时间步的权重

                # 使用时间步自适应权重融合 state 和 global_memory
                enhanced_state = time_weight * state + (1 - time_weight) * global_memory
                # enhanced_state = state + (1 - self.a) * global_memory
                # 更新全局记忆
                global_memory = (1-self.a) * global_memory + self.a * enhanced_state
                state = state + self.b * enhanced_state
                inner_states.append(state)
            output_hidden.append(state)
            current_inputs = torch.stack(inner_states, dim=1)
        return current_inputs, output_hidden


    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.DGCRM_cells[i].init_hidden_state(batch_size))
        return torch.stack(init_states, dim=0)

# from data.PeMS08.generate_adj_mx import get_adjacency_matrix_2direction
#from data.PeMS04.generate_adj_mx import get_adjacency_matrix_2direction

class WRMDGC(nn.Module):

    def __init__(self, args):
        super(WRMDGC, self).__init__()
        self.num_nodes = args.num_nodes
        self.input_dim = args.input_dim
        self.hidden_dim = args.rnn_units
        self.output_dim = args.output_dim
        self.horizon = args.horizon
        self.num_layers = args.num_layers
        self.use_D = args.use_day
        self.use_W = args.use_week
        self.diffusion_steps = args.diffusion_steps
        self.embed_dim = args.embed_dim
        self.dropout1 = nn.Dropout(p=0.2)
        self.dropout2 = nn.Dropout(p=0.2)
        self.dropout3 = nn.Dropout(p=0.2)
        self.dropout4 = nn.Dropout(p=0.2)
        self.dropout5 = nn.Dropout(p=0.2)
        self.fc = nn.Linear(self.hidden_dim, 1)
        self.default_graph = args.default_graph

        # 初始化节点嵌入
        self.node_embeddings1 = nn.Parameter(torch.randn(self.num_nodes, args.embed_dim), requires_grad=True)
        self.node_embeddings2 = nn.Parameter(torch.randn(self.num_nodes, args.embed_dim), requires_grad=True)

        self.T_i_D_emb = nn.Parameter(torch.empty(288, args.embed_dim))
        self.D_i_W_emb = nn.Parameter(torch.empty(7, args.embed_dim))
        # self.T_i_D_rope = rotary_position_embedding(288, args.embed_dim)
        # self.D_i_W_rope = rotary_position_embedding(7, args.embed_dim)

        #初始化编码器和卷积层
        self.encoder1 = DGCRM(args.num_nodes, args.input_dim, args.rnn_units, args.cheb_k,
                              args.embed_dim, args.num_layers)
        self.encoder2 = DGCRM(args.num_nodes, args.input_dim, args.rnn_units, args.cheb_k,
                              args.embed_dim, args.num_layers)
        self.encoder3 = GWCRM(args.num_nodes, args.input_dim, args.rnn_units, args.diffusion_steps,
                              args.embed_dim, args.num_layers)
        self.encoder4 = GWCRM(args.num_nodes, args.input_dim, args.rnn_units, args.diffusion_steps,
                              args.embed_dim, args.num_layers)

        # predictor
        self.end_conv1 = nn.Conv2d(1, args.horizon * self.output_dim, kernel_size=(1, self.hidden_dim), bias=True)
        self.end_conv2 = nn.Conv2d(1, args.horizon * self.output_dim, kernel_size=(1, self.hidden_dim), bias=True)
        self.end_conv3 = nn.Conv2d(1, args.horizon * self.output_dim, kernel_size=(1, self.hidden_dim), bias=True)
        self.end_conv4 = nn.Conv2d(1, args.horizon * self.output_dim, kernel_size=(1, self.hidden_dim), bias=True)
        self.end_conv5 = nn.Conv2d(1, args.horizon * self.output_dim, kernel_size=(1, self.hidden_dim), bias=True)
        self.end_conv6 = nn.Conv2d(1, args.horizon * self.output_dim, kernel_size=(1, self.hidden_dim), bias=True)
        self.end_conv7 = nn.Conv2d(1, args.horizon * self.output_dim, kernel_size=(1, self.hidden_dim), bias=True)

    def forward(self, source, i=2):
        # source: B, T_1, N, D 3
        # target: B, T_2, N, D

        node_embedding2 = self.node_embeddings2  # 突变信号
        node_embedding1 = self.node_embeddings1  # 突变信号
        period_signal, anomaly_signal = separate_signals(source)

        if self.use_D:
            t_i_d_data = period_signal[..., 1]
            T_i_D_emb = self.T_i_D_emb[(t_i_d_data * 288).type(torch.LongTensor)]
            node_embedding1 = torch.mul(node_embedding1, T_i_D_emb)

            t_i_d_data1 = anomaly_signal[..., 1]
            T_i_D_emb1 = self.T_i_D_emb[(t_i_d_data1 * 288).type(torch.LongTensor)]
            node_embedding2 = torch.mul(node_embedding2, T_i_D_emb1)
        if self.use_W:
            d_i_w_data = period_signal[..., 2]
            D_i_W_emb = self.D_i_W_emb[(d_i_w_data).type(torch.LongTensor)]
            node_embedding1 = torch.mul(node_embedding1, D_i_W_emb)

            d_i_w_data1 = anomaly_signal[..., 2]
            D_i_W_emb1 = self.D_i_W_emb[(d_i_w_data1).type(torch.LongTensor)]
            node_embedding2 = torch.mul(node_embedding2, D_i_W_emb1)

        node_embeddings = [node_embedding1, self.node_embeddings1]
        node_embeddings1 = [node_embedding2, self.node_embeddings2]

        source0 = period_signal[..., 0].unsqueeze(-1)
        time_in_day = period_signal[..., 1].unsqueeze(-1)
        day_in_week = period_signal[..., 2].unsqueeze(-1)
        speed1 = period_signal[..., 3].unsqueeze(-1)
        occupy1 = period_signal[..., 4].unsqueeze(-1)
        weekly1 = period_signal[..., 5].unsqueeze(-1)
        daily1 = period_signal[..., 6].unsqueeze(-1)
        recent1 = period_signal[..., 7].unsqueeze(-1)

        source00 = anomaly_signal[..., 0].unsqueeze(-1)
        speed2 = anomaly_signal[..., 3].unsqueeze(-1)
        occupy2 = anomaly_signal[..., 4].unsqueeze(-1)
        # weekly2 = anomaly_signal[..., 5].unsqueeze(-1)
        # daily2 = anomaly_signal[..., 6].unsqueeze(-1)
        # recent2 = anomaly_signal[..., 7].unsqueeze(-1)

        # _, _, A = get_adjacency_matrix_2direction("data/PeMS04/PEMS04.csv", 307, None)
        # A = torch.tensor(A).to(source.device)  # 将 numpy.ndarray 转换为 torch.Tensor

        if i == 1:
            init_state1 = self.encoder1.init_hidden(source0.shape[0])  # [2,64,307,64] 前面是2是因为有两层GRU
            output_p, _ = self.encoder1(source0, init_state1, node_embeddings,time_in_day,day_in_week,speed1,occupy1,weekly1,daily1,recent1)  # B, T, N, hidden
            # output_p, _ = self.encoder1(source0, init_state1, node_embeddings, is_peak)  # B, T, N, hidden
            output_p = self.dropout1(output_p[:, -1:, :, :])

            # CNN based predictor
            output1 = self.end_conv1(output_p)  # B, T*C, N, 1

            init_state4 = self.encoder3.init_hidden(source00.shape[0])  # [2,64,307,64] 前面是2是因为有两层GRU
            output_a, _ = self.encoder3(source00, init_state4, node_embeddings1,speed2,occupy2)  # B, T, N, hidden
            output_a = self.dropout4(output_a[:, -1:, :, :])
            # CNN based predictor
            output3 = self.end_conv5(output_a)  # B, T*C, N, 1

            return output1 + output3

        else:

            init_state1 = self.encoder1.init_hidden(source0.shape[0])  # [2,64,307,64] 前面是2是因为有两层GRU
            output, _ = self.encoder1(source0, init_state1, node_embeddings,time_in_day,day_in_week,speed1,occupy1, weekly1,daily1,recent1)  # B, T, N, hidden
            output = self.dropout1(output[:, -1:, :, :])
            output1 = self.end_conv1(output)  # B, T*C, N, 1
            source1 = self.end_conv2(output)
            source2 = source0 - source1
            init_state2 = self.encoder2.init_hidden(source2.shape[0])  # [2,64,307,64] 前面是2是因为有两层GRU
            output2, _ = self.encoder2(source2, init_state2, node_embeddings,time_in_day,day_in_week, speed1,occupy1, weekly1,daily1,recent1)  # B, T, N, hidden
            output2 = self.dropout2(output2[:, -1:, :, :])
            output2 = self.end_conv3(output2)

            init_state4 = self.encoder3.init_hidden(source00.shape[0])  # [2,64,307,64] 前面是2是因为有两层GRU
            output_a, _ = self.encoder3(source00, init_state4, node_embeddings1,speed2,occupy2)  # B, T, N, hidden
            output_a = self.dropout4(output_a[:, -1:, :, :])
            # CNN based predictor
            output3 = self.end_conv5(output_a)  # B, T*C, N, 1
            source000 = self.end_conv6(output_a)
            source3 = source00 - source000
            init_state5 = self.encoder4.init_hidden(source3.shape[0])
            output_a1, _ = self.encoder4(source3, init_state5, node_embeddings1,speed2,occupy2)
            output_a1 = self.dropout5(output_a1[:, -1:, :, :])
            output4 = self.end_conv7(output_a1)

            return output1  + output3 +output2 +output4
