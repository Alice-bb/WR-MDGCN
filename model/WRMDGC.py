
from model.DDGCRNCell import DDGCRNCell

from model.GWCCell import GWCCell
import numpy as np
import torch
import torch.nn as nn
torch.cuda.empty_cache()
import pywt


def separate_signals(x, wavelet='sym3', level=4):
    # === SSM① (Signal Separation Module in the paper) ===
    # This function corresponds to "wavelet‑based decomposition followed by residual refinement"
    # Split input into low‑frequency (periodic) & high‑frequency (anomaly) signals
    period_signal = torch.zeros_like(x)
    anomaly_signal = torch.zeros_like(x)

    for i in range(1, x.shape[1]):
        coeffs = pywt.wavedec(x[:, i, :, 0].cpu().numpy(), wavelet, level=level, mode='per')
        period_coeffs = [coeffs[0]] + [np.zeros_like(c) for c in coeffs[1:]]  # 高频为0，保留低频（周期性）
        anomaly_coeffs = [np.zeros_like(coeffs[0])] + coeffs[1:]  # 低频为0，保留高频（突变）
        period_reconstructed = torch.tensor(pywt.waverec(period_coeffs, wavelet, mode='per')).to(x.device)
        anomaly_reconstructed = torch.tensor(pywt.waverec(anomaly_coeffs, wavelet, mode='per')).to(x.device)
        if period_reconstructed.shape[1] != x[:, i, :, :].shape[1]:
            period_reconstructed = period_reconstructed[:, :x[:, i, :,:].shape[1]]
        if anomaly_reconstructed.shape[1] != x[:, i, :, :].shape[1]:
            anomaly_reconstructed = anomaly_reconstructed[:, :x[:, i, :,:].shape[1]]

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

    return period_signal, anomaly_signal



class GWCRM(nn.Module):
    # === ASPM (Anomaly Signal Processing Module) ===
    # This module corresponds to the anomaly branch in the paper
    # Hierarchical diffusive modeling of anomaly components
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

    def forward(self, x, init_state, node_embeddings,time, day,speed,occupy):
        #shape of x: (B, T, N, D)
        #shape of init_state: (num_layers, B, N, hidden_dim)
        assert x.shape[2] == self.node_num and x.shape[3] == self.input_dim
        seq_length = x.shape[1]
        current_inputs = x
        output_hidden = []
        for i in range(self.num_layers):

            state = init_state[i]
            inner_states = []
            for t in range(seq_length):
                state = self.GWCRM_cells[i](current_inputs[:, t, :, :], state, [node_embeddings[0][:, t, :, :],
                            node_embeddings[1]],time[:, t, :, :], day[:, t, :, :],speed[:, t, :, :],occupy[:, t, :, :])#state=[batch,steps,nodes,input_dim]
                inner_states.append(state)
            output_hidden.append(state)
            current_inputs = torch.stack(inner_states, dim=1)

        return current_inputs, output_hidden

    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.GWCRM_cells[i].init_hidden_state(batch_size))
        return torch.stack(init_states, dim=0)      #(num_layers, B, N, hidden_dim)

class DGCRM(nn.Module):
    # === PSPM (Periodic Signal Processing Module) ===
    # This module corresponds to the periodic branch in the paper
    # Captures periodic dynamics + includes Global Memory Mechanism (GMM)
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim, num_layers=1):
        super(DGCRM, self).__init__()
        assert num_layers >= 1, 'At least one DCRNN layer in the Encoder.'
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers
        self.embed_dim = embed_dim
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

    def forward(self, x, init_state, node_embeddings, time, day, speed, occupy,weekly, daily, recent):
        assert x.shape[2] == self.node_num and x.shape[3] == self.input_dim
        seq_length = x.shape[1]
        batch_size = x.shape[0]
        global_memory = torch.zeros(batch_size, self.node_num, self.DGCRM_cells[0].hidden_dim).to(x.device)

        # ===  Multi-Scale Periodic Input Enhancement mechanism (weekly/daily/recent) ===
        importance_scores0 = torch.abs(x - weekly)  # Shape: [B, T, N, 1]
        importance_scores0 = importance_scores0.mean(dim=2, keepdim=True)  # Reduce over node dimension [B, T, 1, 1]
        # Normalize importance scores to get attention weights
        importance_weights0 = importance_scores0 / (
                    importance_scores0.sum(dim=1, keepdim=True) + 1e-6)  # Shape: [B, T, 1, 1]
        weighted_input0 = x * importance_weights0  # Shape: [B, T, N, 1]

        importance_scores1 = torch.abs(x - daily)  # Shape: [B, T, N, 1]
        importance_scores1 = importance_scores1.mean(dim=2, keepdim=True)  # Reduce over node dimension [B, T, 1, 1]
        # Normalize importance scores to get attention weights
        importance_weights1 = importance_scores1 / (
                importance_scores1.sum(dim=1, keepdim=True) + 1e-6)  # Shape: [B, T, 1, 1]
        weighted_input1 = x * importance_weights1  # Shape: [B, T, N, 1]

        importance_scores2 = torch.abs(x - recent)  # Shape: [B, T, N, 1]
        importance_scores2 = importance_scores2.mean(dim=2, keepdim=True)  # Reduce over node dimension [B, T, 1, 1]
        importance_weights2 = importance_scores2 / (
                importance_scores2.sum(dim=1, keepdim=True) + 1e-6)  # Shape: [B, T, 1, 1]
        # Apply importance weights to the input
        weighted_input2 = x * importance_weights2  # Shape: [B, T, N, 1]
        current_inputs = x + weighted_input0 + weighted_input1 + weighted_input2

        output_hidden = []
        for i in range(self.num_layers):
            state = init_state[i]
            inner_states = []

            # ===  Global Memory Mechanism (GMM) to retain long-term periodic patterns ===
            for t in range(seq_length):
                state = self.DGCRM_cells[i](current_inputs[:, t, :, :], state,
                                            [node_embeddings[0][:, t, :, :], node_embeddings[1]],
                                            time[:, t, :, :], day[:, t, :, :],speed[:, t, :, :], occupy[:, t, :, :])
                time_input = torch.cat([time[:, t, :, :], day[:, t, :, :]], dim=-1)  # 结合时间信息
                time_weight = self.time_weight_generator(time_input)  # [B, N, 1]：每个时间步的权重

                enhanced_state = time_weight * state + (1 - time_weight) * global_memory
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


class WRMDGC(nn.Module):
    # === WR-MDGCN Main Architecture ===
    # This class implements the overall model architecture of WR-MDGCN
    # Integrates both periodic and anomalous branches with shared interface

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
        self.node_embeddings1 = nn.Parameter(torch.randn(self.num_nodes, args.embed_dim), requires_grad=True)
        self.node_embeddings2 = nn.Parameter(torch.randn(self.num_nodes, args.embed_dim), requires_grad=True)

        self.T_i_D_emb = nn.Parameter(torch.empty(288, args.embed_dim))
        self.D_i_W_emb = nn.Parameter(torch.empty(7, args.embed_dim))


        # === Encoder Initialization ===
        # encoder1: periodic encoder for original signal
        self.encoder1 = DGCRM(args.num_nodes, args.input_dim, args.rnn_units, args.cheb_k,
                              args.embed_dim, args.num_layers)
        # encoder2: periodic encoder for residual signal (after subtracting prediction)
        self.encoder2 = DGCRM(args.num_nodes, args.input_dim, args.rnn_units, args.cheb_k,
                              args.embed_dim, args.num_layers)
        # encoder3: anomalous encoder for original signal
        self.encoder3 = GWCRM(args.num_nodes, args.input_dim, args.rnn_units, args.diffusion_steps,
                              args.embed_dim, args.num_layers)
        # encoder4: anomalous encoder for residual signal (after subtracting prediction)
        self.encoder4 = GWCRM(args.num_nodes, args.input_dim, args.rnn_units, args.diffusion_steps,
                              args.embed_dim, args.num_layers)

        # predictor
        self.end_conv1 = nn.Conv2d(1, args.horizon * self.output_dim, kernel_size=(1, self.hidden_dim), bias=True)
        self.end_conv2 = nn.Conv2d(1, args.horizon * self.output_dim, kernel_size=(1, self.hidden_dim), bias=True)
        self.end_conv3 = nn.Conv2d(1, args.horizon * self.output_dim, kernel_size=(1, self.hidden_dim), bias=True)
        self.end_conv4 = nn.Conv2d(1, args.horizon * self.output_dim, kernel_size=(1, self.hidden_dim), bias=True)
        self.end_conv5 = nn.Conv2d(1, args.horizon * self.output_dim, kernel_size=(1, self.hidden_dim), bias=True)
        self.end_conv6 = nn.Conv2d(1, args.horizon * self.output_dim, kernel_size=(1, self.hidden_dim), bias=True)

    def forward(self, source, i=2, epoch=None):
        # source: B, T_1, N, D 3
        # target: B, T_2, N, D

        node_embedding2 = self.node_embeddings2
        node_embedding1 = self.node_embeddings1
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

        if i == 1:
            init_state1 = self.encoder1.init_hidden(source0.shape[0])
            output_p, _ = self.encoder1(source0, init_state1, node_embeddings,time_in_day,day_in_week,speed1,weekly1,daily1,recent1)  # B, T, N, hidden
            output_p = self.dropout1(output_p[:, -1:, :, :])
            # CNN based predictor
            output1 = self.end_conv1(output_p)  # B, T*C, N, 1
            init_state4 = self.encoder3.init_hidden(source00.shape[0])
            output_a, _ = self.encoder3(source00, init_state4, node_embeddings1,speed2)  # B, T, N, hidden
            output_a = self.dropout4(output_a[:, -1:, :, :])
            # CNN based predictor
            output3 = self.end_conv5(output_a)  # B, T*C, N, 1

            return output1 + output3

        else:

            # === Periodic branch  ===
            # Initialize hidden state for periodic encoder 1
            init_state1 = self.encoder1.init_hidden(source0.shape[0])
            # Encode original periodic signal with time and external features
            output, _ = self.encoder1(source0, init_state1, node_embeddings, time_in_day, day_in_week, speed1, occupy1,
                                      weekly1, daily1, recent1)
            # Take final timestep and apply dropout
            output = self.dropout1(output[:, -1:, :, :])
            # CNN-based predictor to produce first output
            output1 = self.end_conv1(output)  # B, T*C, N, 1
            # Predict intermediate value for residual modeling
            source1 = self.end_conv2(output)
            # Compute residual signal
            source2 = source0 - source1

            # === Periodic branch (residual modeling) ===
            # Initialize hidden state for periodic encoder 2
            init_state2 = self.encoder2.init_hidden(source2.shape[0])
            # Encode residual periodic signal
            output2, _ = self.encoder2(source2, init_state2, node_embeddings, time_in_day, day_in_week, speed1, occupy1,
                                       weekly1, daily1, recent1)
            # Take final timestep and apply dropout
            output2 = self.dropout2(output2[:, -1:, :, :])
            # CNN-based predictor on residual output
            output2 = self.end_conv3(output2)

            # === Anomalous branch ===
            # Initialize hidden state for anomalous encoder 1
            init_state3 = self.encoder3.init_hidden(source00.shape[0])
            # Encode original anomalous signal with time and external features
            output_a, _ = self.encoder3(source00, init_state3, node_embeddings1, time_in_day, day_in_week, speed2,
                                        occupy2)
            # Take final timestep and apply dropout
            output_a = self.dropout3(output_a[:, -1:, :, :])
            # CNN-based predictor to produce third output
            output3 = self.end_conv4(output_a)  # B, T*C, N, 1
            # Predict intermediate value for residual modeling
            source3 = self.end_conv5(output_a)
            # Compute residual anomalous signal
            source4 = source00 - source3

            # === Anomalous branch (residual modeling) ===
            # Initialize hidden state for anomalous encoder 2
            init_state4 = self.encoder4.init_hidden(source4.shape[0])
            # Encode residual anomalous signal
            output_a1, _ = self.encoder4(source4, init_state4, node_embeddings1, time_in_day, day_in_week, speed2,
                                         occupy2)
            # Take final timestep and apply dropout
            output_a1 = self.dropout4(output_a1[:, -1:, :, :])
            # CNN-based predictor on residual output
            output4 = self.end_conv6(output_a1)

            # === Final output fusion ===
            # Sum all branches' predictions to form final forecast
            return output1 + output2 + output3 + output4

