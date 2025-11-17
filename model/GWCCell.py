import torch
import torch.nn as nn
from model.DCNN import DCNN  # Dynamic Diffusion Graph Convolution module
torch.cuda.empty_cache()

# === ASPM Core Module:Dynamic Hierarchical Diffusion Graph Convolutional Recurrent Unit (HDGCRU) ===
# This module is the fundamental unit of ASPM (Anomalous Signal Processing Module)
# It replaces the standard GRU gates and update operations with dynamic periodic graph convolutions (DCNN)
class GWCCell(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, diffusion_steps, embed_dim):
        super(GWCCell, self).__init__()
        self.node_num = node_num
        self.hidden_dim = dim_out
        self.gate = DCNN(dim_in + self.hidden_dim, 2 * dim_out, diffusion_steps, embed_dim)
        self.update = DCNN(dim_in + self.hidden_dim, dim_out, diffusion_steps, embed_dim)

    def forward(self, x, state, node_embeddings,time,day,speed,occupy):
        state = state.to(x.device)
        input_and_state = torch.cat((x, state), dim=-1)
        z_r = torch.sigmoid(self.gate(input_and_state, node_embeddings,time,day,speed,occupy))
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        candidate = torch.cat((x, z*state), dim=-1)
        hc = torch.tanh(self.update(candidate, node_embeddings,time,day,speed,occupy))
        h = r*state + (1-r)*hc
        return h

    def init_hidden_state(self, batch_size):
        return torch.zeros(batch_size, self.node_num, self.hidden_dim)
