import torch
import torch.nn as nn
from model.DDGCN import DGCN
torch.cuda.empty_cache()

# === PSPM Core Module: Periodic  Graph Convolutional Recurrent Unit (PGCRU) ===
# This cell is the main recurrent unit used in the PSPM (Periodic Signal Processing Module).
# It replaces the standard GRU gates and update operations with dynamic periodic graph convolutions (DGCN),
class DDGCRNCell(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim):
        super(DDGCRNCell, self).__init__()
        self.node_num = node_num
        self.hidden_dim = dim_out
        self.gate = DGCN(dim_in + self.hidden_dim, 2 * dim_out, cheb_k, embed_dim)
        self.update = DGCN(dim_in + self.hidden_dim, dim_out, cheb_k, embed_dim)
        self.embed_dim = embed_dim

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




