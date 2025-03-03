import torch
import torch.nn as nn
from model.DCNN import DCNN
torch.cuda.empty_cache()


class GWCCell(nn.Module):  #这个模块只进行GRU内部的更新，所以需要修改的是AGCN里面的东西
    def __init__(self, node_num, dim_in, dim_out, diffusion_steps, embed_dim):
        super(GWCCell, self).__init__()
        self.node_num = node_num
        self.hidden_dim = dim_out
        self.gate = DCNN(dim_in + self.hidden_dim, 2 * dim_out, diffusion_steps, embed_dim)
        self.update = DCNN(dim_in + self.hidden_dim, dim_out, diffusion_steps, embed_dim)

    def forward(self, x, state, node_embeddings,speed,occupy):
        #x: B, num_nodes, input_dim
        #state: B, num_nodes, hidden_dim
        state = state.to(x.device)

        input_and_state = torch.cat((x, state), dim=-1)
        z_r = torch.sigmoid(self.gate(input_and_state, node_embeddings,speed,occupy))
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        candidate = torch.cat((x, z*state), dim=-1)
        hc = torch.tanh(self.update(candidate, node_embeddings,speed,occupy))
        h = r*state + (1-r)*hc
        return h

    def init_hidden_state(self, batch_size):
        return torch.zeros(batch_size, self.node_num, self.hidden_dim)
