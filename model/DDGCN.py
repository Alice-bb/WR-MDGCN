import torch
torch.cuda.empty_cache()
import torch.nn.functional as F
import torch.nn as nn
from collections import OrderedDict #OrderedDict 是一个有序字典，用于定义网络层的顺序

# === PSPM Core Module: Dynamic Periodic Graph Convolution (Θ_Pe) ===
# This module serves as the periodic graph convolution component of PSPM.
# It dynamically constructs the spatial graph using dynamic periodic graph and applies node-adaptive graph convolutions to capture dynamic periodic correlations.

class DGCN(nn.Module):
    def __init__(self, dim_in, dim_out, cheb_k, embed_dim):
        super(DGCN, self).__init__()
        self.cheb_k = cheb_k
        self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, cheb_k, dim_in , dim_out))
        self.weights = nn.Parameter(torch.FloatTensor(cheb_k,dim_in, dim_out))
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_out))
        self.bias = nn.Parameter(torch.FloatTensor(dim_out))
        self.hyperGNN_dim = 16
        self.middle_dim = 2
        self.hyperGNN_dim1 = 16
        self.middle_dim1 = 2
        self.embed_dim = embed_dim
        self.fc=nn.Sequential(
                OrderedDict([('fc1', nn.Linear(dim_in, self.hyperGNN_dim)),
                             ('sigmoid1', nn.Sigmoid()),
                             ('fc2', nn.Linear(self.hyperGNN_dim, self.middle_dim)),
                             ('sigmoid2', nn.Sigmoid()),
                             ('fc3', nn.Linear(self.middle_dim, self.embed_dim))]))

        self.fc1 = nn.Sequential(
            OrderedDict([('fc1', nn.Linear(1, self.hyperGNN_dim1)),
                         ('sigmoid1', nn.Sigmoid()),
                         ('fc2', nn.Linear(self.hyperGNN_dim1, self.middle_dim1)),
                         ('sigmoid2', nn.Sigmoid()),
                         ('fc3', nn.Linear(self.middle_dim1, self.embed_dim))]))

    def forward(self, x, node_embeddings, time, day, speed, occupy):
        # ===  Prepare graph structure ===
        node_num = node_embeddings[0].shape[1]
        supports1 = torch.eye(node_num).to(node_embeddings[0].device)  # Identity adjacency (static)

        # ===  Generate node-wise filters from features and context ===
        filter = self.fc(x)           # From input feature x: shape [B, N, embed_dim]
        filter1 = self.fc1(speed)     # From speed: shape [B, N, embed_dim]
        filter2 = self.fc1(occupy)    # From occupancy: shape [B, N, embed_dim]

        # ===  Node-adaptive embedding transformation with contextual filters ===
        nodevec = torch.tanh(torch.mul(node_embeddings[0], filter))      # [B, N, E]
        nodevec = torch.tanh(torch.mul(nodevec, filter1))                # Apply filter1
        nodevec = torch.tanh(torch.mul(nodevec, filter2))                # Apply filter2

        # ===  Construct adaptive adjacency matrix (learned Laplacian) ===
        supports2 = DGCN.get_laplacian(F.relu(torch.matmul(nodevec, nodevec.transpose(2, 1))), supports1)

        # ===  Graph signal propagation  ===
        x_g1 = torch.einsum("nm,bmc->bnc", supports1, x)  # Static support
        x_g2 = torch.einsum("bnm,bmc->bnc", supports2, x) # Dynamic support
        x_g = torch.stack([x_g1, x_g2], dim=1)            # [B, K=2, N, C_in]

        # ===  Node-adaptive weights & bias generation ===
        weights = torch.einsum('nd,dkio->nkio', node_embeddings[1], self.weights_pool)  # [N, K, C_in, C_out]
        bias = torch.matmul(node_embeddings[1], self.bias_pool)  # [N, C_out]

        # ===  Graph convolution with node-wise weights ===
        x_g = x_g.permute(0, 2, 1, 3)  # [B, N, K, C_in]
        x_gconv = torch.einsum('bnki,nkio->bno', x_g, weights) + bias  # [B, N, C_out]

        return x_gconv

    @staticmethod
    def get_laplacian(graph, I, normalize=True):
        if normalize:
            D = torch.diag_embed(torch.sum(graph, dim=-1) ** (-1 / 2))
            L = torch.matmul(torch.matmul(D, graph), D)
        else:
            graph = graph + I
            D = torch.diag_embed(torch.sum(graph, dim=-1) ** (-1 / 2))
            L = torch.matmul(torch.matmul(D, graph), D)
        return L

