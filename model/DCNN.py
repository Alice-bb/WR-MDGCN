import torch
from collections import OrderedDict
torch.cuda.empty_cache()
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.cuda.empty_cache()

# === ASPM Diffusion Graph Convolution Block：Hierarchical Diffusion Graph Convolutional (HDGC) ===
# Implements HDGC + DGP based dynamic hierarchical diffusion graph convolution
# This module  captures spatiotemporal anomalies via short-/long-range diffusion and attention-based dynamic global pooling.
class DCNN(nn.Module):
    def __init__(self, dim_in, dim_out, diffusion_steps, embed_dim):
        super(DCNN, self).__init__()
        self.diffusion_steps = diffusion_steps
        self.hyperGNN_dim = 16
        self.middle_dim = 2
        self.hyperGNN_dim1 = 16
        self.middle_dim1 = 2
        self.embed_dim = embed_dim
        self.fc1 = nn.Sequential(
            OrderedDict([('fc1', nn.Linear(1, self.hyperGNN_dim1)),
                         ('sigmoid1', nn.Sigmoid()),
                         ('fc2', nn.Linear(self.hyperGNN_dim1, self.middle_dim1)),
                         ('sigmoid2', nn.Sigmoid()),
                         ('fc3', nn.Linear(self.middle_dim1, self.embed_dim))]))
        self.fc = nn.Sequential(
            OrderedDict([('fc1', nn.Linear(dim_in, self.hyperGNN_dim)),
                         ('sigmoid1', nn.Sigmoid()),
                         ('fc2', nn.Linear(self.hyperGNN_dim, self.middle_dim)),
                         ('sigmoid2', nn.Sigmoid()),
                         ('fc3', nn.Linear(self.middle_dim, self.embed_dim))]))
        self.fc2 = nn.LeakyReLU()
        self.alpha = nn.Parameter(torch.FloatTensor([0.6]))  # 正向扩散权重
        self.beta = nn.Parameter(torch.FloatTensor([0.4]))  # 逆向扩散权重
        self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, diffusion_steps, dim_in , dim_out))
        self.weights = nn.Parameter(torch.FloatTensor(diffusion_steps, dim_in, dim_out))

        self.backward_weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, diffusion_steps, dim_in , dim_out))
        self.backward_weights = nn.Parameter(torch.FloatTensor(diffusion_steps, dim_in , dim_out))
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_out))
        self.bias = nn.Parameter(torch.FloatTensor(dim_out))
        self.weights1 = nn.Parameter(torch.FloatTensor(dim_in, dim_out))

        self.global_attention_fc = nn.Linear(dim_in, 1)  # 通过线性变换来计算注意力得分
        self.global_attention_softmax = nn.Softmax(dim=1)  # 对所有节点的注意力得分进行归一化

    def forward(self, x, node_embeddings,time,day,speed,occupy):

        # Generate dynamic support matrix via filtered node embeddings ===
        N = x.shape[1]
        x_g = [x]
        node_num = node_embeddings[0].shape[1]
        supports1 = torch.eye(node_num).to(node_embeddings[0].device)
        filter = self.fc(x)  # 通过三层 FC 网络生成一个滤波器
        filter1 = self.fc1(speed)
        filter2 = self.fc1(occupy)
        nodevec = torch.tanh(torch.mul(node_embeddings[0], filter))  # [B,N,dim_in]  nodevec = torch.tanh(torch.mul(nodevec, filter1))  # [B,N,dim_in]
        nodevec = torch.tanh(torch.mul(nodevec, filter1))  # [B,N,dim_in]
        nodevec = torch.tanh(torch.mul(nodevec, filter2))  # [B,N,dim_in]
        supports2 = F.relu(torch.matmul(nodevec, nodevec.transpose(1, 2)))

        # ===  Short-range diffusion (forward) ===
        for k in range(1, self.diffusion_steps):
            x_k = torch.matmul(supports2, x_g[-1])
            x_k = self.fc2(x_k)
            x_g.append(x_k)
        x_g1 = torch.stack(x_g, dim=1)

        # === DGP for short-range diffusion ===
        node_features = x_g1.mean(dim=1)
        attention_scores = self.global_attention_fc(node_features)  # [B, N, 1]
        attention_weights = self.global_attention_softmax(attention_scores)  # [B, N, 1]
        global_feature = torch.sum(node_features * attention_weights, dim=1).unsqueeze(1).expand(-1, N, -1)
        global_feature = torch.einsum('bni,io->bno', global_feature, self.weights1)
        x_g1 = x_g1.permute(0, 2, 1, 3)
        weights = torch.einsum('nd,dfio->nfio', node_embeddings[1], self.weights_pool)

        # ===  Long-range diffusion (forward) ===
        x_g_backward = [x_g[-1]]
        for k in range(1, self.diffusion_steps):
            x_k_backward = torch.matmul(supports2.transpose(1, 2), x_g_backward[-1])  # 逆向扩散
            x_k_backward = self.fc2(x_k_backward)
            x_g_backward.append(x_k_backward)
        x_g2 = torch.stack(x_g_backward, dim=1)

        # === DGP for long-range diffusion  ===
        node_features = x_g2.mean(dim=1)  # 对扩散步骤的输出进行平均 [B, N, D]
        attention_scores = self.global_attention_fc(node_features)  # [B, N, 1]
        attention_weights = self.global_attention_softmax(attention_scores)  # [B, N, 1]
        global_feature1 = torch.sum(node_features * attention_weights, dim=1).unsqueeze(1).expand(-1, N, -1)  # [B, D]
        global_feature1 = torch.einsum('bni,io->bno', global_feature1, self.weights1)
        x_g2 = x_g2.permute(0, 2, 1, 3)
        backward_weights = torch.einsum('nd,dfio->nfio', node_embeddings[1],self.backward_weights_pool)

        # === Combine short/long-range + global features  ===
        bias = torch.matmul(node_embeddings[1], self.bias_pool)
        x_diffusion1 = torch.einsum('bnfi,nfio->bno', x_g1, weights)
        x_diffusion2 = torch.einsum('bnfi,nfio->bno',x_g2, backward_weights)
        x_diffusion = self.alpha * x_diffusion1 + self.beta * x_diffusion2 +bias

        # Final fusion: HDGC + DGP → ASPM output
        final_output = x_diffusion + global_feature +global_feature1
        return final_output

