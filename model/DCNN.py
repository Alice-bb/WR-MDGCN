import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import math
torch.cuda.empty_cache()


class DCNN(nn.Module):
    def __init__(self, dim_in, dim_out, diffusion_steps, embed_dim):
        super(DCNN, self).__init__()
        self.diffusion_steps = diffusion_steps
        self.hyperGNN_dim = 16
        self.middle_dim = 2
        self.hyperGNN_dim1 = 16
        self.middle_dim1 = 2
        self.embed_dim = embed_dim


        # 使用三层线性层来生成滤波器
        self.fc1 = nn.Sequential(  # 疑问，这里为什么要用三层linear来做，为什么激活函数是sigmoid
            OrderedDict([('fc1', nn.Linear(1, self.hyperGNN_dim1)),
                         # ('sigmoid1', nn.ReLU()),
                         ('sigmoid1', nn.Sigmoid()),
                         ('fc2', nn.Linear(self.hyperGNN_dim1, self.middle_dim1)),
                         # ('sigmoid1', nn.ReLU()),
                         ('sigmoid2', nn.Sigmoid()),
                         ('fc3', nn.Linear(self.middle_dim1, self.embed_dim))]))
        self.fc = nn.Sequential(  # 疑问，这里为什么要用三层linear来做，为什么激活函数是sigmoid
            OrderedDict([('fc1', nn.Linear(dim_in, self.hyperGNN_dim)),
                         # ('sigmoid1', nn.ReLU()),
                         ('sigmoid1', nn.Sigmoid()),
                         ('fc2', nn.Linear(self.hyperGNN_dim, self.middle_dim)),
                         # ('sigmoid1', nn.ReLU()),
                         ('sigmoid2', nn.Sigmoid()),
                         ('fc3', nn.Linear(self.middle_dim, self.embed_dim))]))
        self.fc2 = nn.LeakyReLU()


        # 加权参数
        self.alpha = nn.Parameter(torch.FloatTensor([0.6]))  # 正向扩散权重
        self.beta = nn.Parameter(torch.FloatTensor([0.4]))  # 逆向扩散权重
        self.back = nn.Parameter(torch.FloatTensor([0.8]))  # 逆向扩散权重


        self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, diffusion_steps, dim_in , dim_out))
        self.weights = nn.Parameter(torch.FloatTensor(diffusion_steps, dim_in, dim_out))

        self.backward_weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, diffusion_steps, dim_in , dim_out))
        self.backward_weights = nn.Parameter(torch.FloatTensor(diffusion_steps, dim_in , dim_out))
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_out))
        self.bias = nn.Parameter(torch.FloatTensor(dim_out))
        self.weights1 = nn.Parameter(torch.FloatTensor(dim_in, dim_out))




        # 这里加入一个简单的全局注意力池化层
        self.global_attention_fc = nn.Linear(dim_in, 1)  # 通过线性变换来计算注意力得分
        self.global_attention_softmax = nn.Softmax(dim=1)  # 对所有节点的注意力得分进行归一化
        #self.norm_layer = nn.LayerNorm(dim_in)  # dim_out 指输出特征维度



    def forward(self, x, node_embeddings,speed,occupy):
      #  x= self.spa_attn(x)
      #   x = self.spatial_local_attention(x)  # [B, N, dim_out]
        N = x.shape[1]
        x_g = [x]


        # output shape [B, N, C]
        node_num = node_embeddings[0].shape[1]
        supports1 = torch.eye(node_num).to(node_embeddings[0].device)

        filter = self.fc(x)  # 通过三层 FC 网络生成一个滤波器
        filter1 = self.fc1(speed)
        filter2 = self.fc1(occupy)
        # 使用 tanh 激活生成节点向量 nodevec
        # 计算注意力权重矩阵
        nodevec = torch.tanh(torch.mul(node_embeddings[0], filter))  # [B,N,dim_in]  nodevec = torch.tanh(torch.mul(nodevec, filter1))  # [B,N,dim_in]
        nodevec = torch.tanh(torch.mul(nodevec, filter1))  # [B,N,dim_in]
        nodevec = torch.tanh(torch.mul(nodevec, filter2))  # [B,N,dim_in]


        supports2 = F.relu(torch.matmul(nodevec, nodevec.transpose(2, 1)))


        for k in range(1, self.diffusion_steps):

            x_k = torch.matmul(supports2, x_g[-1])  # 扩散步骤 k

            x_k = self.fc2(x_k)  # 或者使用 torch.leaky_relu(x_k) 替代 ReLU
            x_g.append(x_k)
        x_g1 = torch.stack(x_g, dim=1)

        node_features = x_g1.mean(dim=1)  # 对扩散步骤的输出进行平均 [B, N, D]
        attention_scores = self.global_attention_fc(node_features)  # [B, N, 1]
        attention_weights = self.global_attention_softmax(attention_scores)  # [B, N, 1]
        global_feature = torch.sum(node_features * attention_weights, dim=1).unsqueeze(1).expand(-1, N, -1)
        global_feature = torch.einsum('bni,io->bno', global_feature, self.weights1)



        x_g1 = x_g1.permute(0, 2, 1, 3)
        weights = torch.einsum('nd,dfio->nfio', node_embeddings[1], self.weights_pool)



        x_g_backward = [x_g[-1]]
        for k in range(1, self.diffusion_steps):
            # 确保 x_g_backward[-1] 的形状为 [B, N, K]，并与 supports2 的转置相匹配
            x_k_backward = torch.matmul(supports2.transpose(1, 2), x_g_backward[-1])  # 逆向扩散
            x_k_backward = self.fc2(x_k_backward)

            x_g_backward.append(x_k_backward)
            # 确保 x_g_backward 的维度匹配权重
        x_g2 = torch.stack(x_g_backward, dim=1)

        node_features = x_g2.mean(dim=1)  # 对扩散步骤的输出进行平均 [B, N, D]
        attention_scores = self.global_attention_fc(node_features)  # [B, N, 1]
        attention_weights = self.global_attention_softmax(attention_scores)  # [B, N, 1]
        global_feature1 = torch.sum(node_features * attention_weights, dim=1).unsqueeze(1).expand(-1, N, -1)  # [B, D]
        global_feature1 = torch.einsum('bni,io->bno', global_feature1, self.weights1)

        x_g2 = x_g2.permute(0, 2, 1, 3)
        backward_weights = torch.einsum('nd,dfio->nfio', node_embeddings[1],self.backward_weights_pool)

        bias = torch.matmul(node_embeddings[1], self.bias_pool)
        x_diffusion1 = torch.einsum('bnfi,nfio->bno', x_g1, weights)
        x_diffusion2 = torch.einsum('bnfi,nfio->bno',x_g2, backward_weights)
        # 使用 alpha 和 beta 进行加权平均
        x_diffusion = self.alpha * x_diffusion1 + self.beta * x_diffusion2 +bias


        final_output = x_diffusion + global_feature + global_feature1

        return final_output

