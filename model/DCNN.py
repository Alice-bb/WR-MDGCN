import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import math



torch.cuda.empty_cache()
# class GATLayer(nn.Module):
#     def __init__(self, input_dim, hidden_dim, num_heads=1):
#         super(GATLayer, self).__init__()
#         self.input_dim = input_dim
#         self.hidden_dim = hidden_dim
#         self.num_heads = num_heads
#
#         # 定义可学习的权重矩阵
#         self.W = nn.Parameter(torch.Tensor(input_dim, hidden_dim * num_heads))
#         self.a = nn.Parameter(torch.Tensor(2 * hidden_dim, 1))  # 用于计算注意力的参数
#         self.leakyrelu = nn.LeakyReLU(0.2)
#
#         nn.init.xavier_uniform_(self.W)
#         nn.init.xavier_uniform_(self.a)
#
#     def forward(self, h, adj):
#         # h 是节点的特征，adj 是邻接矩阵
#         Wh = torch.matmul(h, self.W)  # [N, input_dim] -> [N, hidden_dim * num_heads]
#         Wh = Wh.view(-1, self.num_heads, self.hidden_dim)  # [N, num_heads, hidden_dim]
#
#         # 计算注意力系数
#         a_input = torch.cat([Wh.unsqueeze(1).repeat(1, Wh.size(0), 1, 1),
#                              Wh.unsqueeze(0).repeat(Wh.size(0), 1, 1, 1)], dim=-1)  # [N, N, num_heads, 2*out_features]
#         print("000000000000",a_input.shape)
#
#         e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(-1))  # [N, N, num_heads]
#         print("11111111111",e.shape)
#         attention = F.softmax(e.masked_fill(adj == 0, -1e9), dim=1)  # [N, N, num_heads]
#         h_prime = torch.einsum('bnm,bmd->bnd', attention, Wh)  # 通过注意力加权计算输出
#         return h_prime

#
# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, max_len=5000):
#         super(PositionalEncoding, self).__init__()
#
#         # 确保 d_model 是偶数，如果是奇数就自动增加1，防止广播问题
#         if d_model % 2 != 0:
#             d_model += 1
#
#         self.encoding = torch.zeros(max_len, d_model)
#         self.encoding.requires_grad = False
#
#         pos = torch.arange(0, max_len).unsqueeze(1).float()
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
#
#         self.encoding[:, 0::2] = torch.sin(pos * div_term)
#         self.encoding[:, 1::2] = torch.cos(pos * div_term)
#
#     def forward(self, x):
#         seq_len = x.size(1)
#         # 确保 self.encoding 的尺寸与 x 匹配，裁剪到与 x 相同的维度
#         x = x + self.encoding[:seq_len, :x.size(2)].to(x.device)
#         return x


# class SpatialSelfAttention(nn.Module):
#     def __init__(self, embed_dim):
#         super(SpatialSelfAttention, self).__init__()
#         self.query = nn.Linear(embed_dim, embed_dim)
#         self.key = nn.Linear(embed_dim, embed_dim)
#         self.value = nn.Linear(embed_dim, embed_dim)
#         self.softmax = nn.Softmax(dim=-1)
#
#     def forward(self, x):
#         # x: [B, N, embed_dim], adj: [B, N, N] 邻接矩阵
#         Q = self.query(x)  # [B, N, embed_dim]
#         K = self.key(x)    # [B, N, embed_dim]
#         V = self.value(x)  # [B, N, embed_dim]
#
#         attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (x.size(-1) ** 0.5)  # [B, N, N]
#
#         # 使用邻接矩阵进行遮罩，将无连接的节点赋予很小的权重
#         # attention_scores = attention_scores.masked_fill(adj == 0, -1e9)
#
#         attention_probs = self.softmax(attention_scores)  # [B, N, N]
#         context = torch.matmul(attention_probs, V)  # [B, N, embed_dim]
#         return context
#
# class MultiHeadSpatialAttention(nn.Module):
#     def __init__(self, embed_dim, num_heads=1):
#         super(MultiHeadSpatialAttention, self).__init__()
#         assert embed_dim % num_heads == 0, "嵌入维度必须能被注意力头数量整除"
#         self.num_heads = num_heads
#         self.head_dim = embed_dim // num_heads
#         self.embed_dim = embed_dim
#         # 为每个注意力头分别定义 query、key、value
#         self.query = nn.Linear(embed_dim, embed_dim)
#         self.key = nn.Linear(embed_dim, embed_dim)
#         self.value = nn.Linear(embed_dim, embed_dim)
#         # 最终的线性变换，用于将多头输出合并
#         self.out = nn.Linear(embed_dim, embed_dim)
#         self.softmax = nn.Softmax(dim=-1)
#
#
#     def forward(self, x):
#         B, N, embed_dim = x.shape
#         # 线性变换，分成多头
#         Q = self.query(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, N, head_dim]
#         K = self.key(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
#         V = self.value(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
#         # 计算注意力得分
#         attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [B, num_heads, N, N]
#         # softmax 获得注意力权重
#         attention_probs = self.softmax(attention_scores)  # [B, num_heads, N, N]
#         # 通过注意力权重加权求和
#         context = torch.matmul(attention_probs, V)  # [B, num_heads, N, head_dim]
#         # 将多头合并
#         context = context.transpose(1, 2).contiguous().view(B, N, embed_dim)  # [B, N, embed_dim]
#         # 通过最终的线性层进行变换
#         out = self.out(context)  # [B, N, embed_dim]
#         return out


# class AttentionAdjacency(nn.Module):
#     def __init__(self, embed_dim, node_num):
#         super(AttentionAdjacency, self).__init__()
#         self.embed_dim = embed_dim
#         self.node_num = node_num
#
#         # 定义节点嵌入投影层
#         self.node_proj = nn.Linear(embed_dim, embed_dim)
#
#         # 可学习的注意力参数
#         self.attn_param = nn.Parameter(torch.Tensor(node_num, node_num))
#         nn.init.xavier_uniform_(self.attn_param)  # 参数初始化
#
#     def forward(self, node_embeddings, adj_matrix):
#         """
#         :param node_embeddings: 节点嵌入特征 [B, N, D]
#         :param adj_matrix: 原始邻接矩阵 [N, N]
#         :return: 调整后的邻接矩阵 [B, N, N]
#         """
#         # 对节点嵌入进行投影
#         projected = self.node_proj(node_embeddings)  # [B, N, D]
#
#         # 计算注意力得分
#         attn_scores = torch.matmul(projected, projected.transpose(-1, -2))  # [B, N, N]
#         attn_scores = attn_scores / (self.embed_dim ** 0.5)  # 缩放
#
#         # 添加可学习的偏置
#         attn_scores = attn_scores + self.attn_param.unsqueeze(0)  # [B, N, N]
#
#         # 归一化权重
#         attn_weights = F.softmax(attn_scores, dim=-1)  # [B, N, N]
#
#         # 动态调整邻接矩阵
#         adjusted_adj = adj_matrix * attn_weights  # [B, N, N]
#
#         return adjusted_adj


# 单向  ---------效果好，先做实验，后面再看怎么加双向！！！

import torch
import torch.nn as nn
import torch.nn.functional as F

#
# class GlobalAttentionPooling(nn.Module):
#     def __init__(self, dim_in, dim_out,num_heads=1):
#         super(GlobalAttentionPooling, self).__init__()
#         self.num_heads = num_heads
#         # 用于计算节点特征变换的线性层
#         self.node_fc = nn.Linear(dim_in, dim_in)
#         # 用于计算节点的注意系数的线性层
#         self.attention_fc = nn.Linear(dim_in, 1)
#
#         self.dim_in = dim_in
#         # 多头注意力机制
#         self.multihead_attention = nn.MultiheadAttention(dim_in, num_heads)
#
#         # 额外的线性层，将 `gap_output` 转换为 dim_out
#         self.fc_gap_to_dim_out = nn.Linear(dim_in, dim_out)

    # def forward(self, x, node_embeddings):
    #     # 计算节点特征的变换
    #     transformed_x = self.node_fc(x)  # [B, N, dim_in]
    #     # 计算节点的注意系数
    #     attention_scores = self.attention_fc(transformed_x)  # [B, N, 1]
    #     attention_weights = F.softmax(attention_scores, dim=1)  # [B, N, 1]
    #
    #     # 进行全局加权池化
    #     global_rep = torch.sum(attention_weights * x, dim=1)  # [B, dim_in]
    #
    #     # 使用多头注意力进行进一步的特征学习
    #     global_rep = global_rep.unsqueeze(0)  # [1, B, dim_in]
    #     x = x.transpose(0, 1)  # [N, B, dim_in]
    #     attn_output, _ = self.multihead_attention(global_rep, x, x)  # [1, B, dim_in]
    #     attn_output = attn_output.squeeze(0)  # [B, dim_in]
    #
    #     # 将 `attn_output` 映射到 `dim_out`
    #     attn_output = self.fc_gap_to_dim_out(attn_output)  # [B, dim_out]
    #
    #     return attn_output

torch.cuda.empty_cache()
# from data.PeMS08.generate_adj_mx import get_adjacency_matrix_2direction
from data.PeMS04.generate_adj_mx import get_adjacency_matrix_2direction

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

