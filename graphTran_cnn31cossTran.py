import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import torch
from torch import nn
import torch_geometric.nn as gnn
from torch_geometric.nn import GCNConv, global_max_pool
# -*- coding: utf-8 -*-

from .layers2coss import TransformerEncoderLayer
from einops import repeat

class GraphTransformerEncoder(nn.TransformerEncoder):
    def forward(self, x, edge_index, complete_edge_index,
                subgraph_node_index=None, subgraph_edge_index=None,
                subgraph_edge_attr=None, subgraph_indicator_index=None, edge_attr=None, degree=None,
                ptr=None, return_attn=False):
        #
        output = x

        for mod in self.layers:
            output = mod(output, edge_index, complete_edge_index,
                         edge_attr=edge_attr, degree=degree,
                         subgraph_node_index=subgraph_node_index,
                         subgraph_edge_index=subgraph_edge_index,
                         subgraph_indicator_index=subgraph_indicator_index,
                         subgraph_edge_attr=subgraph_edge_attr,
                         ptr=ptr,
                         return_attn=return_attn
                         )
        if self.norm is not None:
            output = self.norm(output)
        return output


class GraphTransformer(nn.Module):
    def __init__(self, in_size=78, num_class=2, d_model=128, num_heads=8,
                 dim_feedforward=512, dropout=0.0, num_layers=1,
                 batch_norm=True, abs_pe=False, abs_pe_dim=0,
                 n_output=1, n_filters=32, embed_dim=128, num_features_xt=25, output_dim=128,
                 gnn_type="gcn", se="gnn", use_edge_attr=False, num_edge_features=4,
                 in_embed=False, edge_embed=True, use_global_pool=True, max_seq_len=None,
                 global_pool='mean', **kwargs):
        super().__init__()

        # compound:gcn+Transformer
        # protein:cnn block+Transformer

        self.abs_pe = abs_pe
        self.abs_pe_dim = abs_pe_dim
        self.embedding = nn.Linear(in_features=in_size,out_features=d_model,bias=False)#（78，128，false）

        self.use_edge_attr = use_edge_attr#边的属性,false

        kwargs['edge_dim'] = None
        self.gnn_type = gnn_type#graph
        self.se = se#gnn
        #Transformer
        encoder_layer = TransformerEncoderLayer(
            d_model, num_heads, dim_feedforward, dropout, batch_norm=batch_norm,
            gnn_type=gnn_type, se=se, **kwargs)#(128,8,512,0.0,true,graph,gnn,none)
        self.encoder = GraphTransformerEncoder(encoder_layer, num_layers)#(encoder_layer,4)
        self.global_pool = global_pool#mean
        if global_pool == 'mean':

            self.pooling= gnn.global_add_pool

            self.pooling2=gnn.global_max_pool
            self.out_Lin=nn.Linear(128*2,128)
            # self.pooling = gnn.global_mean_pool



        elif global_pool == 'add':
            self.pooling = gnn.global_add_pool
        elif global_pool == 'cls':
            self.cls_token = nn.Parameter(torch.randn(1, d_model))
            self.pooling = None
        self.use_global_pool = use_global_pool#true

        self.max_seq_len = max_seq_len#none
        if max_seq_len is None:
            self.classifier = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(True),
                nn.Linear(d_model, num_class)
            )#((128,128),relu,(128,2))
        else:
            self.classifier = nn.ModuleList()
            for i in range(max_seq_len):
                self.classifier.append(nn.Linear(d_model, num_class))

        self.n_output = n_output#1



         # protein sequence branch (1d conv)cnn（3次卷积）+transformer
        self.p_embed = nn.Embedding(num_features_xt + 1, embed_dim)
        # target：512，1000 ，512，1000，128
        self.p_conv1 = nn.Conv1d(1000, 512, kernel_size=3, padding=1, stride=1)
        #输入的通道数为1000，输出的通道数为1000，卷积核的大小为3，添加到输入两侧是1，卷积的步幅是1.
        self.p_bn1 = nn.BatchNorm1d(512)#归一化函数，需要归一化的维度是1000
        # 512，1000，128
        self.p_conv2 = nn.Conv1d(512, 128, kernel_size=3, padding=1, stride=1)
        self.p_bn2 = nn.BatchNorm1d(128)

        self.p_conv3 = nn.Conv1d(in_channels=128, out_channels=n_filters, kernel_size=8)
        self.p_bn3 = nn.BatchNorm1d(32)
        # 512，32，121
        #Transformer的预处理：
        self.p_fc1 = nn.Linear(121, 128)  # 输入神经元个数为121，输出神经元个数是128
        # 512，32，128
        self.cnn_attn = TransformerEncoder(d_model=128, n_head=8, nlayers=3)
        # 512，32，128
        # Q,K,V

        # 512，32，16


        self.p_fc2 = nn.Linear(32 * 128, output_dim)
        self.p_bn4 = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

        # combined layers.py
        self.com_fc1 = nn.Linear(2* output_dim, 1024)
        self.com_bn1 = nn.BatchNorm1d(1024)
        self.com_fc2 = nn.Linear(1024, 512)
        self.com_bn2 = nn.BatchNorm1d(512)
        self.out = nn.Linear(512, self.n_output)
        #使用注意力机制提取信息
    def forward(self, data, return_attn=False):
        # get graph input
        x, edge_index, batch, edge_attr = data.x, data.edge_index, data.batch, data.edge_attr
        # get protein input
        target = data.target
        node_depth = data.node_depth if hasattr(data, "node_depth") else None

        if self.se == "khopgnn":
            subgraph_node_index = data.subgraph_node_idx
            subgraph_edge_index = data.subgraph_edge_index
            subgraph_indicator_index = data.subgraph_indicator
            subgraph_edge_attr = data.subgraph_edge_attr if hasattr(data, "subgraph_edge_attr") \
                else None
        else:
            subgraph_node_index = None
            subgraph_edge_index = None
            subgraph_indicator_index = None
            subgraph_edge_attr = None

        complete_edge_index = data.complete_edge_index if hasattr(data, 'complete_edge_index') else None
        abs_pe = data.abs_pe if hasattr(data, 'abs_pe') else None
        degree = data.degree if hasattr(data, 'degree') else None

        # x = x.long()
        output = self.embedding(x) if node_depth is None else self.embedding(x, node_depth.view(-1, ))
        #

        if self.abs_pe and abs_pe is not None:
            abs_pe = self.embedding_abs_pe(abs_pe)
            output = output + abs_pe
        if self.use_edge_attr and edge_attr is not None:
            edge_attr = self.embedding_edge(edge_attr)
            if subgraph_edge_attr is not None:
                subgraph_edge_attr = self.embedding_edge(subgraph_edge_attr)
        else:
            edge_attr = None
            subgraph_edge_attr = None

        if self.global_pool == 'cls' and self.use_global_pool:
            bsz = len(data.ptr) - 1
            if complete_edge_index is not None:
                new_index = torch.vstack((torch.arange(data.num_nodes).to(data.batch), data.batch + data.num_nodes))
                new_index2 = torch.vstack((new_index[1], new_index[0]))
                idx_tmp = torch.arange(data.num_nodes, data.num_nodes + bsz).to(data.batch)
                new_index3 = torch.vstack((idx_tmp, idx_tmp))
                complete_edge_index = torch.cat((
                    complete_edge_index, new_index, new_index2, new_index3), dim=-1)
            if subgraph_node_index is not None:
                idx_tmp = torch.arange(data.num_nodes, data.num_nodes + bsz).to(data.batch)
                subgraph_node_index = torch.hstack((subgraph_node_index, idx_tmp))
                subgraph_indicator_index = torch.hstack((subgraph_indicator_index, idx_tmp))
            degree = None
            cls_tokens = repeat(self.cls_token, '() d -> b d', b=bsz)
            output = torch.cat((output, cls_tokens))

        output = self.encoder(
            output,
            edge_index,
            complete_edge_index,
            edge_attr=edge_attr,
            degree=degree,
            subgraph_node_index=subgraph_node_index,
            subgraph_edge_index=subgraph_edge_index,
            subgraph_indicator_index=subgraph_indicator_index,
            subgraph_edge_attr=subgraph_edge_attr,
            ptr=data.ptr,
            return_attn=return_attn
        )
        # readout step
        if self.use_global_pool:
            if self.global_pool == 'cls':
                output = output[-bsz:]
            else:
                output1 = self.pooling(output, data.batch)
                output2 = self.pooling2(output, data.batch)
                output=torch.cat((output1, output2), 1)
                output=self.relu(self.out_Lin(output))
        # return self.classifier(output)

        # protein--forward
        p_embed = self.p_embed(target)
        # target: 512,1000--> 512,1000,128
        protein = self.p_conv1(p_embed)
        protein = self.p_bn1(protein)
        protein = self.relu(protein)

        protein = self.p_conv2(protein)
        protein = self.p_bn2(protein)
        protein = self.relu(protein)

        protein = self.p_conv3(protein)
        protein = self.p_bn3(protein)
        protein = self.relu(protein)


        protein = self.p_fc1(protein)
        protein = self.relu(protein)
        protein = self.dropout(protein)
        protein = self.cnn_attn(protein)#进行了transformerencoder进行得到。
        # 512,32,128
        # flatten
        protein = protein.view(-1, 32 * 128)#32*128维，-1代表动态调整这个维度上的元素个数，以保证元素的总数不变。
        protein = self.p_fc2(protein)
        protein = self.relu(protein)
        protein = self.p_bn4(protein)
        protein = self.dropout(protein)

        # 512,128

        # concat
        #将药物分子指纹和药物分子图拼接

        #三个都是256*128
        xc = torch.cat((output, protein), 1)

        xc = self.com_fc1(xc)
        xc = self.relu(xc)
        xc = self.com_bn1(xc)
        xc = self.dropout(xc)

        xc = self.com_fc2(xc)
        xc = self.relu(xc)
        xc = self.com_bn2(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        return out


















#####
# by xmm
#Attention(Q,K,V) = softmax(Q*Kt/sqrt(dk)) *V
class ScaledDotProductAttention(nn.Module):#Attention(Q,K,V) = softmax(Q*Kt/sqrt(dk)) *V
    """
    Compute 'Scaled Dot Product Attention'
    Attention(Q,K,V) = softmax(Q*Kt/sqrt(dk)) *V
    """
    """ for test 
            q = torch.randn(4, 8, 10, 64)  # (batch, n_head, seqLen, dim)
            k = torch.randn(4, 8, 10, 64)
            v = torch.randn(4, 8, 10, 64)
            mask = torch.ones(4, 8, 10, 10)
            model = ScaledDotProductAttention()
            res = model(q, k, v, mask)
            print(res[0].shape)  # torch.Size([4, 8, 10, 64])
    """

    def forward(self, query, key, value, attn_mask=None, dropout=None):
        """
        当QKV来自同一个向量的矩阵变换时称作self-attention;
        当Q和KV来自不同的向量的矩阵变换时叫soft-attention;
        url:https://www.e-learn.cn/topic/3764324
        url:https://my.oschina.net/u/4228078/blog/4497939
          :param query: (batch, n_head, seqLen, dim)  其中n_head表示multi-head的个数，且n_head*dim = embedSize
          :param key: (batch, n_head, seqLen, dim)
          :param value: (batch, n_head, seqLen, dim)
          :param mask:  (batch, n_head, seqLen,seqLen) 这里的mask应该是attn_mask；原来attention的位置为0，no attention部分为1
          :param dropout:
          """
        # (batch, n_head, seqLen,seqLen) attention weights的形状是L*L，因为每个单词两两之间都有一个weight
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, -1e9)  # 保留位置为0的值，其他位置填充极小的数

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn  # (batch, n_head, seqLen, dim)


# by xmm
class MultiHeadAttention(nn.Module):
    """
    for test :
                q = torch.randn(4, 10, 8 * 64)  # (batch, n_head, seqLen, dim)
                k = torch.randn(4, 10, 8 * 64)
                v = torch.randn(4, 10, 8 * 64)
                mask = torch.ones(4, 8, 10, 10)
                model = MultiHeadAttention(h=8, d_model=8 * 64)
                res = model(q, k, v, mask)
                print(res.shape)  # torch.Size([4, 10, 512])
    """

    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = ScaledDotProductAttention()#获得注意力矩阵

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, attn_mask=None):
        """
        :param query: (batch,seqLen, d_model)
        :param key: (batch,seqLen, d_model)
        :param value: (batch,seqLen, d_model)
        :param mask: (batch, seqLen,seqLen)
        :return: (batch,seqLen, d_model)
        """
        batch_size = query.size(0)

        # 1, Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # 2,Apply attention on all the projected vectors in batch.
        # if attn_mask:
        #     attn_mask = attn_mask.unsqueeze(1).repeat(1, self.h, 1, 1)  # (batch, n_head,seqLen,seqLen)
        x, atten = self.attention(query, key, value, attn_mask=attn_mask, dropout=self.dropout)#这个X就是公式中的Z，atten就是softmax中的那一坨内积

        # 3, "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)#view函数表示要重新定义矩阵的形状。
        return self.output_linear(x)


# by xmm
class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, dim_feedforward, dropout, activation):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, dim_feedforward)
        self.w_2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, x):
        return self.dropout(self.w_2(self.activation(self.w_1(x))))


# by xmm
class aTransformerEncoderLayer(nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    Example:
    """

    def __init__(self, d_model, n_head, dim_feedforward, dropout=0.1, activation="relu"):
        """
        :param d_model:
        :param n_head:
        :param dim_feedforward:
        :param dropout:
        :param activation: default :relu
        """

        super().__init__()
        self.self_attn = MultiHeadAttention(h=n_head, d_model=d_model, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        if activation == "relu":
            self.activation = F.relu
        if activation == "gelu":
            self.activation = F.gelu

        self.PositionwiseFeedForward = PositionwiseFeedForward(d_model=d_model, dim_feedforward=dim_feedforward,
                                                               dropout=dropout, activation=self.activation)

    def forward(self, x, atten_mask):
        """
        :param x: (batch, seqLen, em_dim)
        :param mask: attn_mask
        :return:
        """
        # add & norm 1
        attn = self.dropout(self.self_attn(x, x, x, attn_mask=atten_mask))
        x = self.norm1((x + attn))#残差连接和归一化处理

        # # add & norm 2
        x = self.norm2(x + self.PositionwiseFeedForward(x))#
        return x


class TransformerEncoder(nn.Module):
    """
    Example:
           x = torch.randn(4, 10, 128)  # (batch, seqLen, em_dim)
        model = TransformerEncoder(d_model=128, n_head=8, nlayers=3)
        res = model.forward(x)
        print(res.shape)  # torch.Size([4, 10, 128])
    """

    def __init__(self, d_model, n_head, nlayers, dim_feedforward=1024, dropout=0.1, activation="relu"):
        super(TransformerEncoder, self).__init__()
        self.encoder = nn.ModuleList([aTransformerEncoderLayer(d_model, n_head, dim_feedforward, dropout, activation)
                                      for _ in range(nlayers)])

    def forward(self, x, atten_mask=None):
        """
        :param x: input dim == out dim
        :param atten_mask: 对应源码的src_mask，没有实现src_key_padding_mask
        :return:
        """
        for layer in self.encoder:
            x = layer.forward(x, atten_mask)
        return x

