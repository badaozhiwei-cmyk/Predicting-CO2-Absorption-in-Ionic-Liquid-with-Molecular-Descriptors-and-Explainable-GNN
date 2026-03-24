import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,GATv2Conv,MessagePassing
from torch_geometric.nn import global_max_pool,global_mean_pool,global_add_pool
from torch_geometric.utils import add_self_loops, degree, softmax

# global argument
num_atom_type = 119 # including the extra mask tokens
num_Hbrid = 8
num_Aro = 2
num_degree = 7
num_charge = 3

num_bond_type = 5 # including aromatic and self-loop edge
num_bond_isAromatic = 2
num_bond_isInRing = 2

# GCN
class IL_Net_GCN(torch.nn.Module):
    def __init__(self, args):
        super(IL_Net_GCN,self).__init__()
        self.args = args
        args 字典通过键值索引
        n_features = self.args['n_features']
        self.l1 = GCNConv(n_features, 512, normalize = True)
        self.l2 = GCNConv(512, 1024, normalize=True)
        self.l3 = GCNConv(1024, 1024, normalize=True)
        self.l4 = GCNConv(1024, 512, normalize=True)

        self.l5 = nn.Sequential(
            nn.Linear(514, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(p=0.4),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.3),

            nn.Linear(512, 1),
        )

        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=args['dropout_rate'])

    def extract(self,x,batch):
batch 是一维张量，记录了每个节点属于哪个图（例：[0, 0, 0, 1, 1, 2, 2, 2, 2]）。
torch.unique 统计出 batch 中每个独立图的节点总数，存入 count（例：图0有3个节点，图1有2个节点，图2有4个节点，count 为 [3, 2, 4]）。  
        
        output, count= torch.unique(batch, return_counts=True)
        count = count.tolist()
累加计算终止索引（for i in count: 循环）
l 列表用于存储大图中每个独立图的切片终止位置。
根据上述例子，循环结束后 l 的值为 [3, 5, 9]。这意味着图0的节点索引范围是 0~2，图1是 3~4，图2是 5~8。
        l = []
        cur = 0
        for i in count:
            cur += i
            l.append(cur)



提取尾部节点（for j in l: 循环）
j 代表当前图在整个大图中的累加总长度。
j - 1 正好计算出当前图的最后一个节点在大图 x 中的绝对索引（例：索引 2、4、8）。
x[j - 1] 提取该节点（即全局节点）的特征向量。
.reshape(1, -1) 确保提取出的一维向量保持二维矩阵形状 [1, n_features]，以便后续拼接。
        re = []
        for j in l:
            [1, n_features]进行append在0维度
            re.append(x[j - 1].reshape(1,-1))



拼接输出,将提取出的所有全局节点特征按行拼接。
输出张量 g 的最终维度为 [batch_size, n_features]，作为表示各个分子图整体结构的特征向量，送入下游 MLP 层进行溶解度回归预测。
        g = torch.cat(re,dim = 0)

        return g

    def forward(self, data_i, cond):
        x, edge_index = data_i.x.to(torch.float), data_i.edge_index
       
        edge_weight（将边缘属性求和降维后作为边权重），参与图卷积的信息聚合。
        edge_weight = torch.sum(data_i.edge_attr,dim=1).to(torch.float)

        x = self.l1(x, edge_index,edge_weight )
        x = self.act(x)
        x = self.dropout(x)

        x = self.l2(x, edge_index,edge_weight )
        x = self.act(x)
        x = self.dropout(x)

        x = self.l3(x, edge_index,edge_weight )
        x = self.act(x)
        x = self.dropout(x)

        x = self.l4(x, edge_index,edge_weight )
        x = self.act(x)
        x = self.dropout(x)
全局信息读取 (Readout)： 卷积完成后，不使用传统的全局池化层（如平均池化或最大池化），而是调用自定义的 extract 函数。该函数通过统计批量（batch）中每个图的节点数量，直接索引出每个图的最后一个节点（即 x[j - 1]）。对应上游数据处理逻辑，该节点即为连接全图所有原子的虚拟全局节点。
        x = self.extract(x,data_i.batch)

溶解度预测： 获取到 512 维的全局节点特征后，使用 torch.cat 将其与 2 维的实验条件特征 cond（温度和压力）在维度 1 上拼接，形成 514 维的向量。最终通过由多个全连接层、批归一化（BatchNorm1d）和 ReLU 组成的 MLP (self.l5) 输出标量预测值。
        x = torch.cat([x, cond], dim=1)
        x = self.l5(x)

        return x

# GAT
class IL_GAT(torch.nn.Module):
    def __init__(self, args):
        super(IL_GAT,self).__init__()
        self.args = args
        n_features = self.args['n_features']

        self.l1 = GATv2Conv(n_features, 512)
        self.l2 = GATv2Conv(512, 1024)
        self.l3 = GATv2Conv(1024, 1024)
        self.l4 = GATv2Conv(1024, 512)

        self.l5 = nn.Sequential(
            nn.Linear(514, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(p=0.4),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.3),

            nn.Linear(512, 1),
        )

        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=args['dropout_rate'])

    def extract(self,x,batch):
注意力权重返回：return_attention_weights=True。这使得网络在输出节点特征的同时，也会输出边索引及其对应的注意力权重分布（即代码中的 edge 和 attention 元组）。这些权重数据为后续构建论文中的“IL Explainer”提供了原始依据。        
        output, count= torch.unique(batch, return_counts=True)
        count = count.tolist()

        l = []
        cur = 0
        for i in count:
            cur += i
            l.append(cur)
        re = []
        for j in l:
            re.append(x[j - 1].reshape(1,-1))

        g = torch.cat(re,dim = 0).to('cuda')

        return g






    def forward(self, data_i, cond):
        x, edge_index = data_i.x.to(torch.float), data_i.edge_index

        x,(edge1,attention1) = self.l1(x, edge_index, return_attention_weights = True )
        x = self.act(x)
        x = self.dropout(x)

        x,(edge2,attention2) = self.l2(x, edge_index,return_attention_weights = True )
        x = self.act(x)
        x = self.dropout(x)

        x,(edge3,attention3) = self.l3(x, edge_index,return_attention_weights = True )
        x = self.act(x)
        x = self.dropout(x)

        x,(edge4,attention4) = self.l4(x, edge_index,return_attention_weights = True )
        x = self.act(x)
        x = self.dropout(x)

        x = self.extract(x,data_i.batch)

        x = torch.cat([x, cond], dim=1)
        x = self.l5(x)

        return x










Data(
    x=[num_nodes, 5], 
    edge_index=[2, num_edges], 
    edge_attr=[num_edges, 3], 
    y=[1],  # 目标值：CO2 溶解度
    cond=[1, 2] # 实验条件：温度和压力
)



经过 DataLoader 组合后，输出的对象在 PyG 中具体称为 torch_geometric.data.Batch（它是 Data 类的子类）。假设你在 DataLoader 中设定的 batch_size = B
Batch(
    x=[num_nodes_total, 5],            # B 个图的节点数总和。
    edge_index=[2, num_edges_total],   # B 个图的边数总和（内部索引已自动加上偏移量）。
    edge_attr=[num_edges_total, 3],    # B 个图的边特征总和。
    y=[B, 1],                          # B 个图的 CO2 溶解度目标值堆叠。
    cond=[B, 2],                       # B 个图的温度和压力条件堆叠。
    batch=[num_nodes_total],           # 长度为 num_nodes_total 的一维归属标识向量。
    ptr=[B + 1]                        # PyG 自动生成的切片指针，记录每个图在合并大图中的起始和终止索引。
)




在 PyTorch Geometric 的底层机制中，这三个核心属性在传播时各司其职，共同完成“消息传递（Message Passing）”：

edge_index（连接索引）：充当“路由表”
它决定了信息应该如何流动。网络会根据 edge_index 指定的起点和终点，告诉底层计算图“把节点 A 的特征传递给节点 B”。

x（节点特征）：充当“流动的数据实体”
它是真正被传递和更新的数值。沿着 edge_index 铺设的路径，源节点的特征向量会被发送到目标节点。

edge_attr（边特征）：充当“途径处理站”
当源节点 x_i 的特征流向目标节点 x_j 时，它们之间这根连线本身的特征 edge_attr 会参与计算。例如，在代码中，消息传递的公式是源节点特征与边特征直接相加（x_j + edge_attr）。



# GIN
class GINEConv(MessagePassing):
    def __init__(self, emb_dim):
        #super().__init__()
        super(GINEConv, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, 2*emb_dim),
            nn.ReLU(),
            nn.Linear(2*emb_dim, emb_dim)
        )
分别用于处理化学键的三种离散属性：键类型（bond type）、是否在环中（isInRing）、是否是芳香键（isAromatic）
        self.edge_embedding1 = nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = nn.Embedding(num_bond_isInRing, emb_dim)
        self.edge_embedding3 = nn.Embedding(num_bond_isAromatic, emb_dim)
用 Xavier 均匀分布进行初始化浮点向量
        nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        nn.init.xavier_uniform_(self.edge_embedding2.weight.data)
        nn.init.xavier_uniform_(self.edge_embedding3.weight.data)


节点特征 x、边索引 edge_index 和边特征 edge_attribute
    def forward(self, x, edge_index, edge_attr):
       
        添加自环 (Add Self-loops)# 目的：在消息传递 (propagate) 时，强制节点聚合自身上一层的特征，防止固有属性被邻居节点覆盖。     
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))[0]
     
        # 为新增的 x.size(0) 条自环边生成特征矩阵，特征维度为3（对应键类型、是否在环、是否芳香键）。
        self_loop_attr = torch.zeros(x.size(0), 3)

        # 3. 分配专属隔离索引 (避免特征混淆)
        # 目的：自环是人为添加的虚拟边，为了防止其与真实化学键共享嵌入权重，
        # 强制为其分配每种离散属性的最大索引值 (类别总数 - 1)。
        # 结果：当自环边通过共享的 nn.Embedding 层时，将固定提取权重矩阵的最后一行，实现参数隔离更新
        self_loop_attr[:, 0] = num_bond_type - 1
        self_loop_attr[:, 1] = num_bond_isInRing - 1  
        self_loop_attr[:, 2] = num_bond_isAromatic - 1  

        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
         # 将构造好的自环边特征矩阵拼接到真实化学键特征矩阵的最下方。
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)
      
对于图中的任意一条特定边（无论它是真实的化学键还是虚拟的自环），分别计算其三种离散属性对应的独立嵌入向量，然后将这三个同维度的向量相加，压缩成一个统一的 edge_embeddings 向量。这个最终向量综合代表了该边（或自环）的完整物理属性，随后被送入 propagate 参与消息传递。
        edge_embeddings = self.edge_embedding1(edge_attr[:,0]) + \
                          self.edge_embedding2(edge_attr[:,1]) + \
                          self.edge_embedding3(edge_attr[:,2])
触发消息传递
        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings)

message 函数来定义邻居节点传过来的“信息”是什么。
    def message(self, x_j, edge_attr):
        return x_j + edge_attr
aggr_out 是目标节点收集到的所有邻居消息的聚合结果。
    def update(self, aggr_out):
        return self.mlp(aggr_out)

class GIN(nn.Module):
    def __init__(self, args):
        super(GIN, self).__init__()
        self.num_layer = args['num_gin_layer']
        self.emb_dim = args['emb_dim']
        self.feat_dim = args['feat_dim']
        self.drop_ratio = args['drop_ratio']
        pool = args['pool']

2. 节点张量 (x)
在 GIN 模型中，节点特征需要通过 nn.Embedding 层，因此输入的节点张量 x 存储的是整数类别的索引。对象类型： torch.Tensor (通常数据类型为 torch.long 或 int64)张量维度： [num_nodes, 5]
num_nodes 表示该离子液体对中总原子数（阳离子原子数 + 阴离子原子数 + 1个全局节点）。
5 对应我们在 Model.py 中看到的 5 个离散属性的索引：
原子类型（如 C, N, O）
杂化轨道类型（如 SP2, SP3）
是否芳香族
节点度数（连接的边数）
形式电荷      

        self.x_embedding1 = nn.Embedding(num_atom_type, self.emb_dim)
        self.x_embedding2 = nn.Embedding(num_Hbrid, self.emb_dim)
        self.x_embedding3 = nn.Embedding(num_Aro, self.emb_dim)
        self.x_embedding4 = nn.Embedding(num_degree, self.emb_dim)
        self.x_embedding5 = nn.Embedding(num_charge, self.emb_dim)

        nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        nn.init.xavier_uniform_(self.x_embedding2.weight.data)
        nn.init.xavier_uniform_(self.x_embedding3.weight.data)
        nn.init.xavier_uniform_(self.x_embedding4.weight.data)
        nn.init.xavier_uniform_(self.x_embedding5.weight.data)

        # List of MLPs
        self.gnns = nn.ModuleList()
        for layer in range(self.num_layer):
            self.gnns.append(GINEConv(self.emb_dim))

        # List of batchnorms
        self.batch_norms = nn.ModuleList()
        for layer in range(self.num_layer):
            self.batch_norms.append(nn.BatchNorm1d(self.emb_dim))

        if pool == 'mean':
            self.pool = global_mean_pool
        elif pool == 'add':
            self.pool = global_add_pool
        elif pool == 'max':
            self.pool = global_max_pool
        else:
            raise ValueError('Not defined pooling!')

        self.feat_lin = nn.Linear(self.emb_dim, self.feat_dim)

        self.pred_head = nn.Sequential(
            nn.Linear(self.feat_dim + 2, self.feat_dim),
            nn.Softplus(),
            nn.Linear(self.feat_dim, int(self.feat_dim/2)),
            nn.Softplus(),
            nn.Linear(int(self.feat_dim/2), 1)
        )
    def extract(self,x,batch):
        output, count= torch.unique(batch, return_counts=True)
        count = count.tolist()

        l = []
        cur = 0
        for i in count:
            cur += i
            l.append(cur)
        re = []
        for j in l:
            re.append(x[j - 1].reshape(1,-1))

        g = torch.cat(re,dim = 0)

        return g
    def forward(self, pair_graph, cond):
        # GIN layer
        h = self.x_embedding1(pair_graph.x[:, 0]) + \
            self.x_embedding2(pair_graph.x[:, 1]) + \
            self.x_embedding3(pair_graph.x[:, 2]) + \
            self.x_embedding4(pair_graph.x[:, 3]) + \
            self.x_embedding5(pair_graph.x[:, 4])

        for layer in range(self.num_layer):
            h = self.gnns[layer](h, pair_graph.edge_index, pair_graph.edge_attr)
            h = self.batch_norms[layer](h)
            h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)

        h = self.feat_lin(h)
        h_pair = self.extract(h, pair_graph.batch)
        h = torch.cat([h_pair, cond], dim=1)

        return self.pred_head(h)
