Dataset.py 生成了包含节点特征（x）、边索引（edge_index）和全局节点的大型图（Combine_Graph）。

1. 原始阶段：CSV 数据
输入：data/smiles.csv（包含阳离子 SMILES、阴离子 SMILES、温度 T、压力 P、CO2 溶解度）。
形态：此时数据是化学字符串和数值

2 .预处理阶段：SMILES 
 NumPy (核心修正点)
这里不是“读取分子描述符”，而是“结构特征提取”：
工具：通常使用 RDKit。
逻辑：RDKit 读取 SMILES，将其解析为分子的原子（Atoms）和化学键（Bonds）。
输出：生成 data.npy。这个文件里存的不是简单的“分子描述符”（如分子量、偶极矩等单一数值），而是每个原子的特征向量（如原子序数、手性、杂化轨道等）。
对应关系：
cation: 存储了阳离子的原子矩阵和邻接矩阵。
anion: 存储了阴离子的原子矩阵和邻接矩阵。

4. 格式转化：mol2graph (拆分并标准化)
逻辑：把 NumPy 数组塞进 torch_geometric.data.Data 对象。
目的：让数据符合 PyG (PyTorch Geometric) 的标准格式。
关键点：这里不是“拆分”，而是“封装”。它明确了哪些是节点特征 (x)，哪些是连接关系  edgeindex


5. 系统集成：combine_Graph (阴阳离子合体)
逻辑：把阳离子的图和阴离子的图合并成一个大的图。
意义：离子液体是一个整体。虽然阴阳离子在化学上是独立的，但在模型里它们必须作为一个样本同时输入。

6. 特征增强：add_global (全局节点)
逻辑：手动添加一个虚拟全局节点（Global Node）。
作用：
信息桥梁：原本阴离子和阳离子在图中是“断开”的，信息无法交流。
全局特征捕获：通过让虚节点连接所有原子，它能提取出离子液体整体的表征，而不仅仅是单个原子的局部特征。

7. 预测阶段：GCN 运算
输入：(Combine_Graph, condition, label)。
逻辑：
GCN 层：在原子间传播信息（包括通过虚节点在阴阳离子间交换信息）。
融合：将图提取到的结构特征与 T,P 等物性条件拼接。
输出：预测该离子液体在特定压力和温度下的 CO2 吸附量


是不是“分子描述符变成原子”？ 不完全是。准确地说是：通过 RDKit 将分子（SMILES）拆解为原子，并给每个原子分配一套“原子描述符”（Atom Descriptors）。
add_global 是为了预测离子层面吗？ 是的。它就像一个“信息聚合器”，把分散在各个原子上的化学信息汇聚起来，形成一个离子级别的整体画像，从而提高预测准确度。
简单一句话： 它是把微观的原子信息（通过图结构）和宏观的物性条件（TP）结合起来，利用全局节点做中转，最后预测出整体的物理化学性质。


import numpy as np
import torch
from torch_geometric.data import Batch, Data, Dataset, DataLoader

args = {
    'add_global':True,
    'bi_direction':True
}


def combine_Graph(Graph_list):

    x = Batch.from_data_list(Graph_list).x
    edge_index = Batch.from_data_list(Graph_list).edge_index
    edge_attr = Batch.from_data_list(Graph_list).edge_attr

    combined_Graph = Data(x = x,edge_index = edge_index,edge_attr = edge_attr)

    return combined_Graph

def add_global(graph):
    node = torch.tensor([0,0,0,0,0]).reshape(1, -1)
    # node.shape
    x = torch.cat([graph.x, node], dim=0)
    num_node = x.shape[0] - 1
    new_node = x.shape[0] - 1
    start = []
    end = []
    attr = []
    for i in range(num_node):
        # print(i)
        start.append(i)
        end.append(new_node)
        attr.append([0, 0, 0])
    if args['bi_direction'] == True:
        for i in range(num_node):
            # print(i)
            start.append(new_node)
            end.append(i)
            attr.append([0, 0, 0])

    start = torch.tensor(start).reshape(1, -1)
    end = torch.tensor(end).reshape(1, -1)
    new_edge = torch.cat([start, end], dim=0)
    edge_index = torch.cat([graph.edge_index, new_edge], dim=1)
    attr = torch.tensor(attr)
    edge_attr = torch.cat([graph.edge_attr, attr], dim=0)
    g = Data(x = x,edge_index = edge_index,edge_attr = edge_attr)

    return g


class IL_set(torch.utils.data.Dataset):
    """
    torch dataset
    """
    #代码从指定的 path（通常是 clean/ 目录）加载预处理好的 .npy 文件
    def __init__(self,path):
        super(IL_set, self).__init__()
        data_path = path + 'data.npy'
        label_path = path + 'label.npy'
        # data.npy: 包含离子液体的结构信息和物理条件
        self.data = np.load(data_path,allow_pickle=True)
        # label.npy: 包含对应的 CO2 吸附量。
        self.label = np.load(label_path,allow_pickle=True)
        self.length = self.label.shape[0]
        # show basic information
        print("----info----")
        print("data_length",self.length)
        print("------------")


    def __len__(self):
        return self.length

    
    #2. 特征转换与合并阶段 (__getitem__)
    def __getitem__(self, idx):
阳离子与阴离子处理 (mol2graph):
代码从 data.npy 中取出阳离子 (cation) 和阴离子 (anion) 的原始数据（索引 0 和 1）。
通过 mol2graph 函数，将原本以数组形式存储的原子特征 (x)、邻接矩阵 (edge_index) 和边特征 (edge_attr) 封装成 torch_geometric.data.Data 对象。
    
        cation = self.data[idx][0]
        anion = self.data[idx][1]
        T = self.data[idx][2]
        P = self.data[idx][3]
        # # debug
        # print("cation",cation)
        # print("anion",anion)
        cation = self.mol2graph(cation)
        anion = self.mol2graph(anion)

图合并 (combine_Graph):
由于离子液体由阴阳离子组成，代码调用 combine_Graph 将两个独立的分子图合并成一个大的非连通图（节点和边简单堆叠）。
        Combine_Graph = combine_Graph([cation, anion])
        # print('before',Combine_Graph.x.shape)


增加全局节点 (add_global):
核心操作：如果 add_global 为 True，程序会向图中添加一个“虚拟节点”（Global Node）。
这个虚拟节点与图中所有原子节点都建立双向连接（bi_direction=True）。
目的：这在 GNN 中常用于全局信息聚合，让模型能更好地学习分子整体的表示。
        if args['add_global'] == True:
            Combine_Graph = add_global(Combine_Graph)
        # print('after',Combine_Graph.x.shape )

环境条件 (condition):
取出温度 T 和压力 P（索引 2 和 3），转换为 float 类型的张量。
        condition = torch.tensor([T,P],dtype=torch.float)


        label = torch.tensor(self.label[idx],dtype=torch.float)

每一个样本返回给 DataLoader 的是一个元组：(Combine_Graph, condition, label)。
        return Combine_Graph,condition,label









数据的“化身”：mol2graph(self, mol) (特征提取函数)

这个函数负责把 npy 里的数组变成 PyTorch Geometric (PyG) 的 Data 对象。 将离子的化学性质转为节点特征向量（原子描述符）

    def mol2graph(self,mol):
        x = torch.tensor(mol[0],dtype=torch.long)
        edge_index = torch.tensor(mol[1],dtype=torch.long)
        edge_attr = torch.tensor(mol[2],dtype=torch.long)
        # debug
        # print("mol",x.shape,edge_index.shape,edge_attr.shape)
        Graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        return Graph



数据对齐：通过 collate_fn，这些变长的图数据会被 torch_geometric 的 DataLoader 自动处理成 Batch 对象，确保 GPU 可以高效并行计算。
    def collate_fn(batch):
        batch_x = torch.as_tensor(batch)
        return batch_x


if __name__ == '__main__':
    args = {
        'data_path':"clean/"
    }
    D = IL_set(path = args['data_path'])
    print(len(D))
    for item in D:
        G,c,l = item
        print(c,l)
