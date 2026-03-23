import numpy as np
import torch
from torch_geometric.data import Batch, Data, Dataset, DataLoader

args = {
    'add_global':True,
    'bi_direction':True
}

这里开发者手动把阴离子和阳离子合成了一个 Data 对象。
在 __getitem__ 层面，它返回的是一个“包含阴阳离子的单一图”。
这意味着对模型来说，一个样本 = 一个图（这个图里有两个不连通的小图）。
def combine_Graph(Graph_list):
    """
    merge a Graph with multiple subgraph
    Args:
        Graph_list: list() of torch_geometric.data.Data object

    Returns: torch_geometric.data.Data object

    """
    x = Batch.from_data_list(Graph_list).x
    edge_index = Batch.from_data_list(Graph_list).edge_index
    edge_attr = Batch.from_data_list(Graph_list).edge_attr

    combined_Graph = Data(x = x,edge_index = edge_index,edge_attr = edge_attr)

    return combined_Graph

def add_global(graph):
    """
    add a global point, all the attribute are set to zero
    :param graph: pyg.data
    :return: pyg.data
    """
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
