#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: XiaShan
@Contact: 153765931@qq.com
@Time: 2024/3/23 14:22
"""

import glob, json, torch
import numpy as np

train_graphs = glob.glob("data/train/" + "*.json")
test_graphs = glob.glob("data/test/" + "*.json")

def node_mapping():
    """
    将节点的标识符统一，包括训练和测试数据集
    node_id: 节点标识映射
    """
    nodes_id = set()
    graph_pairs = train_graphs + test_graphs

    for graph_pair in graph_pairs:
        graph = json.load(open(graph_pair))
        nodes_id = nodes_id.union(set(graph["labels_1"]))
        nodes_id = nodes_id.union(set(graph["labels_2"]))
    nodes_id = sorted(nodes_id)
    nodes_id = {id: index for index, id in enumerate(nodes_id)}
    num_nodes_id = len(nodes_id)

    return nodes_id, num_nodes_id


def process_data(graph, nodes_id):
    data = dict()

    # 获取每个图的邻接矩阵（无向图）
    edges_1 = graph["graph_1"] + [[y, x] for x, y in graph["graph_1"]]
    edges_2 = graph["graph_2"] + [[y, x] for x, y in graph["graph_2"]]
    edges_1 = torch.from_numpy(np.array(edges_1, dtype=np.int64).T).type(torch.long)
    edges_2 = torch.from_numpy(np.array(edges_2, dtype=np.int64).T).type(torch.long)

    data["edge_index_1"] = edges_1
    data["edge_index_2"] = edges_2

    # 对每个图的节点进行one-hot编码作为节点特征
    features_1, features_2 = [], []
    for n in graph["labels_1"]:
        features_1.append([1.0 if nodes_id[n] == i else 0.0 for i in nodes_id.values()])
    for n in graph["labels_2"]:
        features_2.append([1.0 if nodes_id[n] == i else 0.0 for i in nodes_id.values()])

    features_1 = torch.FloatTensor(np.array(features_1))
    features_2 = torch.FloatTensor(np.array(features_2))

    data["features_1"] = features_1
    data["features_2"] = features_2

    # 根据GED计算每对图的ground truth
    norm_ged = graph["ged"] / (0.5 * (len(graph["labels_1"]) + len(graph["labels_2"])))
    data["norm_ged"] = norm_ged

    # 指数函数映射ground truth得到相似性得分
    data["target"] = torch.from_numpy(np.exp(-norm_ged).reshape(1, 1)).view(-1).float().unsqueeze(0)

    return data


def load_dataset():
    train_dataset = [] # 列表：存储处理后的训练集
    test_dataset = [] # 列表：存储处理后的测试集
    nodes_id, num_nodes_id = node_mapping()

    for graph_pair in train_graphs:
        graph = json.load(open(graph_pair))
        data = process_data(graph, nodes_id)
        train_dataset.append(data)

    for graph_pair in test_graphs:
        graph = json.load(open(graph_pair))
        data = process_data(graph, nodes_id)
        test_dataset.append(data)

    return train_dataset, test_dataset, num_nodes_id
