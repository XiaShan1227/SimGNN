#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: XiaShan
@Contact: 153765931@qq.com
@Time: 2024/3/23 21:18
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class AttentionModule(nn.Module):
    "图嵌入注意力模块"
    def __init__(self, args):
        super(AttentionModule, self).__init__()
        self.args = args
        self.setup_weights()
        self.init_parameters()

    def setup_weights(self):
        self.weight_matrix = nn.Parameter(torch.Tensor(self.args.filters_3, self.args.filters_3))

    def init_parameters(self):
        nn.init.xavier_uniform_(self.weight_matrix)

    def forward(self, embedding):
        # embedding:(num_nodes=14, num_features=32)
        # 在每个特征维度上，取节点平均值
        # global_context:(32)
        global_context = torch.mean(torch.matmul(embedding, self.weight_matrix), dim=0)

        transformed_global = torch.tanh(global_context)

        # sigmoid_scores计算每个节点与图之间的相似性得分(注意力)
        # sigmoid_scores:(14)
        sigmoid_scores = torch.sigmoid(torch.mm(embedding, transformed_global.view(-1, 1)))

        # representation:图的嵌入(32)
        representation = torch.mm(torch.t(embedding), sigmoid_scores)

        return representation


class TenorNetworkModule(nn.Module):
    "Neural Tensor Network"
    def __init__(self, args):
        super(TenorNetworkModule, self).__init__()
        self.args = args
        self.setup_weights()
        self.init_parameters()

    def setup_weights(self):
        self.weight_matrix = nn.Parameter(torch.Tensor(self.args.filters_3, self.args.filters_3, self.args.tensor_neurons))
        self.weight_matrix_block = nn.Parameter(torch.Tensor(self.args.tensor_neurons, 2*self.args.filters_3))
        self.bias = nn.Parameter(torch.Tensor(self.args.tensor_neurons, 1))

    def init_parameters(self):
        nn.init.xavier_uniform_(self.weight_matrix)
        nn.init.xavier_uniform_(self.weight_matrix_block)
        nn.init.xavier_uniform_(self.bias)

    def forward(self, embedding_1, embedding_2):
        scoring = torch.mm(torch.t(embedding_1), self.weight_matrix.view(self.args.filters_3, -1))
        scoring = scoring.view(self.args.filters_3, self.args.tensor_neurons) # shape(32, 16)
        scoring = torch.mm(torch.t(scoring), embedding_2) # shape(16, 1)

        # combined_representation:shape(64, 1)
        combined_representation = torch.cat((embedding_1, embedding_2))

        # block_scoring:shape(16, 1)
        block_scoring = torch.mm(self.weight_matrix_block, combined_representation)

        scores = F.relu(scoring + block_scoring + self.bias)

        return scores


class SimGNN(nn.Module):
    def __init__(self, args, num_nodes_id):
        super(SimGNN, self).__init__()
        self.args = args
        self.num_nodes_id = num_nodes_id
        self.setup_layers()

    def calculate_bottleneck_features(self):
        if self.args.histogram == True:
            self.feature_count = self.args.tensor_neurons + self.args.bins
        else:
            self.feature_count = self.args.tensor_neurons

    def setup_layers(self):
        self.calculate_bottleneck_features()

        self.convolution_1 = GCNConv(self.num_nodes_id, self.args.filters_1)
        self.convolution_2 = GCNConv(self.args.filters_1, self.args.filters_2)
        self.convolution_3 = GCNConv(self.args.filters_2, self.args.filters_3)

        self.attention = AttentionModule(self.args)

        self.tensor_network = TenorNetworkModule(self.args)

        self.fully_connected_first = nn.Linear(self.feature_count, self.args.bottle_neck_neurons)
        self.scoring_layer = nn.Linear(self.args.bottle_neck_neurons, 1)

    def calculate_histogram(self, abstract_features_1, abstract_features_2):
        "Pairwise Node Comparison"

        # abstract_features_1:(num_nodes1, num_features=32)
        # abstract_features_2:(num_features=32, num_nodes2)
        scores = torch.mm(abstract_features_1, abstract_features_2).detach()
        scores = scores.view(-1, 1)

        hist = torch.histc(scores, bins=self.args.bins) # 统计得分在每个区间的个数
        hist = hist/torch.sum(hist) # 归一化
        hist = hist.view(1, -1)

        return hist

    def convolutional_pass(self, edge_index, features):
        features = self.convolution_1(features, edge_index)
        features = F.relu(features)
        features = F.dropout(features, p=self.args.dropout, training=self.training)
        features = self.convolution_2(features, edge_index)
        features = F.relu(features)
        features = F.dropout(features, p=self.args.dropout, training=self.training)
        features = self.convolution_3(features, edge_index)

        return features

    def forward(self, data):
        edge_index_1 = data["edge_index_1"]
        edge_index_2 = data["edge_index_2"]
        features_1 = data["features_1"] # (num_nodes1, num_features=16)
        features_2 = data["features_2"] # (num_nodes2, num_features=16)

        abstract_features_1 = self.convolutional_pass(edge_index_1, features_1) # (num_nodes1, num_features=16) ——> (num_nodes1, num_features=32)
        abstract_features_2 = self.convolutional_pass(edge_index_2, features_2) # (num_nodes2, num_features=16) ——> (num_nodes2, num_features=32)

        if self.args.histogram == True:
            hist = self.calculate_histogram(abstract_features_1, torch.t(abstract_features_2))

        pooled_features_1 = self.attention(abstract_features_1) # (num_nodes1, num_features=32) ——> (num_features=32)
        pooled_features_2 = self.attention(abstract_features_2) # (num_nodes2, num_features=32) ——> (num_features=32)

        scores = self.tensor_network(pooled_features_1, pooled_features_2)
        scores = torch.t(scores)

        if self.args.histogram == True:
            scores = torch.cat((scores, hist), dim=1).view(1, -1)

        scores = F.relu(self.fully_connected_first(scores))
        score = torch.sigmoid(self.scoring_layer(scores))

        return score
