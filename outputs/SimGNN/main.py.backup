#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: XiaShan
@Contact: 153765931@qq.com
@Time: 2024/3/23 16:12
"""

import os, torch, random, math
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm, trange

from data import load_dataset
from model import SimGNN
from parameter import parameter_parser, IOStream, table_printer

def train(args, IO, train_dataset, num_nodes_id):
    # 使用GPU or CPU
    device = torch.device('cpu' if args.gpu_index < 0 else 'cuda:{}'.format(args.gpu_index))
    if args.gpu_index < 0:
        IO.cprint('Using CPU')
    else:
        IO.cprint('Using GPU: {}'.format(args.gpu_index))
        torch.cuda.manual_seed(args.seed)  # 设置PyTorch GPU随机种子

    # 加载模型及参数量统计
    model = SimGNN(args, num_nodes_id).to(device)
    IO.cprint(str(model))
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    IO.cprint('Model Parameter: {}'.format(total_params))

    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    IO.cprint('Using Adam')

    epochs = trange(args.epochs, leave=True, desc="Epoch")

    for epoch in epochs:

        random.shuffle(train_dataset)
        train_batches = []
        for graph in range(0, len(train_dataset), 16):
            train_batches.append(train_dataset[graph:graph + 16])

        loss_epoch = 0 # 一个epoch，所有样本损失

        for index, batch in tqdm(enumerate(train_batches), total=len(train_batches), desc="Train_Batches"):
            optimizer.zero_grad()

            loss_batch = 0 # 一个batch，样本损失

            for data in batch:
                # 数据变成GPU支持的数据类型
                data["edge_index_1"], data["edge_index_2"] = data["edge_index_1"].to(device), data["edge_index_2"].to(device)
                data["features_1"], data["features_2"] = data["features_1"].to(device), data["features_2"].to(device)

                prediction = model(data)
                loss_batch = loss_batch + F.mse_loss(data["target"], prediction.cpu())

            loss_epoch = loss_epoch + loss_batch.item()

            loss_batch.backward()
            optimizer.step()

        IO.cprint('Epoch #{}, Train_Loss: {:.6f}'.format(epoch, loss_epoch/len(train_dataset)))

    torch.save(model, 'outputs/%s/model.pth' % args.exp_name)
    IO.cprint('The current best model is saved in: {}'.format('******** outputs/%s/model.pth *********' % args.exp_name))


def test(args, IO, test_dataset):
    """测试模型"""
    device = torch.device('cpu' if args.gpu_index < 0 else 'cuda:{}'.format(args.gpu_index))

    # 输出内容保存在之前的训练日志里
    IO.cprint('********** TEST START **********')
    IO.cprint('Reload Best Model')
    IO.cprint('The current best model is saved in: {}'.format('******** outputs/%s/model.pth *********' % args.exp_name))

    model = torch.load('outputs/%s/model.pth' % args.exp_name).to(device)
    #model = model.eval()  # 创建一个新的评估模式的模型对象，不覆盖原模型

    ground_truth = [] # 存放data["norm_ged"]
    scores = [] # 存放模型预测与ground_truth的损失

    for data in test_dataset:
        data["edge_index_1"], data["edge_index_2"] = data["edge_index_1"].to(device), data["edge_index_2"].to(device)
        data["features_1"], data["features_2"] = data["features_1"].to(device), data["features_2"].to(device)

        prediction = model(data)

        scores.append((-math.log(prediction.item()) - data["norm_ged"]) ** 2) # MSELoss
        ground_truth.append(data["norm_ged"])

    model_error = np.mean(scores)

    norm_ged_mean = np.mean(ground_truth)
    baseline_error = np.mean([(gt - norm_ged_mean) ** 2 for gt in ground_truth])
    IO.cprint('Baseline_Error: {:.6f}, Model_Test_Error: {:.6f}'.format(baseline_error, model_error))


def exp_init():
    """实验初始化"""
    if not os.path.exists('outputs'):
        os.mkdir('outputs')
    if not os.path.exists('outputs/' + args.exp_name):
        os.mkdir('outputs/' + args.exp_name)

    # 跟踪执行脚本，windows下使用copy命令，且使用双引号
    os.system(f"copy main.py outputs\\{args.exp_name}\\main.py.backup")
    os.system(f"copy data.py outputs\\{args.exp_name}\\data.py.backup")
    os.system(f"copy model.py outputs\\{args.exp_name}\\model.py.backup")
    os.system(f"copy parameter.py outputs\\{args.exp_name}\\parameter.py.backup")
    # os.system('cp main.py outputs' + '/' + args.exp_name + '/' + 'main.py.backup')
    # os.system('cp data.py outputs' + '/' + args.exp_name + '/' + 'data.py.backup')
    # os.system('cp model.py outputs' + '/' + args.exp_name + '/' + 'model.py.backup')
    # os.system('cp parameter.py outputs' + '/' + args.exp_name + '/' + 'parameter.py.backup')


if __name__ == '__main__':
    args = parameter_parser()
    random.seed(args.seed)  # 设置Python随机种子
    torch.manual_seed(args.seed)  # 设置PyTorch随机种子
    exp_init()

    IO = IOStream('outputs/' + args.exp_name + '/run.log')
    IO.cprint(str(table_printer(args)))  # 参数可视化

    train_dataset, test_dataset, num_nodes_id = load_dataset()

    train(args, IO, train_dataset, num_nodes_id)
    test(args, IO, test_dataset)