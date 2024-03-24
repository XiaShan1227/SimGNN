#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: XiaShan
@Contact: 153765931@qq.com
@Time: 2024/3/23 16:50
"""

import argparse
from texttable import Texttable


def parameter_parser():
    parser = argparse.ArgumentParser(description="Run SimGNN.") # 创建解析器

    parser.add_argument('--seed', type=int, default=16, help='Random seed of the experiment')
    parser.add_argument('--exp-name', type=str, default='Exp', help='Name of the experiment')
    parser.add_argument('--gpu-index', type=int, default=0, help='Index of GPU(set <0 to use CPU)')
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs. Default is 5.")
    parser.add_argument("--filters-1", type=int, default=128, help="Filters (neurons) in 1st convolution. Default is 128.")
    parser.add_argument("--filters-2", type=int, default=64, help="Filters (neurons) in 2nd convolution. Default is 64.")
    parser.add_argument("--filters-3", type=int, default=32, help="Filters (neurons) in 3rd convolution. Default is 32.")
    parser.add_argument("--tensor-neurons", type=int, default=16, help="Neurons in tensor network layer. Default is 16.")
    parser.add_argument("--bottle-neck-neurons", type=int, default=16, help="Bottle neck layer neurons. Default is 16.")
    parser.add_argument("--batch-size", type=int, default=16, help="Number of graph pairs per batch. Default is 16.")
    parser.add_argument("--bins", type=int, default=16, help="Similarity score bins. Default is 16.")
    parser.add_argument("--dropout", type=float, default=0.6, help="Dropout probability. Default is 0.5.")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate. Default is 0.001.")
    parser.add_argument("--weight-decay", type=float, default=5*10^-4, help="Adam weight decay. Default is 5*10^-4.")
    parser.add_argument('--histogram', type=bool, default=True, help='Use histogram or not. Default is True.')

    return parser.parse_args() # 解析参数


class IOStream():
    """训练日志文件"""
    def __init__(self, path):
        self.file = open(path, 'a') # 附加模式：用于在文件末尾添加内容，如果文件不存在则创建新文件

    def cprint(self, text):
        print(text)
        self.file.write(text + '\n')
        self.file.flush() # 确保将写入的内容刷新到文件中，以防止数据在缓冲中滞留

    def close(self):
        self.file.close()


def table_printer(args):
    """绘制参数表格"""
    args = vars(args) # 转成字典类型
    keys = sorted(args.keys()) # 按照字母顺序进行排序
    table = Texttable()
    table.set_cols_dtype(['t', 't']) # 列的类型都为文本(str)
    rows = [["Parameter", "Value"]] # 设置表头
    for k in keys:
        rows.append([k.replace("_", " ").capitalize(), str(args[k])]) # 下划线替换成空格，首字母大写
    table.add_rows(rows)
    return table.draw()
