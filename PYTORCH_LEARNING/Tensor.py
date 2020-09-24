#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
from __future__ import print_function
__author__ = 'Gary'

import torch

'''
     Tensor
'''
# 创建一个没有初始化的 5 * 3 矩阵
# tensor([[3.0245e+35, 1.4153e-43, 0.0000e+00],
#         [0.0000e+00, 0.0000e+00, 0.0000e+00],
#         [1.5290e-33, 1.4013e-45, 3.0166e-14],
#         [4.5897e-41, 3.0166e-14, 4.5897e-41],
#         [3.0166e-14, 4.5897e-41, 1.4605e-19]])
x = torch.empty(5, 3)
print(x)

# 创建一个随机初始化的矩阵
x = torch.rand(5, 3)
print(x)

# 创建一个全是0且数据类型为long的矩阵
x = torch.zeros(5, 3, dtype=torch.long)
print(x)

# 直接从数据创造张量
x = torch.tensor([5.5, 3])
print(x)

# 根据已有的tensor创造新的tensor
x = x.new_ones(5, 3, dtype=torch.double)
print(x)

# 重载 dtype
x = torch.randn_like(x, dtype=torch.float)
print(x)

# 获取tensor形状
print(x.size())  # torch.Size 本质上是tuple， 支持tuple的一切操作

'''
     运算
'''
# 加法运算
y = torch.rand(5, 3)
print(x + y)
# or
print(torch.add(x, y))
# or 给定输出张量
result = torch.empty(5, 3)
torch.add(x, y, out= result)
print(result)
