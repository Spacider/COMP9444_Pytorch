#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
__author__ = 'Gary'

from typing import List

import torch

# 创造一个张量
# 并且 requires_grad=True 来追踪其计算历史
x = torch.ones(2, 2, requires_grad=True)

# 做一次运算
y = x + 2

z = y * y * 3
out = z.mean()

print(z, out)

'''
    梯度
'''
out.backward()
print(x.gard)










