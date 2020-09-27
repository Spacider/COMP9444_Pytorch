#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
__author__ = 'Gary'

# 导入训练需要的包
import torch
import torch.utils.data
import torch.nn.functional as F
import argparse

# 传入 学习率 曲率 和 初始权重
parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=0.1, help='learning rate')
parser.add_argument("--mom", type=float, default=0.0, help='momentum')
parser.add_argument("--init", type=float, default=1.0, help='inital weight size')

args = parser.parse_args()

# 名字自取 MyModel 可以改成其他名字, e.g. XOR_MODEL
class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # 定义网络结构（层数，层的名字）
        # Args:
        # in_features: size of each input sample
        # out_features: size of each output sample
        # bias: If set to ``False``, the layer will not learn an additive bias.
        #     Default: ``True``

        # 相当与生成一个 2 --- 2 --- 1 的训练模型
        #            *         *
        #                 -->       --->     *
        #            *         *

        # in_features 代表了输入张量
        # out_features 代表了输出张量
        self.in_hid = torch.nn.Linear(2, 2)
        self.out_hid = torch.nn.Linear(2, 1)

    # 前向传播函数
    # 设计网络以及定义返回值
    # 即可理解为 图中的线
    # 先由输入经过 in_hid 方法得到和，再通过 tanh 激活函数
    # 再求hid_out 的和， 通过 sigmoid 激活函数得到最终的结果
    def forward(self, input):
        hid_sum = self.in_hid(input)
        hidden = torch.tanh_(hid_sum)
        out_sum = self.out_hid(hidden)
        output = torch.sigmoid(out_sum)
        return output


# 如果想用显卡运算，则使用 gpu
# 判断， 若支持 cuda，则使用 gpu 加速训练
# 若不支持，则使用 cpu
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# 传入输入值和输出值
# 由 input 要训练成 target， 也是LOSS FUNCTION 里的 Zi 和 Ti
input = torch.Tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
target = torch.Tensor([[0], [1], [1], [0]])

# 定义一个数据集
xor_dataset = torch.utils.data.TensorDataset(input, target)
# 通过 DataLoader 把数据加载并转换成 batch（批量） 的对象
# todo 判断如果 batch_size 减少会发生什么现象
# todo 理解 batch 打包的意义
train_loader = torch.utils.data.DataLoader(xor_dataset, batch_size=4)

# 导入 cpu 或 gpu 创造网络
net = MyModel().to(device)

# net.in_hid.weight 本质上是个 Parameter 类
# Parameter 类的意义是将不可寻刘安的 Tensor 类型转化为可训练的 Parameter 类型并且绑定在 module 里
# 目的是为了定义权重
# self.weight = Parameter(torch.Tensor(out_features, in1_features, in2_features))
net.in_hid.weight.data.normal_(0, args.init)
net.out_hid.weight.data.normal_(0, args.init)


# 选择一种优化函数 SGD 加 momentum
# SGD 采用 mini batches 的方法训练，梯度下降
# momentum 的方法是在每次计算梯度时增加之前的一次的梯度，保证在可能存在的 global minimum 后可以继续寻找
optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.mom)


# 运行 100000 次 (下为实际训练代码)
epochs = 100000
# todo 理解梯度
for epoch in range(1, epochs):
    # train(net, device, train_loader, optimizer)
    for batch_id, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()  # 梯度初始化为 0
        output = net(data)  # 数据导入网络
        loss = F.binary_cross_entropy(output, target)  # 损失函数(用于计算 target 和 output之间的损失值)
        # 下面两部可以理解为 Exercise 2 中手工计算的数据
        loss.backward()  # 反向传播并计算当前的梯度
        optimizer.step()  # 根据梯度更新网络梯度
        # 每 10 次检测一次
        if epoch % 10 == 0:
            print('ep%3d: loss = %7.4f' % (epoch, loss.item()))  # loss.item() 返回当前的损失值
        # 如果损失值小于 0.1， 停止训练
        if loss < 0.0001:
            print(output)
            exit()

# 可以打印训练完的结果集
print(output)












