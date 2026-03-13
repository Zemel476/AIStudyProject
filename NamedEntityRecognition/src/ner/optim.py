# -*- coding: utf-8 -*-
# @Time    : 2026/3/11 23:51
# @Author  : 老冰棍
# @File    : optim.py
# @Software: PyCharm
from torch import nn, optim


def build_optim(net: nn.Module, lr: float, lr_update_total_iters: int):
   opt = optim.SGD(net.parameters(), lr=lr)
   lr_scheduler = optim.lr_scheduler.LinearLR(
       opt, start_factor=1.0, end_factor=0.01,
       total_iters=lr_update_total_iters
   )

   return opt, lr_scheduler