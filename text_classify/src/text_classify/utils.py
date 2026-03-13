# -*- coding: utf-8 -*-
# @Time    : 2026/1/4 17:03
# @Author  : 老冰棍
# @File    : utils.py
# @Software: PyCharm
import json
import os

from torch import nn, optim


def load_json(json_file):
    with open(json_file, "r", encoding="utf-8") as reader:
        return json.load(reader)


def save_json(json_file, json_obj):
    os.makedirs(os.path.dirname(json_file), exist_ok=True)
    with open(json_file, 'w', encoding='utf-8') as writer:
        json.dump(json_obj, writer, ensure_ascii=False, indent=2)


def build_loss():
    return nn.CrossEntropyLoss()


def build_optim(net: nn.Module, lr: float):
    return optim.SGD(params=net.parameters(), lr=lr)