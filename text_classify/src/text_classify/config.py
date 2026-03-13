# -*- coding: utf-8 -*-
# @Time    : 2026/1/5 17:42
# @Author  : 老冰棍
# @File    : config.py
# @Software: PyCharm
from dataclasses import dataclass
from typing import Optional, Union

from text_classify.dataset.tokenizer import TokenizerBase


@dataclass
class Config:
    model_output_dir: Optional[str] = None # 模型输出文件路径
    summary_dir: Optional[str] = None  # 日志输出路径
    tokenizer: Optional[TokenizerBase] = None  # 分词器

    train_file: Optional[str] = None # 训练数据对应文件
    eval_file: Optional[str] = None  # 模型评估对应文件

    total_epoch : Optional[int] = None
    batch_size: Optional[int] = None # 批次大小
    hidden_size: Optional[int] = None  # 网络隐层大小
    lr: Optional[float] = None # 模型训练学习率

    network_type: str = "lstm" # 网络类型 可选lstm, bert
    bert_path: Optional[str] = None # bert模型迁移路径
    freeze: Optional[Union[bool, int]] = None # 给定迁移模型的时候冻结参数

    max_no_improved_epoch: int = 10 # 提前停止器参数 当连续n个epoch模型效果均没有提升的时候，提前结束模型训练

    device: str = "cpu"