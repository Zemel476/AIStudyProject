# -*- coding: utf-8 -*-
# @Time    : 2026/1/15 17:53
# @Author  : 老冰棍
# @File    : config.py
# @Software: PyCharm
import os
from dataclasses import dataclass
from typing import Optional, Union, Dict

from ner.datas.tokenizer import Tokenizer


@dataclass
class Config:
    output_dir: Optional[str] = None # 输出文件夹根目标
    tokenizer: Optional[Tokenizer] = None # 分词器
    vocab_size: Optional[int] = None  # 词汇表大小
    label2id: Optional[Union[str, Dict[str, int]]] = None # 类别标签名称到id的映射
    num_classes: Optional[int] = None # 类别数目

    train_file: Optional[str] = None # 训练数据对应文件
    eval_file: Optional[str] = None  # 模型评估数据对应文件

    total_epoch: Optional[int] = None
    batch_size: Optional[int] = None  #批次大小
    lr: Optional[float] = None # 模型训练学习率

    network_type: str = "lstm"
    lstm_layers: int = 1 #lstm模型层数
    lstm_hidden_size: int = 758
    max_length: int = 512 # 最大输入文本长度限制，仅在部分模型结构中生效
    bert_path: Optional[str] = None # bert模型迁移路径
    freeze: Optional[Union[bool, int]] = None # 给定迁移模型的时候冻结参数

    max_no_improved_epoch: int = 10 #提前停止器参数 当连续n个epoch模型效果均没有提升的时候, 提前结束模型训练

    device: str = "cpu"

    @property
    def model_output_dir(self) -> str:
        return os.path.join(self.output_dir, self.network_type, "models")

    @property
    def summary_dir(self) -> str:
        return os.path.join(self.output_dir, self.network_type, "logs")