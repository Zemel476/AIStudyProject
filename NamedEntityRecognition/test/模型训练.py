# -*- coding: utf-8 -*-
# @Time    : 2026/3/6 14:34
# @Author  : 老冰棍
# @File    : 模型训练.py
# @Software: PyCharm

import os

from ner.config import Config
from ner.datas.tokenizer import Tokenizer
from ner.trainer.trainer import Trainer

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


def t0():
    import sys

    sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..", "src")))

    bert_path = r"D:\cache\huggingface\hub\models--bert-base-chinese"
    if not os.path.exists(bert_path):
        bert_path = "bert-base-chinese"

    tokenizer = Tokenizer(vocabs=r"./datas/medical/vocab.txt")
    config = Config(
        output_dir="./output/medical/",
        tokenizer=tokenizer,
        label2id="./datas/medical/label2id.json",  # 类别标签名称到id的映射
        train_file="./datas/medical/training.txt", # 训练数据对应文件
        eval_file="./datas/medical/test.json",   # 模型评估数据对应文件
        total_epoch=50,
        batch_size=32, # 批次大小
        lr=0.01,  # 模型训练学习率

        network_type="bert", # 可选 lstm
        lstm_layers=3,  # LSTM的层数
        lstm_hidden_size=768,
        bert_path=bert_path,  # Bert模型迁移路径
        max_length=512,
        freeze=True,  # 给定迁移模型的时候冻结参数
        device="cuda",
        max_no_improved_epoch=20
    )

    trainer = Trainer(config)
    trainer.training()


if __name__ == '__main__':
    t0()