# -*- coding: utf-8 -*-
# @Time    : 2026/1/15 14:48
# @Author  : 老冰棍
# @File    : __init__.py.py
# @Software: PyCharm
from ner.config import Config
from ner.models.bert_ner import BertNerNetwork
from ner.models.bilstm_ner import BiLSTMNerNetwork


def build_network(config: Config):
    net_type = config.network_type.lower()
    if net_type == 'lstm':
        return BiLSTMNerNetwork(
            vocab_size=config.vocab_size,
            hidden_size=config.lstm_hidden_size,
            num_classes=config.num_classes,
            num_layers=config.lstm_layers,
        )
    else:
        return BertNerNetwork(
            bert_path=config.bert_path,
            num_classes=config.num_classes,
            freeze=config.freeze
        )
