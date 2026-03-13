# -*- coding: utf-8 -*-
# @Time    : 2026/1/15 17:09
# @Author  : 老冰棍
# @File    : bilstm_ner.py
# @Software: PyCharm
import torch
from torch import nn

from ner.models.base import TokenClassifyNetwork


class BiLSTMNerNetwork(TokenClassifyNetwork):
    network_type = "lstm"

    def __init__(self, vocab_size, hidden_size, num_classes, num_layers=3):
        super().__init__()

        self.emb_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=hidden_size)
        self.lstm_layer = nn.ModuleList([
            nn.LSTM(
                input_size=hidden_size if i == 0 else hidden_size * 2,
                hidden_size=hidden_size, # 控制单向 LSTM 的隐藏维度
                num_layers=1,
                batch_first=True,
                bidirectional=True # 启用双向lstm
            ) for i in range(num_layers)
        ])

        self.classify_layer = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size * 4),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_size * 4, num_classes)
        )

    def extract_token_features(self, token_ids, token_masks):
        # embedding得到token向量 [bs, t, e]
        token_embs = self.emb_layer(token_ids)

        # lstm的迭代特征向量的提取, [bs, t, 2e]
        # input_embs = torch.concat([token_embs, token_embs], dim=-1)
        input_embs = token_embs
        for i, lstm in enumerate(self.lstm_layer):
           lstm_output,_ = lstm(input_embs)
           if i == 0:
               input_embs = lstm_output
           else:
               input_embs = input_embs + lstm_output # 残差
        return input_embs

    def classify_scores(self, token_embs):
        return self.classify_layer(token_embs)





