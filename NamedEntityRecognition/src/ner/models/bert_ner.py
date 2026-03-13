# -*- coding: utf-8 -*-
# @Time    : 2026/1/15 15:16
# @Author  : 老冰棍
# @File    : bert_ner.py
# @Software: PyCharm
from typing import Optional, Union

from torch import nn
from transformers import BertModel

from ner.models.base import TokenClassifyNetwork


class BertNerNetwork(TokenClassifyNetwork):
    network_type: str = "bert"

    def __init__(self, bert_path, num_classes, freeze: Optional[Union[bool, int]] = None):
        super().__init__()

        self.bert = BertModel.from_pretrained(bert_path, weights_only=False)

        # 是否需要冻结参数
        if freeze is not None:
            if isinstance(freeze, bool):
                if freeze:
                    # 需要冻结bert的所有参数
                    for name, param in self.bert.named_parameters():
                        # True → 可学习参数  False → 固定权重 / 常量
                        param.requires_grad = False
                        print(f"冻结参数: {name}")
            elif isinstance(freeze, int) and freeze > 0:
                # 冻结前多少层(EncoderLayer层)的参数
                freeze_layers = ["embeddings."]
                for layer_idx in range(freeze):
                    freeze_layers.append(f"encoder.layer.{layer_idx}.")

                for name, param in self.bert.named_parameters():
                    param.requires_grad = True
                    if any(name.startswith(p) for p in freeze_layers):
                        param.requires_grad = False
                        print(f"冻结参数: {name}")

        _hidden_size: int = self.bert.config.hidden_size

        self.classify_layer = nn.Sequential(
            nn.Linear(_hidden_size, _hidden_size * 4),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(_hidden_size * 4, num_classes),
        )

    def extract_token_features(self, token_ids, token_masks):
        bert_output = self.bert(
            input_ids=token_ids,
            attention_mask=token_masks,
        )

        # BERT 最后一层 encoder 输出的 token 级特征 [bs, t, e]
        last_hidden_state = bert_output[0]
        return last_hidden_state

    def classify_scores(self, token_embs):
        return self.classify_layer(token_embs)


