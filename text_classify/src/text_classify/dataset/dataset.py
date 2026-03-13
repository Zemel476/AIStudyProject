# -*- coding: utf-8 -*-
# @Time    : 2026/1/4 15:40
# @Author  : 老冰棍
# @File    : dataset.py
# @Software: PyCharm
from typing import List

import torch
from torch.utils.data import Dataset

from text_classify.dataset.tokenizer import Tokenizer


class TextClassifyDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[str], tokenizer: Tokenizer):
        super().__init__()
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        # 获取index对应的原始文本和原始标签字符串
        text = self.texts[index]
        label = self.labels[index]

        # 分词 +token
        tokenizer_output = self.tokenizer(text=text, label=label)

        return{
            "text": tokenizer_output.text,
            "tokens": tokenizer_output.tokens,
            "token_ids": torch.tensor(tokenizer_output.token_ids, dtype=torch.int64),
            "token_masks": torch.ones(len(tokenizer_output.token_ids), dtype=torch.float32),
            "label": tokenizer_output.label,
            "label_id": torch.tensor(tokenizer_output.label_id, dtype=torch.int64)
        }
