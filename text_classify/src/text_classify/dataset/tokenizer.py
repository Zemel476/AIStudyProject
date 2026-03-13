# -*- coding: utf-8 -*-
# @Time    : 2026/1/4 15:43
# @Author  : 老冰棍
# @File    : tokenizer.py
# @Software: PyCharm
# DESC : 分词器
from dataclasses import dataclass
from typing import List, Optional, Dict

from transformers import BertTokenizer

from text_classify.dataset.utils import split_text_to_tokens


@dataclass
class TokenizerOutput:
    text: str
    tokens: List[str]
    token_ids: List[int]
    label: Optional[str] = None
    label_id: Optional[int] = 0


class TokenizerBase:

    def __call__(self, text: str, label: Optional[str] = None):
        raise NotImplementedError("子类实现")

    @property
    def vocab_size(self):
        raise NotImplementedError("子类实现")

    @property
    def num_classes(self):
        raise NotImplementedError("子类实现")

    @property
    def pad_token_id(self):
        raise NotImplementedError("子类实现")

    @property
    def unk_token_id(self):
        raise NotImplementedError("子类实现")

    @property
    def token2ids(self):
        raise NotImplementedError("子类实现")

    @property
    def label2ids(self):
        raise NotImplementedError("子类实现")


class Tokenizer(TokenizerBase):
    def __init__(self,
                 token2ids: Dict[str, int],
                 label2ids: Dict[str, int],
                 unk_token = "<UNK>",
                 pad_token = "<PAD>"
                 ):
        super().__init__()
        self._token2ids = token2ids
        self._unk_token_id = self._token2ids[unk_token]
        self._pad_token_id = self._token2ids[pad_token]
        self._label2ids = label2ids

    def __call__(self, text: str, label: Optional[str] = None) -> TokenizerOutput:
        # 分词
        tokens = split_text_to_tokens(text)

        # 将token转换为id
        token_ids = [self._token2ids.get(token, self._unk_token_id) for token in tokens]

        # 标签转换
        label_id = None
        if label is not None:
            label = str(label)
            label_id = self._label2ids.get(label)

        return TokenizerOutput(text=text, tokens=tokens, token_ids=token_ids, label=label, label_id=label_id)


    @property
    def vocab_size(self):
        return len(self._token2ids)

    @property
    def num_classes(self):
        return len(self._label2ids)

    @property
    def pad_token_id(self):
        return self._pad_token_id

    @property
    def unk_token_id(self):
        return self._unk_token_id

    @property
    def token2ids(self):
        return self._token2ids

    @property
    def label2ids(self):
        return self._label2ids

# 使用Bert分词
class ProxyBertTokenizer(TokenizerBase):
    def __init__(self,
                 bert_tokenizer_file: str,
                 label2ids: Dict[str, int] # 标签名到ID的映射字典
                 ):
        super().__init__()

        self.proxy: BertTokenizer = BertTokenizer.from_pretrained(bert_tokenizer_file)
        self._label2ids = label2ids


    def __call__(self, text: str, label: Optional[str] = None) -> TokenizerOutput:
        # 分词
        tokens = self.proxy.tokenize(text)

        # 将每个token转换为 id
        token_ids = self.proxy(text)["input_ids"]

        #标签转换
        label_id = None
        if label is not None:
            label = str(label)
            label_id = self._label2ids.get(label)

        return TokenizerOutput(text=text, tokens=tokens, token_ids=token_ids, label=label, label_id=label_id)

    @property
    def vocab_size(self):
        return self.proxy.vocab_size

    @property
    def num_classes(self):
        return len(self._label2ids)

    @property
    def pad_token_id(self):
        return self.proxy.pad_token_id

    @property
    def unk_token_id(self):
        return self.proxy.unk_token_id

    @property
    def token2ids(self):
        return dict(self.proxy.vocab.copy())

    @property
    def label2ids(self):
        return self._label2ids
