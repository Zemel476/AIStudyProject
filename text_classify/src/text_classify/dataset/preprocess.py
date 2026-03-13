# -*- coding: utf-8 -*-
# @Time    : 2026/1/4 16:52
# @Author  : 老冰棍
# @File    : preprocess.py
# @Software: PyCharm
from text_classify.dataset.utils import split_text_to_tokens
from text_classify.utils import save_json


def intention_process(intention_data_file, token2id_file, label2id_file):
    """
    意图原始数据的解析构造
    """
    import pandas as pd

    df = pd.read_csv(intention_data_file, sep='\t', header=None, names=["text", "label"])
    token2cnt = {} # 以token字符串为key,以该token出现的次数为value
    label2cnt = {} # 以label 字符串为key, 以该label出现的次数为value
    text_lens = []
    for items in df.iterrows():
        text = items[1]['text'].strip()
        label = items[1]['label'].strip()
        tokens = split_text_to_tokens(text)
        for token in tokens:
            token2cnt[token] = token2cnt.get(token, 0) + 1

        label2cnt[label] = label2cnt.get(label, 0) + 1
        text_lens.append(len(tokens))

    #　基于单词数量构建词典
    token2ids = {
        "<PAD>": 0,
        "<UNK>": 1
    }

    for token, cnt in token2cnt.items():
        # 一般情况下出现次数太少的单词直接过滤
        if cnt < 3:
            continue
        token2ids[token] = len(token2ids)
    save_json(token2id_file, token2ids)

    label2ids = {}
    for label in label2cnt.keys():
        label2ids[label] = len(label2ids)
    save_json(label2id_file, label2ids)


def senti_corp_process(sentiment_data_file, token2id_file, label2id_file):
    import pandas as pd

    df = pd.read_csv(sentiment_data_file, sep=',')
    df = df[["label", "review"]]
    df.columns = ["label", "text"]

    token2cnt = {}
    label2cnt = {}
    text_lens = []
    for items in df.iterrows():
        text = str(items[1]["text"]).strip()
        label = str(items[1]["label"]).strip()
        tokens = split_text_to_tokens(text)
        for token in tokens:
            token2cnt[token] = token2cnt.get(token, 0) + 1

        label2cnt[label] = label2cnt.get(label, 0) + 1
        text_lens.append(len(tokens))


    token2ids = {
        "<PAD>": 0,
        "<UNK>": 1
    }
    for token, cnt in token2cnt.items():
        if cnt < 3:
            continue
        token2ids[token] = len(token2ids)
    save_json(token2id_file, token2ids)

    label2ids = {}
    for label in label2cnt.keys():
        label2ids[label] = len(label2ids)
    save_json(label2id_file, label2ids)












