# -*- coding: utf-8 -*-
# @Time    : 2026/3/13 16:51
# @Author  : 老冰棍
# @File    : china_people_daily.py
# @Software: PyCharm
import json
import os
import sys

from ner.config import Config
from ner.datas.tokenizer import Tokenizer
from ner.trainer.trainer import Trainer

project_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(project_root, "src"))

def t0():
    """
    将原始数据进行转换  --> 转换为当前训练点支持的结构
    :return:
    """
    in_file = "./datas/china-people-daily-ner-corpus/min_example.train"
    out_file = "./datas/china-people-daily-ner-corpus/min_train.txt"

    # 创建输出文件夹
    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    records = []

    def _build_record(_tokens, _labels):
        if len(_tokens) == 0:
            return None

        _entities = []
        _label_type = None
        _start_pos = None
        _end_pos = None
        for _i, _label in enumerate(_labels):
            if _label.startswith("B-"):
                if _label_type is not None:
                    _entities.append({
                        "label_type": _label_type,
                        "overlap": 0,
                        "start_pos": _start_pos, # 包含
                        "end_pos": _end_pos # 不包含
                    })
                _start_pos = _i # 表示从当前token开始属于某个实体
                _label_type = _label[2:]
                _end_pos = _i + 1
            elif _label.startswith("I-"):
                _end_pos = _i + 1
            else:
                if _label_type is not None:
                    _entities.append({
                        "label_type": _label_type,
                        "overlap": 0,
                        "start_pos": _start_pos, # 包含
                        "end_pos": _end_pos  # 不包含
                    })
                _label_type = None
                start_pos = None
                end_pos = None
        # 最后最后的token属于实体的情况
        if _label_type is not None:
            _entities.append({
                "label_type": _label_type,
                "overlap": 0,
                "start_pos": _start_pos, # 包含
                "end_pos": _end_pos  # 不包含
            })

        return {
            "originalText": "".join(_tokens),
            "entities": _entities,
        }

    with open(in_file, "r", encoding="utf-8") as reader:
        tokens = []
        labels = []
        for line in reader:
            line = line.strip()
            if line == "":
                # 表示当前行的token属于当前文本
                token, label = line.split(" ")
                tokens.append(token)
                labels.append(label)
            else:
                record = _build_record(tokens, labels)
                if record is not None:
                    records.append(record)
                # 开启新一个样本
                tokens = []
                labels = []
        # 处理最后一个样本
        record = _build_record(tokens, labels)
        if record is not None:
            records.append(record)

    with open(out_file, "w", encoding="utf-8") as writer:
        for record in records:
            record = json.dumps(record, ensure_ascii=False)
            writer.writelines(f"{record}\n")

def t1():
    """
    训练
    基于数据构造训练依赖的元数据
    执行下列训练代码 -- 给定好对应的各种参数
    """

    bert_path = r"D:\cache\huggingface\hub\models--bert-base-chinese"
    if not os.path.exists(bert_path):
        bert_path = "bert-base-chinese"

    tokenizer = Tokenizer(vocabs=r"./datas/vocab.txt")
    data_root_dir= r"./datas/china-people-daily-ner-corpus/ner"
    config = Config(
        output_dir="./output/china-people-daily-ner-corpus/",
        tokenizer=tokenizer,

        label2id=os.path.join(data_root_dir, "label2id.json"), # 类别标签名称到id的映射
        train_file=os.path.join(data_root_dir, "min_training.txt"), # 训练数据对应文件
        eval_file= os.path.join(data_root_dir, "min_test.txt"),

        total_epoch=3,
        batch_size=32, # 批次大小
        lr=0.01, # 模型训练学习率

        network_type="bert",  # 网络类型 可选lstm、bert
        lstm_layers=3, # LSTM 层数
        lstm_hidden_size=768,
        bert_path=bert_path, # Bert模型迁移路径
        max_length=512,
        freeze=True,  # 给定迁移模型的时候冻结参数
        device="cuda",
        max_no_improved_epoch=20
    )

    trainer = Trainer(config=config)
    trainer.training()


def t2():
    import requests

    url = "http://127.0.0.1:9001/predict"

    def tt_get():
        text = "在发达国家，急救保险十分普及，已成为社会保障体系的重要组成部分。"
        response = requests.get(
            url, json={'text': text}
        )
        if response.status_code == 200:
            print("访问服务器成功GET")
            # 明确的知道服务器返回的是json格式，那么可以直接调用json()这个方法转换为字典对象
            result = response.json()
            print(result)
            print(type(result))
            if result['code'] == 0:
                print(f"调用模型成功GET，结果为:{result['data']}")
            else:
                print(f"调用模型服务器处理异常，异常信息为:{result['msg']}")
        else:
            print("访问服务器失败：网络异常等等")

    def tt_post():
        # NOTE: 在fastapi框架中，post请求的参数必须通过json参数给定
        text = "在发达国家，急救保险十分普及，已成为社会保障体系的重要组成部分。"
        response = requests.post(
            url,
            json={'text': text},
        )
        if response.status_code == 200:
            print("访问服务器成功POST")
            # 明确的知道服务器返回的是json格式，那么可以直接调用json()这个方法转换为字典对象
            result = response.json()
            print(result)
            print(type(result))
            if result['code'] == 0:
                print(f"调用模型成功POST，结果为:{result['data']}")
            else:
                print(f"调用模型服务器处理异常，异常信息为:{result['msg']}")
        else:
            print("访问服务器失败：网络异常等等")

    tt_get()
    tt_post()

if __name__ == '__main__':
    # t0()  # 将原始数据转换
    # t1()  # 模型训练
    t2()  # 访问接口模型
