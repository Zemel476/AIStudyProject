# -*- coding: utf-8 -*-
# @Time    : 2026/1/9 17:35
# @Author  : 老冰棍
# @File    : tt_sentiment.py
# @Software: PyCharm
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), "src")))

def split_train_test():
    import pandas as pd
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(r"./datas/sentiment/ChnSentiCorp_htl_all.csv", sep=",")
    df = df[["review", "label"]]
    df.columns = ["text", "label"]
    df = df.dropna() # 有一条数据的text为空

    df0, df1 = train_test_split(df, test_size=0.1, random_state=42)
    print(df0.shape, df1.shape)

    df0.to_csv(r"./datas/sentiment/train0.csv", sep="\t",index=None,header=None)
    df1.to_csv(r"./datas/sentiment/val0.csv", sep="\t",index=None,header=None)


def run_preprocess():
    """
    数据预处理
    """
    # 下列代码能够正常运行的前提是: text_classify 它所在的文件夹必须在sys.path环境变量中
    from text_classify.dataset import preprocess

    preprocess.senti_corp_process(
        sentiment_data_file=r"./datas/sentiment/ChnSentiCorp_htl_all.csv",
        token2id_file = r"./output/sentiment/token2id.json",
        label2id_file = r"./output/sentiment/label2id.json",
    )

def run_train_lstm():
    from text_classify.dataset.tokenizer import Tokenizer
    from text_classify.config import Config
    from text_classify.trainer.trainer import Trainer
    from text_classify.utils import load_json

    tokenizer = Tokenizer(
        token2ids=load_json(r"./output/sentiment/token2id.json"),
        label2ids=load_json(r"./output/sentiment/label2id.json")
    )

    config = Config(
        model_output_dir="./output/sentiment/lstm/models",
        summary_dir="./output/sentiment/lstm/logs",
        tokenizer=tokenizer,
        train_file="./datas/sentiment/train0.csv",
        eval_file="./datas/sentiment/val0.csv",
        total_epoch=3,
        batch_size=32,
        hidden_size=128,
        lr=0.01,
        device="cuda",
    )

    trainer = Trainer(config)
    trainer.training()
    # 针对转换后的静态结构，我们可以通过 https://netron.app/ 查看结构

def run_predictor():
    from text_classify.deploy.jit_predictor import Predictor

    p = Predictor(
        jit_model_path="./output/sentiment/lstm/models/best.pt"
    )

    while True:
        v = input("请输入文字: ")
        if v == "q":
            break

        result = p.predict(v)
        print(result)


if __name__ == '__main__':
    # split_train_test()
    # run_preprocess()
    # run_train_lstm()
    run_predictor()