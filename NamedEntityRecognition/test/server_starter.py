# -*- coding: utf-8 -*-
# @Time    : 2026/3/13 16:46
# @Author  : 老冰棍
# @File    : server_starter.py
# @Software: PyCharm
import os
import sys

sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..", "src")))

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

if __name__ == '__main__':
    from ner.deploy.fastapi_app_onnx import start_server

    start_server(
        model_path="./output/medical/bert/models/best.onnx",
        host="0.0.0.0",
        port=9000,
    )


