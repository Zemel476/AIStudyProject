# -*- coding: utf-8 -*-
# @Time    : 2026/1/13 16:57
# @Author  : 老冰棍
# @File    : server_runner.py
# @Software: PyCharm
import os
import sys

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), "src")))


def start_flask_jit():
    from text_classify.deploy.flask_app import start_server


    os.environ['MODEL_PATH'] = './output/intention/bert/models/best.pt'
    start_server(
        model_path='./output/intention/bert/models/best.pt',
        host="0.0.0.0",
        port=5000
    )


def start_flask_onnx():
    from text_classify.deploy.flask_app_onnx import start_server

    root_dir = os.path.dirname(__file__)
    model_path = os.path.join(root_dir, 'output', 'intention', 'bert', 'models', 'best.onnx')
    os.environ['MODEL_PATH'] = os.path.abspath(model_path)

    start_server(
        # model_path='./output/intention/bert1/deploy/best.pt',
        # host="0.0.0.0",
        # port=5000
    )


def start_fastapi_onnx():
    from text_classify.deploy.fastapi_app_onnx import start_server

    root_dir = os.path.dirname(__file__)
    model_path = os.path.join(root_dir, 'output', 'intention', 'bert', 'models', 'best.onnx')
    os.environ['MODEL_PATH'] = os.path.abspath(model_path)

    start_server(
        # model_path='./output/intention/bert1/deploy/best.pt',
        # host="0.0.0.0",
        # port=5000
    )

if __name__ == '__main__':
    # start_flask_jit()
    # start_flask_onnx()
    start_fastapi_onnx()

