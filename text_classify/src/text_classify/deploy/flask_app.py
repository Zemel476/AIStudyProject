# -*- coding: utf-8 -*-
# @Time    : 2026/1/13 15:51
# @Author  : 老冰棍
# @File    : flask_app.py
# @Software: PyCharm
import logging
import os
from typing import Optional

from flask import Flask, request, jsonify
from flask_cors import CORS

from text_classify.deploy.jit_predictor import Predictor

app = Flask(__name__)
CORS(app)  # 支出跨域请求


predictor: Optional[Predictor] = None

@app.route('/')
def index():
    return "基于Flask的算法API部署后端框架 + 文本分类模型API"

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    try:
        _args = None
        if request.method == 'POST':
            # 当请求方式为POST的时候，一般的参数直接从request.form(form是一个字典)中按照参数名称直接获取即可
            # 当请求方式为POST的时候，如果传递的数据为文件的话，需要从request.files中获取
            _args = request.form
            if len(_args) == 0:
                try:
                    _args = request.get_json()
                except Exception as e:
                    raise ValueError(f"从request json中获取参数数据异常 {e}")
        elif request.method == 'GET':
            # 当请求方式为GET的时候，参数直接从request.args(args是一个字典)中按照参数名称直接获取即可
            _args = request.args
        else:
            raise ValueError("当前服务仅支持GET和POST请求方式!")

        if _args is None:
            raise ValueError("获取请求参数对象为None， 请检查!")

        text = _args['text']
        topk = int(_args.get('topk', 1))

        # 参加的检查 过滤 转换
        if not text:
            return jsonify({'code': 2, 'msg': f'请求参数异常，请给定有效请求参数 [{text}]'})

        if topk <= 0:
            topk = 1

        pred_result = predictor.predict(
            x=text,
            k=topk
        )

        # 将模型预测结果转换返回给调用方
        return jsonify({'code': 0, 'msg': '成功', 'data': pred_result, 'text': text})
    except Exception as e:
        error_msg = f"服务器后端异常 {e}"
        logging.error(error_msg, exc_info=e)
        return jsonify({'code': 1, 'msg': error_msg})


def start_server(model_path, host="0.0.0.0", port=5000):
    global predictor

    predictor = Predictor(
        jit_model_path=model_path or os.environ['MODEL_PATH']
    )

    app.run(host=host, port=port)