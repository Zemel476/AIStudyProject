# -*- coding: utf-8 -*-
# @Time    : 2026/1/13 15:51
# @Author  : 老冰棍
# @File    : fastapi_app_onnx.py
# @Software: PyCharm
import logging
import os
from typing import Optional

import uvicorn
from fastapi import FastAPI, Body

from text_classify.deploy.onnx_predictor import Predictor

# 构建应用
app = FastAPI()
predictor: Optional[Predictor] = None

@app.get("/")
async def index():
    return "基于FastAPI的算法API部署后端框架 + 文本分类 + ONNX"


async def predict(
        text: str=Body(..., description="待遇测文本text"),
        topk: int=Body(..., description="获取预测概率最大的前K个值")
):
    try:
        if not text:
            return {"code": 2, "msg": f"参数请求异常，请给定有效请求参数 [{text}]"}

        if topk <= 0:
            topk = 1

        pred_result = predictor.predict(x=text, k=topk)

        return {"code": 0, "msg": "success", "data": pred_result, "text": text}
    except Exception as e:
        error_msg = f"服务器后端异常{e}"
        logging.error(error_msg)
        return {"code": 1, "msg": error_msg}


def start_server(model_path=None, host="0.0.0.0", port=9001):
    global predictor
    app.get("/predict", summary="文本分类预测方法")(predict)
    app.post("/predict", summary="文本分类预测方法")(predict)

    predictor = Predictor(
        onnx_model_path=model_path or os.environ['MODEL_PATH']
    )

    uvicorn.run(app, host=host, port=port, log_level="info")