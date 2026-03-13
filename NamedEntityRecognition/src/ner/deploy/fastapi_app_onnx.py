# -*- coding: utf-8 -*-
# @Time    : 2026/3/6 17:50
# @Author  : 老冰棍
# @File    : fastapi_app_onnx.py
# @Software: PyCharm
import logging
import os
from typing import Optional

import uvicorn
from fastapi import FastAPI, Body
from ner.deploy.onnx_predictor import Predictor

app = FastAPI()
# 定义一个变量
predictor: Optional[Predictor] = None

@app.get("/")
def index():
    return "基于FastAPI的算法API部署后端框架 + 命名实体识别 + ONNX"


async def predict(
        text: str = Body(..., description="待预测的文本text"),
        empty_param: int = Body(1, description="占位")
):
    try:
        if text is None or len(text) == 0:
            return {"code": 2, "msg": f"请求参数异常，请给定有效请求参数 [{text}]"}

        pred_result = predictor.predict(text=text)

        # 将模型预测结果转换返回给调用方
        return {"code": 0, "msg": "success", "data": pred_result, "text": text}
    except Exception as e:
        error_msg = f"服务器后端异常 {e}"
        logging.error(error_msg, exc_info=True)

        return {"code": 1, "msg": error_msg}


def start_server(model_path=None, host="0.0.0.0", port=9000):
    global predictor

    app.get("/predict", summary="命名实体识别")(predict)
    app.post("/predict", summary="命名实体识别")(predict)

    predictor = Predictor(
        onnx_model_path=model_path or os.environ["MODEL_PATH"]
    )

    uvicorn.run(app, host=host, port=port, log_level="info")
