# -*- coding: utf-8 -*-
# @Time    : 2026/1/13 14:03
# @Author  : 老冰棍
# @File    : onnx_predictor.py
# @Software: PyCharm
import json
import os

import numpy as np
import onnxruntime

from text_classify.dataset.tokenizer import ProxyBertTokenizer


class Predictor(object):

    def __init__(self, onnx_model_path):
        super().__init__()

        # 模型恢复
        # providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
        providers = ["CPUExecutionProvider"]
        session = onnxruntime.InferenceSession(onnx_model_path, providers=providers)
        self.session = session
        meta = session.get_modelmeta().custom_metadata_map

        label2ids = json.loads(meta["label2ids.txt"])
        self.id2labels = {_id: _label for _label, _id in label2ids.items()}

        network_type = meta["network_type.txt"]
        if not isinstance(network_type, str):
            network_type = str(network_type, encoding="utf-8")

        if network_type == "bert":
            self.tokenizer = ProxyBertTokenizer(os.path.dirname(onnx_model_path), label2ids)
        else:
            raise ValueError("不支持非bert模型恢复！")

    def predict(self, x:str, k: int=1):
        # 分词
        token_result = self.tokenizer(x)
        # 构造模型输入数据
        token_ids = np.asarray([token_result.token_ids], dtype=np.int64)
        token_masks  = np.ones_like(token_ids, dtype=np.float32)

        # 调用模型
        scores = self.session.run(
            ["scores"],
            {"token_ids": token_ids, "token_masks": token_masks}
        )
        scores = scores[0][0]

        # 模型结果处理
        k = max(min(k, len(self.id2labels)), 1)
        probs = np.exp(scores) / np.sum(np.exp(scores))
        topk_indices = np.argsort(probs)[-k:][::-1]
        topk_class_names = [self.id2labels[_id] for _id in topk_indices]
        final_result = []
        for cls_idx, cls_name in zip(topk_indices, topk_class_names):
            final_result.append({
                "cls_idx": int(cls_idx),
                "cls_name": cls_name,
                "prob": float(probs[cls_idx].round(3)),
            })

        return final_result
