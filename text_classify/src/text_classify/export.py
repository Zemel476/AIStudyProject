# -*- coding: utf-8 -*-
# @Time    : 2026/1/5 18:18
# @Author  : 老冰棍
# @File    : export.py
# @Software: PyCharm
import json
import os.path

import torch

from text_classify.config import Config

# 模型静态化转换
def export_jit(cfg: Config, model_name: str = "best.pkl"):
    """
    将Pytorch训练好的模型转换为TorchScript
    :return:
    """
    # 模型恢复
    tokenizer = cfg.tokenizer
    ckpt = torch.load(os.path.join(cfg.model_output_dir, model_name), map_location="cpu")
    net = ckpt["net"]
    net.cpu().eval()

    # 静态转换
    jit_net = torch.jit.trace(
        net,
        (torch.randint(0, 100, (10, 128)),  # # input_ids
         torch.ones(10, 128)), #  # attention_mask
    )

    _network_type = ""
    try:
        _network_type = net.network_type
    except:
        pass

    torch.jit.save(
        jit_net,
        os.path.join(cfg.model_output_dir, f"{os.path.splitext(model_name)[0]}.pt"),
        _extra_files={
            "token2ids.txt": json.dumps(tokenizer.token2ids, ensure_ascii=False),
            "label2ids.txt": json.dumps(tokenizer.label2ids, ensure_ascii=False),
            "network_type.txt": _network_type,
        }
    )


def export_onnx(cfg: Config, model_name: str = "best.pkl"):
    """
    将Pytorch训练好的模型转换为onnx
    """
    tokenizer = cfg.tokenizer
    ckpt = torch.load(os.path.join(cfg.model_output_dir, model_name), map_location="cpu", weights_only=False)
    net = ckpt["net"]
    net.cpu().eval()
    _network_type = ""
    try:
        _network_type = net.network_type
    except Exception as e:
        pass

    # 转换为onnx格式
    onnx_file = os.path.join(cfg.model_output_dir, f"{os.path.splitext(model_name)[0]}.onnx")
    torch.onnx.export(
        net.cpu(),
        (torch.randint(0, 100, (10, 128)), torch.ones((10, 128))),
        onnx_file,
        verbose=False,
        opset_version=14,
        do_constant_folding=True,
        input_names=["token_ids", "token_masks"],
        output_names=["scores"],
        dynamic_axes={
            "token_ids": {0: "bs", 1: "t"},
            "token_masks": {0: "bs", 1: "t"},
            "scores": {0: "bs"},
        }
    )

    # checks
    import onnx
    model_onnx = onnx.load(onnx_file)
    onnx.checker.check_model(model_onnx)

    # 添加元数据
    d = {
        "label2ids.txt": json.dumps(tokenizer.label2ids, ensure_ascii=False),
        # "token2ids": json.dumps(tokenizer.token2ids, ensure_ascii=False),
        "network_type.txt": _network_type,
    }

    for k, v in d.items():
        meta = model_onnx.metadata_props.add()
        meta.key = k
        meta.value = str(v)

    onnx.save(model_onnx, onnx_file)
