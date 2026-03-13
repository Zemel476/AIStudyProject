# -*- coding: utf-8 -*-
# @Time    : 2026/3/6 18:27
# @Author  : 老冰棍
# @File    : entity_metrics.py
# @Software: PyCharm
import copy
from typing import List, Dict

import torch
from torch import Tensor

from ner.utils import trans_entity2tuple


def list_entity_metrics(
        pred_scores_lst: List[Tensor],
        input_masks_lst: List[Tensor],
        label_ids_lst: List[Tensor],
        label_id2names: Dict[int, str],
        label_pad_id=-100
):
    """
    针对多个批次的数据计算总的准确率
    :param pred_scores_lst: 预测置信度
    :param input_masks_lst: 实际的填充信息
    :param label_ids_lst: 实际类别id
    :param label_id2names: 类别id到名称的映射
    :param label_pad_id: 填充类别id
    :return: 一个字典的评估指标
    """
    label_id2names = copy.deepcopy(label_id2names)
    _lst_num = len(pred_scores_lst)

    all_pred_entities = []
    all_true_entities = []
    sample_offset = 0
    for _i in range(_lst_num):
        pred_score = pred_scores_lst[_i].cpu()
        input_masks = input_masks_lst[_i].cpu().numpy()
        label_ids = label_ids_lst[_i].cpu().numpy()

        # 基于预测置信度得到预测的类别id
        pred_ids = torch.argmax(pred_score, dim=-1).numpy() # [bs,t,c] -> [bs,t]
        pred_ids = pred_ids * input_masks + label_pad_id * (1 - input_masks)
        pred_entities = trans_entity2tuple(
            label_ids=pred_ids,
            label_id2names=label_id2names,
            label_pad_id=label_pad_id,
            offset=sample_offset,
        )
        all_pred_entities.extend(pred_entities)

        # 实际的实体列表
        true_entities = trans_entity2tuple(
            label_ids=label_ids,
            label_id2names=label_id2names,
            label_pad_id=label_pad_id,
            offset=sample_offset,
        )
        all_true_entities.extend(true_entities)

        # 更新偏移值
        sample_offset += len(label_ids)

    # 基于预测实体列表、实际的实体列表 计算评估指标
    is_eq_num = 0 # 正确的数量
    for entity in all_pred_entities:
        if entity in all_true_entities:
            is_eq_num += 1
    recall = is_eq_num / (len(all_true_entities) + 1e-6) # 召回率：正确预测数 / 实际实体数
    precision = is_eq_num / (len(all_pred_entities) + 1e-6) # 精确率：正确预测数 / 预测实体数

    def _calc_f_beta(_beta):
        _v = (1 + _beta ** 2) * (precision * recall) / (_beta ** 2 * precision + recall + 1e-6)
        return _v

    return {
        "entity_recall": recall,
        "entity_precision": precision,
        "entity_f1": _calc_f_beta(1.0), # 精确率和召回率的调和平均
        "entity_f2": _calc_f_beta(2.0), # 更重视召回率
        "entity_f0.5": _calc_f_beta(0.5), # 更重视精确率
    }














