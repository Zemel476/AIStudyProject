# -*- coding: utf-8 -*-
# @Time    : 2026/3/6 18:27
# @Author  : 老冰棍
# @File    : token_metrics.py
# @Software: PyCharm
from typing import List

import torch
from torch import Tensor


@torch.no_grad()
def token_accuracy(pred_scores: Tensor, input_masks: Tensor, label_ids: Tensor):
    """
    计算token级别的准确率 = 预测相同的token数目 / 总的有效token数目
    :param pred_scores: [bs, t, c]
    :param input_masks: [bs, t] 1表示实际token 0表示当前位置为填充
    :param label_ids: [bs, t]
    :return:
    """
    pred_label_id = torch.argmax(pred_scores, dim=-1) # [bs, t, C] -> [bs, t] 预测类别id
    is_equals = pred_label_id == label_ids # 预测token和实际token相等的位置为true， 不相等的位置为false
    is_equals = is_equals.to(torch.float32) # # true->1, false->0
    is_equals = is_equals * input_masks # [bs, t] * [bs, t]  ---> 实际token位置保持原样，填充token位置x0
    equal_tokens = torch.sum(is_equals.cpu())
    effective_tokens = torch.sum(input_masks.cpu())

    return equal_tokens.item(), effective_tokens.item()


def list_token_accuracy(pred_scores_lst: List[Tensor], input_masks_lst: List[Tensor], label_ids_lst: List[Tensor]):
    """
    针对多个批次的数据计算总的准确率
    :param pred_scores_lst: 预测置信度
    :param input_masks_lst: 实际的填充信息
    :param label_ids_lst: 实际类别id
    :return float准确率
    """
    _lst_num = len(pred_scores_lst)
    total_equal_tokens = 0
    total_effective_tokens = 0
    for _i in range(_lst_num):
        _equal_tokens, _effective_tokens = token_accuracy(
            pred_scores_lst[_i],
            input_masks_lst[_i],
            label_ids_lst[_i]
        )
        total_equal_tokens += _equal_tokens
        total_effective_tokens += _effective_tokens

    if total_effective_tokens <= 0:
        return 0.0

    return 1.0 * total_equal_tokens / total_effective_tokens
