import torch
import torch.nn as nn
import numpy as np


class Multilabel_Categorical_CrossEntropy(nn.Module):
    """多标签分类的交叉熵
        说明：y_true和y_pred的shape一致，y_true的元素非0即1，
             1表示对应的类为目标类，0表示对应的类为非目标类。
        警告：请保证y_pred的值域是全体实数，换言之一般情况下y_pred
             不用加激活函数，尤其是不能加sigmoid或者softmax！预测
             阶段则输出y_pred大于0的类。如有疑问，请仔细阅读并理解
             本文。
        文章链接：https://kexue.fm/archives/7359
    """
    def __init__(self):
        super(Multilabel_Categorical_CrossEntropy, self).__init__()

    # noinspection PyMethodMayBeStatic
    def forward(self, y_pred, y_true):
        y_pred = (1 - 2 * y_true) * y_pred
        y_pred_neg = y_pred - y_true * 1e12
        y_pred_pos = y_pred - (1 - y_true) * 1e12
        zeros = torch.zeros_like(y_pred[..., :1])
        y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
        y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
        neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
        pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
        return (neg_loss + pos_loss).mean()


def getresult(outputs, targets):
    assert len(outputs) == len(targets), "预测值与实际值数量不等，请检查！"
    tp = fp = fn = tn = macro_accuracy = 0
    for i in range(len(outputs)):
        for j in range(len(outputs[i])):
            if outputs[i][j] == 1 and targets[i][j] == 1:
                tp += 1
            elif outputs[i][j] == 1 and targets[i][j] == 0:
                fp += 1
            elif outputs[i][j] == 0 and targets[i][j] == 1:
                fn += 1
            elif outputs[i][j] == 0 and targets[i][j] == 0:
                tn += 1
        if outputs[i] == targets[i]:
            macro_accuracy += 1
    macro_accuracy /= len(outputs)
    micro_accuracy = (tp + tn) / (tp + fp + fn + tn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    return macro_accuracy, micro_accuracy, f1





