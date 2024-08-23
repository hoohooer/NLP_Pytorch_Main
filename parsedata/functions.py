import torch
import torch.nn as nn
import pandas as pd


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


def getresult(args, outputs, targets):
    assert len(outputs) == len(targets), "预测值与实际值数量不等，请检查！"
    tp = fp = fn = tn = macro_accuracy = 0
    if args.task_type_detail == "singlelabel" or args.task_type_detail == "pipeline_nered":
        for i in range(len(outputs)):
            if outputs[i] == targets[i]:
                tp += 1
                tn += args.num_tags - 1
                macro_accuracy += 1
            else:
                fn += 1
                fp += 1
                tn += args.num_tags - 2
        macro_accuracy /= len(outputs)
    elif args.task_type_detail == "multilabels":
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
    elif args.task_type == "ner":
        for i in range(len(outputs)):
            for j in range(len(outputs[i])):
                if outputs[i][j] == targets[i][j]:
                    tp += 1
                    tn += args.num_tags - 1
                    macro_accuracy += 1
                else:
                    fn += 1
                    fp += 1
                    tn += args.num_tags - 2
        macro_accuracy /= len(outputs) * args.max_seq_len
    
    micro_accuracy = (tp + tn) / (tp + fp + fn + tn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    return macro_accuracy, micro_accuracy, f1


def getreport(args, outputs, targets, labels):
    assert len(outputs) == len(targets), "预测值与实际值数量不等，请检查！"
    if args.task_type == "ner":  # 去除填充标签，将实体内容相同标签合为一个
        unique_labels = [label.replace('B-', '').replace('I-', '').replace('E-', '').replace('S-', '') for label in labels if label not in {'O','[PAD]','[CLS]','[SEP]'}]
        unique_labels = list(set(unique_labels))
        reportdf = pd.DataFrame(0, index=unique_labels, columns=['TP','FP','FN','precision','recall','f1-score','support']).astype(float)
    else:
        reportdf = pd.DataFrame(0, index=labels, columns=['TP','FP','FN','precision','recall','f1-score','support']).astype(float)
    if args.task_type_detail == "singlelabel" or args.task_type_detail == "pipeline_nered":
        for i in range(len(outputs)):
            if outputs[i] == targets[i]:
                reportdf.iloc[outputs[i]]['TP'] += 1
            else:
                reportdf.iloc[outputs[i]]['FP'] += 1
                reportdf.iloc[targets[i]]['FN'] += 1
    elif args.task_type_detail == "multilabels":
        for i in range(len(outputs)):
            for j in range(len(outputs[i])):
                if outputs[i][j] == 1 and targets[i][j] == 1:
                    reportdf.iloc[j]['TP'] += 1
                elif outputs[i][j] == 1 and targets[i][j] == 0:
                    reportdf.iloc[j]['FP'] += 1
                elif outputs[i][j] == 0 and targets[i][j] == 1:
                    reportdf.iloc[j]['FN'] += 1
    elif args.task_type == "ner":
        for i in range(len(outputs)):
            outputs_labels = []
            targets_labels = []
            for j in range(len(outputs[i])):
                if labels[targets[i][j]][0] == 'S':
                    targets_labels.append((labels[targets[i][j]].replace('S-', ''), targets[i][j], targets[i][j]))
                elif labels[targets[i][j]][0] == 'B':
                    targets_labels.append((labels[targets[i][j]].replace('B-', ''), j, next((index for index in range(j, len(targets[i])) if labels[targets[i][index]][0] == 'E'), len(targets[i]) - 2)))
                if labels[outputs[i][j]][0] == 'S':
                    outputs_labels.append((labels[outputs[i][j]].replace('S-', ''), outputs[i][j], outputs[i][j]))
                elif labels[outputs[i][j]][0] == 'B':
                    outputs_labels.append((labels[outputs[i][j]].replace('B-', ''), j, next((index for index in range(j, len(outputs[i])) if labels[outputs[i][index]][0] == 'E'), len(targets[i]) - 2)))
            for p in range(len(outputs_labels) - 1, 0, -1):
                for q in range(len(targets_labels) - 1, 0, -1):
                    if outputs_labels[p] == targets_labels[q]:
                        reportdf.loc[outputs_labels[p][0]]['TP'] += 1
                        outputs_labels.pop(p)
                        targets_labels.pop(q)
                        break
            for outputs_label in outputs_labels:
                reportdf.loc[outputs_label[0]]['FP'] += 1
            for targets_label in targets_labels:
                reportdf.loc[targets_label[0]]['FN'] += 1
    for i in range(len(reportdf)):
        reportdf.iloc[i]['precision'] = reportdf.iloc[i]['TP'] / (1e-10 + reportdf.iloc[i]['TP'] + reportdf.iloc[i]['FP'])
        reportdf.iloc[i]['recall'] = reportdf.iloc[i]['TP'] / (1e-10 + reportdf.iloc[i]['TP'] + reportdf.iloc[i]['FN'])
        reportdf.iloc[i]['f1-score'] = 2 * reportdf.iloc[i]['precision'] * reportdf.iloc[i]['recall'] / (1e-10 + reportdf.iloc[i]['precision'] + reportdf.iloc[i]['recall'])
        reportdf.iloc[i]['support'] = reportdf.iloc[i]['TP'] + reportdf.iloc[i]['FN']
    reportdf.loc['合计'] = [0, 0, 0, 0, 0, 0, 0]
    column_sums = reportdf.iloc[:-1, :3].sum().values
    reportdf.iloc[-1, :3] = column_sums
    reportdf.iloc[-1]['precision'] = reportdf.iloc[-1]['TP'] / (1e-10 + reportdf.iloc[-1]['TP'] + reportdf.iloc[-1]['FP'])
    reportdf.iloc[-1]['recall'] = reportdf.iloc[-1]['TP'] / (1e-10 + reportdf.iloc[-1]['TP'] + reportdf.iloc[-1]['FN'])
    reportdf.iloc[-1]['f1-score'] = 2 * reportdf.iloc[-1]['precision'] * reportdf.iloc[-1]['recall'] / (1e-10 + reportdf.iloc[-1]['precision'] + reportdf.iloc[-1]['recall'])
    reportdf.iloc[-1]['support'] = reportdf.iloc[-1]['TP'] + reportdf.iloc[-1]['FN']
    return reportdf.iloc[:, 3:].to_string()
