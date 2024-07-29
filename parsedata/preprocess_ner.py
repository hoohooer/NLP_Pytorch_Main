import json
import pickle
import sys
import os
from transformers import BertTokenizer
from parsedata.config import *
import numpy as np
import pandas as pd


class InputExample:
    def __init__(self, text, labels=None):
        self.text = text
        self.labels = labels


class BaseFeature:
    def __init__(self, token_ids, attention_masks, token_type_ids):
        # BERT 输入
        self.token_ids = token_ids
        self.attention_masks = attention_masks
        self.token_type_ids = token_type_ids


class BertFeature(BaseFeature):
    def __init__(self, token_ids, attention_masks, token_type_ids, labels=None):
        super(BertFeature, self).__init__(
            token_ids=token_ids,
            attention_masks=attention_masks,
            token_type_ids=token_type_ids)
        self.labels = labels


def convert_bert_example(example: InputExample, tokenizer: BertTokenizer,
                         max_seq_len, label2id):
    raw_text = example.text
    labels = example.labels
    # 文本元组
    callback_info = (raw_text,)
    callback_labels = labels
    callback_info += (callback_labels,)
    encode_dict = tokenizer.encode_plus(text=raw_text,
                                        add_special_tokens=True,
                                        max_length=max_seq_len,
                                        truncation_strategy='longest_first',
                                        padding="max_length",
                                        return_token_type_ids=True,
                                        return_attention_mask=True)
    token_ids = encode_dict['input_ids']
    attention_masks = encode_dict['attention_mask']
    token_type_ids = encode_dict['token_type_ids']

    feature = BertFeature(
        token_ids=token_ids,
        attention_masks=attention_masks,
        token_type_ids=token_type_ids,
        labels=labels)
    return feature, callback_info


def convert_examples_to_features(examples, max_seq_len, bert_dir, label2id):
    tokenizer = BertTokenizer.from_pretrained("../model/chinese-roberta-small-wwm-cluecorpussmall")
    features = []
    callback_info = []
    for i, example in enumerate(examples):
        feature, tmp_callback = convert_bert_example(
            example=example,
            max_seq_len=max_seq_len,
            tokenizer=tokenizer,
            label2id=label2id,
        )
        if feature is None:
            continue

        features.append(feature)
        callback_info.append(tmp_callback)
    out = (features,)
    if not len(callback_info):
        return out
    out += (callback_info,)
    return out


def preprocess_ner(args, data_all):  # 实体识别任务预处理程序
    label2id = {}
    id2label = {}
    labels = []
    labels.append("O")
    labels.append("[PAD]")  # 用于填充文本序列
    labels.append("[CLS]")  # 用于标识句子开头
    labels.append("[SEP]")  # 用于标识句子结尾
    for sample in data_all:
        for annotation_result in sample["annotations"][0]["result"]:
            label = annotation_result["value"]["labels"][0]
            if "B-" + label not in labels:
                labels.append("B-" + label)
                labels.append("I-" + label)
                labels.append("E-" + label)
                labels.append("S-" + label)
    for i, label in enumerate(labels):
        label2id[label] = i
        id2label[i] = label
    with open(args.data_dir + '{}_id2label.json'.format(args.task_name), 'w', encoding='utf-8') as file:
        file.write(json.dumps(id2label, ensure_ascii=False))
    examples = []
    for sample in data_all:
        examples.append(parsetextandlabels(args, sample, label2id))
    out = convert_examples_to_features(examples, args.max_seq_len, args.bert_dir, label2id)
    with open(args.data_dir + '{}_data.pkl'.format(args.task_name), 'wb') as f:
        pickle.dump(out, f)

def parsetextandlabels(args, sample, label2id):
    labels = [0] * len(sample["data"]["text"])  # 首先默认全是"O",即非标签文字
    if len(sample["annotations"][0]["result"]) > 0:
        for annotation in sample["annotations"][0]["result"]:
            if annotation["value"]["end"] == annotation["value"]["start"]:
                labels[annotation["value"]["start"]] = label2id["S-" + annotation["value"]["labels"][0]]
            else:
                labels[annotation["value"]["start"]] = label2id["B-" + annotation["value"]["labels"][0]]
                labels[annotation["value"]["end"]] = label2id["E-" + annotation["value"]["labels"][0]]
                if annotation["value"]["end"] - annotation["value"]["start"] > 1:
                    labels[annotation["value"]["start"] + 1:annotation["value"]["end"]] \
                        = [label2id["I-" + annotation["value"]["labels"][0]]] * (annotation["value"]["end"] - annotation["value"]["start"] - 1)
    if len(sample["data"]["text"]) > args.max_seq_len - 2:  # 若是长度过剩，做截断。-2是因为编码时会自动在开头补一个[CLS]，在结尾补一个[SEP]。
        sample["data"]["text"] = sample["data"]["text"][:args.max_seq_len - 3]
        labels.insert(0, label2id["[CLS]"])
        labels[args.max_seq_len - 1] = label2id["[SEP]"]
        labels = labels[:args.max_seq_len]
    elif len(sample["data"]["text"]) < args.max_seq_len - 2:  # 若是长度不足，在分词器编码时会自动补全，这里只需要处理一下label真实值就好。
        labels.insert(0, label2id["[CLS]"])
        labels.append(label2id["[SEP]"])
        labels.extend([label2id["[PAD]"]] * (args.max_seq_len - len(sample["data"]["text"]) - 2))
    else:
        labels.insert(0, label2id["[CLS]"])
        labels.append(label2id["[SEP]"])
    return InputExample(text=sample["data"]["text"], labels=labels)


def test_out(data, tokenizer: BertTokenizer, max_seq_len=512):
    examples = []
    for sample in data:
        example = {}
        encode_dict = tokenizer.encode_plus(text=sample,
                                            add_special_tokens=True,
                                            max_length=max_seq_len,
                                            truncation_strategy='longest_first',
                                            padding="max_length",
                                            return_token_type_ids=True,
                                            return_attention_mask=True)
        example["token_ids"] = encode_dict["input_ids"]
        example["attention_masks"] = encode_dict["attention_mask"]
        example["token_type_ids"] = encode_dict["token_type_ids"]
        examples.append(example)
    return examples


if __name__ == '__main__':
    args = Args().get_parser()
    print('========实体识别预处理程序========')
    with open(args.data_dir + 'datas.json', encoding='utf-8') as file:
        data_all = json.load(file)
    preprocess_ner(args, data_all)
    print("已生成预处理数据文件。")
