import json
import pickle
import sys
import os
from transformers import BertTokenizer
from parsedata.config import *
import random
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
    # 转换为one-hot编码
    label_ids = [0 for _ in range(len(label2id))]
    for label in labels:
        label_ids[label2id[label]] = 1
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
        labels=label_ids)
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


def preprocess_tc(args, data_all):  # 多标签分类任务预处理程序
    random.shuffle(data_all)
    label2id = {}
    id2label = {}
    labels = []
    for sample in data_all:
        for label in sample["labels"]:
            if label not in labels:
                labels.append(label)
    for i, label in enumerate(labels):
        label2id[label] = i
        id2label[i] = label
    with open(args.data_dir + '{}_id2label.json'.format(args.task_name), 'w', encoding='utf-8') as file:
        file.write(json.dumps(id2label, ensure_ascii=False))
    examples = []
    max_text_len = 0
    for sample in data_all:
        examples.append(InputExample(text=sample["text"],
                                     labels=sample["labels"]))
        if len(sample["text"]) > max_text_len:
            max_text_len = len(sample["text"])
    print("最大文本长度：" + str(max_text_len))
    out = convert_examples_to_features(examples, args.max_seq_len, args.bert_dir, label2id)
    with open(args.data_dir + '{}_data.pkl'.format(args.task_name), 'wb') as f:
        pickle.dump(out, f)


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
    print('========多标签分类预处理程序========')
    with open(args.data_dir + 'datas.json', encoding='utf-8') as file:
        data_all = json.load(file)
    preprocess_tc(args, data_all)
    print("已生成预处理数据文件。")
