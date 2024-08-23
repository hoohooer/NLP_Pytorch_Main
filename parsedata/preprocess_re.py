import json
import pickle
import sys
import os
from transformers import BertTokenizer
from parsedata.argsconfig import *
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
    encode_dict = tokenizer.encode_plus(text=raw_text,
                                        add_special_tokens=True,
                                        max_length=max_seq_len,
                                        truncation='longest_first',
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
    tokenizer = BertTokenizer.from_pretrained(bert_dir)
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


def preprocess_re(args, data_all):  # 关系抽取任务预处理程序
    random.shuffle(data_all)
    label2id = {}
    id2label = {}
    labels = []
    examples = []
    if args.task_type_detail == "pipeline":
        labels.append("无关")
        for sample in data_all:
            for annotation_result in sample["annotations"][0]["result"]:
                if annotation_result["type"] == "relation":
                    if len(annotation_result["labels"]) == 1 and annotation_result["labels"][0] not in labels:
                        labels.append(annotation_result["labels"][0])
                    elif len(annotation_result["labels"]) == 0:  # 如果语料的关系没有标签，就将其头尾实体的标签相加作为关系的标签
                        annotation_result["labels"].append(next((item for item in sample["annotations"][0]["result"] if item["id"] == annotation_result["from_id"]), None)["value"]["labels"][0] \
                        + "+" + next((item for item in sample["annotations"][0]["result"] if item["id"] == annotation_result["to_id"]), None)["value"]["labels"][0])
                        if annotation_result["labels"][0] not in labels:
                            labels.append(annotation_result["labels"][0])
        for i, label in enumerate(labels):
            label2id[label] = i
            id2label[i] = label
        with open(args.data_dir + '{}_id2label.json'.format(args.task_name), 'w', encoding='utf-8') as file:
            file.write(json.dumps(id2label, ensure_ascii=False))
        for sample in data_all:
            relations = [[item["from_id"], item["to_id"], item["labels"][0]] for item in sample["annotations"][0]["result"] if item["type"] == "relation"]
            negative_samples = []
            for i in range(len(sample["annotations"][0]["result"])):
                for j in range(len(sample["annotations"][0]["result"])):
                    if i != j and sample["annotations"][0]["result"][i]["type"] == "labels" and sample["annotations"][0]["result"][j]["type"] == "labels":
                        relation = next((item[2] for item in relations if item[0] == sample["annotations"][0]["result"][i]["id"] and item[1] == sample["annotations"][0]["result"][j]["id"]), "无关")
                        if relation != "无关":
                            examples.append(InputExample(text=sample["annotations"][0]["result"][i]["value"]["text"] + "[SEP]" + sample["annotations"][0]["result"][j]["value"]["text"] + "[SEP]" + sample["data"]["text"], \
                                                        labels=label2id[relation]))
                        else:
                            negative_samples.append(InputExample(text=sample["annotations"][0]["result"][i]["value"]["text"] + "[SEP]" + sample["annotations"][0]["result"][j]["value"]["text"] + "[SEP]" + sample["data"]["text"], \
                                                        labels=label2id[relation]))
            if len(negative_samples) > 5:  # 适量选取负样本
                negative_samples = random.sample(negative_samples, 5)
            examples.extend(negative_samples)
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
                                            truncation='longest_first',
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
    print('========关系抽取预处理程序========')
    with open(args.data_dir + 'datas.json', encoding='utf-8') as file:
        data_all = json.load(file)
    preprocess_re(args, data_all)
    print("已生成预处理数据文件。")
