import os
import pickle
import shutil
from sklearn.metrics import accuracy_score, f1_score, multilabel_confusion_matrix, classification_report
import torch
import torch.nn as nn
from torchcrf import CRF
import numpy as np
from torch.utils.data import DataLoader, RandomSampler
from transformers import BertTokenizer
import sys
import config
import preprocess_mlc, preprocess_ner
import dataset
import models
import json
from tqdm import tqdm
from datetime import datetime
import pandas as pd
import functions


class Trainer:
    def __init__(self, args, train_loader, dev_loader, test_loader):
        self.args = args
        gpu_ids = args.gpu_ids.split(',')
        self.device = torch.device("cpu" if gpu_ids[0] == '-1' else "cuda:" + gpu_ids[0])
        # self.model = models.BertMLClf(args)
        self.model = models.BertLSTMMLClf(args)
        # self.criterion = nn.BCEWithLogitsLoss()
        self.criterion = functions.Multilabel_Categorical_CrossEntropy()
        self.optimizer = torch.optim.AdamW(params=self.model.parameters(), lr=self.args.lr)
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.test_loader = test_loader

    def load_ckp(self, model, optimizer, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        return model, optimizer, epoch, loss

    def save_ckp(self, state, checkpoint_path):
        torch.save(state, checkpoint_path)

    def train(self):
        global_step = 0
        eval_step = len(self.train_loader)
        best_dev_macro_f1 = 0.0
        for epoch in range(args.train_epochs):
            losslist = []
            bar = tqdm(self.train_loader, ncols=80, position=0, desc='【train】epoch{}'.format(epoch + 1), dynamic_ncols=False)
            for train_data in bar:
                self.model.train()
                token_ids = train_data['token_ids'].to(self.device)
                attention_masks = train_data['attention_masks'].to(self.device)
                token_type_ids = train_data['token_type_ids'].to(self.device)
                data_labels = train_data['labels'].to(self.device)
                outputs = self.model(token_ids, attention_masks, token_type_ids)
                train_loss = self.criterion(outputs, data_labels)
                self.optimizer.zero_grad()
                train_loss.backward()
                self.optimizer.step()
                losslist.append(train_loss.detach().item())
                bar.set_postfix({'loss': '{:.6f}'.format(np.mean(losslist))})
                global_step += 1
                if global_step % eval_step == 0:
                    dev_loss, dev_outputs, dev_targets = self.dev()
                    accuracy, micro_f1, macro_f1 = self.get_metrics(dev_outputs, dev_targets)
                    print(
                        "【dev】 loss：{:.6f} accuracy：{:.4f} micro_f1：{:.4f} macro_f1：{:.4f} former_best_dev_macro_f1：{:.4f}".format(dev_loss, accuracy, micro_f1, macro_f1, best_dev_macro_f1))
                    if macro_f1 > best_dev_macro_f1:
                        print("------------>save the best")
                        checkpoint = {
                            'epoch': epoch,
                            'loss': dev_loss,
                            'state_dict': self.model.state_dict(),
                            'optimizer': self.optimizer.state_dict(),
                        }
                        best_dev_macro_f1 = macro_f1
                        checkpoint_path = os.path.join(self.args.output_dir, 'best.pt')
                        if not os.path.exists(self.args.output_dir):
                            os.makedirs(self.args.output_dir)
                        self.save_ckp(checkpoint, checkpoint_path)

    def dev(self):
        self.model.eval()
        total_loss = 0.0
        dev_outputs = []
        dev_targets = []
        with torch.no_grad():
            # X, Y, Z = 1e-15, 1e-15, 1e-15
            bar = tqdm(self.dev_loader, ncols=80, position=0, desc='【dev】')
            for dev_data in bar:
                token_ids = dev_data['token_ids'].to(self.device)
                attention_masks = dev_data['attention_masks'].to(self.device)
                token_type_ids = dev_data['token_type_ids'].to(self.device)
                data_labels = dev_data['labels'].to(self.device)
                outputs = self.model(token_ids, attention_masks, token_type_ids)
                dev_loss = self.criterion(outputs, data_labels)
                total_loss += dev_loss.item()
                if self.args.task_type == "mlc":
                    outputs = torch.sigmoid(outputs).cpu().detach().numpy().tolist()
                    outputs = (np.array(outputs) > 0.5).astype(int)
                    dev_outputs.extend(outputs.tolist())
                else:
                    dev_outputs.extend(outputs)
                dev_targets.extend(data_labels.cpu().detach().numpy().tolist())

                # R = set(functions.get_gp_classify(dev_outputs))
                # T = set(functions.get_gp_classify(dev_targets))
                # X += len(R & T)
                # Y += len(R)
                # Z += len(T)
                # f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
        return total_loss, dev_outputs, dev_targets

    def test(self, checkpoint_path):
        model = self.model
        optimizer = self.optimizer
        model, optimizer, epoch, loss = self.load_ckp(model, optimizer, checkpoint_path)
        model.eval()
        model.to(self.device)
        total_loss = 0.0
        test_outputs = []
        test_targets = []
        with torch.no_grad():
            bar = tqdm(self.test_loader, ncols=80, position=0, desc='【test】')
            for test_data in bar:
                token_ids = test_data['token_ids'].to(self.device)
                attention_masks = test_data['attention_masks'].to(self.device)
                token_type_ids = test_data['token_type_ids'].to(self.device)
                data_labels = test_data['labels'].to(self.device)
                outputs = model(token_ids, attention_masks, token_type_ids)
                test_loss = self.criterion(outputs, data_labels)
                total_loss += test_loss.item()
                outputs = torch.sigmoid(outputs).cpu().detach().numpy().tolist()
                outputs = (np.array(outputs) > 0.5).astype(int)
                test_outputs.extend(outputs.tolist())
                test_targets.extend(data_labels.cpu().detach().numpy().tolist())

        return total_loss, test_outputs, test_targets

    def get_metrics(self, outputs, targets):
        accuracy = accuracy_score(targets, outputs)
        micro_f1 = f1_score(targets, outputs, average='micro')
        macro_f1 = f1_score(targets, outputs, average='macro')
        return accuracy, micro_f1, macro_f1

    def get_classification_report(self, outputs, targets, labels):
        report = classification_report(targets, outputs, target_names=labels)
        return report


if __name__ == '__main__':
    args = bert_config.Args().get_parser()
    labels = []
    if os.path.exists(args.data_dir + '{}_id2label.json'.format(args.task_name)) and os.path.exists(args.data_dir + '{}_data.pkl'.format(args.task_name)):
        print('========读取预处理文件========')
    else:
        print('========开始预处理========')
        with open(args.data_dir + 'datas.json', encoding='utf-8') as file:
            data_all = json.load(file)
            if args.task_type == "mlc":
                preprocess_mlc.preprocess(args, data_all)
            else:
                preprocess_ner.preprocess(args, data_all)
    with open(args.data_dir + '{}_id2label.json'.format(args.task_name), 'r', encoding='utf-8') as f:
        id2label = json.load(f)
        labels = [str(value) for value in id2label.values()]
    with open(args.data_dir + '{}_data.pkl'.format(args.task_name), 'rb') as f:
        data_out = pickle.load(f)
    args.num_tags = len(labels)
    data_train_out = (data_out[0][:int(len(data_out[0]) * 0.9)],data_out[1][:int(len(data_out[1]) * 0.9)])
    data_dev_out = (data_out[0][int(len(data_out[0]) * 0.9):],data_out[1][int(len(data_out[1]) * 0.9):])
    # data_train_out = (data_out[0][int(len(data_out[0]) * 0.9):],data_out[1][int(len(data_out[0]) * 0.9):])
    # data_dev_out = (data_out[0][int(len(data_out[0]) * 0.9):],data_out[1][int(len(data_out[0]) * 0.9):])
    features, callback_info = data_train_out
    train_dataset = dataset.MLDataset(features)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=args.batch_size,
                              num_workers=0)
    dev_features, dev_callback_info = data_dev_out
    dev_dataset = dataset.MLDataset(dev_features)
    dev_loader = DataLoader(dataset=dev_dataset,
                            batch_size=args.batch_size,
                            num_workers=0)
    # 训练和验证
    print('========开始训练========')
    trainer = Trainer(args, train_loader, dev_loader, dev_loader)  # 测试集此处同dev
    trainer.train()
    # 测试
    print('========开始测试========')
    checkpoint_path = './checkpoints/best.pt'
    total_loss, test_outputs, test_targets = trainer.test(checkpoint_path)
    accuracy, micro_f1, macro_f1 = trainer.get_metrics(test_outputs, test_targets)
    report = trainer.get_classification_report(test_outputs, test_targets, labels)
    print(report)
