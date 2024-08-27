import os
from sklearn.metrics import classification_report
import torch
import torch.nn as nn
from torchcrf import CRF
import numpy as np
from torch.utils.data import DataLoader, RandomSampler
import pickle
from parsedata.preprocess_tc import preprocess_tc
from parsedata.preprocess_ner import preprocess_ner
from parsedata.preprocess_re import preprocess_re
from parsedata.dataset import MLDataset
from parsedata.models import *
from parsedata.functions import *
from parsedata.argsconfig import Args
import json
import pandas as pd
import time
import re
from PyQt5.QtCore import QTimer
import shutil


args = Args().get_parser()
class Trainer():
    def __init__(self, args, train_loader, dev_loader, test_loader, MyMainWindow):
        self.MyMainWindow = MyMainWindow
        gpu_ids = args.gpu_ids.split(',')
        self.device = torch.device("cpu" if gpu_ids[0] == '-1' else "cuda:" + gpu_ids[0])
        self.model = BertLSTMMLClf(args)
        self.optimizer = torch.optim.AdamW(params=self.model.parameters(), lr=args.lr)
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.test_loader = test_loader

    def load_ckp(self, model):
        checkpoint = torch.load(os.path.join(args.checkpoint_path, 'bestmodel.pt'))
        model.load_state_dict(checkpoint['state_dict'])
        return model

    def save_ckp(self, state, checkpoint_path):
        torch.save(state, checkpoint_path)

    def train(self):
        global_step = 0
        eval_step = len(self.train_loader)
        former_best_f1 = 0.0
        for epoch in range(args.train_epochs):
            losslist = []
            self.MyMainWindow.cpercentage_Train = 0
            self.MyMainWindow.start_time_Train = time.time()
            self.MyMainWindow.timer_Train = QTimer(self.MyMainWindow)  # 创建 QTimer 实例
            self.MyMainWindow.timer_Train.timeout.connect(self.MyMainWindow.updateElapsedTime_Train)  # 连接信号和槽
            self.MyMainWindow.timer_Train.start(1000)  # 设置 QTimer 每秒触发一次
            self.MyMainWindow.progressBar_Train.reset()
            self.MyMainWindow.progressBar_Dev.reset()
            self.MyMainWindow.label_TrainLoss.setText("Loss:0.000000")
            self.MyMainWindow.label_Train.setText("训练进度  第{}/{}轮".format(epoch + 1, args.train_epochs))
            for train_data in self.train_loader:
                self.model.train()
                token_ids = train_data['token_ids'].to(self.device)
                attention_masks = train_data['attention_masks'].to(self.device)
                token_type_ids = train_data['token_type_ids'].to(self.device)
                data_labels = train_data['labels'].to(self.device)
                _, train_loss = self.model(token_ids, attention_masks, token_type_ids, data_labels)
                self.optimizer.zero_grad()
                train_loss.backward()
                self.optimizer.step()
                losslist.append(train_loss.detach().item())
                self.MyMainWindow.cpercentage_Train += 1 / len(self.train_loader)  # 这代表当前进度
                self.MyMainWindow.update_Train(self.MyMainWindow.cpercentage_Train)
                self.MyMainWindow.label_TrainLoss.setText("Loss:{:.6f}".format(np.mean(losslist)))
                global_step += 1
                if global_step % eval_step == 0:
                    dev_loss, macro_accuracy, micro_accuracy, f1 = self.dev()
                    if args.task_type_detail == "multilabels":
                        self.MyMainWindow.update_TextBrowser(
                        "【dev】 轮数：{}, loss：{:.6f}, 样本正确率：{:.4f}, 标签正确率：{:.4f}, f1：{:.4f}, 历史最高f1：{:.4f}"
                        .format(epoch + 1, dev_loss, macro_accuracy, micro_accuracy, f1, former_best_f1))
                    elif args.task_type_detail == "singlelabel" or args.task_type_detail == "pipeline_nered":
                        self.MyMainWindow.update_TextBrowser(
                        "【dev】 轮数：{}, loss：{:.6f}, 样本正确率：{:.4f}, f1：{:.4f}, 历史最高f1：{:.4f}"
                        .format(epoch + 1, dev_loss, macro_accuracy, f1, former_best_f1))
                    else:
                        self.MyMainWindow.update_TextBrowser(
                        "【dev】 轮数：{}, loss：{:.6f}, 标签正确率：{:.4f}, f1：{:.4f}, 历史最高f1：{:.4f}"
                        .format(epoch + 1, dev_loss, macro_accuracy, f1, former_best_f1))
                    if f1 > former_best_f1:
                        self.MyMainWindow.update_TextBrowser("--------保存模型--------")
                        checkpoint = {
                            'state_dict': self.model.state_dict(),
                            'pretrained_model': args.bert_dir,
                            'task_type': args.task_type,
                            'task_type_detail': args.task_type_detail if args.task_type_detail != "pipeline_nered" else "pipeline"
                        }
                        former_best_f1 = f1
                        if not os.path.exists(args.checkpoint_path):
                            os.makedirs(args.checkpoint_path)
                        self.save_ckp(checkpoint, os.path.join(args.checkpoint_path, 'bestmodel.pt'))
            self.MyMainWindow.timer_Train.stop()
                

    def dev(self):
        self.model.eval()
        total_loss = []
        dev_outputs = []
        dev_targets = []
        self.MyMainWindow.cpercentage_Dev = 0
        self.MyMainWindow.start_time_Dev = time.time()
        self.MyMainWindow.timer_Dev = QTimer(self.MyMainWindow)  # 创建 QTimer 实例
        self.MyMainWindow.timer_Dev.timeout.connect(self.MyMainWindow.updateElapsedTime_Dev)  # 连接信号和槽
        self.MyMainWindow.timer_Dev.start(1000)  # 设置 QTimer 每秒触发一次
        self.MyMainWindow.label_DevLoss.setText("TLoss:0.000000")
        progressvalue = 0
        with torch.no_grad():
            for dev_data in self.dev_loader:
                token_ids = dev_data['token_ids'].to(self.device)
                attention_masks = dev_data['attention_masks'].to(self.device)
                token_type_ids = dev_data['token_type_ids'].to(self.device)
                data_labels = dev_data['labels'].to(self.device)
                logits, dev_loss = self.model(token_ids, attention_masks, token_type_ids, data_labels)
                total_loss.append(dev_loss.detach().item())
                self.MyMainWindow.cpercentage_Dev += 1 / len(self.dev_loader)  # 这代表当前进度
                self.MyMainWindow.update_Dev(self.MyMainWindow.cpercentage_Dev)
                dev_outputs.extend(logits)
                dev_targets.extend(data_labels.cpu().detach().numpy().tolist())
                self.MyMainWindow.label_DevLoss.setText("TLoss:{:.6f}".format(np.mean(total_loss)))
                progressvalue += 1 / len(self.dev_loader)
                self.MyMainWindow.update_Dev(progressvalue)
            self.MyMainWindow.timer_Dev.stop()
        macro_accuracy, micro_accuracy, f1 = getresult(args, dev_outputs, dev_targets)
        return np.mean(total_loss), macro_accuracy, micro_accuracy, f1

    def test(self):
        self.model = self.load_ckp(self.model)
        self.model.eval()
        self.model.to(self.device)
        total_loss = 0.0
        test_outputs = []
        test_targets = []
        with torch.no_grad():
            for test_data in self.test_loader:
                token_ids = test_data['token_ids'].to(self.device)
                attention_masks = test_data['attention_masks'].to(self.device)
                token_type_ids = test_data['token_type_ids'].to(self.device)
                data_labels = test_data['labels'].to(self.device)
                logits, test_loss = self.model(token_ids, attention_masks, token_type_ids, data_labels)
                total_loss += test_loss.item()
                test_outputs.extend(logits)
                test_targets.extend(data_labels.cpu().detach().numpy().tolist())
        return total_loss, test_outputs, test_targets


def train(MyMainWindow):
    labels = []
    UpdateArgs(MyMainWindow)
    assert args.task_name, MyMainWindow.update_TextBrowser("请输入任务名称！")
    assert args.task_type, MyMainWindow.update_TextBrowser("请选择任务类型！")
    if not checkmodel(args, MyMainWindow):
        return
    if os.path.exists(args.data_dir + '{}_id2label.json'.format(args.task_name)) and os.path.exists(args.data_dir + '{}_data.pkl'.format(args.task_name)):
        MyMainWindow.update_TextBrowser('============读取预处理文件============')
    else:
        MyMainWindow.update_TextBrowser('============开始预处理============')
        with open(args.data_dir + args.data_name, encoding='utf-8') as file:
            data_all = json.load(file)
            if args.task_type == "tc":
                preprocess_tc(args, data_all)
            elif args.task_type == "ner":
                preprocess_ner(args, data_all)
            else:
                preprocess_re(args, data_all, MyMainWindow)
    with open(args.data_dir + '{}_id2label.json'.format(args.task_name), 'r', encoding='utf-8') as f:
        if not os.path.exists(args.checkpoint_path):
            os.makedirs(args.checkpoint_path)
        shutil.copyfile(args.data_dir + '{}_id2label.json'.format(args.task_name), args.checkpoint_path + '/id2label.json')
        id2label = json.load(f)
        labels = [str(value) for value in id2label.values()]
    with open(args.data_dir + '{}_data.pkl'.format(args.task_name), 'rb') as f:
        data_out = pickle.load(f)
    args.num_tags = len(labels)
    data_train_out = (data_out[0][:int(len(data_out[0]) * 0.8)],data_out[1][:int(len(data_out[1]) * 0.8)])
    data_dev_out = (data_out[0][int(len(data_out[0]) * 0.8):],data_out[1][int(len(data_out[1]) * 0.8):])
    features, _ = data_train_out
    train_dataset = MLDataset(features)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=args.batch_size,
                              num_workers=0)
    dev_features, _ = data_dev_out
    dev_dataset = MLDataset(dev_features)
    dev_loader = DataLoader(dataset=dev_dataset,
                            batch_size=args.batch_size,
                            num_workers=0)
    # 训练和验证
    MyMainWindow.update_TextBrowser('============开始训练============')
    trainer = Trainer(args, train_loader, dev_loader, dev_loader, MyMainWindow)  # 测试集此处同dev
    trainer.train()
    # 测试
    MyMainWindow.update_TextBrowser('============开始测试============')
    total_loss, test_outputs, test_targets = trainer.test()
    report = getreport(args, test_outputs, test_targets, labels)
    MyMainWindow.update_TextBrowser(report)
    MyMainWindow.update_TextBrowser('========模型训练完成！========')

def UpdateArgs(MyMainWindow):
    args.bert_dir = MyMainWindow.lineEdit_PretrainedModel.text()
    args.output_dir = MyMainWindow.lineEdit_BestModel.text()
    match = re.match(r'(^.*/)([^/]*$)', MyMainWindow.lineEdit_Data.text())
    if match:
        args.data_dir, args.data_name = match.groups()
    args.bert_dir = MyMainWindow.lineEdit_PretrainedModel.text()
    args.task_name = MyMainWindow.lineEdit_TaskName.text() or None
    args.checkpoint_path = args.output_dir + args.task_name
    task_type = MyMainWindow.comboBox_TaskType.currentText() or None
    if task_type == "文本分类":
        args.task_type = "tc"
    elif task_type == "实体识别":
        args.task_type = "ner"
    elif task_type == "关系抽取":
        args.task_type = "re"
    task_type_detail = MyMainWindow.comboBox_TaskTypeDetail.currentText() or None
    if task_type_detail == "单标签分类":
        args.task_type_detail = "singlelabel"
    elif task_type_detail == "多标签分类":
        args.task_type_detail = "multilabels"
    elif task_type_detail == "流水线式":
        args.task_type_detail = "pipeline"
    elif task_type_detail == "联合式":
        args.task_type_detail = "joint"
    else:
        args.task_type_detail = None
    args.max_seq_len = int(MyMainWindow.lineEdit_MaxSeqLen.text())
    args.batch_size = int(MyMainWindow.lineEdit_BatchSize.text())
    args.train_epochs = int(MyMainWindow.lineEdit_Epochs.text())


def checkmodel(args, MyMainWindow):  # 流水线式关系抽取任务需要前置的NER模型，如未检测到训练好的NER模型要先训练之
    if args.task_type_detail == "pipeline":
        if os.path.exists(args.checkpoint_path + "_ner"):
            args.task_type_detail = "pipeline_nered"
        else:
            MyMainWindow.update_TextBrowser("<span style='font-family:Arial; font-size:12pt; color:#FF0000;'>未检测到训练好的NER模型，流水线式关系抽取任务无法训练。 \
                                            请用相同语料先训练NER模型，并确保其模型保存根路径与关系抽取任务相同，任务名称为关系抽取任务的名称加\"_ner\"后缀。</span>")
        return False
    return True
