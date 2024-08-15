from transformers import BertModel, BertConfig
from parsedata.functions import *
# from parsedata.functions import *
import torch
import torch.nn as nn
from torchcrf import CRF
import numpy as np

    
class BertLSTMMLClf(nn.Module):
    def __init__(self, args):
        super(BertLSTMMLClf, self).__init__()
        self.task_type = args.task_type
        self.task_type_detail = args.task_type_detail
        self.device = torch.device("cpu" if args.gpu_ids[0] == '-1' else "cuda:" + args.gpu_ids[0])
        self.bert = BertModel.from_pretrained(args.bert_dir).to(self.device)
        self.bert_config = self.bert.config
        out_dims = self.bert_config.hidden_size
        self.bidirectional = args.lstm_bidirectional
        self.lstm = nn.LSTM(input_size=out_dims,
                            hidden_size=out_dims,
                            num_layers=1,
                            bidirectional=self.bidirectional,
                            batch_first=True,
                            dropout=0.1).to(self.device)
        self.crf = CRF(args.num_tags, batch_first=True).to(self.device)
        # 为了适应双向lstm的输出维数，改变了全连接层的输入维数
        if self.bidirectional and self.task_type == "ner":
            self.linear = nn.Linear(out_dims * 2, args.num_tags).to(self.device)
        else:
            self.linear = nn.Linear(out_dims, args.num_tags).to(self.device)
        # 损失函数
        if self.task_type == "tc" and self.task_type_detail == "multilabels":
            self.criterion = Multilabel_Categorical_CrossEntropy()
        elif self.task_type == "tc" and self.task_type_detail == "singlelabel":
            self.criterion = nn.CrossEntropyLoss()
        elif self.task_type == "ner":
            self.criterion = nn.CrossEntropyLoss()

    def forward(self, token_ids, attention_masks, token_type_ids, labels=None):
        assert self.task_type in ("tc", "ner"), "任务类型不正确！"
        bert_outputs = self.bert(
            input_ids=token_ids,
            attention_mask=attention_masks,
            token_type_ids=token_type_ids,
        )
        if self.task_type == "tc":
            seq_out = bert_outputs[1]
            seq_out = self.linear(seq_out)
            loss = None
            if labels is not None:
                loss = self.criterion(seq_out, labels)
            if self.task_type_detail == "singlelabel":
                outputs = nn.functional.softmax(seq_out, dim=1).cpu().detach().numpy().tolist()
                logits = [np.argmax(outputs[i]) for i in range(len(outputs))]
            else:
                outputs = torch.sigmoid(seq_out).cpu().detach().numpy().tolist()
                logits = (np.array(outputs) > 0.5).astype(int).tolist()
        else:
            seq_out = bert_outputs[0]
            seq_out, h = self.lstm(seq_out)
            seq_out = self.linear(seq_out)
            logits = self.crf.decode(seq_out)
            loss = None
            if labels is not None:
                loss = -self.crf(seq_out, labels, reduction='mean')
        return logits, loss
            

