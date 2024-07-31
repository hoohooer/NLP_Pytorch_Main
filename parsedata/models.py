from transformers import BertModel
import torch
import torch.nn as nn
from torchcrf import CRF
    
class BertLSTMMLClf(nn.Module):
    def __init__(self, args):
        super(BertLSTMMLClf, self).__init__()
        self.task_type = args.task_type
        self.device = torch.device("cpu" if args.gpu_ids[0] == '-1' else "cuda:" + args.gpu_ids[0])
        self.bert = BertModel.from_pretrained(args.bert_dir).to(self.device)
        self.bert_config = self.bert.config
        out_dims = self.bert_config.hidden_size
        self.bidirectional = args.lstm_bidirectional
        self.bidirectional = False
        self.lstm = nn.LSTM(input_size=out_dims,
                            hidden_size=out_dims,
                            num_layers=2,
                            bidirectional=self.bidirectional,
                            batch_first=True,
                            dropout=args.dropout_prob).to(self.device)
        # self.crf = CRF(args.num_tags, batch_first=True).to(self.device)
        # 为了适应双向lstm的输出维数，改变了全连接层的输入维数
        if self.bidirectional:
            self.linear = nn.Linear(out_dims * 2, args.num_tags).to(self.device)
        else:
            self.linear = nn.Linear(out_dims, args.num_tags).to(self.device)

    def forward(self, token_ids, attention_masks, token_type_ids):
        assert self.task_type in ("tc", "ner"), "任务类型不正确！"
        bert_outputs = self.bert(
            input_ids=token_ids,
            attention_mask=attention_masks,
            token_type_ids=token_type_ids,
        )
        if self.task_type == "tc":
            seq_out = bert_outputs[1]
            seq_out = self.linear(seq_out)
            return seq_out
        # else:
        #     seq_out = bert_outputs[0]
        #     seq_out, h = self.lstm(seq_out)
        #     seq_out = self.linear(seq_out)
        #     if labels is not None:
        #         labels = labels.to(torch.int32)
        #         loss = -self.crf(seq_out, labels, mask=attention_masks.bool(), reduction='mean')
        #         seq_out = self.crf.decode(seq_out, mask=attention_masks.bool())
        #         return seq_out, loss
        #     else:
        #         seq_out = self.crf.decode(seq_out, mask=attention_masks.bool())
        #         return seq_out
