from transformers import BertTokenizer
import models
import torch
import json
import torch.nn as nn
from parsedata.config import *
import os
from parsedata.preprocess_tc import test_out

checkpoint_path = './checkpoints/best.pt'


class Tester:
    def __init__(self, args):
        self.args = args
        gpu_ids = args.gpu_ids.split(',')
        self.device = torch.device("cpu" if gpu_ids[0] == '-1' else "cuda:" + gpu_ids[0])
        # self.model = models.BertMLClf(args)
        self.model = models.BertLSTMMLClf(args)
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.args.lr)
        self.criterion = nn.BCEWithLogitsLoss()

    def load_ckp(self, model, optimizer, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        return model, optimizer, epoch, loss

    def get_result(self, input_examples):
        result_list = []
        model = self.model
        optimizer = self.optimizer
        model, optimizer, epoch, loss = self.load_ckp(model, optimizer, checkpoint_path)
        model.eval()
        model.to(self.device)
        for input_example in input_examples:
            token_ids = torch.tensor(input_example['token_ids']).unsqueeze(0).to(self.device)
            attention_masks = torch.tensor(input_example['attention_masks']).unsqueeze(0).to(self.device)
            token_type_ids = torch.tensor(input_example['token_type_ids']).unsqueeze(0).to(self.device)
            output = model(token_ids, attention_masks, token_type_ids)
            output = torch.squeeze(output)
            active_labels_index = torch.squeeze(torch.nonzero(output > 0), dim=0).tolist()
            active_labels = [id2label[str(index[0])] for index in active_labels_index]
            if active_labels == []:
                active_labels.append("其它")
            result_list.append(active_labels)
        return result_list

if __name__ == '__main__':
    args = Args().get_parser()
    assert os.path.exists(args.data_dir + '{}_id2label.json'.format(args.task_name)), '缺少id2label文件，请检查。'
    with open(args.data_dir + '{}_id2label.json'.format(args.task_name), 'r', encoding='utf-8') as f:
        id2label = json.load(f)
    args.num_tags = len(id2label)
    tester = Tester(args)
    input = ['海螺新材：第九届董事会第三十八次会议决议公告']
    tokenizer = BertTokenizer.from_pretrained("../model/chinese-roberta-small-wwm-cluecorpussmall")
    input_examples = test_out(input, tokenizer=tokenizer)
    output_results = tester.get_result(input_examples)
    print(output_results)


