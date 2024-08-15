from transformers import BertTokenizer
from parsedata.models import *
from parsedata.functions import *
from parsedata.argsconfig import Args
import torch
import json
import torch.nn as nn
import os
from parsedata import preprocess_tc
from parsedata import preprocess_ner


args = Args().get_parser()
class Tester:
    def __init__(self, args):
        gpu_ids = args.gpu_ids.split(',')
        self.device = torch.device("cpu" if gpu_ids[0] == '-1' else "cuda:" + gpu_ids[0])
        self.checkpoint = torch.load(os.path.join(args.checkpoint_path, 'bestmodel.pt'))
        args.task_type = self.checkpoint['task_type']
        args.task_type_detail = self.checkpoint['task_type_detail']
        self.model = BertLSTMMLClf(args)
        self.model.load_state_dict(self.checkpoint['state_dict'])
        self.model.pretrained_model = self.checkpoint['pretrained_model']
        self.optimizer = torch.optim.AdamW(params=self.model.parameters(), lr=args.lr)

    def get_result(self, input_examples, id2label):
        result_list = []
        self.model.eval()
        self.model.to(self.device)
        for input_example in input_examples:
            token_ids = torch.tensor(input_example['token_ids']).unsqueeze(0).to(self.device)
            attention_masks = torch.tensor(input_example['attention_masks']).unsqueeze(0).to(self.device)
            token_type_ids = torch.tensor(input_example['token_type_ids']).unsqueeze(0).to(self.device)
            logits, _ = self.model(token_ids, attention_masks, token_type_ids)
            if self.model.task_type_detail == "singlelabel":
                active_label = id2label[str(logits[0])]
                result_list.append(active_label)
            elif self.model.task_type_detail == "multilabels":
                active_label = [id2label[str(i)] for i in range(len(logits[0])) if logits[0][i] == 1]
                if active_label == []:
                    active_label.append("未匹配标签")
                result_list.append(active_label)
            elif self.model.task_type == "ner":
                active_labellist = [id2label[str(logits[0][i])] for i in range(len(logits[0]))]
                for i in range(len(active_labellist)):
                    if active_labellist[i][0] == "S":
                        result_list.append([i, i+1, active_labellist[i][2:]])
                    elif active_labellist[i][0] == "B" and i < len(active_labellist) - 2:
                        for j in range(i + 1, len(active_labellist) - 1):
                            if active_labellist[j][0] == "E" and active_labellist[i][2:] == active_labellist[j][2:]:
                                result_list.append([i, j+1, active_labellist[i][2:]])
                                i = j
                                break
                            elif active_labellist[j][0] == "I" and active_labellist[i][2:] == active_labellist[j][2:] and \
                                active_labellist[j + 1] in ("O", "[SEP]"):  # 考虑没有结束标志的特殊情况，后期可以去除
                                result_list.append([i, j+1, active_labellist[i][2:]])
                                i = j
                                break
                    elif i > 0 and active_labellist[i][0] == "I" and active_labellist[i - 1] in ("O", "[CLS]") and i < len(active_labellist) - 2:  # 考虑没有开始标志的特殊情况，后期可以去除
                        for j in range(i + 1, len(active_labellist) - 1):
                            if active_labellist[j][0] == "E" and active_labellist[i][2:] == active_labellist[j][2:]:
                                result_list.append([i, j+1, active_labellist[i][2:]])
                                i = j
                                break
        return result_list
    

def test(MyDeploymentDialog):
    UpdateArgs(MyDeploymentDialog)
    input_text = MyDeploymentDialog.textBrowser_Input.toPlainText()
    input = [input_text]
    results = parsetext(input, args)
    MyDeploymentDialog.textBrowser_Output.setText(str(results[0]))


def UpdateArgs(MyDeploymentDialog):
    args.checkpoint_path = MyDeploymentDialog.lineEdit_TrainedModel.text()
    args.port = MyDeploymentDialog.lineEdit_Port.text()
    args.useport = MyDeploymentDialog.checkBox_UsePort.isChecked()


def parsetext(input, args):
    with open(args.checkpoint_path + '/id2label.json', 'r', encoding='utf-8') as f:
        id2label = json.load(f)
    args.num_tags = len(id2label)
    tester = Tester(args)
    tokenizer = BertTokenizer.from_pretrained(tester.model.pretrained_model)
    if tester.model.task_type == "tc":
        input_examples = preprocess_tc.test_out(input, tokenizer=tokenizer)
    elif tester.model.task_type == "ner":
        input_examples = preprocess_ner.test_out(input, tokenizer=tokenizer)
    output_results = tester.get_result(input_examples, id2label)
    return output_results
    

