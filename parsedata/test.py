from transformers import BertTokenizer
from parsedata.models import *
from parsedata.functions import *
from parsedata.argsconfig import Args
import torch
import json
import torch.nn as nn
import os
from parsedata import preprocess_tc, preprocess_ner, preprocess_re
import requests


args = Args().get_parser()
class Tester:
    def __init__(self, args):
        gpu_ids = args.gpu_ids.split(',')
        self.device = torch.device("cpu" if gpu_ids[0] == '-1' else "cuda:" + gpu_ids[0])
        self.checkpoint = torch.load(os.path.join(args.checkpoint_path, 'bestmodel.pt'))
        args.task_type = self.checkpoint['task_type']
        if args.task_type_detail != "pipeline_nered":
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
                result = self.get_result_ner(id2label, logits)
                result_list.append(result)
            elif self.model.task_type_detail == "pipeline_nered":
                result_list.append(logits[0])
        return result_list
    
    def get_result_ner(self, id2label, logits):
        active_labellist = [id2label[str(logits[0][i])] for i in range(len(logits[0]))]
        result = []
        for i in range(len(active_labellist)):
            if active_labellist[i][0] == "S":
                result.append([i, i+1, active_labellist[i][2:]])
            elif active_labellist[i][0] == "B" and i < len(active_labellist) - 2:
                for j in range(i + 1, len(active_labellist) - 1):
                    if active_labellist[j][0] == "E" and active_labellist[i][2:] == active_labellist[j][2:]:
                        result.append([i, j+1, active_labellist[i][2:]])
                        i = j
                        break
                    elif active_labellist[j][0] == "I" and active_labellist[i][2:] == active_labellist[j][2:] and \
                        active_labellist[j + 1] in ("O", "[SEP]"):  # 考虑没有结束标志的特殊情况，后期可以去除
                        result.append([i, j+1, active_labellist[i][2:]])
                        i = j
                        break
            elif i > 0 and active_labellist[i][0] == "I" and active_labellist[i - 1] in ("O", "[CLS]") and i < len(active_labellist) - 2:  # 考虑没有开始标志的特殊情况，后期可以去除
                for j in range(i + 1, len(active_labellist) - 1):
                    if active_labellist[j][0] == "E" and active_labellist[i][2:] == active_labellist[j][2:]:
                        result.append([i, j+1, active_labellist[i][2:]])
                        i = j
                        break
        return [[item[0] - 1, item[1] - 1, item[2]] for item in result]  # 为了去掉编码时开头添加的'[CLS]'
    
    def get_result_re_pipeline(self, args, input, input_examples):
        args.checkpoint_path += "_ner"
        if not os.path.exists(args.checkpoint_path):
            return ["未检测到训练好的NER模型！请用相同语料先训练NER模型，并确保其模型保存根路径与关系抽取任务相同，任务名称为关系抽取任务的名称加\"_ner\"后缀。"]
        self.model.task_type = "ner"  # 先通过NER模型抽取实体
        with open(os.path.join(args.checkpoint_path, 'id2label.json'), 'r', encoding='utf-8') as f:
            id2label = json.load(f)
        args.num_tags = len(id2label)
        tester = Tester(args)
        output_results_ner = tester.get_result(input_examples, id2label)
        output_results_re = []
        for k in range(len(output_results_ner)):
            output_result_re = []
            if len(output_results_ner[k]) >= 2:
                re_input = []
                for i in range(len(output_results_ner[k])):
                    for j in range(len(output_results_ner[k])):
                        if i != j:
                            re_input.append(input[k][output_results_ner[k][i][0]:output_results_ner[k][i][1]] + "[SEP]" + input[k][output_results_ner[k][j][0]:output_results_ner[k][j][1]] + "[SEP]" + input[k])
                args.checkpoint_path = args.checkpoint_path.replace("_ner", "")
                with open(os.path.join(args.checkpoint_path, 'id2label.json'), 'r', encoding='utf-8') as f:
                    id2label = json.load(f)
                args.task_type = "re"
                args.task_type_detail = "pipeline_nered"  # 再通过RE模型抽取关系
                results = parsetext(re_input, args)
                for num, re_result in enumerate(results):
                    if re_result != 0:
                        i, j = divmod(num, len(output_results_ner[k]) - 1)
                        if j >= i:
                            j += 1
                        output_result_re.append([output_results_ner[k][i], output_results_ner[k][j], id2label[str(re_result)]])
            output_results_re.append(output_result_re)
        output_results = [[output_results_ner[i], output_results_re[i]] for i in range(len(output_results_ner))]
        return output_results
        

def test(MyDeploymentDialog):
    UpdateArgs(MyDeploymentDialog)
    input_text = MyDeploymentDialog.textEdit_Input.toPlainText()
    input = [input_text]
    if not args.useport:
        results = parsetext(input, args)
    else:
        params = {
            "serialized": True
        }
        data = {
            "data": json.dumps(input)
        }
        response = requests.post(args.port, params=params, json=data)
        results = json.loads(response.text)
        # 打印响应内容
        print('请求成功，响应内容：', response.json())
    MyDeploymentDialog.textBrowser_Output.append(str(results[0]))


def UpdateArgs(MyDeploymentDialog):
    args.checkpoint_path = MyDeploymentDialog.lineEdit_TrainedModel.text()
    args.port = MyDeploymentDialog.lineEdit_Port.text()
    args.useport = MyDeploymentDialog.checkBox_UsePort.isChecked()


def parsetext(input, args):
    with open(os.path.join(args.checkpoint_path, 'id2label.json'), 'r', encoding='utf-8') as f:
        id2label = json.load(f)
    args.num_tags = len(id2label)
    tester = Tester(args)
    tokenizer = BertTokenizer.from_pretrained(tester.model.pretrained_model)
    if tester.model.task_type == "tc":
        input_examples = preprocess_tc.test_out(input, tokenizer=tokenizer)
    elif tester.model.task_type == "ner":
        input_examples = preprocess_ner.test_out(input, tokenizer=tokenizer)
    elif tester.model.task_type == "re":
        input_examples = preprocess_re.test_out(input, tokenizer=tokenizer)
    if tester.model.task_type_detail != "pipeline":
        output_results = tester.get_result(input_examples, id2label)
    else:
        output_results = tester.get_result_re_pipeline(args, input, input_examples)
    return output_results
    

