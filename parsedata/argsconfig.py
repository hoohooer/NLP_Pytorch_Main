import argparse
import os


class Args:
    @staticmethod
    def parse():
        parser = argparse.ArgumentParser()
        return parser

    @staticmethod
    def initialize(parser):
        parser.add_argument('--data_dir', default="./data/",
                        help='the dir for training data')
        parser.add_argument('--data_name', default="datas.json",
                        help='the name for training data')
        parser.add_argument('--output_dir', default="./checkpoints/",
                        help='the output dir for model checkpoints')
        parser.add_argument('--bert_dir', default='../model/chinese-roberta-small-wwm-cluecorpussmall',
        help='bert dir for uer')
        parser.add_argument('--task_type', default='tc', help='your task type')
        parser.add_argument('--task_type_detail', default='singlelabel', help='your task type detail')
        parser.add_argument('--task_name', default='zhaotoubiaonew_projectclassification',
        help='your task name')
        # other args
        parser.add_argument('--num_tags', default=0, type=int,
                        help='number of tags')  # 标签类别数，由程序自动获取
        parser.add_argument('--seed', type=int, default=123, help='random seed')
        parser.add_argument('--gpu_ids', type=str, default="0",
                        help='gpu ids to use, -1 for cpu, "0,1" for multi gpu')
        parser.add_argument('--max_seq_len', default=64, type=int)
        parser.add_argument('--batch_size', default=32, type=int)
        parser.add_argument('--swa_start', default=2, type=int,
                        help='the epoch when swa start')
        parser.add_argument('--train_epochs', default=5, type=int,
                        help='Max training epoch')
        parser.add_argument('--dropout_prob', default=0.3, type=float,
                        help='drop out probability')
        parser.add_argument('--lr', default=3e-5, type=float,
                        help='learning rate for the bert module')
        parser.add_argument('--other_lr', default=3e-4, type=float,
                        help='learning rate for the module except bert')
        parser.add_argument('--max_grad_norm', default=1, type=float,
                        help='max grad clip')
        parser.add_argument('--warmup_proportion', default=0.1, type=float)
        parser.add_argument('--weight_decay', default=0.01, type=float)
        parser.add_argument('--adam_epsilon', default=1e-8, type=float)
        parser.add_argument('--eval_model', default=True, action='store_true',
                        help='whether to eval model after training')
        parser.add_argument('--lstm_bidirectional', default=True,
                        help='whether to use bidirectional lstm function or not')
        return parser

    def get_parser(self):
        parser = self.parse()
        parser = self.initialize(parser)
        return parser.parse_args()