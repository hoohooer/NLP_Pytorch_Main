import sys
from Ui_MainWindow import Ui_MainWindow  
import qtawesome
from PyQt5 import QtWidgets
from PyQt5.QtCore import QCoreApplication
from PyQt5.QtGui import QTextCursor
from parsedata import config, preprocess_tc, preprocess_ner, dataset, train
import pickle
import os
import json
from torch.utils.data import DataLoader


args = config.Args().get_parser()
class MyMainWindow(QtWidgets.QMainWindow,Ui_MainWindow):
    def __init__(self,parent=None):
        super(MyMainWindow,self).__init__(parent)
        self.setupUi(self)
        self.pushButton_RunParse.clicked.connect(self.runparse)
    
    def browse_PretrainedModel(self):
        directory = QtWidgets.QFileDialog.getExistingDirectory(None,"选取文件夹","C:/")
        self.lineEdit_PretrainedModel.setText(directory)

    def browse_Data(self):
        directory = QtWidgets.QFileDialog.getExistingDirectory(None,"选取文件夹","C:/")
        self.lineEdit_Data.setText(directory)

    def browse_BestModel(self):
        directory = QtWidgets.QFileDialog.getExistingDirectory(None,"选取文件夹","C:/")
        self.lineEdit_BestModel.setText(directory)
    
    def updateTextBrowser(self, text):
        self.textBrowser.append(text)
        QCoreApplication.processEvents()  # 确保立即更新
    
    def runparse(self):        
        labels = []
        if os.path.exists(args.data_dir + '{}_id2label.json'.format(args.task_name)) and os.path.exists(args.data_dir + '{}_data.pkl'.format(args.task_name)):
            self.updateTextBrowser("========读取预处理文件========")
        else:
            self.updateTextBrowser("========开始预处理========")
            with open(args.data_dir + 'datas.json', encoding='utf-8') as file:
                data_all = json.load(file)
                if args.task_type == "tc":
                    preprocess_tc(args, data_all)
                else:
                    preprocess_ner(args, data_all)
        with open(args.data_dir + '{}_id2label.json'.format(args.task_name), 'r', encoding='utf-8') as f:
            id2label = json.load(f)
            labels = [str(value) for value in id2label.values()]
        with open(args.data_dir + '{}_data.pkl'.format(args.task_name), 'rb') as f:
            data_out = pickle.load(f)
        args.num_tags = len(labels)
        data_train_out = (data_out[0][:int(len(data_out[0]) * 0.8)],data_out[1][:int(len(data_out[1]) * 0.8)])
        data_dev_out = (data_out[0][int(len(data_out[0]) * 0.8):],data_out[1][int(len(data_out[1]) * 0.8):])
        features, _ = data_train_out
        train_dataset = dataset.MLDataset(features)
        train_loader = DataLoader(dataset=train_dataset,
                                batch_size=args.batch_size,
                                num_workers=0)
        dev_features, _ = data_dev_out
        dev_dataset = dataset.MLDataset(dev_features)
        dev_loader = DataLoader(dataset=dev_dataset,
                                batch_size=args.batch_size,
                                num_workers=0)
        # 训练和验证
        self.updateTextBrowser('========开始训练========')
        trainer = train.Trainer(args, train_loader, dev_loader, dev_loader, self)  # 测试集此处同dev
        trainer.train()
        # 测试
        self.updateTextBrowser('========开始测试========')
        checkpoint_path = './checkpoints/best.pt'
        total_loss, test_outputs, test_targets = trainer.test(checkpoint_path)
        # accuracy, micro_f1, macro_f1 = trainer.get_metrics(test_outputs, test_targets)
        report = trainer.get_classification_report(test_outputs, test_targets, labels)
        self.updateTextBrowser(report)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    myWin = MyMainWindow()
    myWin.show()
    sys.exit(app.exec_())
