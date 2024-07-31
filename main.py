import sys
from Ui_MainWindow import Ui_MainWindow  
import qtawesome
from PyQt5 import QtWidgets
from PyQt5.QtCore import QCoreApplication, QTimer
from PyQt5.QtGui import QTextCursor
from parsedata import config, preprocess_tc, preprocess_ner, dataset, train
import time
import os
import json
from torch.utils.data import DataLoader


args = config.Args().get_parser()
class MyMainWindow(QtWidgets.QMainWindow,Ui_MainWindow):
    def __init__(self,parent=None):
        super(MyMainWindow,self).__init__(parent)
        self.setupUi(self)
        self.pushButton_RunParse.clicked.connect(self.runparse)
        self.comboBox_TaskType.currentTextChanged.connect(self.changedetailtext)

    def updateElapsedTime_Train(self):  # 实时更新训练计时
        elapsed = time.time() - self.start_time_Train
        if self.cpercentage_Train == 0:
            self.label_TrainTimer.setText(f"{self.format_elapsed_time(elapsed)}<<00:00:00")
        else:
            self.label_TrainTimer.setText(f"{self.format_elapsed_time(elapsed)}<<{self.format_elapsed_time(elapsed / self.cpercentage_Train  - elapsed)}")
        QCoreApplication.processEvents()   # 确保立即更新

    def updateElapsedTime_Dev(self):  # 实时更新评估计时
        elapsed = time.time() - self.start_time_Dev
        if self.cpercentage_Dev == 0:
            self.label_DevTimer.setText(f"{self.format_elapsed_time(elapsed)}<<00:00:00")
        else:
            self.label_DevTimer.setText(f"{self.format_elapsed_time(elapsed)}<<{self.format_elapsed_time(elapsed / self.cpercentage_Dev  - elapsed)}")
        QCoreApplication.processEvents()   # 确保立即更新
    
    def format_elapsed_time(self, seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        return f"{hours:02}:{minutes:02}:{seconds:02}"
    
    def browse_PretrainedModel(self):  # 选择预训练模型存储路径
        directory = QtWidgets.QFileDialog.getExistingDirectory(None,"选取文件夹","../model/chinese-roberta-small-wwm-cluecorpussmall")
        self.lineEdit_PretrainedModel.setText(directory)

    def browse_Data(self):  # 选择数据存储路径
        directory = QtWidgets.QFileDialog.getExistingDirectory(None,"选取文件夹","./data/data.json")
        self.lineEdit_Data.setText(directory)

    def browse_BestModel(self):  # 选择训练完成的模型存储路径
        directory = QtWidgets.QFileDialog.getExistingDirectory(None,"选取文件夹","./checkpoints/")
        self.lineEdit_BestModel.setText(directory)
    
    def update_TextBrowser(self, text):  # 更新主文本框
        self.textBrowser.append(text)
        QCoreApplication.processEvents()

    def update_Train(self, value):  # 更新训练进度条
        self.progressBar_Train.setValue(int(value * 100))
        self.progressBar_Train.setFormat(f"{value * 100:.2f}%")
        QCoreApplication.processEvents()

    def update_Dev(self, value):  # 更新评估进度条
        self.progressBar_Dev.setValue(int(value * 100))
        self.progressBar_Dev.setFormat(f"{value * 100:.2f}%")
        QCoreApplication.processEvents()
    
    def runparse(self):  # 调用训练主函数
        self.start_time = time.time()
        try:
            train.train(self)
        except Exception as e:
            self.update_TextBrowser(e)
        self.updateElapsedTime()

    def updateElapsedTime(self):  # 计算总用时
        elapsed = time.time() - self.start_time
        self.update_TextBrowser(f"总用时  {self.format_elapsed_time(elapsed)}")

    def changedetailtext(self):  # 根据第一个下拉框的值改变第二个下拉框的选项
        self.comboBox_TaskTypeDetail.clear()
        self.comboBox_TaskTypeDetail.addItems([""])
        if self.comboBox_TaskType.currentText() == "文本分类":
            self.comboBox_TaskTypeDetail.addItems(["单标签分类", "多标签分类"])
        

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    myWin = MyMainWindow()
    myWin.show()
    sys.exit(app.exec_())
