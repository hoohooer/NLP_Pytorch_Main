import sys
from Ui_MainWindow import Ui_MainWindow  
from Ui_DeploymentDialog import Ui_DeploymentDialog
import qtawesome
from PyQt5 import QtWidgets
from PyQt5.QtCore import QCoreApplication
from PyQt5.QtGui import QTextCursor
from parsedata import argsconfig, train, test
import time
import requests
from torch.utils.data import DataLoader
import configparser
import traceback


config = configparser.ConfigParser()
args = argsconfig.Args().get_parser()
class MyMainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
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
        file = QtWidgets.QFileDialog.getOpenFileName(None,"选取文件","./data","JSON Files (*.json)")
        self.lineEdit_Data.setText(file[0])

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
            self.textBrowser.clear()
            train.train(self)
        except Exception:
            self.update_TextBrowser("<span style='font-family:Arial; font-size:12pt; color:#FF0000;'>Error:{}</span>".format(traceback.format_exc()))
            print(traceback.format_exc())
        self.updateElapsedTime()

    def updateElapsedTime(self):  # 计算总用时
        elapsed = time.time() - self.start_time
        self.update_TextBrowser(f"总用时  {self.format_elapsed_time(elapsed)}")

    def changedetailtext(self):  # 根据第一个下拉框的值改变第二个下拉框的选项
        self.comboBox_TaskTypeDetail.clear()
        if self.comboBox_TaskType.currentText() == "文本分类":
            self.comboBox_TaskTypeDetail.addItems(["单标签分类", "多标签分类"])
    
    def saveconfig(self):  # 保存配置
        config['DEFAULT'] = {'lineEdit_PretrainedModel': self.lineEdit_PretrainedModel.text(),
                             'lineEdit_BestModel': self.lineEdit_BestModel.text(),
                             'lineEdit_Data': self.lineEdit_Data.text(),
                             'lineEdit_TaskName': self.lineEdit_TaskName.text(),
                             'lineEdit_MaxSeqLen': self.lineEdit_MaxSeqLen.text(),
                             'lineEdit_BatchSize': self.lineEdit_BatchSize.text(),
                             'lineEdit_Epochs': self.lineEdit_Epochs.text(),
                             'comboBox_TaskType': self.comboBox_TaskType.currentText(),
                             'comboBox_TaskTypeDetail': self.comboBox_TaskTypeDetail.currentText()}
        with open('config.ini', 'w') as configfile:
            config.write(configfile)
        QtWidgets.QMessageBox.information(self,"保存配置","已保存配置！")

    def loadconfig(self):  # 加载配置
        config.read('config.ini')
        self.lineEdit_PretrainedModel.setText(config['DEFAULT']['lineEdit_PretrainedModel'])
        self.lineEdit_BestModel.setText(config['DEFAULT']['lineEdit_BestModel'])
        self.lineEdit_Data.setText(config['DEFAULT']['lineEdit_Data'])
        self.lineEdit_TaskName.setText(config['DEFAULT']['lineEdit_TaskName'])
        self.lineEdit_MaxSeqLen.setText(config['DEFAULT']['lineEdit_MaxSeqLen'])
        self.lineEdit_BatchSize.setText(config['DEFAULT']['lineEdit_BatchSize'])
        self.lineEdit_Epochs.setText(config['DEFAULT']['lineEdit_Epochs'])
        self.comboBox_TaskType.setCurrentText(config['DEFAULT']['comboBox_TaskType'])
        self.comboBox_TaskTypeDetail.setCurrentText(config['DEFAULT']['comboBox_TaskTypeDetail'])
        QtWidgets.QMessageBox.information(self,"加载配置","已加载配置！")

    def openwindow(self):
        self.MyDeploymentDialog = MyDeploymentDialog()
        self.MyDeploymentDialog.show()


class MyDeploymentDialog(QtWidgets.QMainWindow, Ui_DeploymentDialog):
    def __init__(self,parent=None):
        super(MyDeploymentDialog,self).__init__(parent)
        self.setupUi(self)
        self.pushButton_Confirm.clicked.connect(self.runtest)
        self.pushButton_PortStatus.clicked.connect(self.testport)

    def browse_TrainedModel(self):  # 选择训练完成的模型存储路径
        directory = QtWidgets.QFileDialog.getExistingDirectory(None,"选取文件夹","./checkpoints/")
        self.lineEdit_TrainedModel.setText(directory)

    def runtest(self):  # 调用测试主函数
        self.start_time = time.time()
        try:
            test.test(self)
        except Exception:
            self.update_TextBrowser("<span style='font-family:Arial; font-size:12pt; color:#FF0000;'>Error:{}</span>".format(traceback.format_exc()))
            print(traceback.format_exc())
    
    def testport(self):  # 测试端口开启与否
        try:
            requests.head(self.lineEdit_Port.text(), timeout=1)
            QtWidgets.QMessageBox.information(self, "端口状态", self.lineEdit_Port.text() + "端口已开启！")
        except requests.exceptions.RequestException:
            QtWidgets.QMessageBox.information(self, "端口状态", self.lineEdit_Port.text() + "端口未开启！")
    
    def update_TextBrowser(self, text):  # 更新主文本框
        self.textBrowser_Output.append(text)
        QCoreApplication.processEvents()
    

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    myWin = MyMainWindow()
    myWin.show()
    sys.exit(app.exec_())
