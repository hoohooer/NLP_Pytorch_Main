# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'd:\project\liyunhao\NLP_Pytorch_Main\MainWindow.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1000, 800)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("d:\\project\\liyunhao\\NLP_Pytorch_Main\\OGAS.jpg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        MainWindow.setWindowIcon(icon)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.pushButton_PretrainedModel = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_PretrainedModel.setGeometry(QtCore.QRect(40, 40, 60, 20))
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(10)
        self.pushButton_PretrainedModel.setFont(font)
        self.pushButton_PretrainedModel.setStyleSheet("")
        self.pushButton_PretrainedModel.setObjectName("pushButton_PretrainedModel")
        self.label_PretrainedModel = QtWidgets.QLabel(self.centralwidget)
        self.label_PretrainedModel.setGeometry(QtCore.QRect(40, 10, 320, 30))
        self.label_PretrainedModel.setStyleSheet("font: 12pt \"宋体\";")
        self.label_PretrainedModel.setObjectName("label_PretrainedModel")
        self.lineEdit_PretrainedModel = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_PretrainedModel.setGeometry(QtCore.QRect(120, 40, 320, 20))
        self.lineEdit_PretrainedModel.setReadOnly(False)
        self.lineEdit_PretrainedModel.setObjectName("lineEdit_PretrainedModel")
        self.lineEdit_Data = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_Data.setGeometry(QtCore.QRect(120, 100, 320, 20))
        self.lineEdit_Data.setReadOnly(False)
        self.lineEdit_Data.setObjectName("lineEdit_Data")
        self.pushButton_Data = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_Data.setGeometry(QtCore.QRect(40, 100, 60, 20))
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(10)
        self.pushButton_Data.setFont(font)
        self.pushButton_Data.setStyleSheet("")
        self.pushButton_Data.setObjectName("pushButton_Data")
        self.label_Data = QtWidgets.QLabel(self.centralwidget)
        self.label_Data.setGeometry(QtCore.QRect(40, 70, 320, 30))
        self.label_Data.setStyleSheet("font: 12pt \"宋体\";")
        self.label_Data.setObjectName("label_Data")
        self.label_BestModel = QtWidgets.QLabel(self.centralwidget)
        self.label_BestModel.setGeometry(QtCore.QRect(40, 130, 320, 30))
        self.label_BestModel.setStyleSheet("font: 12pt \"宋体\";")
        self.label_BestModel.setObjectName("label_BestModel")
        self.lineEdit_BestModel = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_BestModel.setGeometry(QtCore.QRect(120, 160, 320, 20))
        self.lineEdit_BestModel.setReadOnly(False)
        self.lineEdit_BestModel.setObjectName("lineEdit_BestModel")
        self.pushButton_BestModel = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_BestModel.setGeometry(QtCore.QRect(40, 160, 60, 20))
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(10)
        self.pushButton_BestModel.setFont(font)
        self.pushButton_BestModel.setStyleSheet("")
        self.pushButton_BestModel.setObjectName("pushButton_BestModel")
        self.label_TaskName = QtWidgets.QLabel(self.centralwidget)
        self.label_TaskName.setGeometry(QtCore.QRect(480, 10, 100, 30))
        self.label_TaskName.setStyleSheet("font: 12pt \"宋体\";")
        self.label_TaskName.setObjectName("label_TaskName")
        self.textBrowser = QtWidgets.QTextBrowser(self.centralwidget)
        self.textBrowser.setGeometry(QtCore.QRect(40, 240, 920, 400))
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(12)
        self.textBrowser.setFont(font)
        self.textBrowser.setObjectName("textBrowser")
        self.pushButton_RunParse = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_RunParse.setGeometry(QtCore.QRect(870, 200, 70, 30))
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(10)
        self.pushButton_RunParse.setFont(font)
        self.pushButton_RunParse.setStyleSheet("")
        self.pushButton_RunParse.setObjectName("pushButton_RunParse")
        self.progressBar_Train = QtWidgets.QProgressBar(self.centralwidget)
        self.progressBar_Train.setGeometry(QtCore.QRect(210, 660, 400, 20))
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(10)
        self.progressBar_Train.setFont(font)
        self.progressBar_Train.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.progressBar_Train.setProperty("value", 0)
        self.progressBar_Train.setAlignment(QtCore.Qt.AlignCenter)
        self.progressBar_Train.setObjectName("progressBar_Train")
        self.label_Train = QtWidgets.QLabel(self.centralwidget)
        self.label_Train.setGeometry(QtCore.QRect(40, 660, 160, 20))
        self.label_Train.setStyleSheet("font: 12pt \"宋体\";")
        self.label_Train.setObjectName("label_Train")
        self.progressBar_Dev = QtWidgets.QProgressBar(self.centralwidget)
        self.progressBar_Dev.setGeometry(QtCore.QRect(210, 690, 400, 20))
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(10)
        self.progressBar_Dev.setFont(font)
        self.progressBar_Dev.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.progressBar_Dev.setProperty("value", 0)
        self.progressBar_Dev.setAlignment(QtCore.Qt.AlignCenter)
        self.progressBar_Dev.setObjectName("progressBar_Dev")
        self.label_Dev = QtWidgets.QLabel(self.centralwidget)
        self.label_Dev.setGeometry(QtCore.QRect(40, 690, 160, 20))
        self.label_Dev.setStyleSheet("font: 12pt \"宋体\";")
        self.label_Dev.setObjectName("label_Dev")
        self.label_TrainTimer = QtWidgets.QLabel(self.centralwidget)
        self.label_TrainTimer.setGeometry(QtCore.QRect(800, 660, 161, 20))
        self.label_TrainTimer.setStyleSheet("font: 12pt \"宋体\";")
        self.label_TrainTimer.setObjectName("label_TrainTimer")
        self.label_DevTimer = QtWidgets.QLabel(self.centralwidget)
        self.label_DevTimer.setGeometry(QtCore.QRect(800, 690, 161, 20))
        self.label_DevTimer.setStyleSheet("font: 12pt \"宋体\";")
        self.label_DevTimer.setObjectName("label_DevTimer")
        self.lineEdit_TaskName = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_TaskName.setGeometry(QtCore.QRect(480, 40, 460, 20))
        self.lineEdit_TaskName.setReadOnly(False)
        self.lineEdit_TaskName.setObjectName("lineEdit_TaskName")
        self.comboBox_TaskType = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox_TaskType.setGeometry(QtCore.QRect(560, 80, 120, 20))
        font = QtGui.QFont()
        font.setFamily("宋体")
        self.comboBox_TaskType.setFont(font)
        self.comboBox_TaskType.setObjectName("comboBox_TaskType")
        self.comboBox_TaskType.addItem("")
        self.comboBox_TaskType.addItem("")
        self.comboBox_TaskType.addItem("")
        self.label_TaskType = QtWidgets.QLabel(self.centralwidget)
        self.label_TaskType.setGeometry(QtCore.QRect(480, 80, 70, 20))
        self.label_TaskType.setStyleSheet("font: 12pt \"宋体\";")
        self.label_TaskType.setObjectName("label_TaskType")
        self.comboBox_TaskTypeDetail = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox_TaskTypeDetail.setGeometry(QtCore.QRect(810, 80, 130, 20))
        font = QtGui.QFont()
        font.setFamily("宋体")
        self.comboBox_TaskTypeDetail.setFont(font)
        self.comboBox_TaskTypeDetail.setObjectName("comboBox_TaskTypeDetail")
        self.comboBox_TaskTypeDetail.addItem("")
        self.comboBox_TaskTypeDetail.addItem("")
        self.label_TaskTypeDetail = QtWidgets.QLabel(self.centralwidget)
        self.label_TaskTypeDetail.setGeometry(QtCore.QRect(700, 80, 100, 20))
        self.label_TaskTypeDetail.setStyleSheet("font: 12pt \"宋体\";")
        self.label_TaskTypeDetail.setObjectName("label_TaskTypeDetail")
        self.label_MaxSeqLen = QtWidgets.QLabel(self.centralwidget)
        self.label_MaxSeqLen.setGeometry(QtCore.QRect(480, 120, 100, 20))
        self.label_MaxSeqLen.setStyleSheet("font: 12pt \"宋体\";")
        self.label_MaxSeqLen.setObjectName("label_MaxSeqLen")
        self.lineEdit_MaxSeqLen = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_MaxSeqLen.setGeometry(QtCore.QRect(590, 120, 50, 20))
        self.lineEdit_MaxSeqLen.setReadOnly(False)
        self.lineEdit_MaxSeqLen.setObjectName("lineEdit_MaxSeqLen")
        self.lineEdit_BatchSize = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_BatchSize.setGeometry(QtCore.QRect(740, 120, 50, 20))
        self.lineEdit_BatchSize.setReadOnly(False)
        self.lineEdit_BatchSize.setObjectName("lineEdit_BatchSize")
        self.label_BatchSize = QtWidgets.QLabel(self.centralwidget)
        self.label_BatchSize.setGeometry(QtCore.QRect(660, 120, 70, 20))
        self.label_BatchSize.setStyleSheet("font: 12pt \"宋体\";")
        self.label_BatchSize.setObjectName("label_BatchSize")
        self.label_Epochs = QtWidgets.QLabel(self.centralwidget)
        self.label_Epochs.setGeometry(QtCore.QRect(810, 120, 70, 20))
        self.label_Epochs.setStyleSheet("font: 12pt \"宋体\";")
        self.label_Epochs.setObjectName("label_Epochs")
        self.lineEdit_Epochs = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_Epochs.setGeometry(QtCore.QRect(890, 120, 50, 20))
        self.lineEdit_Epochs.setReadOnly(False)
        self.lineEdit_Epochs.setObjectName("lineEdit_Epochs")
        self.label_TrainLoss = QtWidgets.QLabel(self.centralwidget)
        self.label_TrainLoss.setGeometry(QtCore.QRect(630, 660, 160, 20))
        self.label_TrainLoss.setStyleSheet("font: 12pt \"宋体\";")
        self.label_TrainLoss.setObjectName("label_TrainLoss")
        self.label_DevLoss = QtWidgets.QLabel(self.centralwidget)
        self.label_DevLoss.setGeometry(QtCore.QRect(630, 690, 160, 20))
        self.label_DevLoss.setStyleSheet("font: 12pt \"宋体\";")
        self.label_DevLoss.setObjectName("label_DevLoss")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1000, 23))
        self.menubar.setObjectName("menubar")
        self.menu_files = QtWidgets.QMenu(self.menubar)
        self.menu_files.setObjectName("menu_files")
        self.menu_Deployment = QtWidgets.QMenu(self.menubar)
        self.menu_Deployment.setObjectName("menu_Deployment")
        MainWindow.setMenuBar(self.menubar)
        self.action_LoadConfig = QtWidgets.QAction(MainWindow)
        self.action_LoadConfig.setObjectName("action_LoadConfig")
        self.action_SaveConfig = QtWidgets.QAction(MainWindow)
        self.action_SaveConfig.setObjectName("action_SaveConfig")
        self.action_Deploy = QtWidgets.QAction(MainWindow)
        self.action_Deploy.setObjectName("action_Deploy")
        self.menu_files.addAction(self.action_LoadConfig)
        self.menu_files.addAction(self.action_SaveConfig)
        self.menu_Deployment.addAction(self.action_Deploy)
        self.menubar.addAction(self.menu_files.menuAction())
        self.menubar.addAction(self.menu_Deployment.menuAction())

        self.retranslateUi(MainWindow)
        self.pushButton_PretrainedModel.clicked.connect(MainWindow.browse_PretrainedModel) # type: ignore
        self.pushButton_BestModel.clicked.connect(MainWindow.browse_BestModel) # type: ignore
        self.pushButton_Data.clicked.connect(MainWindow.browse_Data) # type: ignore
        self.action_LoadConfig.triggered.connect(MainWindow.loadconfig) # type: ignore
        self.action_SaveConfig.triggered.connect(MainWindow.saveconfig) # type: ignore
        self.action_Deploy.triggered.connect(MainWindow.openwindow) # type: ignore
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "NLP模型训练系统"))
        self.pushButton_PretrainedModel.setText(_translate("MainWindow", "选择"))
        self.label_PretrainedModel.setText(_translate("MainWindow", "预训练模型路径"))
        self.lineEdit_PretrainedModel.setText(_translate("MainWindow", "../model/chinese-roberta-small-wwm-cluecorpussmall"))
        self.lineEdit_Data.setText(_translate("MainWindow", "./data/data.json"))
        self.pushButton_Data.setText(_translate("MainWindow", "选择"))
        self.label_Data.setText(_translate("MainWindow", "语料路径"))
        self.label_BestModel.setText(_translate("MainWindow", "模型保存根路径"))
        self.lineEdit_BestModel.setText(_translate("MainWindow", "./checkpoints/"))
        self.pushButton_BestModel.setText(_translate("MainWindow", "选择"))
        self.label_TaskName.setText(_translate("MainWindow", "任务名称"))
        self.pushButton_RunParse.setText(_translate("MainWindow", "运行"))
        self.progressBar_Train.setFormat(_translate("MainWindow", "%p%"))
        self.label_Train.setText(_translate("MainWindow", "训练进度"))
        self.progressBar_Dev.setFormat(_translate("MainWindow", "%p%"))
        self.label_Dev.setText(_translate("MainWindow", "评估进度"))
        self.label_TrainTimer.setText(_translate("MainWindow", "00:00:00<<00:00:00"))
        self.label_DevTimer.setText(_translate("MainWindow", "00:00:00<<00:00:00"))
        self.lineEdit_TaskName.setText(_translate("MainWindow", "test"))
        self.comboBox_TaskType.setItemText(0, _translate("MainWindow", "文本分类"))
        self.comboBox_TaskType.setItemText(1, _translate("MainWindow", "实体识别"))
        self.comboBox_TaskType.setItemText(2, _translate("MainWindow", "关系抽取"))
        self.label_TaskType.setText(_translate("MainWindow", "任务类型"))
        self.comboBox_TaskTypeDetail.setItemText(0, _translate("MainWindow", "单标签分类"))
        self.comboBox_TaskTypeDetail.setItemText(1, _translate("MainWindow", "多标签分类"))
        self.label_TaskTypeDetail.setText(_translate("MainWindow", "详细任务类型"))
        self.label_MaxSeqLen.setText(_translate("MainWindow", "最大序列长度"))
        self.lineEdit_MaxSeqLen.setText(_translate("MainWindow", "64"))
        self.lineEdit_BatchSize.setText(_translate("MainWindow", "32"))
        self.label_BatchSize.setText(_translate("MainWindow", "批次大小"))
        self.label_Epochs.setText(_translate("MainWindow", "训练轮数"))
        self.lineEdit_Epochs.setText(_translate("MainWindow", "50"))
        self.label_TrainLoss.setText(_translate("MainWindow", "Loss:0.000000"))
        self.label_DevLoss.setText(_translate("MainWindow", "TLoss:0.000000"))
        self.menu_files.setTitle(_translate("MainWindow", "文件"))
        self.menu_Deployment.setTitle(_translate("MainWindow", "测试"))
        self.action_LoadConfig.setText(_translate("MainWindow", "加载配置"))
        self.action_SaveConfig.setText(_translate("MainWindow", "保存配置"))
        self.action_Deploy.setText(_translate("MainWindow", "模型测试"))
