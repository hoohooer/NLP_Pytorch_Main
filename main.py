import sys
from Ui_MainWindow import Ui_MainWindow  
import qtawesome
from PyQt5 import QtWidgets
from PyQt5.QtCore import QObject, pyqtSignal, QEventLoop, QTimer, QThread, QTime
from PyQt5.QtGui import QTextCursor
from parsedata.train import *


class MyMainWindow(QtWidgets.QMainWindow,Ui_MainWindow): 
    def __init__(self,parent=None):
        super(MyMainWindow,self).__init__(parent)
        self.setupUi(self)
        self.pushButton_RunParse.clicked.connect(self.runparse)
        self.th = MyThread()
        self.th.signalForText.connect(self.onUpdateText)
        sys.stdout = self.th
    
    def browse_PretrainedModel(self):
        directory = QtWidgets.QFileDialog.getExistingDirectory(None,"选取文件夹","C:/")
        self.lineEdit_PretrainedModel.setText(directory)

    def browse_Data(self):
        directory = QtWidgets.QFileDialog.getExistingDirectory(None,"选取文件夹","C:/")
        self.lineEdit_Data.setText(directory)

    def browse_BestModel(self):
        directory = QtWidgets.QFileDialog.getExistingDirectory(None,"选取文件夹","C:/")
        self.lineEdit_BestModel.setText(directory)

    def onUpdateText(self,text):
        cursor = self.textBrowser.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(text)
        self.textBrowser.setTextCursor(cursor)
        self.textBrowser.ensureCursorVisible()
    
    def search(self):
        try:
            self.t = MyThread()
            self.t.start()
        except Exception as e:
            raise e

    def runparse(self):
        """Runs the main function."""
        print('Running...')
        self.search()

        loop = QEventLoop()
        QTimer.singleShot(2000, loop.quit)
        loop.exec_()

    def closeEvent(self, event):
        """Shuts down application on close."""
        # Return stdout to defaults.
        sys.stdout = sys.__stdout__
        super().closeEvent(event)


class MyThread(QThread):
    signalForText = pyqtSignal(str)

    def __init__(self,data=None, parent=None):
        super(MyThread, self).__init__(parent)
        self.data = data

    def write(self, text):
        self.signalForText.emit(str(text))  # 发射信号

    def run(self):
        main()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    myWin = MyMainWindow()
    myWin.show()
    sys.exit(app.exec_())    
