import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QProgressBar
from PyQt5.QtCore import QBasicTimer
from PyQt5.QtGui import QFont
import time

class MyClass(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.resize(500, 300)
        # 载入进度条控件
        self.pgb = QProgressBar(self)
        self.pgb.move(50, 50)
        self.pgb.resize(250, 20)
        self.pgb.setStyleSheet("QProgressBar { border: 2px solid grey; border-radius: 5px; color: rgb(20,20,20);  background-color: #FFFFFF; text-align: center;}QProgressBar::chunk {background-color: rgb(100,200,200); border-radius: 10px; margin: 0.1px;  width: 1px;}")
        ## 其中 width 是设置进度条每一步的宽度
        ## margin 设置两步之间的间隔
        #设置字体
        font = QFont()
        font.setBold(True)
        font.setWeight(30)
        self.pgb.setFont(font)
        # 设置一个值表示进度条的当前进度
        self.pv = 0
        # 申明一个时钟控件
        self.timer1 = QBasicTimer()

        # 设置进度条的范围
        self.pgb.setMinimum(0)
        self.pgb.setMaximum(100)
        self.pgb.setValue(self.pv)
        ## 设置进度条文字格式
        self.pgb.setFormat('Loaded  %p%, {}'.format(self.pgb.value()-self.pgb.minimum(), time.time))
        # 加载pushbutton1
        self.btn_start = QPushButton("begin", self)
        self.btn_start.move(50, 100)
        self.btn_start.clicked.connect(self.myTimerState)
        # 加载 pushbutton 2
        self.btn_update = QPushButton("update", self)
        self.btn_update.move(150, 100)
        self.btn_update.clicked.connect(self.update_event)
        
    def myTimerState(self):
        if self.timer1.isActive():
            self.timer1.stop()
            self.btn_start.setText("begin")
        else:
            self.timer1.start(100, self)
            self.btn_start.setText("stop")

    def timerEvent(self, e):
        if self.pv == 100:
            self.timer1.stop()
            self.btn_start.setText("Finish")
        else:
            self.pv += 1
            self.pgb.setValue(self.pv)

    def update_event(self):
        if self.timer1.isActive():
            self.timer1.stop()
        self.btn_start.setText("begin")
        self.pv = 0
        self.pgb.setValue(self.pv)
        
if __name__ == "__main__":
    app = QApplication(sys.argv)
    mytask = MyClass()
    mytask.show()
    app.exec_()
