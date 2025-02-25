from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QPushButton, QAction, QLineEdit, QMessageBox
import sys
from test import judge
import cv2 as cv

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        self.al_path = False
        self.emotion_list = ["afraid", "angry", "disgust", "happy", "neutral", "sad", "surprise"]

        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(600, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.front = QtWidgets.QLabel(self.centralwidget)
        self.front.setGeometry(QtCore.QRect(30, 30, 541, 61))
        self.front.setTextFormat(QtCore.Qt.RichText)
        self.front.setObjectName("front")
        self.layoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget.setGeometry(QtCore.QRect(30, 130, 541, 31))
        self.layoutWidget.setObjectName("layoutWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.layoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.filepath = QtWidgets.QLabel(self.layoutWidget)
        self.filepath.setObjectName("filepath")
        self.horizontalLayout.addWidget(self.filepath)
        self.path = QtWidgets.QLineEdit(self.layoutWidget)
        self.path.setObjectName("path")
        self.horizontalLayout.addWidget(self.path)
        self.scaner = QtWidgets.QPushButton(self.layoutWidget)
        self.scaner.setObjectName("scaner")
        self.horizontalLayout.addWidget(self.scaner)
        self.layoutWidget1 = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget1.setGeometry(QtCore.QRect(30, 200, 541, 331))
        self.layoutWidget1.setObjectName("layoutWidget1")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.layoutWidget1)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.pic = QtWidgets.QLabel(self.layoutWidget1)
        self.pic.setFrameShape(QtWidgets.QFrame.Panel)
        self.pic.setLineWidth(1)
        self.pic.setMidLineWidth(0)
        self.pic.setText("")
        self.pic.setObjectName("pic")
        self.verticalLayout.addWidget(self.pic)
        self.check = QtWidgets.QPushButton(self.layoutWidget1)
        self.check.setObjectName("check")
        self.verticalLayout.addWidget(self.check)
        self.verticalLayout_2.addLayout(self.verticalLayout)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 600, 26))
        self.menubar.setObjectName("menubar")
        self.menu = QtWidgets.QMenu(self.menubar)
        self.menu.setObjectName("menu")
        self.menu_2 = QtWidgets.QMenu(self.menubar)
        self.menu_2.setObjectName("menu_2")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.action1 = QtWidgets.QAction(MainWindow)
        self.action1.setObjectName("action1")
        self.actiond = QtWidgets.QAction(MainWindow)
        self.actiond.setObjectName("actiond")
        self.menu.addAction(self.action1)
        self.menu_2.addAction(self.actiond)
        self.menubar.addAction(self.menu.menuAction())
        self.menubar.addAction(self.menu_2.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        # 完成点击事件
        self.scaner.clicked.connect(self.openfile)
        self.path.returnPressed.connect(self.openfile_straight)
        self.check.clicked.connect(self.face_recognition)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.front.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:36pt; font-weight:600;\">面部情绪识别</span></p></body></html>"))
        self.filepath.setText(_translate("MainWindow", "图片路径"))
        self.scaner.setText(_translate("MainWindow", "浏览"))
        self.check.setText(_translate("MainWindow", "确认"))
        self.menu.setTitle(_translate("MainWindow", "项目说明"))
        self.menu_2.setTitle(_translate("MainWindow", "使用介绍"))
        self.action1.setText(_translate("MainWindow", "该项目目的在于完成一个简单的面部情绪识别任务"))
        self.actiond.setText(_translate("MainWindow", "输入图像路径后，点击确认，之后即会呈现分析数据"))

    # 打开文件
    def openfile(self):
        file, _ = QtWidgets.QFileDialog.getOpenFileName(None, "选择文件", "./", "All Files(*)")
        self.path.clear()
        self.path.setText(file)
        pic = QtGui.QPixmap(file)
        pic = pic.scaled(292,537,Qt.KeepAspectRatio)
        self.pic.setPixmap(pic)
        self.al_path = True

    # 直接输入路径打开文件
    def openfile_straight(self):
        file = self.path.text()  
        pic = QtGui.QPixmap(file)
        if pic.isNull():
            QMessageBox.warning(None, "警告", "无效的图片文件")
            return
        pic = pic.scaled(292,537,Qt.KeepAspectRatio)
        self.pic.setPixmap(pic)
        self.al_path = True

    def face_recognition(self):
        if not self.al_path:
            QMessageBox.warning(None, "警告", "请先选择图片")
            return
        # 利用opencv查看是否为照片
        img = cv.imread(self.path.text())
        
        if img is None:
            QMessageBox.warning(None, "警告", "请确认图片格式")
            return
        emotion_probabilities = judge(self.path.text())
        emotion = dict(zip(self.emotion_list, emotion_probabilities))
        # 将所有情绪及其百分比显示在弹窗中
        msg = ""
        for key in emotion:
            msg += key + ": " + "{:.2f}".format(emotion[key]) + "%\n"
        
        QMessageBox.information(None, "情绪识别", msg)

#测试
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
