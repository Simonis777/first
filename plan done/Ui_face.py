# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'e:\MyFirstAI\face.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.front = QtWidgets.QLabel(self.centralwidget)
        self.front.setGeometry(QtCore.QRect(61, 34, 661, 61))
        self.front.setObjectName("front")
        self.layoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget.setGeometry(QtCore.QRect(20, 140, 741, 31))
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
        self.layoutWidget1.setGeometry(QtCore.QRect(30, 200, 721, 341))
        self.layoutWidget1.setObjectName("layoutWidget1")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.layoutWidget1)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
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
        self.horizontalLayout_2.addLayout(self.verticalLayout)
        self.bar_chart = QtWidgets.QLabel(self.layoutWidget1)
        self.bar_chart.setFrameShape(QtWidgets.QFrame.Panel)
        self.bar_chart.setText("")
        self.bar_chart.setObjectName("bar_chart")
        self.horizontalLayout_2.addWidget(self.bar_chart)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 26))
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
