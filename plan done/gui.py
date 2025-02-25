from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QPushButton, QAction, QLineEdit, QMessageBox

class Ui_MainWindow(QMainWindow):
    #所需数据
    total_main_theme = {}
    total_minor_theme = {}
    cul_main_theme = {}
    cul_minor_theme = {}
    sci_main_theme= {}
    sci_minor_theme={}
    whole_theme=[total_main_theme,total_minor_theme,cul_main_theme,cul_minor_theme,sci_main_theme,sci_minor_theme]
    cr=crawler() #爬虫类
    i=False #是否已经确认关键词

    def setupUi(self, MainWindow):
        MainWindow.setWindowIcon(QtGui.QIcon(r'.\FirstProject\image\cnkl1.png'))
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label_main = QtWidgets.QLabel(self.centralwidget)
        self.label_main.setGeometry(QtCore.QRect(200, 20, 400, 100))
        self.label_main.setObjectName("label_main")
        self.label_mid = QtWidgets.QLabel(self.centralwidget)
        self.label_mid.setGeometry(QtCore.QRect(60, 180, 150, 40))
        self.label_mid.setObjectName("label_mid")
        self.text = QtWidgets.QLineEdit(self.centralwidget)
        self.text.setGeometry(QtCore.QRect(250, 185, 350, 30))
        self.text.setObjectName("text")
        self.line = QtWidgets.QFrame(self.centralwidget)
        self.line.setGeometry(QtCore.QRect(0, 130, 800, 5))
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.line_2 = QtWidgets.QFrame(self.centralwidget)
        self.line_2.setGeometry(QtCore.QRect(0, 270, 800, 3))
        self.line_2.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.sure = QtWidgets.QPushButton(self.centralwidget)
        self.sure.setGeometry(QtCore.QRect(660, 185, 80, 30))
        self.sure.setObjectName("sure")
        self.label_last2 = QtWidgets.QLabel(self.centralwidget)
        self.label_last2.setGeometry(QtCore.QRect(20, 410, 120, 23))
        self.label_last2.setObjectName("label_last2")
        self.done = QtWidgets.QPushButton(self.centralwidget)
        self.done.setGeometry(QtCore.QRect(670, 490, 111, 50))
        self.done.setIconSize(QtCore.QSize(300, 20))
        self.done.setObjectName("done")
        self.label_last1 = QtWidgets.QLabel(self.centralwidget)
        self.label_last1.setGeometry(QtCore.QRect(320, 400, 120, 46))
        self.label_last1.setObjectName("label_last1")
        self.layoutWidget_2 = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget_2.setGeometry(QtCore.QRect(470, 290, 171, 271))
        self.layoutWidget_2.setObjectName("layoutWidget_2")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.layoutWidget_2)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.theme_1 = QtWidgets.QCheckBox(self.layoutWidget_2)
        self.theme_1.setObjectName("theme_1")
        self.verticalLayout.addWidget(self.theme_1)
        self.theme_2 = QtWidgets.QCheckBox(self.layoutWidget_2)
        self.theme_2.setObjectName("theme_2")
        self.verticalLayout.addWidget(self.theme_2)
        self.theme_3 = QtWidgets.QCheckBox(self.layoutWidget_2)
        self.theme_3.setObjectName("theme_3")
        self.verticalLayout.addWidget(self.theme_3)
        self.theme_4 = QtWidgets.QCheckBox(self.layoutWidget_2)
        self.theme_4.setObjectName("theme_4")
        self.verticalLayout.addWidget(self.theme_4)
        self.theme_5 = QtWidgets.QCheckBox(self.layoutWidget_2)
        self.theme_5.setObjectName("theme_5")
        self.verticalLayout.addWidget(self.theme_5)
        self.theme_6 = QtWidgets.QCheckBox(self.layoutWidget_2)
        self.theme_6.setObjectName("theme_6")
        self.verticalLayout.addWidget(self.theme_6)
        self.max = QtWidgets.QLineEdit(self.centralwidget)
        self.max.setGeometry(QtCore.QRect(660, 420, 113, 21))
        self.max.setInputMask("")
        self.max.setObjectName("max")
        self.label_last_last = QtWidgets.QLabel(self.centralwidget)
        self.label_last_last.setGeometry(QtCore.QRect(680, 400, 72, 15))
        self.label_last_last.setObjectName("label_last_last")
        self.widget = QtWidgets.QWidget(self.centralwidget)
        self.widget.setGeometry(QtCore.QRect(160, 330, 141, 181))
        self.widget.setObjectName("widget")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.widget)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.checkBox_bar = QtWidgets.QCheckBox(self.widget)
        self.checkBox_bar.setObjectName("checkBox_bar")
        self.verticalLayout_2.addWidget(self.checkBox_bar)
        self.checkBox_pie = QtWidgets.QCheckBox(self.widget)
        self.checkBox_pie.setObjectName("checkBox_pie")
        self.verticalLayout_2.addWidget(self.checkBox_pie)
        self.checkBox_wordcloud = QtWidgets.QCheckBox(self.widget)
        self.checkBox_wordcloud.setObjectName("checkBox_wordcloud")
        self.verticalLayout_2.addWidget(self.checkBox_wordcloud)
        self.checkBox_cnkl = QtWidgets.QCheckBox(self.widget)
        self.checkBox_cnkl.setObjectName("checkBox_cnkl")
        self.verticalLayout_2.addWidget(self.checkBox_cnkl)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 26))
        self.menubar.setObjectName("menubar")
        self.main = QtWidgets.QMenu(self.menubar)
        self.main.setObjectName("main")
        self.menu_1 = QtWidgets.QMenu(self.main)
        self.menu_1.setObjectName("menu_1")
        self.menu_2 = QtWidgets.QMenu(self.main)
        self.menu_2.setObjectName("menu_2")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.ProjectDescription = QtWidgets.QAction(MainWindow)
        self.ProjectDescription.setObjectName("ProjectDescription")
        self.instruction = QtWidgets.QAction(MainWindow)
        self.instruction.setObjectName("instruction")
        self.menu_1.addAction(self.ProjectDescription)
        self.menu_2.addAction(self.instruction)
        self.main.addAction(self.menu_1.menuAction())
        self.main.addAction(self.menu_2.menuAction())
        self.menubar.addAction(self.main.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        # 当点击确认按钮时，触发的事件
        self.sure.clicked.connect(self.kaipa)

        # 当点击生成按钮时，触发的事件
        self.done.clicked.connect(self.doing)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "论文分析工具"))
        self.label_main.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:36pt; font-weight:600;\">论文分析工具</span></p></body></html>"))
        self.label_mid.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:18pt;\">论文关键词</span></p></body></html>"))
        self.text.setText(_translate("MainWindow", "此处键入"))
        self.sure.setText(_translate("MainWindow", "确认"))
        self.label_last2.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:12pt;\">选择数据类型</span></p></body></html>"))
        self.done.setText(_translate("MainWindow", "生成"))
        self.label_last1.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:12pt;\">选择相关主题<br/>生成excel</span></p></body></html>"))
        self.theme_1.setText(_translate("MainWindow", "请先确认关键词"))
        self.theme_2.setText(_translate("MainWindow", "请先确认关键词"))
        self.theme_3.setText(_translate("MainWindow", "请先确认关键词"))
        self.theme_4.setText(_translate("MainWindow", "请先确认关键词"))
        self.theme_5.setText(_translate("MainWindow", "请先确认关键词"))
        self.theme_6.setText(_translate("MainWindow", "请先确认关键词"))
        self.max.setText(_translate("MainWindow", "?"))
        self.label_last_last.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:7pt;\">excel条目数</span></p></body></html>"))
        self.checkBox_bar.setText(_translate("MainWindow", "数据柱形图"))
        self.checkBox_pie.setText(_translate("MainWindow", "数据占比饼图"))
        self.checkBox_wordcloud.setText(_translate("MainWindow", "数据占比词云图"))
        self.checkBox_cnkl.setText(_translate("MainWindow", "知网网页快照"))
        self.main.setTitle(_translate("MainWindow", "说明文档"))
        self.menu_1.setTitle(_translate("MainWindow", "项目说明"))
        self.menu_2.setTitle(_translate("MainWindow", "使用方法"))
        self.ProjectDescription.setText(_translate("MainWindow", "本项目是基于知网查询的一款对已有论文的数据分析程序。"))
        self.instruction.setText(_translate("MainWindow", "在“论文关键词”处输入关键词后，点击“确认”；之后选择所需数据类型以及选择excle所额外需求的主题之后点击“生成”，便会在上生成一个文件夹，其中包含了你需要的数据类型。"))

    '''def kaipa(self):
        self.cr=crawler()
        if self.text.text()=='此处键入' or self.text.text()=='':
            QMessageBox.warning(self, "警告", "请先确认关键词", QMessageBox.Yes)
            return
        _translate = QtCore.QCoreApplication.translate
        #将keyword传入，返回一个字典
        keyword=self.text.text()
        #爬，返回字典（类变量）
        self.cr.kaipa(new_input=keyword,new_theme=self.whole_theme)
        theme_needchange=self.whole_theme[0]
        #对于字典theme_needchange,按照顺序将其内容显示在对应的checkBox上
        theme=list(theme_needchange.keys())
        #对于theme里面的顺序，按照theme的顺序，将theme的内容显示在对应的checkBox上
        self.theme_1.setText(_translate("MainWindow", theme[0]))
        self.theme_2.setText(_translate("MainWindow", theme[1]))
        self.theme_3.setText(_translate("MainWindow", theme[2]))
        self.theme_4.setText(_translate("MainWindow", theme[3]))
        self.theme_5.setText(_translate("MainWindow", theme[4]))
        self.theme_6.setText(_translate("MainWindow", theme[5]))

        #完成操作后，将所有checkBox的状态设置为未选中
        self.theme_1.setChecked(False)
        self.theme_2.setChecked(False)
        self.theme_3.setChecked(False)
        self.theme_4.setChecked(False)
        self.theme_5.setChecked(False)
        self.theme_6.setChecked(False)
        self.checkBox_bar.setChecked(False)
        self.checkBox_pie.setChecked(False)
        self.checkBox_wordcloud.setChecked(False)
        self.checkBox_cnkl.setChecked(False)

        #i为True，表示已经确认关键词
        self.i=True

        #完成操作后，生成一个弹窗
        QMessageBox.information(self, "提示", "关键词已确认", QMessageBox.Yes)

    def doing(self):
        #判断是否已经确认关键词
        if self.text.text()=='此处键入' or self.text.text()=='':
            QMessageBox.warning(self, "警告", "请先确认关键词", QMessageBox.Yes)
            return
        if self.i==False:
            QMessageBox.warning(self, "警告", "请先点击确认", QMessageBox.Yes)
            return
        #判断max.text()是否为数字
        if self.max.text().isdigit()==False:
            QMessageBox.warning(self, "警告", "请在excel条目数处输入数字", QMessageBox.Yes)
            return

        #处理图像部分
        pic=picture(self.whole_theme,self.text.text())
        if self.checkBox_pie.isChecked():
            pic.pie_chart()
        if self.checkBox_bar.isChecked():
            pic.bar_chart()
        if self.checkBox_wordcloud.isChecked():
            pic.wordcloud_chart()
        #是否需要网页快照
        if self.checkBox_cnkl.isChecked():
            self.cr.get_stru()
            
        #将用户选择的主题和数据类型传入，返回一个文件夹
        #将theme_1~6和checkBox_pie,checkBox_wordcloud,checkBox_cnkl的选定状态存入两个字典为String：bool
        theme={}
        theme['theme_1']=self.theme_1.isChecked()
        theme['theme_2']=self.theme_2.isChecked()
        theme['theme_3']=self.theme_3.isChecked()
        theme['theme_4']=self.theme_4.isChecked()
        theme['theme_5']=self.theme_5.isChecked()
        theme['theme_6']=self.theme_6.isChecked()
        self.cr.excel(theme,int(self.max.text()))'''

        #完成操作后，生成一个弹窗
        #QMessageBox.information(self, "提示", "数据已生成", QMessageBox.Yes)

if __name__ == "__main__":
    #模拟运行
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())