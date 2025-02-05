# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'gui_2.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1669, 1010)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.index = QtWidgets.QFrame(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.index.sizePolicy().hasHeightForWidth())
        self.index.setSizePolicy(sizePolicy)
        self.index.setMinimumSize(QtCore.QSize(0, 200))
        self.index.setMaximumSize(QtCore.QSize(150, 16777215))
        self.index.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.index.setFrameShadow(QtWidgets.QFrame.Raised)
        self.index.setObjectName("index")
        self.page_1 = QtWidgets.QPushButton(self.index)
        self.page_1.setGeometry(QtCore.QRect(10, 60, 93, 28))
        self.page_1.setObjectName("page_1")
        self.page_2 = QtWidgets.QPushButton(self.index)
        self.page_2.setGeometry(QtCore.QRect(10, 110, 93, 28))
        self.page_2.setObjectName("page_2")
        self.gridLayout.addWidget(self.index, 0, 0, 3, 1)
        self.window = QtWidgets.QFrame(self.centralwidget)
        self.window.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.window.setFrameShadow(QtWidgets.QFrame.Raised)
        self.window.setObjectName("window")
        self.stackedWidget = QtWidgets.QStackedWidget(self.window)
        self.stackedWidget.setGeometry(QtCore.QRect(70, 0, 1411, 921))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.stackedWidget.sizePolicy().hasHeightForWidth())
        self.stackedWidget.setSizePolicy(sizePolicy)
        self.stackedWidget.setObjectName("stackedWidget")
        self.page = QtWidgets.QWidget()
        self.page.setObjectName("page")
        self.tableWidget = QtWidgets.QTableWidget(self.page)
        self.tableWidget.setGeometry(QtCore.QRect(25, 111, 1081, 771))
        self.tableWidget.setObjectName("tableWidget")
        self.tableWidget.setColumnCount(4)
        self.tableWidget.setRowCount(0)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignCenter)
        self.tableWidget.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignCenter)
        self.tableWidget.setHorizontalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignCenter)
        self.tableWidget.setHorizontalHeaderItem(2, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignCenter)
        self.tableWidget.setHorizontalHeaderItem(3, item)
        self.tableWidget.horizontalHeader().setDefaultSectionSize(270)
        self.tableWidget.verticalHeader().setVisible(False)
        self.groupBox_4 = QtWidgets.QGroupBox(self.page)
        self.groupBox_4.setGeometry(QtCore.QRect(20, 9, 1081, 91))
        self.groupBox_4.setObjectName("groupBox_4")
        self.remove_patient = QtWidgets.QPushButton(self.groupBox_4)
        self.remove_patient.setGeometry(QtCore.QRect(40, 20, 341, 28))
        self.remove_patient.setObjectName("remove_patient")
        self.change_patient = QtWidgets.QPushButton(self.groupBox_4)
        self.change_patient.setGeometry(QtCore.QRect(430, 20, 341, 28))
        self.change_patient.setObjectName("change_patient")
        self.number = QtWidgets.QComboBox(self.groupBox_4)
        self.number.setGeometry(QtCore.QRect(100, 60, 111, 22))
        self.number.setObjectName("number")
        self.label = QtWidgets.QLabel(self.groupBox_4)
        self.label.setGeometry(QtCore.QRect(50, 60, 72, 15))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.groupBox_4)
        self.label_2.setGeometry(QtCore.QRect(250, 60, 72, 15))
        self.label_2.setObjectName("label_2")
        self.age = QtWidgets.QLineEdit(self.groupBox_4)
        self.age.setGeometry(QtCore.QRect(300, 60, 113, 21))
        self.age.setText("")
        self.age.setObjectName("age")
        self.gender = QtWidgets.QLineEdit(self.groupBox_4)
        self.gender.setGeometry(QtCore.QRect(500, 60, 113, 21))
        self.gender.setText("")
        self.gender.setObjectName("gender")
        self.label_3 = QtWidgets.QLabel(self.groupBox_4)
        self.label_3.setGeometry(QtCore.QRect(440, 60, 72, 15))
        self.label_3.setObjectName("label_3")
        self.medicine = QtWidgets.QLineEdit(self.groupBox_4)
        self.medicine.setGeometry(QtCore.QRect(710, 60, 113, 21))
        self.medicine.setText("")
        self.medicine.setObjectName("medicine")
        self.label_4 = QtWidgets.QLabel(self.groupBox_4)
        self.label_4.setGeometry(QtCore.QRect(650, 60, 72, 15))
        self.label_4.setObjectName("label_4")
        self.add = QtWidgets.QPushButton(self.groupBox_4)
        self.add.setGeometry(QtCore.QRect(880, 50, 93, 28))
        self.add.setObjectName("add")
        self.groupBox_5 = QtWidgets.QGroupBox(self.page)
        self.groupBox_5.setGeometry(QtCore.QRect(1119, 19, 281, 861))
        self.groupBox_5.setObjectName("groupBox_5")
        self.label_7 = QtWidgets.QLabel(self.groupBox_5)
        self.label_7.setGeometry(QtCore.QRect(20, 30, 151, 41))
        self.label_7.setObjectName("label_7")
        self.label_8 = QtWidgets.QLabel(self.groupBox_5)
        self.label_8.setGeometry(QtCore.QRect(21, 84, 241, 681))
        font = QtGui.QFont()
        font.setFamily("Adobe 宋体 Std L")
        font.setPointSize(14)
        self.label_8.setFont(font)
        self.label_8.setObjectName("label_8")
        self.stackedWidget.addWidget(self.page)
        self.page_3 = QtWidgets.QWidget()
        self.page_3.setObjectName("page_3")
        self.groupBox_2 = QtWidgets.QGroupBox(self.page_3)
        self.groupBox_2.setGeometry(QtCore.QRect(0, 470, 1181, 450))
        self.groupBox_2.setMinimumSize(QtCore.QSize(700, 450))
        self.groupBox_2.setMaximumSize(QtCore.QSize(1600, 600))
        self.groupBox_2.setObjectName("groupBox_2")
        self.label_5 = QtWidgets.QLabel(self.groupBox_2)
        self.label_5.setGeometry(QtCore.QRect(10, 20, 611, 231))
        self.label_5.setObjectName("label_5")
        self.left = QtWidgets.QPushButton(self.page_3)
        self.left.setGeometry(QtCore.QRect(1190, 120, 141, 28))
        self.left.setObjectName("left")
        self.right = QtWidgets.QPushButton(self.page_3)
        self.right.setGeometry(QtCore.QRect(1190, 160, 141, 28))
        self.right.setObjectName("right")
        self.comboBox = QtWidgets.QComboBox(self.page_3)
        self.comboBox.setGeometry(QtCore.QRect(1190, 20, 141, 30))
        self.comboBox.setMaximumSize(QtCore.QSize(200, 30))
        self.comboBox.setObjectName("comboBox")
        self.groupBox = QtWidgets.QGroupBox(self.page_3)
        self.groupBox.setGeometry(QtCore.QRect(0, 0, 1181, 450))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.groupBox.sizePolicy().hasHeightForWidth())
        self.groupBox.setSizePolicy(sizePolicy)
        self.groupBox.setMinimumSize(QtCore.QSize(800, 450))
        self.groupBox.setMaximumSize(QtCore.QSize(1600, 600))
        self.groupBox.setObjectName("groupBox")
        self.img1_3 = QtWidgets.QLabel(self.groupBox)
        self.img1_3.setGeometry(QtCore.QRect(10, 20, 611, 231))
        self.img1_3.setObjectName("img1_3")
        self.groupBox_3 = QtWidgets.QGroupBox(self.page_3)
        self.groupBox_3.setGeometry(QtCore.QRect(1190, 200, 211, 271))
        self.groupBox_3.setObjectName("groupBox_3")
        self.label_6 = QtWidgets.QLabel(self.groupBox_3)
        self.label_6.setGeometry(QtCore.QRect(10, 30, 91, 16))
        self.label_6.setObjectName("label_6")
        self.result_3 = QtWidgets.QLabel(self.groupBox_3)
        self.result_3.setGeometry(QtCore.QRect(11, 54, 181, 211))
        font = QtGui.QFont()
        font.setFamily("Adobe 宋体 Std L")
        font.setPointSize(11)
        self.result_3.setFont(font)
        self.result_3.setObjectName("result_3")
        self.pushButton = QtWidgets.QPushButton(self.page_3)
        self.pushButton.setGeometry(QtCore.QRect(1190, 60, 141, 50))
        self.pushButton.setMaximumSize(QtCore.QSize(200, 50))
        self.pushButton.setObjectName("pushButton")
        self.stackedWidget.addWidget(self.page_3)
        self.gridLayout.addWidget(self.window, 0, 1, 3, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1669, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.stackedWidget.setCurrentIndex(1)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.page_1.setText(_translate("MainWindow", "页面一"))
        self.page_2.setText(_translate("MainWindow", "页面二"))
        item = self.tableWidget.horizontalHeaderItem(0)
        item.setText(_translate("MainWindow", "编号"))
        item = self.tableWidget.horizontalHeaderItem(1)
        item.setText(_translate("MainWindow", "年龄"))
        item = self.tableWidget.horizontalHeaderItem(2)
        item.setText(_translate("MainWindow", "性别"))
        item = self.tableWidget.horizontalHeaderItem(3)
        item.setText(_translate("MainWindow", "药物"))
        self.groupBox_4.setTitle(_translate("MainWindow", "GroupBox"))
        self.remove_patient.setText(_translate("MainWindow", "删除病人信息"))
        self.change_patient.setText(_translate("MainWindow", "修改病人信息"))
        self.label.setText(_translate("MainWindow", "编号："))
        self.label_2.setText(_translate("MainWindow", "年龄："))
        self.label_3.setText(_translate("MainWindow", "性别："))
        self.label_4.setText(_translate("MainWindow", "药物："))
        self.add.setText(_translate("MainWindow", "添加"))
        self.groupBox_5.setTitle(_translate("MainWindow", "GroupBox"))
        self.label_7.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:16pt;\">查询结果：</span></p></body></html>"))
        self.label_8.setText(_translate("MainWindow", "<html><head/><body><p align=\"justify\"><span style=\" font-size:12pt;\">TextLabel</span></p></body></html>"))
        self.groupBox_2.setTitle(_translate("MainWindow", "GroupBox"))
        self.label_5.setText(_translate("MainWindow", "TextLabel"))
        self.left.setText(_translate("MainWindow", "左移"))
        self.right.setText(_translate("MainWindow", "右移"))
        self.groupBox.setTitle(_translate("MainWindow", "GroupBox"))
        self.img1_3.setText(_translate("MainWindow", "TextLabel"))
        self.groupBox_3.setTitle(_translate("MainWindow", "GroupBox"))
        self.label_6.setText(_translate("MainWindow", "经检查发现："))
        self.result_3.setText(_translate("MainWindow", "TextLabel"))
        self.pushButton.setText(_translate("MainWindow", "检测"))
