# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'register.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_registerWindow(object):
    def setupUi(self, registerWindow):
        registerWindow.setObjectName("registerWindow")
        registerWindow.resize(385, 203)
        self.centralwidget = QtWidgets.QWidget(registerWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.formLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.formLayoutWidget.setGeometry(QtCore.QRect(50, 20, 261, 101))
        self.formLayoutWidget.setObjectName("formLayoutWidget")
        self.formLayout = QtWidgets.QFormLayout(self.formLayoutWidget)
        self.formLayout.setContentsMargins(0, 0, 0, 0)
        self.formLayout.setObjectName("formLayout")
        self.name = QtWidgets.QLabel(self.formLayoutWidget)
        self.name.setObjectName("name")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.name)
        self.name_edit = QtWidgets.QLineEdit(self.formLayoutWidget)
        self.name_edit.setObjectName("name_edit")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.name_edit)
        self.male = QtWidgets.QRadioButton(self.formLayoutWidget)
        self.male.setObjectName("male")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.male)
        self.female = QtWidgets.QRadioButton(self.formLayoutWidget)
        self.female.setObjectName("female")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.female)
        self.phone_number = QtWidgets.QLabel(self.formLayoutWidget)
        self.phone_number.setObjectName("phone_number")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.phone_number)
        self.phone_edit = QtWidgets.QLineEdit(self.formLayoutWidget)
        self.phone_edit.setObjectName("phone_edit")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.phone_edit)
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(240, 140, 71, 31))
        self.pushButton.setObjectName("pushButton")
        registerWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(registerWindow)
        self.statusbar.setObjectName("statusbar")
        registerWindow.setStatusBar(self.statusbar)

        self.retranslateUi(registerWindow)
        self.pushButton.clicked.connect(registerWindow.registerFace)
        QtCore.QMetaObject.connectSlotsByName(registerWindow)

    def retranslateUi(self, registerWindow):
        _translate = QtCore.QCoreApplication.translate
        registerWindow.setWindowTitle(_translate("registerWindow", "childWindow"))
        self.name.setText(_translate("registerWindow", "姓名："))
        self.male.setText(_translate("registerWindow", "男"))
        self.female.setText(_translate("registerWindow", "女"))
        self.phone_number.setText(_translate("registerWindow", "手机号码："))
        self.pushButton.setText(_translate("registerWindow", "注册"))